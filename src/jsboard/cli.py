"""Command line entry point.

    jsboard sim                     synthetic market, live board
    jsboard live --symbol BTCUSDT   real Binance depth, paper fills
    jsboard record --out f.jsonl    capture a live session
    jsboard replay f.jsonl          replay a capture
    jsboard backtest                headless run, prints the P&L summary

Nothing here places a real order. `live` uses genuine Binance market data and
fills against the public tape through the paper venue; there is no exchange
credential anywhere in this project and no order-entry path to a real venue.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import logging
import math
import sys
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .core.market import MarketView
from .core.types import Instrument, Side
from .feed import multi
from .feed.base import DepthDelta, DepthSnapshot, Feed, TradeTick
from .feed.binance import BinanceFeed
from .feed.binance_futures import FALLBACK_MODES, BinanceFuturesFeed
from .feed.replay import JsonlRecorder, ReplayFeed, SyntheticFeed, iter_tagged
from .mm.fair_value import FairValueConfig, FairValueEstimator
from .mm.inventory import FeeSchedule, Position
from .mm.quoter import Quoter, QuoterConfig
from .mm.risk import RiskLimits, RiskManager
from .mm.strategy import MarketMaker, StrategyConfig
from .net import describe_tls_error, make_session
from .research import dynamic, triage
from .research.archive import days_ending, fetch_day, load_seconds
from .research.events import combine, score, simulate, threshold
from .research.features import attach_open_interest, build, load_open_interest, to_minutes
from .research.horizon import analyse, round_trip_cost_bps
from .research.predict import (
    Series,
    always_long,
    candidates,
    evaluate,
    split,
    volatility_threshold,
)
from .research.scan import (
    ScanFilters,
    fetch_market,
    fetch_tick_sizes,
    rank_persistence,
    scan,
    summarise,
    watch,
)
from .sim.capture import MultiCapture, write_meta
from .sim.hedge import HedgeConfig, Hedger
from .sim.paper import PAPER_OWNER, PaperConfig, PaperVenue
from .sim.runner import attach_virtual_clock, run
from .ui.board import Board

console = Console()

# Tick/lot sizes for the usual suspects, so `sim` works with no network.
KNOWN_INSTRUMENTS = {
    "BTCUSDT": ("0.01", "0.00001", "BTC", "USDT"),
    "ETHUSDT": ("0.01", "0.0001", "ETH", "USDT"),
    "SOLUSDT": ("0.01", "0.001", "SOL", "USDT"),
    "XRPUSDT": ("0.0001", "1", "XRP", "USDT"),
}
DEFAULT_SPEC = ("0.01", "0.00001", "", "")


def build_instrument(symbol: str, tick: str | None, lot: str | None) -> Instrument:
    spec = KNOWN_INSTRUMENTS.get(symbol.upper(), DEFAULT_SPEC)
    return Instrument(
        symbol=symbol.upper(),
        tick_size=Decimal(tick or spec[0]),
        lot_size=Decimal(lot or spec[1]),
        base=spec[2],
        quote=spec[3],
    )


async def fetch_instrument(symbol: str) -> Instrument:
    """Read the real tick/lot size from Binance exchangeInfo."""
    import aiohttp

    url = "https://api.binance.com/api/v3/exchangeInfo"
    async with make_session() as session, session.get(
        url, params={"symbol": symbol.upper()}, timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    info = payload["symbols"][0]
    tick = lot = None
    for f in info["filters"]:
        if f["filterType"] == "PRICE_FILTER":
            tick = f["tickSize"]
        elif f["filterType"] == "LOT_SIZE":
            lot = f["stepSize"]
    if tick is None or lot is None:
        raise RuntimeError(f"exchangeInfo for {symbol} had no price/lot filter")

    return Instrument(
        symbol=info["symbol"],
        tick_size=Decimal(tick).normalize(),
        lot_size=Decimal(lot).normalize(),
        base=info["baseAsset"],
        quote=info["quoteAsset"],
    )


async def fetch_futures_instrument(symbol: str) -> Instrument:
    """Tick and lot for the USDⓈ-M perpetual.

    They differ from spot for the same ticker — BTCUSDT is 0.01/0.00001 on
    spot and 0.1/0.001 on the perp — so the two must be fetched separately.
    The futures exchangeInfo takes no symbol filter, so the whole list comes
    back and is searched locally.
    """
    import aiohttp

    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    async with make_session() as session, session.get(
        url, timeout=aiohttp.ClientTimeout(total=30)
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()

    wanted = symbol.upper()
    for info in payload["symbols"]:
        if info["symbol"] != wanted:
            continue
        tick = lot = None
        for f in info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick = f["tickSize"]
            elif f["filterType"] == "LOT_SIZE":
                lot = f["stepSize"]
        if tick is None or lot is None:
            raise RuntimeError(f"futures exchangeInfo for {wanted} had no price/lot filter")
        return Instrument(
            symbol=info["symbol"],
            tick_size=Decimal(tick).normalize(),
            lot_size=Decimal(lot).normalize(),
            base=info["baseAsset"],
            quote=info["quoteAsset"],
        )
    raise RuntimeError(f"{wanted} is not listed on USDⓈ-M futures")


class ConfigError(Exception):
    """A setting that would make the session do nothing, caught at startup."""


def _check_sizes(instrument: Instrument, args: argparse.Namespace) -> None:
    """Refuse sizes that round away to nothing.

    Quantities round *down* to the lot size, so an order smaller than one lot
    becomes zero and is never placed. The defaults are BTC-shaped: 0.01 is a
    sensible order there and is below the minimum on WIFUSDT, whose lot is
    0.1. Left unchecked this runs for as long as it is asked to, reports
    "QUOTE: ok" every cycle, and places nothing — which is the hardest kind
    of failure to read.
    """
    lot = instrument.lot_size
    size_lots = instrument.to_lots(args.size)
    position_lots = instrument.to_lots(args.max_position)

    if size_lots <= 0:
        suggestion = f"{lot * 100:g}"
        raise ConfigError(
            f"--size {args.size} は {instrument.symbol} の最小単位 {lot:g} 未満です。\n"
            f"  数量は切り捨てられるので、注文は0になり一度も置かれません。\n"
            f"  --size {suggestion} 以上を指定してください。"
        )
    if position_lots < size_lots:
        raise ConfigError(
            f"--max-position {args.max_position} が --size {args.size} より小さいため、\n"
            f"  最初の1枚も建てられません。建玉上限を広げてください。"
        )


def build_maker(instrument: Instrument, args: argparse.Namespace) -> MarketMaker:
    _check_sizes(instrument, args)
    market = MarketView(instrument=instrument, depth=args.depth)

    quoter = Quoter(
        QuoterConfig(
            gamma=args.gamma,
            kappa=args.kappa,
            levels=args.levels,
            level_step_ticks=args.level_step,
            base_size_lots=instrument.to_lots(args.size),
            max_position_lots=instrument.to_lots(args.max_position),
            min_half_spread_ticks=args.min_half_spread,
            max_distance_ticks=getattr(args, "max_distance", None),
            # Quote at least wide enough to cover what the venue charges us.
            min_edge_bps=args.min_edge_bps if args.min_edge_bps is not None else args.maker_bps,
        )
    )
    risk = RiskManager(
        limits=RiskLimits(
            max_position_lots=instrument.to_lots(args.max_position),
            max_notional=args.max_notional,
            max_drawdown=args.max_drawdown,
        )
    )
    venue = PaperVenue(
        instrument=instrument,
        config=PaperConfig(latency_ms=args.latency_ms, cancel_ahead_ratio=args.cancel_ahead),
    )
    position = Position(
        instrument=instrument,
        fees=FeeSchedule(maker_bps=args.maker_bps, taker_bps=args.taker_bps),
    )

    return MarketMaker(
        instrument=instrument,
        market=market,
        venue=venue,
        position=position,
        quoter=quoter,
        fair_value=FairValueEstimator(FairValueConfig()),
        risk=risk,
        config=StrategyConfig(requote_interval_ms=args.requote_ms),
    )


async def drive(feed: Feed, mm: MarketMaker, args: argparse.Namespace, *, headless: bool) -> None:
    if headless:
        result = await run(feed, mm, duration_s=args.duration, max_events=args.max_events)
        _print_report(mm, result)
        return

    board = Board(mm)
    with Live(board.render(), console=console, refresh_per_second=12, screen=True) as live:
        def on_update(_mm: MarketMaker, _event) -> None:
            live.update(board.render())

        result = await run(
            feed, mm, duration_s=args.duration, max_events=args.max_events, on_update=on_update
        )
    _print_report(mm, result)


def _price_decimals(instrument) -> int:
    """Enough digits to show a tick. A cent default hides most alt prices."""
    return max(2, -instrument.tick_size.as_tuple().exponent)


def _capture_line(mm: MarketMaker, s: dict) -> str | None:
    """Gross spread capture per round trip, against the fee that eats it.

    This is the number the whole exercise turns on. Total P&L mixes three
    different things — spread captured, fees paid, and an open position
    marked to market — and only the first is the edge. Isolating it says
    whether the strategy loses because it has no edge or because the edge is
    smaller than the fee, which are not the same problem.
    """
    matched = min(s["bought"], s["sold"])
    mid_ticks = s.get("mid_ticks")
    if matched <= 0 or not mid_ticks:
        return None
    mid = float(mid_ticks) * float(mm.instrument.tick_size)
    closed_notional = matched * mid
    if closed_notional <= 0:
        return None

    gross_bps = s["gross"] / closed_notional * 10_000.0
    fee_bps = 2.0 * mm.position.fees.maker_bps
    edge = gross_bps - fee_bps
    colour = "green" if edge > 0 else "red"
    return (
        f"  spread capture : {gross_bps:+.2f} bps gross per round trip "
        f"vs {fee_bps:.2f} bps of fees  "
        f"[{colour}]({edge:+.2f} bps)[/{colour}]"
    )


def _markout_line(s: dict) -> str | None:
    """Adverse selection: how far the mid moved against us after each fill.

    Measured mid-to-mid, so the spread we earned is excluded — that half
    lives in the attribution's spread-capture term instead. Reading a
    fill-referenced mark-out as adverse selection double-counts the spread
    and, on a wide-tick symbol, flips the sign.
    """
    windows = s.get("markout") or []
    parts = []
    for w in windows:
        # Sub-second horizons round to "0s" under a seconds format, which
        # reads as no horizon at all.
        h = w["horizon_s"]
        label = f"{h * 1000:.0f}ms" if h < 1 else f"{h:.0f}s"
        mean = w["mean_bps"]
        if math.isnan(mean):
            parts.append(f"+{label} n/a")
            continue
        colour = "green" if mean >= 0 else "red"
        parts.append(f"+{label} [{colour}]{mean:+.2f}[/{colour}]bps(n={int(w['n'])})")
    return "  逆選択(mid基準): " + "  ".join(parts) if parts else None


def _attribution_lines(mm: MarketMaker, s: dict) -> list[str]:
    """The identity the whole diagnosis rests on, in currency and in bps."""
    a = s.get("attribution")
    if not a:
        return []

    quote = mm.instrument.quote
    lines = [
        f"  P&L 分解       : スプレッド取り {a['spread_capture']:+,.2f}"
        f"  在庫 {a['inventory']:+,.2f}"
        f"  手数料 {-a['fees']:+,.2f}"
        f"  = {a['total']:+,.2f} {quote}"
    ]
    if a.get("unpriced_fills"):
        lines.append(
            f"  [yellow]うち {int(a['unpriced_fills'])} 件はミッド不明のため分解できていません[/yellow]"
        )

    matched = min(s["bought"], s["sold"])
    bps = mm.attribution.per_round_trip_bps(matched)
    if bps:
        colour = "green" if bps["total"] > 0 else "red"
        lines.append(
            f"  往復あたり     : スプレッド {bps['spread_capture']:+.2f}"
            f"  在庫 {bps['inventory']:+.2f}"
            f"  手数料 {bps['fees']:+.2f}"
            f"  = [{colour}]{bps['total']:+.2f} bps[/{colour}]"
        )
        buckets = "  ".join(
            f"{label} {value:+.2f}"
            for label, _, value in mm.attribution.age_buckets(matched)
        )
        lines.append(f"  在庫の内訳     : {buckets}  bps (保有時間別)")
        ceiling = mm.attribution.max_maker_bps(matched)
        lines.append(
            f"  許容メイカー料 : [bold]{ceiling:+.2f} bps/片道[/bold]"
            f"  (現在 {mm.position.fees.maker_bps:g})"
        )
    return lines


def _reach_lines(mm: MarketMaker, s: dict) -> list[str]:
    """Why the fills did or did not happen.

    A session that reports zero fills has said nothing yet: the tape may
    never have come to our price, or it may have come repeatedly and been
    eaten by the queue standing in front of us. The first is fixed by
    quoting tighter, the second only by size, time, or a different symbol.
    """
    placement = s.get("placement") or {}
    total = sum(placement.values())
    if not total:
        return []

    inside = sum(n for d, n in placement.items() if d < 0)
    at_touch = placement.get(0, 0)
    behind = total - inside - at_touch
    ratio = s.get("queue_ahead_ratio", math.nan)
    queue = "—" if math.isnan(ratio) else f"{ratio:,.1f}x our size"

    prints = s.get("prints_seen", 0)
    reached = s.get("prints_at_our_price", 0)
    share = f"{reached / prints * 100:.1f}%" if prints else "—"

    return [
        f"  quote placement: inside {inside / total * 100:.0f}% / "
        f"at touch {at_touch / total * 100:.0f}% / behind {behind / total * 100:.0f}%"
        f"   queue ahead at touch: {queue}",
        f"  tape reach     : {reached:,} of {prints:,} prints ({share}) came to our price, "
        f"{s.get('queue_absorbed', 0):,.0f} {mm.instrument.base} of it absorbed ahead of us",
    ]


def _print_report(mm: MarketMaker, result) -> None:
    s = mm.summary()
    inst = mm.instrument
    dp = _price_decimals(inst)
    console.print()
    console.rule(f"[bold cyan]{inst.symbol} — session report")
    console.print(
        f"  stopped        : {result.stopped_because}\n"
        f"  elapsed        : {result.elapsed_s:,.1f}s over {result.events:,} events\n"
        f"  quote cycles   : {s['cycles']:,}  "
        f"(placed {s['placed']:,} / cancelled {s['cancelled']:,} / kept {s['kept']:,})\n"
        f"  fills          : {int(s['fills']):,}  volume {s['volume']:,.5f} {inst.base}  "
        f"(bought {s['bought']:,.5f} / sold {s['sold']:,.5f})\n"
        f"  position       : {s['position']:+,.5f} {inst.base} @ {s['avg_price']:,.{dp}f}\n"
        f"  realized P&L   : {s['realized']:+,.2f} {inst.quote}  "
        f"(gross {s['gross']:+,.2f} less {s['fees']:,.2f} fees)\n"
        f"  unrealized P&L : {s['unrealized']:+,.2f} {inst.quote}\n"
        f"  [bold]total P&L      : {s['total']:+,.2f} {inst.quote}[/bold]"
    )
    for line in (
        _capture_line(mm, s),
        *_attribution_lines(mm, s),
        _markout_line(s),
        *_reach_lines(mm, s),
    ):
        if line:
            console.print(line)
    console.print(f"  last decision  : {s['decision']}")
    console.rule()


# ---------------------------------------------------------------- commands


async def cmd_sim(args: argparse.Namespace) -> int:
    instrument = build_instrument(args.symbol, args.tick_size, args.lot_size)
    feed = SyntheticFeed(
        instrument,
        start_price=args.start_price,
        tick_interval=args.interval,
        volatility_bps=args.volatility,
        seed=args.seed,
    )
    mm = build_maker(instrument, args)
    if not args.interval:
        # No wall-clock delay between events, so run the whole maker on the
        # feed's simulated timeline instead.
        attach_virtual_clock(mm)
    await drive(feed, mm, args, headless=args.headless)
    return 0


async def _market_feed(args: argparse.Namespace) -> tuple[Instrument, Feed]:
    """The instrument and its live feed, for whichever product was asked for.

    Spot and perp differ in tick and lot for the same ticker — BTCUSDT steps
    by 0.01 on spot and 0.1 on the perp — so the spec has to come from the
    matching exchangeInfo. Quoting a perp against spot's tick would place
    every order at a price the venue rejects.
    """
    perp = getattr(args, "product", "spot") == "perp"
    try:
        instrument = (
            await fetch_futures_instrument(args.symbol)
            if perp
            else await fetch_instrument(args.symbol)
        )
        console.print(
            f"[dim]{'perp' if perp else 'spot'} exchangeInfo: "
            f"tick={instrument.tick_size} lot={instrument.lot_size}[/dim]"
        )
    except Exception as exc:  # noqa: BLE001
        if perp:
            # No built-in perp specs exist, and spot's would be wrong. Better
            # to stop than to quote against a tick size from the other book.
            raise
        console.print(f"[yellow]exchangeInfo unavailable ({exc}); using built-in spec[/yellow]")
        instrument = build_instrument(args.symbol, args.tick_size, args.lot_size)

    if perp:
        depth_ms = args.depth_ms if args.depth_ms in (100, 250, 500) else 100
        return instrument, BinanceFuturesFeed(instrument, depth_ms=depth_ms)
    return instrument, BinanceFeed(instrument, depth_ms=args.depth_ms)


async def cmd_live(args: argparse.Namespace) -> int:
    instrument, feed = await _market_feed(args)
    try:
        mm = build_maker(instrument, args)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    await drive(feed, mm, args, headless=args.headless)
    return 0


async def cmd_record(args: argparse.Namespace) -> int:
    instrument, feed = await _market_feed(args)
    out = Path(args.out)
    count = 0

    console.print(f"[cyan]recording {instrument.symbol} → {out}[/cyan]  (ctrl-c to stop)")
    with JsonlRecorder(out) as rec:
        stream = feed.stream()
        try:
            async for event in stream:
                rec.write(event)
                count += 1
                if count % 100 == 0:
                    console.print(f"[dim]  {count:,} events[/dim]", end="\r")
                if args.max_events and count >= args.max_events:
                    break
                if args.duration and count and getattr(event, "ts_ns", 0):
                    pass
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            with contextlib.suppress(Exception):
                await stream.aclose()

    # Store the instrument alongside the capture so replay needs no guesswork.
    meta = out.with_suffix(out.suffix + ".meta.json")
    meta.write_text(
        json.dumps(
            {
                "symbol": instrument.symbol,
                "tick_size": str(instrument.tick_size),
                "lot_size": str(instrument.lot_size),
                "base": instrument.base,
                "quote": instrument.quote,
                "events": count,
            },
            indent=2,
        )
    )
    console.print(f"\n[green]wrote {count:,} events to {out}[/green] (meta: {meta.name})")
    return 0


async def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.path)
    instrument = _instrument_for_recording(path, args)
    feed = ReplayFeed(instrument, path, speed=args.speed, source=args.source)
    mm = build_maker(instrument, args)
    # A recording carries the timestamps it was captured with. Judged against
    # the wall clock those are always in the past — a day-old capture reads as
    # a book that is a day stale, and the risk gate pulls every quote before
    # one is ever placed. "Now", during a replay, is the timestamp of the
    # event being replayed.
    attach_virtual_clock(mm)
    await drive(feed, mm, args, headless=args.headless)
    return 0


def _instrument_from_spec(spec: dict) -> Instrument:
    return Instrument(
        symbol=spec["symbol"],
        tick_size=Decimal(spec["tick_size"]),
        lot_size=Decimal(spec["lot_size"]),
        base=spec.get("base", ""),
        quote=spec.get("quote", ""),
    )


def _instrument_for_recording(path: Path, args: argparse.Namespace) -> Instrument:
    """Prefer the spec the recorder saved; fall back to the CLI overrides.

    Two recorders write two shapes. `record` stores one instrument at the top
    level; `capture` stores one per venue under "sources", because spot and
    perp do not share a tick size and guessing wrong silently rescales every
    price in the file.
    """
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        return build_instrument(args.symbol, args.tick_size, args.lot_size)

    meta = json.loads(meta_path.read_text())
    sources = meta.get("sources")
    if not sources:
        return _instrument_from_spec(meta)

    wanted = getattr(args, "source", None)
    if wanted is None:
        if len(sources) == 1:
            return _instrument_from_spec(next(iter(sources.values())))
        raise ConfigError(
            f"{meta_path.name} は {', '.join(sorted(sources))} を含んでいます。\n"
            f"  --source でどちらを再生するか指定してください（例: --source perp）。"
        )
    if wanted not in sources:
        raise ConfigError(
            f"--source {wanted} は {meta_path.name} にありません。"
            f"  使えるのは: {', '.join(sorted(sources))}"
        )
    return _instrument_from_spec(sources[wanted])


def _grid(spec: str, cast):
    """Parse a comma-separated sweep axis. "none" means the uncapped setting."""
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(None if piece.lower() in {"none", "off"} else cast(piece))
    return out


def _cast_like(current) -> object:
    """Cast a sweep value the way the parser would have cast the default.

    Sizes are carried as strings so Decimal keeps them exact, floats stay
    floats, and a setting whose default is None gets the narrowest type its
    text will support.
    """
    if isinstance(current, bool):
        return lambda s: s.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int
    if isinstance(current, float):
        return float
    if isinstance(current, str):
        return str

    def guess(s: str):
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                continue
        return s

    return guess


def _sweep_axes(args: argparse.Namespace) -> dict[str, list]:
    """Every setting being varied, as {argument name: values to try}.

    Common quote-placement and latency experiments have shorthands;
    `--axis` reaches every other setting. A name that is not an actual setting
    is refused rather
    than ignored, since a typo would otherwise run the same configuration N
    times and read as a result.
    """
    axes: dict[str, list] = {}
    # Preserve the historical default (distance sweep) only when the caller
    # selected no other axis. An explicit latency experiment must not silently
    # multiply into five quote-distance variants.
    distances = args.distances
    if distances is None and not any(
        (args.sizes, getattr(args, "requotes", ""), getattr(args, "latencies", ""), args.axis)
    ):
        distances = "0,1,2,4,none"
    if distances:
        axes["max_distance"] = _grid(distances, int)
    if args.sizes:
        axes["size"] = _grid(args.sizes, str)
    if getattr(args, "requotes", ""):
        axes["requote_ms"] = _grid(args.requotes, float)
    if getattr(args, "latencies", ""):
        axes["latency_ms"] = _grid(args.latencies, float)

    for spec in args.axis or []:
        name, sep, values = spec.partition("=")
        name = name.strip().replace("-", "_")
        if not sep or not values.strip():
            raise ConfigError(f"--axis {spec} は name=v1,v2 の形で指定してください。")
        if not hasattr(args, name):
            raise ConfigError(
                f"--axis {name} という設定はありません。"
                f"  `jsboard sweep --help` で名前を確認してください。"
            )
        axes[name] = _grid(values, _cast_like(getattr(args, name)))
    return axes


def _label(name: str, value) -> str:
    if name == "max_distance":
        return "touch" if value == 0 else ("none" if value is None else str(value))
    return "none" if value is None else str(value)


def _markout_at(windows: list[dict], horizon_s: float) -> float:
    """Adverse selection at one horizon, by value rather than by position.

    Indexing into the list would silently return a different horizon the
    moment the default set changes, which it just did.
    """
    for w in windows:
        if abs(w["horizon_s"] - horizon_s) < 1e-9:
            return w["mean_bps"]
    return math.nan


def _sweep_row(mm: MarketMaker, s: dict) -> dict:
    """Reduce one replay to the handful of numbers worth comparing."""
    matched = min(s["bought"], s["sold"])
    mid_ticks = s.get("mid_ticks")
    capture = math.nan
    if matched > 0 and mid_ticks:
        notional = matched * float(mid_ticks) * float(mm.instrument.tick_size)
        if notional > 0:
            capture = s["gross"] / notional * 10_000.0
    placement = s.get("placement") or {}
    total = sum(placement.values()) or 1
    at_touch = sum(n for d, n in placement.items() if d <= 0)
    markout = s.get("markout") or []
    bps = mm.attribution.per_round_trip_bps(matched)
    a = s.get("attribution") or {}
    return {
        "fills": int(s["fills"]),
        "capture_bps": capture,
        "net_bps": capture - 2.0 * mm.position.fees.maker_bps,
        "realized": s["realized"],
        "total": s["total"],
        "touch_share": at_touch / total * 100.0,
        "spread_bps": bps.get("spread_capture", math.nan),
        "inventory_bps": bps.get("inventory", math.nan),
        "pre_fee_bps": bps.get("net_before_fees", math.nan),
        "fee_bps": bps.get("fees", math.nan),
        "attributed_bps": bps.get("total", math.nan),
        "markout_100ms": _markout_at(markout, 0.1),
        "markout_1s": _markout_at(markout, 1.0),
        "markout_10s": _markout_at(markout, 10.0),
        "exposed_share": a.get("exposed_share", math.nan) * 100.0,
        "max_maker_bps": mm.attribution.max_maker_bps(matched),
        **{
            f"age_{i}": value
            for i, (_, _, value) in enumerate(mm.attribution.age_buckets(matched))
        },
    }


async def cmd_sweep(args: argparse.Namespace) -> int:
    """Replay one recording under many settings.

    A live session yields one data point per hour of waiting, which is far
    too slow to choose a quote width or a size. A recording can be replayed
    as many times as there are settings to try, against the identical market,
    so the comparison is a controlled one rather than a comparison of two
    different hours.
    """
    path = Path(args.path)
    if not path.exists():
        console.print(f"[red]{path} がありません。まず capture で録画してください。[/red]")
        return 1

    instrument = _instrument_for_recording(path, args)
    axes = _sweep_axes(args)
    if not axes:
        raise ConfigError(
            "掃引する軸がありません。--distances / --requotes / --latencies / --axis "
            "のいずれかを指定してください。"
        )

    names = list(axes)
    combos = list(itertools.product(*(axes[n] for n in names)))

    console.print(
        f"{path.name}: {instrument.symbol}  tick={instrument.tick_size} lot={instrument.lot_size}\n"
        f"{len(combos)} 通りを同じ録画に対して再生します "
        f"(maker {args.maker_bps:g}bps → 往復 {2 * args.maker_bps:g}bps)\n"
        f"軸: {', '.join(f'{n}={len(axes[n])}' for n in names)}"
    )

    rows = []
    for combo in combos:
        settings = dict(zip(names, combo, strict=True))
        run_args = argparse.Namespace(**vars(args))
        for name, value in settings.items():
            setattr(run_args, name, value)
        shown = "  ".join(f"{n}={_label(n, v)}" for n, v in settings.items())
        try:
            _check_sizes(instrument, run_args)
        except ConfigError as exc:
            console.print(f"[yellow]skip {shown}: {exc}[/yellow]")
            continue

        mm = build_maker(instrument, run_args)
        attach_virtual_clock(mm)
        feed = ReplayFeed(instrument, path, speed=0.0, source=args.source)
        await run(feed, mm, duration_s=None, max_events=args.max_events)
        row = _sweep_row(mm, mm.summary())
        row["settings"] = settings
        rows.append(row)
        console.print(f"  {shown}  fills={row['fills']:,}")

    if not rows:
        console.print("[red]走れた組み合わせがありません。[/red]")
        return 1

    # Best net edge first; rows that never filled have nothing to rank and go last.
    rows.sort(key=lambda r: (math.isnan(r["net_bps"]), -(r["net_bps"] or 0.0)))
    columns = [
        ("約定", "fills", "{:,.0f}"),
        ("spread", "spread_bps", "{:+.2f}"),
        ("在庫", "inventory_bps", "{:+.2f}"),
        ("手数料前", "pre_fee_bps", "{:+.2f}"),
        ("手数料", "fee_bps", "{:+.2f}"),
        ("手数料後", "attributed_bps", "{:+.2f}"),
        ("mo100ms", "markout_100ms", "{:+.2f}"),
        ("mo1s", "markout_1s", "{:+.2f}"),
        ("mo10s", "markout_10s", "{:+.2f}"),
        ("在庫時間%", "exposed_share", "{:.0f}"),
        ("在庫0-100ms", "age_0", "{:+.2f}"),
        ("100ms-1s", "age_1", "{:+.2f}"),
        ("1-10s", "age_2", "{:+.2f}"),
        ("10s+", "age_3", "{:+.2f}"),
        ("許容料率", "max_maker_bps", "{:+.2f}"),
    ]

    def cell(row: dict, key: str, fmt: str) -> str:
        value = row.get(key, math.nan)
        return "—" if isinstance(value, float) and math.isnan(value) else fmt.format(value)

    if args.plain:
        # A rich table of this width wraps in most terminals, which makes the
        # numbers unreadable and unpasteable. Tab-separated survives both.
        head = [*names, *(c[0] for c in columns)]
        console.print("\t".join(head), highlight=False)
        for r in rows:
            cells = [_label(n, r["settings"][n]) for n in names]
            cells += [cell(r, key, fmt) for _, key, fmt in columns]
            console.print("\t".join(cells), highlight=False)
        return 0

    table = Table(title=f"{instrument.symbol} — sweep ({path.name})", padding=(0, 1))
    for name in names:
        table.add_column(name.replace("_", " "), justify="right")
    for header, _, _ in columns:
        table.add_column(header, justify="right")

    for r in rows:
        attributed = r["attributed_bps"]
        colour = "dim" if math.isnan(attributed) else ("green" if attributed > 0 else "red")
        cells = [_label(n, r["settings"][n]) for n in names]
        for _, key, fmt in columns:
            text = cell(r, key, fmt)
            cells.append(f"[{colour}]{text}[/{colour}]" if key == "attributed_bps" else text)
        table.add_row(*cells)
    console.print(table)
    console.print(
        "[dim]同じ1時間に対する再生なので、行どうしの比較は公平です。"
        "ただし約定数が二桁に届かない行は、まだ数字として読めません。[/dim]"
    )
    return 0


async def cmd_probe(args: argparse.Namespace) -> int:
    """Measure every candidate at once: does the price outrun the spread?

    The static screen cannot see speed, and USUSDT cost an hour of recording
    to establish that. Watching the whole shortlist for five minutes settles
    the same question for all of them at once, and no paper maker runs: every
    public print is a fill a maker at the front of that queue would have had,
    so the tape alone gives maker-side adverse selection with no fill model
    in the way.
    """
    if args.replay:
        return await _probe_replay(args)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        console.print("[dim]triage で候補を絞っています…[/dim]")
        symbols = await _triage_survivors(args)
        if not symbols:
            console.print("[red]triage の候補がゼロです。動的審査に進む対象がありません。[/red]")
            return 1

    console.print(
        f"{len(symbols)} 銘柄を {args.duration:,.0f} 秒間まとめて観測します"
        f" (逆選択の地平 {args.horizon_ms:g}ms)\n"
        f"  [dim]{', '.join(symbols)}[/dim]"
    )

    probes = {s: dynamic.SymbolProbe(s, horizon_s=args.horizon_ms / 1000.0) for s in symbols}
    seen = 0
    try:
        async for event in multi.stream(symbols, duration_s=args.duration):
            probe = probes.get(event.symbol)
            if probe is None:
                continue
            if isinstance(event, multi.Quote):
                probe.on_book(
                    event.bid, event.ask, event.ts_ns,
                    bid_qty=event.bid_qty, ask_qty=event.ask_qty,
                )
            else:
                probe.on_trade(event.aggressor_sign, event.ts_ns, qty=event.qty)
            seen += 1
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]観測に失敗しました: {exc}[/red]")
        hint = describe_tls_error(exc)
        if hint:
            console.print(f"[yellow]{hint}[/yellow]")
        return 1

    rows = list(probes.values())
    gates = {"min_trades": args.min_trades, "min_sweeps": args.min_sweeps}
    counts = dynamic.tally(rows, **gates)

    console.print()
    console.rule("[bold cyan]動的審査 — 100ms で逆選択がスプレッドを超えるか")
    console.print(
        f"  {seen:,} イベント  →  " + "  ".join(f"{k} {v:,}" for k, v in counts.items())
    )

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("symbol", style="cyan")
    table.add_column("spread\n(bps)", justify="right")
    table.add_column(f"逆選択\n{args.horizon_ms:g}ms", justify="right")
    table.add_column("MO/spread\n(全約定)", justify="right")
    table.add_column("MO/spread\n(板を消した分)", justify="right")
    table.add_column("約定", justify="right")
    table.add_column("うち\n板消し", justify="right")
    table.add_column("判定")

    style = {"研究候補": "green", "見込み薄": "yellow", "不可": "red"}
    def cell(value: float, fmt: str = "{:,.2f}") -> str:
        return "—" if math.isnan(value) else fmt.format(value)

    for probe in dynamic.rank(rows, **gates)[: args.top]:
        verdict = probe.verdict(**gates)
        colour = style.get(verdict, "dim")
        decisive = probe.decisive_ratio(min_sweeps=args.min_sweeps)
        sweep = probe.sweep_ratio
        sweep_text = cell(sweep)
        if not math.isnan(sweep) and probe.sweeps_settled >= args.min_sweeps:
            sweep_text = f"[{colour}]{sweep_text}[/{colour}]"
        table.add_row(
            probe.symbol,
            cell(probe.spread_bps),
            cell(probe.sweep_markout_bps if decisive is sweep else probe.markout_bps, "{:+,.2f}"),
            cell(probe.ratio),
            sweep_text,
            f"{probe.settled:,}",
            f"{probe.sweeps_settled:,}",
            f"[{colour}]{verdict}[/{colour}]",
        )
    console.print()
    console.print(table)

    measured = len(rows) - counts["サンプル不足"]
    if measured == 0:
        # The stop condition must not fire on a failed measurement. Nothing
        # was observed, so nothing was ruled out.
        console.print(
            "\n[yellow]測定できた銘柄がゼロです。判定は出ていません。[/yellow]\n"
            "[dim]--duration を伸ばすか --min-trades を下げてください。"
            "全銘柄で約定ゼロなら、板に到達できていません。[/dim]"
        )
    elif counts["研究候補"] == 0:
        console.print(
            f"\n[red][bold]研究候補ゼロ（{measured} 銘柄を測定）。"
            "100ms で板に貼りつく MM はここで終了。[/bold][/red]\n"
            f"[dim]どの銘柄も、スプレッドが払う以上に価格が {args.horizon_ms:g}ms で動いています。"
            "置き場所・サイズ・在庫規則では届きません。[/dim]"
        )
    else:
        console.print(
            "\n[dim]判定は「板を消した分」の列で行っています — メイカーが実際に約定するのは"
            "前に並んだ枚数を食い切る約定だけで、それは最も攻撃的な側だからです。"
            "全約定の列はその楽観側の上限で、2列が開いている銘柄ほど情報が一気に来る板です。\n"
            "小さいことは必要条件でしかありません。手数料・在庫・約定モデルの誤差の余白が"
            "まだ引かれていないので、候補は capture → sweep → hedge の全コスト後 Net で"
            "判定してください。[/dim]"
        )
    return 0


async def _probe_replay(args: argparse.Namespace) -> int:
    """Run the live screen's own measurement over a recording instead.

    The screen and an hour of replayed depth disagreed about USUSDT by a
    factor of ten, and two explanations fit: the screen measures the wrong
    thing, or the two looked at different hours of a market whose character
    changes. Feeding the recording through the screen's own accumulator
    separates them — same data, same clock, only the method differs. If it
    reproduces the replay's figure, the screen is sound and the gap was the
    market; if it reproduces the screen's, the method is the problem.
    """
    path = Path(args.replay)
    if not path.exists():
        raise ConfigError(f"{path} がありません。")

    instrument = _instrument_for_recording(path, args)
    view = MarketView(instrument=instrument, depth=args.depth)
    probe = dynamic.SymbolProbe(instrument.symbol, horizon_s=args.horizon_ms / 1000.0)
    tick = float(instrument.tick_size)

    console.print(
        f"{path.name} ({args.source or 'all'}) を動的審査の計算式で再生します\n"
        f"  [dim]{instrument.symbol} tick={instrument.tick_size}[/dim]"
    )

    for src, event in iter_tagged(path):
        if args.source is not None and src is not None and src != args.source:
            continue
        view.apply(event)
        ts = getattr(event, "ts_ns", 0) or 0
        if isinstance(event, (DepthDelta, DepthSnapshot)):
            bid, ask = view.book.best_bid(), view.book.best_ask()
            if bid is None or ask is None:
                continue
            probe.on_book(
                bid * tick,
                ask * tick,
                ts,
                bid_qty=instrument.qty_f(view.book.depth_at(Side.BUY, bid)),
                ask_qty=instrument.qty_f(view.book.depth_at(Side.SELL, ask)),
            )
        elif isinstance(event, TradeTick):
            probe.on_trade(event.aggressor.sign, ts, qty=instrument.qty_f(event.qty))

    console.print()
    console.rule(f"[bold cyan]{instrument.symbol} — 録画に対する動的審査")
    console.print(
        f"  spread          : {probe.spread_bps:,.2f} bps\n"
        f"  逆選択 {args.horizon_ms:g}ms  : {probe.markout_bps:+,.2f} bps"
        f"  (全約定 {probe.settled:,})\n"
        f"  逆選択 板消しのみ: {probe.sweep_markout_bps:+,.2f} bps"
        f"  (板消し {probe.sweeps_settled:,})\n"
        f"  [bold]MO/spread       : 全約定 {probe.ratio:,.2f}"
        f"  / 板消し {probe.sweep_ratio:,.2f}[/bold]"
    )
    console.print(
        "\n[dim]同じ録画をペーパー MM で再生した値と突き合わせてください。"
        "一致すれば計算式は正しく、ライブとの差は時間帯の違いです。"
        "一致しなければ計算式の側に原因があります。[/dim]"
    )
    return 0


async def _triage_survivors(args: argparse.Namespace) -> list[str]:
    """The static screen's shortlist, as plain symbol names."""
    market = await fetch_market(product=args.product, samples=args.samples)
    ticks = await fetch_tick_sizes(product=args.product)
    rows = []
    for symbol, stats in market.items():
        if not symbol.endswith(args.quote_asset):
            continue
        if stats.quote_volume < args.min_volume or stats.trades < args.min_trades_24h:
            continue
        tick = ticks.get(symbol)
        if tick is None:
            continue
        row = triage.build(
            symbol,
            bid=stats.bid,
            ask=stats.ask,
            tick_size=tick,
            quote_volume=stats.quote_volume,
            trades=stats.trades,
            spread_bps=stats.spread_bps,
        )
        if row is not None:
            rows.append(row)
    survivors = triage.rank(
        rows, args.maker_bps, min_ticks=args.min_ticks, max_tick_bps=args.max_tick_bps
    )
    return [r.symbol for r in survivors[: args.max_symbols]]


async def cmd_triage(args: argparse.Namespace) -> int:
    """Reject the whole market on three numbers, before writing any code.

    Fee, tick width and spread. The best a maker can do is capture the spread
    once per round trip, so a symbol whose spread does not cover twice the
    maker fee is finished — no simulation, no recording, no strategy work.
    """
    try:
        market = await fetch_market(
            product=args.product, samples=args.samples, interval=args.sample_interval
        )
        ticks = await fetch_tick_sizes(product=args.product)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]取得に失敗しました: {exc}[/red]")
        hint = describe_tls_error(exc)
        if hint:
            console.print(f"[yellow]{hint}[/yellow]")
        return 1

    rows = []
    missing_tick = 0
    for symbol, stats in market.items():
        if not symbol.endswith(args.quote_asset):
            continue
        if stats.quote_volume < args.min_volume or stats.trades < args.min_trades:
            continue
        tick = ticks.get(symbol)
        if tick is None:
            # Listed on the ticker feed but not in exchangeInfo: skipped
            # rather than guessed, since a wrong tick decides the verdict.
            missing_tick += 1
            continue
        row = triage.build(
            symbol,
            bid=stats.bid,
            ask=stats.ask,
            tick_size=tick,
            quote_volume=stats.quote_volume,
            trades=stats.trades,
            # The median of several looks, not one snapshot: a thin book's
            # touch swings enough between polls to reorder the whole table.
            spread_bps=stats.spread_bps,
        )
        if row is not None:
            rows.append(row)

    if not rows:
        console.print("[red]条件に合う銘柄がありません。[/red]")
        return 1

    gates = {"min_ticks": args.min_ticks, "max_tick_bps": args.max_tick_bps}
    counts = triage.tally(rows, args.maker_bps, **gates)
    survivors = triage.rank(rows, args.maker_bps, **gates)

    console.print()
    console.rule("[bold cyan]一次審査")
    console.print(
        f"  {args.product} / {args.quote_asset}建て / 24h出来高 {args.min_volume:,.0f} 以上"
        f" / 約定 {args.min_trades:,} 件以上 / {args.samples} 回観測の中央値\n"
        f"  メイカー {args.maker_bps:g} bps → 往復 {2 * args.maker_bps:g} bps"
        f" / スプレッド {args.min_ticks:g} tick 以上"
        f" / 1tick {args.max_tick_bps:g} bps 以下を要求\n"
        f"  {len(rows):,} 銘柄  →  "
        + "  ".join(f"{k} {v:,}" for k, v in counts.items())
    )
    if missing_tick:
        console.print(f"  [dim]tick 不明で除外: {missing_tick:,}[/dim]")

    if not survivors:
        console.print(
            "\n[red]候補ゼロ。この手数料では、板を全部取っても往復コストに勝てません。[/red]\n"
            "[dim]手数料を下げる以外にできることはありません。"
            "--maker-bps を変えて必要水準を探ってください。[/dim]"
        )
        return 0

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("symbol", style="cyan")
    table.add_column("1tick\n(bps)", justify="right")
    table.add_column("spread\n(bps)", justify="right")
    table.add_column("spread\n(tick)", justify="right")
    table.add_column("余裕\n(bps)", justify="right")
    table.add_column("24h約定", justify="right")
    table.add_column("24h出来高", justify="right")
    for row in survivors[: args.top]:
        head = row.headroom_bps(args.maker_bps)
        table.add_row(
            row.symbol,
            f"{row.tick_bps:,.2f}",
            f"{row.spread_bps:,.2f}",
            f"{row.spread_ticks:,.1f}",
            f"[green]{head:+,.2f}[/green]",
            f"{row.trades:,}",
            f"{row.quote_volume:,.0f}",
        )
    console.print()
    console.print(table)
    console.print(
        "\n[dim]これは落とすための審査で、通ったことは何の保証でもありません。"
        "余裕 bps は「板を全部取れたら」の上限で、逆選択も在庫もヘッジ代も"
        "まだ1銭も引いていません。上位数銘柄だけ capture → sweep → hedge に回してください。[/dim]"
    )
    return 0


def _hedge_verdict(net_by_fee: dict[float, float]) -> tuple[str, str]:
    """The decision, fixed in advance so the data cannot be argued with.

    Stated before the run rather than after: with a free maker fee the
    strategy either survives its own costs or it does not, and if it does,
    the only remaining question is which fee tier it needs.
    """
    free = net_by_fee.get(0.0, float("nan"))
    two = net_by_fee.get(2.0, float("nan"))
    if not (free > 0):
        return (
            "red",
            "終了。手数料ゼロでもコストを回収できていないので、"
            "この銘柄でのマーケットメイクは成立しません。別戦略へ。",
        )
    if not (two > 0):
        return (
            "yellow",
            "戦略は成立するが、低手数料口座が必須。"
            "現在の 10 bps 口座では不可能で、2 bps でも届いていません。",
        )
    return (
        "green",
        "成立。次は期間を変えた out-of-sample → ペーパー → 少額実売買。",
    )


async def cmd_hedge(args: argparse.Namespace) -> int:
    """Make on one venue, hedge immediately on the other, and total it up.

    This is the decisive test, not another diagnostic. The maker leg earns a
    spread and the hedge leg pays to cross; the question is whether anything
    survives both plus the fees. One number decides it, at three fee levels.
    """
    path = Path(args.path)
    if not path.exists():
        console.print(f"[red]{path} がありません。[/red]")
        return 1

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ConfigError(f"{meta_path.name} がありません。capture で録った録画が要ります。")
    sources = json.loads(meta_path.read_text()).get("sources") or {}
    for name in (args.maker_source, args.hedge_source):
        if name not in sources:
            raise ConfigError(
                f"--{'maker' if name == args.maker_source else 'hedge'}-source {name} は "
                f"録画にありません。使えるのは: {', '.join(sorted(sources))}"
            )
    maker_inst = _instrument_from_spec(sources[args.maker_source])
    hedge_inst = _instrument_from_spec(sources[args.hedge_source])

    console.print(
        f"{path.name}\n"
        f"  メイク: {args.maker_source} {maker_inst.symbol} "
        f"tick={maker_inst.tick_size} lot={maker_inst.lot_size}\n"
        f"  ヘッジ: {args.hedge_source} {hedge_inst.symbol} "
        f"tick={hedge_inst.tick_size} lot={hedge_inst.lot_size} "
        f"(taker {args.taker_bps:g}bps, ratio {args.hedge_ratio:g})\n"
        f"  requote {args.requote_ms:g}ms"
    )

    rows = []
    net_by_fee: dict[float, float] = {}
    for maker_bps in _grid(args.maker_fees, float):
        run_args = argparse.Namespace(**vars(args))
        run_args.maker_bps = maker_bps
        _check_sizes(maker_inst, run_args)
        row = await _run_hedged(path, maker_inst, hedge_inst, run_args)
        row["maker_bps"] = maker_bps
        rows.append(row)
        net_by_fee[maker_bps] = row["net_bps"]

    columns = [
        ("maker", "maker_bps", "{:g}"),
        ("約定", "fills", "{:,.0f}"),
        ("ヘッジ", "hedges", "{:,.0f}"),
        ("未ヘッジ", "skipped", "{:,.0f}"),
        ("make.spread", "maker_spread", "{:+.2f}"),
        ("make.在庫", "maker_inventory", "{:+.2f}"),
        ("hedge.cross", "hedge_cross", "{:+.2f}"),
        ("hedge.在庫", "hedge_inventory", "{:+.2f}"),
        ("手数料", "fees", "{:+.2f}"),
        ("Net bps", "net_bps", "{:+.2f}"),
    ]

    def cell(row, key, fmt):
        value = row[key]
        return "—" if isinstance(value, float) and math.isnan(value) else fmt.format(value)

    if args.plain:
        console.print("\t".join(c[0] for c in columns), highlight=False)
        for r in rows:
            console.print("\t".join(cell(r, k, f) for _, k, f in columns), highlight=False)
    else:
        table = Table(title=f"{maker_inst.symbol} — メイク＋即時ヘッジ", padding=(0, 1))
        for header, _, _ in columns:
            table.add_column(header, justify="right")
        for r in rows:
            net = r["net_bps"]
            colour = "dim" if math.isnan(net) else ("green" if net > 0 else "red")
            cells = [cell(r, k, f) for _, k, f in columns]
            cells[-1] = f"[{colour}]{cells[-1]}[/{colour}]"
            table.add_row(*cells)
        console.print(table)

    colour, verdict = _hedge_verdict(net_by_fee)
    console.print(f"\n[{colour}][bold]判定: {verdict}[/bold][/{colour}]")
    console.print(
        "[dim]Net は全コスト込み（メイカー手数料・ヘッジのテイカー手数料・"
        "板を歩いたスリッページ・両脚のベーシス変動）を1往復あたり bps に直したもの。\n"
        "ペーパー約定は板を突き抜ける約定を数えていないので、これでも上限値。[/dim]"
    )
    return 0


async def _run_hedged(path, maker_inst, hedge_inst, args) -> dict:
    """One pass over the recording with the maker and hedge books side by side."""
    mm = build_maker(maker_inst, args)
    attach_virtual_clock(mm)
    hedge_view = MarketView(instrument=hedge_inst, depth=args.depth)
    hedger = Hedger(
        instrument=hedge_inst,
        market=hedge_view,
        config=HedgeConfig(
            ratio=args.hedge_ratio, taker_bps=args.taker_bps, max_levels=args.depth
        ),
    )

    for src, event in iter_tagged(path):
        if src == args.hedge_source:
            hedge_view.apply(event)
            hedger.on_market()
            continue
        if src != args.maker_source:
            continue
        for fill in mm.on_event(event):
            if fill.maker_owner == PAPER_OWNER:
                sign = fill.aggressor.opposite.sign
                hedger.on_maker_fill(sign, maker_inst.qty_f(fill.qty))
        mm.requote()

    s = mm.summary()
    matched = min(s["bought"], s["sold"])
    maker = mm.attribution
    hedge = hedger.attribution
    # One denominator for both legs: the maker notional actually turned over
    # is what the whole exercise is trying to earn on.
    mid = maker.last_mid_ticks
    notional = matched * float(mid or 0) * float(maker_inst.tick_size)
    scale = 10_000.0 / notional if notional > 0 else math.nan

    return {
        "fills": int(s["fills"]),
        "hedges": int(hedger.hedges),
        "skipped": int(hedger.skipped_no_book),
        "maker_spread": maker.spread_capture * scale,
        "maker_inventory": maker.inventory_pnl * scale,
        "hedge_cross": hedge.spread_capture * scale,
        "hedge_inventory": hedge.inventory_pnl * scale,
        "fees": -(maker.fees + hedge.fees) * scale,
        "net_bps": (maker.total + hedge.total) * scale,
    }


# Binance spot maker fees by VIP tier, for reading the breakeven column
# against something concrete. Check the current schedule before relying on it.
BINANCE_MAKER_TIERS = ((10.0, "VIP0"), (9.0, "VIP1"), (8.0, "VIP2"), (4.2, "VIP3"),
                       (3.0, "VIP6"), (1.2, "VIP9"))


def _fee_cell(breakeven: float) -> str:
    """Colour the breakeven fee by which tier would reach it."""
    if breakeven >= 8.0:
        return f"[green]{breakeven:,.2f}[/green]"
    if breakeven >= 1.2:
        return f"[yellow]{breakeven:,.2f}[/yellow]"
    return f"[red]{breakeven:,.2f}[/red]"


def _swing_cell(swing: float) -> str:
    """Flag symbols whose touch moves so much that one look proves nothing."""
    if swing >= 10.0:
        return f"[red]{swing:,.1f}x[/red]"
    if swing >= 3.0:
        return f"[yellow]{swing:,.1f}x[/yellow]"
    return f"[dim]{swing:,.1f}x[/dim]"


async def cmd_watch(args: argparse.Namespace) -> int:
    """Repeat the scan over hours and report which symbols keep qualifying."""
    filters = _scan_filters(args)
    total_min = args.rounds * args.every / 60.0

    console.rule("[bold cyan]持続性を測る")
    console.print(
        f"  {args.rounds} 回 × {args.every / 60:.0f}分間隔 = 約 {total_min:.0f}分\n"
        f"  [dim]1回のスキャンは「今この瞬間」しか答えません。実際に建値を出し続ける\n"
        f"  時間スケールで同じ銘柄が残るかを見ます。Ctrl-C で途中集計を表示します。[/dim]\n"
    )

    tally: dict = {}

    def on_round(n, total, current, considered):
        placeable = sum(1 for p in current.values() if p.tradeable_rounds)
        console.print(
            f"  [dim]{n}/{total} 回目 … これまでに一度でも置けた銘柄 {placeable} 件[/dim]"
        )
        tally.update(current)

    try:
        tally = await watch(
            filters, rounds=args.rounds, every_seconds=args.every, on_round=on_round
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]中断しました。途中までの集計を表示します。[/yellow]")

    if not tally:
        console.print("[red]1回も観測できませんでした。[/red]")
        return 1

    ranked = [p for p in rank_persistence(tally) if p.viable_rounds]
    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("symbol", style="cyan")
    table.add_column("置けた\n回数", justify="right")
    table.add_column("持続率", justify="right")
    table.add_column("出現率", justify="right")
    table.add_column("net中央値\n(bps)", justify="right")
    table.add_column("上限手数料\n(bps)", justify="right")
    table.add_column("板の厚み\n中央値", justify="right")
    table.add_column("板のぶれ\n(観測全体)", justify="right")

    for p in ranked[: args.top]:
        style = "green" if p.persistence >= 0.8 else "yellow" if p.persistence >= 0.5 else "red"
        table.add_row(
            p.symbol,
            f"{p.tradeable_rounds}/{p.total_rounds or p.rounds}",
            f"[{style}]{p.persistence:.0%}[/{style}]",
            f"[dim]{p.presence:.0%}[/dim]",
            f"{p.median_net_bps:+,.2f}",
            _fee_cell(p.breakeven_fee_bps(filters.maker_bps)),
            f"{p.median_depth:,.0f}",
            _swing_cell(p.depth_swing),
        )

    console.print()
    console.print(table)

    reliable = [p for p in ranked if p.persistence >= 0.75]
    console.print(
        f"\n  一度でも手数料を超えた銘柄 [bold]{len(ranked)}[/bold] 件のうち、"
        f"75%以上の回で注文が置けたのは [bold]{len(reliable)}[/bold] 件"
    )
    if reliable:
        console.print("\n  [yellow]どの手数料まで成立するか[/yellow]")
        for fee, tier in BINANCE_MAKER_TIERS:
            passing = [p.symbol for p in reliable if p.breakeven_fee_bps(filters.maker_bps) >= fee]
            mark = "green" if passing else "dim"
            console.print(
                f"  [dim]{tier:<6} {fee:>5.1f}bps →[/dim] "
                f"[{mark}]{len(passing)}件[/{mark}] "
                f"[dim]{', '.join(passing[:6]) if passing else ''}[/dim]"
            )
        console.print(
            "\n  [dim]「上限手数料」は スプレッド÷2。これを下回る手数料が取れれば成立します。\n"
            "  手数料は交渉やVIP昇格で動かせますが、スプレッドは動かせません。[/dim]"
        )
    if not reliable:
        console.print(
            "\n  [yellow]継続して成立する銘柄はありませんでした。[/yellow]\n"
            "  [dim]スキャン1回では数件が「可」になりますが、それは観測した瞬間の話で、\n"
            "  次に見たときには別の銘柄に入れ替わっています。建値を出し続ける戦略の\n"
            "  対象としては使えません。[/dim]"
        )
    console.rule()
    return 0


def _scan_filters(args: argparse.Namespace) -> ScanFilters:
    return ScanFilters(
        quote_asset=args.quote.upper(),
        product=args.product,
        maker_bps=args.maker_bps,
        min_quote_volume=args.min_volume,
        min_trades=args.min_trades,
        size_quote=args.size_quote,
        thin_ratio=args.thin_ratio,
        crowded_ratio=args.crowded_ratio,
        samples=args.samples,
        sample_interval=args.sample_interval,
    )


async def cmd_scan(args: argparse.Namespace) -> int:
    filters = _scan_filters(args)

    def progress(n: int, total: int) -> None:
        console.print(f"[dim]  板を観測中 {n}/{total}[/dim]", end="\r")

    console.print(
        f"[dim]Binance の全銘柄を取得中… "
        f"（板を {filters.samples} 回、{filters.sample_interval:.0f}秒間隔で観測）[/dim]"
    )
    try:
        results, considered = await scan(filters, on_sample=progress)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]取得に失敗しました: {exc}[/red]")
        hint = describe_tls_error(exc)
        console.print(
            f"[yellow]{hint}[/yellow]"
            if hint
            else "[dim]ネットワークから api.binance.com に到達できるか確認してください。[/dim]"
        )
        return 1

    stats = summarise(results, filters)
    breakeven = stats["breakeven_spread_bps"]

    console.print()
    console.rule("[bold cyan]マーケットメイクが成立しうる銘柄")
    console.print(
        f"  メイカー手数料 [bold]{filters.maker_bps:.1f} bps[/bold]"
        f"  →  損益分岐スプレッド [bold]{breakeven:.1f} bps[/bold]\n"
        f"  条件: {filters.quote_asset}建て / 24h出来高 {filters.min_quote_volume:,.0f} 以上"
        f" / 約定 {filters.min_trades:,} 件以上"
    )

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("symbol", style="cyan")
    table.add_column("spread\n(bps)", justify="right")
    table.add_column("net\n(bps)", justify="right")
    table.add_column(f"1往復\n({filters.size_quote:,.0f})", justify="right")
    table.add_column("24h出来高", justify="right")
    table.add_column("板の厚み", justify="right")
    table.add_column("行列/自分", justify="right")
    table.add_column("板のぶれ", justify="right")
    table.add_column("判定")

    verdict_style = {"可": "green", "行列が長い": "yellow", "板が薄い": "yellow"}

    shown = results[: args.top]
    for s in shown:
        net = s.net_bps(filters.maker_bps)
        style = "green" if net > 0 else "red"
        verdict = (
            s.capacity_verdict(
                filters.size_quote, thin=filters.thin_ratio, crowded=filters.crowded_ratio
            )
            if net > 0
            else "手数料割れ"
        )
        ratio = s.queue_ratio(filters.size_quote)
        table.add_row(
            s.symbol,
            f"{s.spread_bps:,.2f}",
            f"[{style}]{net:+,.2f}[/{style}]",
            f"[{style}]{s.profit_per_round_trip(filters.maker_bps, filters.size_quote):+,.4f}[/{style}]",
            f"{s.quote_volume:,.0f}",
            f"{s.top_of_book_quote:,.0f}",
            f"{ratio:,.1f}x",
            _swing_cell(s.depth_swing),
            f"[{verdict_style.get(verdict, 'red')}]{verdict}[/{verdict_style.get(verdict, 'red')}]",
        )

    console.print()
    console.print(table)
    console.print(
        f"\n  {considered:,} 銘柄 → 流動性条件を満たす [bold]{stats['liquid']:,}[/bold] 件"
        f" → 手数料を超える [bold]{stats['viable']:,}[/bold] 件"
        f" → 注文が置ける [bold]{stats['tradeable']:,}[/bold] 件"
    )
    console.print(
        f"  スプレッドの中央値 [bold]{stats['median_spread_bps']:.2f} bps[/bold]"
        f"（損益分岐は {breakeven:.1f} bps）"
    )
    if stats["unstable"]:
        console.print(
            f"  [yellow]うち {stats['unstable']} 件は板が観測中に3倍以上ぶれています。"
            f"1回の観測では判定できません。[/yellow]"
        )

    # Above, symbols are ranked by edge, which pushes the workable ones off
    # the list — a 22bps spread nobody can reach outranks an 11bps one that
    # is actually placeable. These are the only rows worth acting on.
    placeable = stats.get("placeable") or []
    if placeable:
        console.print(f"\n  [bold green]注文が置ける {len(placeable)} 件[/bold green]")
        good = Table(box=None, header_style="bold dim", padding=(0, 1))
        good.add_column("symbol", style="cyan")
        for label in ("spread\n(bps)", "net\n(bps)", "板の厚み", "行列/自分", "板のぶれ"):
            good.add_column(label, justify="right")
        good.add_column("24h出来高", justify="right")
        for s_ in sorted(
            placeable, key=lambda x: x.net_bps(filters.maker_bps), reverse=True
        ):
            good.add_row(
                s_.symbol,
                f"{s_.spread_bps:,.2f}",
                f"[green]{s_.net_bps(filters.maker_bps):+,.2f}[/green]",
                f"{s_.top_of_book_quote:,.0f}",
                f"{s_.queue_ratio(filters.size_quote):,.1f}x",
                _swing_cell(s_.depth_swing),
                f"{s_.quote_volume:,.0f}",
            )
        console.print(good)
        console.print(
            "  [dim]この一覧は1回の観測です。数分で反転することが実測されているので、\n"
            "  watch で持続率を測るまで採用しないでください。[/dim]"
        )

    if stats["viable"] == 0:
        console.print(
            "\n  [yellow]この手数料でスプレッドを超える銘柄はありません。[/yellow]\n"
            "  [dim]手数料を下げる以外に、この戦略が成立する道はありません。[/dim]"
        )
    elif stats["tradeable"] == 0:
        console.print(
            f"\n  [yellow]手数料を超える {stats['viable']} 件は、すべて別の理由で成立しません。[/yellow]\n"
            f"  [dim]行列が長い {stats['too_deep']} 件 … 自分の前に並ぶ枚数が多すぎて順番が回らない\n"
            f"  板が薄い　　 {stats['too_thin']} 件 … 自分の注文が板の大半を占める。抜けられない[/dim]\n"
            f"  [dim]--size-quote を下げれば「板より大きい」は解消しますが、\n"
            f"  1往復の利益も同じ比率で下がります。[/dim]"
        )
    else:
        console.print(
            "\n  [yellow]数字の読み方[/yellow]\n"
            "  [dim]スプレッドが広い銘柄は、誰も建値を置きたがらないから広いのが普通です。\n"
            "  「行列/自分」は自分の前に並んでいる金額の倍率。大きいほど約定しません。\n"
            "  1未満は自分が板より大きいという意味で、そこでは在庫を捌けません。[/dim]"
        )
    console.rule()
    return 0


async def cmd_capture(args: argparse.Namespace) -> int:
    """Record spot and perp together, onto one timeline."""
    if args.spot_only and args.perp_only:
        console.print("[red]--spot-only と --perp-only は同時に指定できません。[/red]")
        return 2

    sources: dict = {}
    specs: dict = {}

    if not args.perp_only:
        try:
            spot = await fetch_instrument(args.symbol)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]spot exchangeInfo unavailable ({exc}); using built-in[/yellow]")
            spot = build_instrument(args.symbol, None, None)
        sources["spot"] = BinanceFeed(spot, depth_ms=args.spot_depth_ms)
        specs["spot"] = _spec_dict(spot, "spot")
        console.print(f"[dim]spot  tick={spot.tick_size} lot={spot.lot_size}[/dim]")

    if not args.spot_only:
        try:
            perp = await fetch_futures_instrument(args.symbol)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]perp exchangeInfo failed: {exc}[/red]")
            hint = describe_tls_error(exc)
            if hint:
                console.print(f"[yellow]{hint}[/yellow]")
            return 1
        sources["perp"] = BinanceFuturesFeed(
            perp,
            depth_ms=args.perp_depth_ms,
            open_interest_interval=args.oi_interval,
            rest_fallback=args.perp_fallback,
            fallback_after_s=args.fallback_after,
            trade_poll_interval=args.trade_poll,
            mark_poll_interval=args.mark_poll,
        )
        specs["perp"] = _spec_dict(perp, "perp")
        console.print(f"[dim]perp  tick={perp.tick_size} lot={perp.lot_size}[/dim]")

    out = Path(args.out)
    capture = MultiCapture(sources, out)

    console.rule(f"[bold cyan]{args.symbol.upper()} を記録")
    console.print(
        f"  出力   : {out}\n"
        f"  対象   : {', '.join(sources)}\n"
        f"  停止   : "
        + (f"{args.duration:.0f}秒後" if args.duration else "Ctrl-C まで")
        + "\n  [dim]取引所時刻と受信時刻の両方を記録します。前者は市場が何をしたか、\n"
        "  後者は戦略が何を知り得たか。片方だけでは後から問い直せません。[/dim]\n"
    )

    last_report = [time.monotonic()]

    def on_event(_name, _event) -> None:
        now = time.monotonic()
        if now - last_report[0] < 5.0:
            return
        last_report[0] = now
        parts = []
        for name, st in capture.stats.items():
            kinds = " ".join(f"{k}={v:,}" for k, v in sorted(st.by_kind.items()))
            parts.append(f"{name}[{st.status}] {st.events:,} ({kinds})")
        console.print(f"  [dim]{' | '.join(parts)}[/dim]")

    capture.on_event = on_event

    try:
        result = await capture.run(duration_s=args.duration, max_events=args.max_events)
    except KeyboardInterrupt:
        console.print("\n[yellow]中断しました。[/yellow]")
        return 130

    meta = write_meta(out, specs)

    console.print()
    console.rule("[bold cyan]記録完了")
    console.print(
        f"  停止理由 : {result.stopped_because}\n"
        f"  時間     : {result.duration_s:,.1f}秒\n"
        f"  イベント : {result.total_events:,} 件\n"
        f"  出力     : {out}  ({out.stat().st_size / 1e6:,.1f} MB)\n"
        f"  メタ     : {meta.name}"
    )
    for name, st in result.stats.items():
        kinds = ", ".join(f"{k} {v:,}" for k, v in sorted(st.by_kind.items()))
        console.print(f"  [bold]{name}[/bold]: {st.events:,} 件  [dim]{kinds}[/dim]")
        if st.errors:
            console.print(f"    [yellow]切断 {st.errors} 回[/yellow]")
        if st.events == 0:
            console.print("    [red]1件も受信していません。接続を確認してください。[/red]")
    console.rule()
    return 0


async def _tick_size_for(symbol: str, product: str, override: str | None) -> tuple[float, str]:
    """The instrument's real price step, not a default borrowed from BTC.

    Slippage is charged in ticks, so a wrong tick size scales the cost
    estimate directly — BTCUSDT perp steps by 0.1 and ETHUSDT by 0.01, so a
    shared default overstates one of them tenfold. It is cheap to ask the
    venue, and only worth guessing when the venue cannot be reached.
    """
    if override is not None:
        return float(override), "指定値"
    try:
        inst = (
            await fetch_futures_instrument(symbol)
            if product == "perp"
            else await fetch_instrument(symbol)
        )
        return float(inst.tick_size), "exchangeInfo"
    except Exception as exc:  # noqa: BLE001 - a guess with a warning beats a crash
        console.print(
            f"  [yellow]刻み幅を取得できませんでした（{type(exc).__name__}）。"
            "0.01 で計算します。--tick-size で上書きできます。[/yellow]"
        )
        return 0.01, "推定"


def _no_data_hint(symbol: str, product: str, missing: int, total: int) -> None:
    """Say what is actually wrong when nothing downloaded.

    Every day 404ing almost always means the symbol is not listed on that
    product, not that a week of history went missing — so say that, rather
    than repeating "not published" once per day and leaving the reader to
    infer it.
    """
    if missing < total:
        console.print("[red]データが1日も取れませんでした。[/red]")
        return
    other = "spot" if product == "perp" else "perp"
    console.print(
        f"\n  [red]{symbol.upper()} は {product} のアーカイブに1日もありません。[/red]\n"
        f"  [dim]銘柄名の綴り、または製品の選択を確認してください。\n"
        f"  現物にしか無い銘柄なら --product {other} で通ります。\n"
        f"  何が公開されているかは次で一覧できます:\n"
        f"    python tools/probe_archive.py {symbol.upper()}[/dim]"
    )


async def cmd_horizon(args: argparse.Namespace) -> int:
    """Ask whether the moves are bigger than the fee, before modelling them."""
    last = (
        date.fromisoformat(args.end)
        if args.end
        else date.today() - timedelta(days=1 if args.product == "spot" else 2)
    )
    wanted = days_ending(last, args.days)

    console.rule(f"[bold cyan]{args.symbol.upper()} {args.product} — 値動きと手数料")
    console.print(
        f"  対象   : {wanted[0]} 〜 {wanted[-1]}（{len(wanted)}日）\n"
        "  [dim]予測モデルの前に、そもそも動きが手数料を超えているかを見ます。\n"
        "  超えていなければ、どんな精度でも勝てません。[/dim]\n"
    )

    bars: list = []
    missing = 0
    session = make_session()
    try:
        for day in wanted:
            try:
                path = await fetch_day(
                    args.product, "aggTrades", args.symbol, day, session=session
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"  [yellow]{day} 取得失敗: {exc}[/yellow]")
                continue
            if path is None:
                missing += 1
                continue
            day_bars = load_seconds(path)
            bars.extend(day_bars)
            console.print(
                f"  [dim]{day}  {len(day_bars):,} 秒  "
                f"{path.stat().st_size / 1e6:,.1f} MB[/dim]"
            )
    finally:
        await session.close()

    if not bars:
        _no_data_hint(args.symbol, args.product, missing, len(wanted))
        return 1

    bars.sort(key=lambda b: b.sec)
    tick, tick_src = await _tick_size_for(args.symbol, args.product, args.tick_size)
    tick_bps = tick / bars[-1].last * 10_000.0
    cost = round_trip_cost_bps(args.taker_bps, args.slippage_ticks, tick_bps)

    console.print(
        f"\n  往復コスト : [bold]{cost:.2f} bps[/bold]  "
        f"[dim](手数料 {args.taker_bps} × 2 + スリッページ {args.slippage_ticks} tick × 2、"
        f"刻み {tick:g}[{tick_src}] = {tick_bps:.4f} bps/tick)[/dim]"
    )

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("保有", justify="right")
    table.add_column("平均\n変動(bps)", justify="right")
    table.add_column("中央値", justify="right")
    table.add_column("上位10%", justify="right")
    table.add_column("上位1%", justify="right")
    table.add_column("手数料超\nの割合", justify="right")
    table.add_column("必要な的中率\n（全場面）", justify="right")
    table.add_column("必要な的中率\n（上位10%）", justify="right")
    table.add_column("上位10%の\n平均変動", justify="right")

    verdicts = []
    for h in args.horizons:
        st = analyse(bars, h, cost)
        verdicts.append(st)
        acc = (
            f"[green]{st.required_accuracy:.1%}[/green]"
            if st.is_possible
            else "[red]不可能[/red]"
        )
        sel = (
            f"[green]{st.required_accuracy_top10:.1%}[/green]"
            if st.selective_is_possible
            else "[red]不可能[/red]"
        )
        table.add_row(
            f"{h}秒",
            f"{st.mean_abs_bps:.2f}",
            f"{st.median_abs_bps:.2f}",
            f"{st.p90_abs_bps:.2f}",
            f"{st.p99_abs_bps:.2f}",
            f"{st.tradeable_fraction:.1%}",
            acc,
            sel,
            f"{st.mean_top10_bps:.2f}",
        )

    console.print()
    console.print(table)

    possible = [v for v in verdicts if v.is_possible]
    selective = [v for v in verdicts if v.selective_is_possible]
    console.print()
    if possible:
        best = min(possible, key=lambda v: v.required_accuracy)
        console.print(
            f"  常時売買でも余地があるのは {', '.join(f'{v.horizon_s}秒' for v in possible)}。\n"
            f"  最も条件が緩いのは [bold]{best.horizon_s}秒[/bold]、"
            f"必要な的中率 [bold]{best.required_accuracy:.1%}[/bold]。"
        )
    elif selective:
        best = min(selective, key=lambda v: v.required_accuracy_top10)
        console.print(
            "  [yellow]平均的な値動きでは、どの保有時間もコストに届きません。[/yellow]\n"
            f"  ただし[bold]大きく動く場面だけ[/bold]に絞れば "
            f"{', '.join(f'{v.horizon_s}秒' for v in selective)} に余地があります。\n"
            f"  最良は [bold]{best.horizon_s}秒[/bold]で、上位10%の場面だけを売買して\n"
            f"  的中率 [bold]{best.required_accuracy_top10:.1%}[/bold]。\n\n"
            "  [dim]ただしこれは条件付きの話です。「これから大きく動く」と事前に\n"
            "  当てること自体が、方向を当てるより難しい予測問題です。\n"
            "  算数として不可能ではない、というだけの意味しかありません。[/dim]"
        )
    else:
        console.print(
            "  [red]常時売買でも、大きく動く場面に絞っても、コストに届きません。[/red]\n"
            "  [dim]テイカーで短時間、という前提そのものが成立しません。\n"
            "  手数料の交渉、メイカー執行、保有時間を伸ばす、のいずれかが要ります。[/dim]"
        )
    console.rule()
    return 0


async def cmd_predict(args: argparse.Namespace) -> int:
    """Measure whether any signal reaches the accuracy the fee demands."""
    last = (
        date.fromisoformat(args.end)
        if args.end
        else date.today() - timedelta(days=1 if args.product == "spot" else 2)
    )
    wanted = days_ending(last, args.days)

    console.rule(f"[bold cyan]{args.symbol.upper()} {args.product} — 予測できるか")
    console.print(
        f"  対象   : {wanted[0]} 〜 {wanted[-1]}（{len(wanted)}日）  "
        f"保有 {args.horizon}秒\n"
        "  [dim]前半で信号を選び、後半で一度だけ答え合わせします。\n"
        "  同じデータで選んで測ると、選び方の上手さを測ることになります。[/dim]\n"
    )

    bars: list = []
    missing = 0
    session = make_session()
    try:
        for day in wanted:
            path = await fetch_day(args.product, "aggTrades", args.symbol, day, session=session)
            if path is None:
                missing += 1
                continue
            bars.extend(load_seconds(path))
    finally:
        await session.close()

    if not bars:
        _no_data_hint(args.symbol, args.product, missing, len(wanted))
        return 1
    bars.sort(key=lambda b: b.sec)

    train_bars, test_bars = split(bars, args.train_fraction)
    if not train_bars or not test_bars:
        console.print("[red]分割できるだけの期間がありません。[/red]")
        return 1
    train, test = Series.build(train_bars), Series.build(test_bars)

    tick, _ = await _tick_size_for(args.symbol, args.product, args.tick_size)
    tick_bps = tick / bars[-1].last * 10_000.0
    cost = round_trip_cost_bps(args.taker_bps, args.slippage_ticks, tick_bps)
    threshold = (
        volatility_threshold(train, args.vol_window, args.vol_quantile)
        if args.vol_quantile > 0
        else None
    )

    console.print(
        f"  往復コスト : [bold]{cost:.2f} bps[/bold]\n"
        f"  学習期間   : {len(train_bars):,} 秒   検証期間 : {len(test_bars):,} 秒"
    )
    if threshold is not None:
        console.print(
            f"  出動条件   : 直近{args.vol_window}秒の変動が {threshold:.2f} bps 以上"
            f"（学習期間の上位{(1 - args.vol_quantile):.0%}）"
        )

    scored = []
    for name, kind, window, sign in candidates():
        s_train = evaluate(
            train, name, kind, window, sign,
            horizon_s=args.horizon, cost_bps=cost,
            vol_window=args.vol_window, vol_threshold=threshold,
        )
        if s_train.samples < args.min_samples:
            continue
        scored.append((s_train, kind, window, sign, name))

    if not scored:
        console.print("[red]十分な標本のある信号がありませんでした。[/red]")
        return 1

    scored.sort(key=lambda row: row[0].accuracy, reverse=True)
    best_train, kind, window, sign, name = scored[0]

    best_test = evaluate(
        test, name, kind, window, sign,
        horizon_s=args.horizon, cost_bps=cost,
        vol_window=args.vol_window, vol_threshold=threshold,
    )
    base = always_long(test, horizon_s=args.horizon, cost_bps=cost)

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("信号", style="cyan")
    table.add_column("学習\n的中率", justify="right")
    for label in ("検証\n的中率", "回数", "平均変動\n(bps)", "1回あたり\n損益(bps)"):
        table.add_column(label, justify="right")

    for s_train, _k, _w, _s, label in scored[:8]:
        mark = "bold" if label == name else "dim"
        table.add_row(f"[{mark}]{label}[/{mark}]", f"{s_train.accuracy:.1%}", "", "", "", "")

    console.print()
    console.print("  [dim]学習期間での順位[/dim]")
    console.print(table)

    result = Table(box=None, header_style="bold dim", padding=(0, 1))
    result.add_column("", style="cyan")
    result.add_column("検証 的中率", justify="right")
    result.add_column("回数", justify="right")
    result.add_column("平均変動", justify="right")
    result.add_column("1回あたり損益", justify="right")
    for label, sc in (("選ばれた信号", best_test), ("常に買い（基準）", base)):
        colour = "green" if sc.edge_bps > 0 else "red"
        result.add_row(
            f"{label}  {sc.name if label.startswith('選') else ''}",
            f"{sc.accuracy:.2%}",
            f"{sc.samples:,}",
            f"{sc.mean_move_bps:.2f}",
            f"[{colour}]{sc.edge_bps:+.2f}[/{colour}]",
        )

    console.print("\n  [dim]検証期間での答え合わせ（1回のみ）[/dim]")
    console.print(result)

    margin = best_test.accuracy - base.accuracy
    console.print()
    if best_test.edge_bps > 0 and margin > 0:
        console.print(
            f"  [green]{name} は検証期間で手数料を上回りました。[/green]\n"
            f"  基準（常に買い）との差は [bold]{margin:+.2%}[/bold]。\n"
            "  [dim]ただし7日程度では偶然の範囲を出ません。期間を延ばし、\n"
            "  複数の期間で再現するかを見る必要があります。[/dim]"
        )
    elif best_test.edge_bps > 0:
        console.print(
            f"  [yellow]{name} は利益が出ていますが、常に買うだけの基準を"
            f"超えていません（差 {margin:+.2%}）。[/yellow]\n"
            "  [dim]上昇相場を捉えただけで、信号が効いた証拠にはなりません。[/dim]"
        )
    else:
        console.print(
            f"  [red]{name} は検証期間で手数料を超えませんでした"
            f"（1回あたり {best_test.edge_bps:+.2f} bps）。[/red]\n"
            f"  [dim]必要だった的中率は "
            f"{(1 + cost / best_test.mean_move_bps) / 2:.1%}、実際は "
            f"{best_test.accuracy:.1%}。[/dim]"
        )
    console.rule()
    return 0


# Conditions are named after what they claim is happening, not after their
# thresholds, so the ranking reads as a list of hypotheses.
EVENT_CONDITIONS: dict[str, tuple[int, list[tuple[str, str, float]]]] = {
    # Thresholds are z-scores, not raw values. A guessed absolute is how a
    # condition ends up firing zero times and being read as "tested and
    # failed" — `futures_lead_bps > 3.0` never fired on BTC, because spot and
    # perp do not diverge by whole basis points in a minute. z > 1.65 means
    # "top ~5% relative to the last day", which is what the rule intends and
    # fires by construction.
    #
    # --- 1. futures-led breakout ------------------------------------------
    "出来高急増": (1, [("volume_z", ">", 1.65)]),
    "成行買い優勢": (1, [("perp_buy_ratio", ">", 0.65)]),
    "先物先行(上)": (1, [("futures_lead_z", ">", 1.65)]),
    "OI増加": (1, [("oi_change_z", ">", 1.65)]),
    "出来高＋買いフロー": (
        1, [("volume_z", ">", 1.65), ("perp_buy_ratio", ">", 0.65)]
    ),
    "出来高＋OI増加": (1, [("volume_z", ">", 1.65), ("oi_change_z", ">", 1.65)]),
    "先物先行＋買いフロー": (
        1, [("futures_lead_z", ">", 1.65), ("perp_buy_ratio", ">", 0.65)]
    ),
    "出来高＋先物先行＋OI増加": (
        1,
        [
            ("volume_z", ">", 1.65),
            ("futures_lead_z", ">", 1.0),
            ("oi_change_z", ">", 1.0),
        ],
    ),
    # --- 2. capitulation and reversal -------------------------------------
    "急落": (1, [("perp_ret_5m", "<", -30.0)]),
    "急落＋OI急減": (1, [("perp_ret_5m", "<", -30.0), ("oi_change_z", "<", -1.65)]),
    "急落＋売り枯れ": (
        1, [("perp_ret_5m", "<", -30.0), ("perp_buy_ratio", ">", 0.5)]
    ),
    "急落＋出来高急増＋売り枯れ": (
        1,
        [
            ("perp_ret_5m", "<", -30.0),
            ("volume_z", ">", 1.0),
            ("perp_buy_ratio", ">", 0.5),
        ],
    ),
    # --- 3. basis mean reversion ------------------------------------------
    "ベーシス過熱(売り)": (-1, [("basis_z", ">", 1.65)]),
    "ベーシス過冷(買い)": (1, [("basis_z", "<", -1.65)]),
    "ベーシス過熱＋フロー反転": (
        -1, [("basis_z", ">", 1.65), ("perp_buy_ratio", "<", 0.5)]
    ),
    "ベーシス過冷＋フロー反転": (
        1, [("basis_z", "<", -1.65), ("perp_buy_ratio", ">", 0.5)]
    ),
    # --- controls ----------------------------------------------------------
    "常時ロング(基準)": (1, []),
    "高ボラのみ": (1, [("volatility_z", ">", 1.65)]),
}


async def cmd_events(args: argparse.Namespace) -> int:
    """Rank conditions by net money, with every acceptance gate shown."""
    last = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=2)
    wanted = days_ending(last, args.days)

    console.rule(f"[bold cyan]{args.symbol.upper()} — 条件別の純損益")
    console.print(
        f"  対象   : {wanted[0]} 〜 {wanted[-1]}（{len(wanted)}日）  "
        f"保有 {args.horizon}分\n"
        "  [dim]的中率ではなく手数料込みの純損益で並べます。\n"
        "  重複する取引は数えません（1時間保有の60分ぶんは60件ではなく1件）。[/dim]\n"
    )

    spot_secs: list = []
    perp_secs: list = []
    oi_marks: list = []
    session = make_session()
    try:
        for day in wanted:
            perp_path = await fetch_day("perp", "aggTrades", args.symbol, day, session=session)
            spot_path = await fetch_day("spot", "aggTrades", args.symbol, day, session=session)
            if perp_path is None or spot_path is None:
                console.print(f"  [yellow]{day} は片側が欠けています[/yellow]")
                continue
            perp_secs.extend(load_seconds(perp_path))
            spot_secs.extend(load_seconds(spot_path))
            metrics = await fetch_day("perp", "metrics", args.symbol, day, session=session)
            if metrics is not None:
                oi_marks.extend(load_open_interest(metrics))
            console.print(f"  [dim]{day}[/dim]")
    finally:
        await session.close()

    if not perp_secs or not spot_secs:
        _no_data_hint(args.symbol, "perp", len(wanted), len(wanted))
        return 1

    perp_secs.sort(key=lambda b: b.sec)
    spot_secs.sort(key=lambda b: b.sec)
    oi_marks.sort()

    bars = to_minutes(spot_secs, perp_secs)
    attach_open_interest(bars, oi_marks)
    rows = build(bars, z_window=args.z_window)
    if not rows:
        console.print("[red]両方の製品が揃った分が足りません。[/red]")
        return 1

    tick, _ = await _tick_size_for(args.symbol, "perp", args.tick_size)
    tick_bps = tick / perp_secs[-1].last * 10_000.0
    cost = round_trip_cost_bps(args.taker_bps, args.slippage_ticks, tick_bps)

    console.print(
        f"\n  分足 {len(rows):,} 本   往復コスト [bold]{cost:.2f} bps[/bold]"
        + (
            f"   利確 +{args.target}bps / 損切 −{args.stop}bps"
            if args.target and args.stop
            else "   利確・損切なし（時間で手仕舞い）"
        )
    )

    # Every condition is simulated long. A short on the same moments is the
    # exact mirror, so `|mean_gross| - cost` already says what the better
    # direction could do — running both to keep the less bad one would answer
    # nothing and double the chance of picking a winner by luck.
    verdicts = []
    for name, (_direction, specs) in EVENT_CONDITIONS.items():
        cond = combine(*(threshold(f, op, v) for f, op, v in specs)) if specs else (
            lambda f: True
        )
        trades = simulate(
            rows, perp_secs, cond,
            direction=1, horizon_min=args.horizon, cost_bps=cost,
            target_bps=args.target, stop_bps=args.stop,
        )
        verdicts.append(score(name, trades))

    verdicts.sort(
        key=lambda v: (v.trades > 0, v.best_direction_net_bps), reverse=True
    )

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("条件", style="cyan")
    for label in (
        "件数", "変動幅", "方向性", "最良方向\nの純益", "上位10除く", "手数料1.5倍",
    ):
        table.add_column(label, justify="right")
    table.add_column("向き", justify="center")
    table.add_column("判定")

    for v in verdicts:
        if v.trades == 0:
            table.add_row(v.name, "0", *["—"] * 5, "—", "[dim]該当なし[/dim]")
            continue
        best = v.best_direction_net_bps
        colour = "green" if best > 0 else "red"
        table.add_row(
            v.name,
            f"{v.trades:,}",
            f"{v.mean_abs_move_bps:.1f}",
            f"{v.mean_gross_bps:+.2f}",
            f"[{colour}]{best:+.2f}[/{colour}]",
            f"{v.mean_without_top10_bps:+.1f}",
            f"{v.mean_at_15x_cost_bps:+.1f}",
            "買" if v.implied_direction > 0 else "売",
            "[green]合格[/green]" if v.passes else f"[dim]{v.failures()[0]}[/dim]",
        )

    console.print()
    console.print(table)

    passed = [v for v in verdicts if v.passes]
    console.print()
    if passed:
        console.print(
            f"  [green]全基準を満たした条件: {len(passed)}件[/green] — "
            + "、".join(v.name for v in passed)
            + "\n  [dim]ただしこれは選定と検証が同じ期間です。別期間で再現するまで\n"
            "  採用しないでください。[/dim]"
        )
    else:
        console.print(
            "  [yellow]全基準を満たした条件はありません。[/yellow]\n"
            "  [dim]「判定」列は最初に落ちた基準を示します。件数不足なら閾値を緩め、\n"
            "  平均は黒字だが上位10件を除くと赤なら、少数の大当たりに乗っただけです。[/dim]"
        )
    console.print(
        f"\n  [dim]「変動幅」は方向を無視した平均の値幅。ここが {cost:.1f} bps を\n"
        "  下回れば、方向を完璧に当てても勝てません（場面選びの失敗）。\n\n"
        "  「方向性」は符号付きの平均。同じ場面で買いと売りは鏡像なので、\n"
        f"  どちらか良いほうでも [bold]|方向性| − {cost:.1f}[/bold] が上限です。\n"
        "  ここが小さければ、条件は方向の情報を持っていません。\n"
        "  買いが−10で売りが−8、という並びは「売りのほうがマシ」ではなく\n"
        "  「情報がゼロで、差は相場のドリフト」という意味です。\n\n"
        "  板の特徴量（板の偏り・キャンセル・microprice）は未計測です。[/dim]"
    )
    console.rule()
    return 0


def _spec_dict(inst: Instrument, market: str) -> dict:
    return {
        "symbol": inst.symbol,
        "market": market,
        "tick_size": str(inst.tick_size),
        "lot_size": str(inst.lot_size),
        "base": inst.base,
        "quote": inst.quote,
    }


# ------------------------------------------------------------------ parser


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--tick-size", default=None, help="override price tick")
    p.add_argument("--lot-size", default=None, help="override size step")
    p.add_argument("--depth", type=int, default=12, help="ladder rows per side")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--headless", action="store_true", help="no live board, report only")

    mm = p.add_argument_group("market maker")
    mm.add_argument("--gamma", type=float, default=0.6, help="risk aversion")
    mm.add_argument("--kappa", type=float, default=1.4, help="order arrival intensity")
    mm.add_argument("--levels", type=int, default=3, help="ladder depth per side")
    mm.add_argument("--level-step", type=int, default=2, help="ticks between levels")
    mm.add_argument("--size", default="0.01", help="base quote size per level")
    mm.add_argument("--max-position", default="0.10", help="inventory limit")
    mm.add_argument("--min-half-spread", type=int, default=1, help="ticks")
    mm.add_argument(
        "--max-distance",
        type=int,
        default=None,
        help="cap ticks behind the touch (0 = join the queue at best)",
    )
    mm.add_argument(
        "--min-edge-bps",
        type=float,
        default=None,
        help="half-spread floor in bps (defaults to the maker fee)",
    )
    mm.add_argument("--requote-ms", type=float, default=250.0)

    risk = p.add_argument_group("risk")
    risk.add_argument("--max-notional", type=float, default=250_000.0)
    risk.add_argument("--max-drawdown", type=float, default=2_000.0)

    sim = p.add_argument_group("simulation")
    sim.add_argument("--latency-ms", type=float, default=5.0)
    sim.add_argument("--cancel-ahead", type=float, default=0.5)
    # Binance's standard spot maker fee is 10bps, which no one market-makes
    # against; 0 is the market-maker / VIP tier this strategy assumes.
    sim.add_argument("--maker-bps", type=float, default=0.0)
    sim.add_argument("--taker-bps", type=float, default=4.0)


def add_scan_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by `scan` and `watch` — watch is scan, repeated."""
    p.add_argument("--quote", default="USDT", help="建て通貨")
    p.add_argument(
        "--product", default="spot", choices=("spot", "perp"),
        help="どちらの板を見るか。先物はメイカー手数料が現物より低い",
    )
    p.add_argument("--maker-bps", type=float, default=10.0, help="自分のメイカー手数料")
    p.add_argument("--min-volume", type=float, default=1_000_000.0, help="24h出来高の下限")
    p.add_argument("--min-trades", type=int, default=1_000, help="24h約定数の下限")
    p.add_argument("--size-quote", type=float, default=1_000.0, help="1回の注文金額")
    p.add_argument("--thin-ratio", type=float, default=5.0,
                   help="板がこの倍率未満なら「薄い」と判定")
    p.add_argument("--crowded-ratio", type=float, default=50.0,
                   help="行列がこの倍率を超えたら「長い」と判定")
    p.add_argument("--samples", type=int, default=5,
                   help="板を観測する回数。1回では薄い板を判定できない")
    p.add_argument("--sample-interval", type=float, default=2.0, help="観測の間隔（秒）")
    p.add_argument("--top", type=int, default=25, help="表示件数")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jsboard", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("sim", help="synthetic market")
    add_common(p_sim)
    p_sim.add_argument("--start-price", type=float, default=64_000.0)
    p_sim.add_argument("--interval", type=float, default=0.1, help="seconds per tick")
    p_sim.add_argument("--volatility", type=float, default=0.1, help="bps per tick")
    p_sim.add_argument("--seed", type=int, default=None)
    p_sim.set_defaults(func=cmd_sim)

    p_live = sub.add_parser("live", help="live Binance depth, paper fills")
    add_common(p_live)
    p_live.add_argument(
        "--product", default="spot", choices=("spot", "perp"),
        help="どちらの板で動かすか",
    )
    p_live.add_argument(
        "--depth-ms", type=int, default=100, choices=(100, 250, 500, 1000),
        help="現物は100/1000、先物は100/250/500",
    )
    p_live.set_defaults(func=cmd_live)

    p_rec = sub.add_parser("record", help="capture a live session to JSONL")
    add_common(p_rec)
    p_rec.add_argument("--out", required=True)
    p_rec.add_argument(
        "--product", default="spot", choices=("spot", "perp"),
        help="どちらの板で動かすか",
    )
    p_rec.add_argument(
        "--depth-ms", type=int, default=100, choices=(100, 250, 500, 1000),
        help="現物は100/1000、先物は100/250/500",
    )
    p_rec.set_defaults(func=cmd_record)

    p_rep = sub.add_parser("replay", help="replay a capture")
    add_common(p_rep)
    p_rep.add_argument("path")
    p_rep.add_argument("--speed", type=float, default=1.0, help="0 = as fast as possible")
    p_rep.add_argument(
        "--source", default=None,
        help="capture 録画のどちらを再生するか (spot / perp)",
    )
    p_rep.set_defaults(func=cmd_replay)

    p_sw = sub.add_parser("sweep", help="replay one recording under many settings")
    add_common(p_sw)
    p_sw.add_argument("path", help="a .jsonl recording from `capture`")
    p_sw.add_argument(
        "--distances",
        default=None,
        help="ticks behind the touch to try; 'none' = uncapped (単独時の既定: 0,1,2,4,none)",
    )
    p_sw.add_argument(
        "--sizes",
        default="",
        help="quote sizes to try (default: just --size)",
    )
    p_sw.add_argument(
        "--requotes",
        default="",
        help="再計算間隔msの一覧（例: 100,50,20,10）",
    )
    p_sw.add_argument(
        "--latencies",
        default="",
        help="注文到着遅延msの一覧（例: 20,10,5,2）",
    )
    p_sw.add_argument(
        "--axis",
        action="append",
        metavar="NAME=V1,V2",
        help="任意の設定を軸にする (例: --axis gamma=0.6,3,9)。繰り返し指定可",
    )
    p_sw.add_argument(
        "--source", default=None,
        help="capture 録画のどちらを再生するか (spot / perp)",
    )
    p_sw.add_argument(
        "--plain",
        action="store_true",
        help="表ではなくタブ区切りで出す（折り返さないので貼り付けやすい）",
    )
    p_sw.set_defaults(func=cmd_sweep, headless=True)

    p_hg = sub.add_parser("hedge", help="メイク＋即時ヘッジの最終損益テスト")
    add_common(p_hg)
    p_hg.add_argument("path", help="capture で録った .jsonl")
    p_hg.add_argument("--maker-source", default="spot", help="どちらでメイクするか")
    p_hg.add_argument("--hedge-source", default="perp", help="どちらでヘッジするか")
    p_hg.add_argument("--hedge-ratio", type=float, default=1.0, help="0 でヘッジなし")
    p_hg.add_argument("--maker-fees", default="0,1,2", help="試すメイカー手数料 bps")
    p_hg.add_argument("--plain", action="store_true", help="タブ区切りで出す")
    # min_edge_bps normally tracks the maker fee, which is right when trading
    # and wrong here: it would make each fee level quote differently, so the
    # three rows would compare three strategies rather than one strategy at
    # three prices. Held at zero so only the cost varies.
    p_hg.set_defaults(
        func=cmd_hedge, headless=True, max_distance=0, requote_ms=100.0, min_edge_bps=0.0
    )

    p_tri = sub.add_parser("triage", help="手数料・tick・スプレッドだけで全銘柄を一次審査")
    p_tri.add_argument("--product", default="perp", choices=("spot", "perp"))
    p_tri.add_argument("--maker-bps", type=float, default=10.0)
    p_tri.add_argument("--min-ticks", type=float, default=2.0, help="必要なスプレッド幅")
    p_tri.add_argument(
        "--max-tick-bps", type=float, default=2.0,
        help="1tick の上限。太いと逆選択もヘッジ代も比例して重くなる",
    )
    p_tri.add_argument("--quote-asset", default="USDT")
    p_tri.add_argument("--min-volume", type=float, default=5_000_000.0)
    p_tri.add_argument(
        "--min-trades", type=int, default=20_000,
        help="24h約定件数の下限。板が広いのは誰も居ないからかもしれない",
    )
    p_tri.add_argument("--samples", type=int, default=3, help="板を何回見るか")
    p_tri.add_argument("--sample-interval", type=float, default=2.0)
    p_tri.add_argument("--top", type=int, default=25)
    p_tri.set_defaults(func=cmd_triage)

    p_prb = sub.add_parser("probe", help="候補を5分まとめて動的審査 (逆選択/スプレッド)")
    p_prb.add_argument("--symbols", default="", help="カンマ区切り。省略すると triage の候補")
    p_prb.add_argument(
        "--replay", default=None,
        help="ライブではなく録画に対して同じ計算式を走らせる（検算用）",
    )
    p_prb.add_argument("--source", default=None, help="録画のどちらを使うか (spot / perp)")
    p_prb.add_argument("--depth", type=int, default=20)
    p_prb.add_argument("--tick-size", default=None)
    p_prb.add_argument("--lot-size", default=None)
    p_prb.add_argument("--duration", type=float, default=300.0)
    p_prb.add_argument("--horizon-ms", type=float, default=100.0)
    p_prb.add_argument("--min-trades", type=int, default=500, help="判定に要る約定数")
    p_prb.add_argument(
        "--min-sweeps", type=int, default=100,
        help="板を消した約定がこの数あれば、そちらで判定する",
    )
    p_prb.add_argument("--max-symbols", type=int, default=20)
    p_prb.add_argument("--top", type=int, default=30)
    # Passed through to the static screen when --symbols is omitted.
    p_prb.add_argument("--product", default="perp", choices=("spot", "perp"))
    p_prb.add_argument("--maker-bps", type=float, default=2.0)
    p_prb.add_argument("--min-ticks", type=float, default=2.0)
    p_prb.add_argument("--max-tick-bps", type=float, default=2.0)
    p_prb.add_argument("--quote-asset", default="USDT")
    p_prb.add_argument("--min-volume", type=float, default=5_000_000.0)
    p_prb.add_argument("--min-trades-24h", type=int, default=20_000)
    p_prb.add_argument("--samples", type=int, default=3)
    p_prb.set_defaults(func=cmd_probe)

    p_bt = sub.add_parser("backtest", help="headless synthetic run")
    add_common(p_bt)
    p_bt.add_argument("--start-price", type=float, default=64_000.0)
    p_bt.add_argument("--interval", type=float, default=0.0, help="0 = no wall-clock delay")
    p_bt.add_argument("--volatility", type=float, default=0.1)
    p_bt.add_argument("--seed", type=int, default=7)
    p_bt.set_defaults(func=cmd_sim, headless=True)

    p_scan = sub.add_parser("scan", help="全銘柄を走査し、手数料を超えるスプレッドを探す")
    add_scan_args(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_watch = sub.add_parser("watch", help="スキャンを繰り返し、持続して成立する銘柄を探す")
    add_scan_args(p_watch)
    p_watch.add_argument("--rounds", type=int, default=12, help="スキャンの回数")
    p_watch.add_argument("--every", type=float, default=300.0, help="スキャンの間隔（秒）")
    p_watch.set_defaults(func=cmd_watch)

    p_hz = sub.add_parser("horizon", help="値動きが手数料を超えるかを過去データで測る")
    p_hz.add_argument("--symbol", default="BTCUSDT")
    p_hz.add_argument("--product", default="perp", choices=("perp", "spot"))
    p_hz.add_argument("--days", type=int, default=7, help="さかのぼる日数")
    p_hz.add_argument("--end", default=None, help="最終日 YYYY-MM-DD。既定は直近")
    p_hz.add_argument(
        "--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 30, 60], help="保有秒数"
    )
    p_hz.add_argument("--taker-bps", type=float, default=4.5, help="片道テイカー手数料")
    p_hz.add_argument("--slippage-ticks", type=float, default=1.0, help="片道の想定滑り")
    p_hz.add_argument("--tick-size", default=None, help="価格の刻み。既定は取引所から取得")
    p_hz.set_defaults(func=cmd_horizon)

    p_pr = sub.add_parser("predict", help="信号が手数料を超える的中率に届くか測る")
    p_pr.add_argument("--symbol", default="BTCUSDT")
    p_pr.add_argument("--product", default="perp", choices=("perp", "spot"))
    p_pr.add_argument("--days", type=int, default=14)
    p_pr.add_argument("--end", default=None, help="最終日 YYYY-MM-DD")
    p_pr.add_argument("--horizon", type=int, default=3600, help="保有秒数")
    p_pr.add_argument("--train-fraction", type=float, default=0.6, help="学習に使う割合")
    p_pr.add_argument("--vol-window", type=int, default=900, help="出動条件の観測窓（秒）")
    p_pr.add_argument(
        "--vol-quantile", type=float, default=0.9, help="この分位を超えたときだけ売買。0で常時"
    )
    p_pr.add_argument("--min-samples", type=int, default=200)
    p_pr.add_argument("--taker-bps", type=float, default=4.5)
    p_pr.add_argument("--slippage-ticks", type=float, default=1.0)
    p_pr.add_argument("--tick-size", default=None, help="価格の刻み。既定は取引所から取得")
    p_pr.set_defaults(func=cmd_predict)

    p_ev = sub.add_parser("events", help="条件別の純損益を並べる")
    p_ev.add_argument("--symbol", default="BTCUSDT")
    p_ev.add_argument("--days", type=int, default=14)
    p_ev.add_argument("--end", default=None, help="最終日 YYYY-MM-DD")
    # 60 rather than 30: the horizon study measured a 6.39bps average move at
    # 30 minutes against a 9.03bps round trip, so that horizon is below the
    # fee before any condition is applied. 60 minutes averages 22.93bps.
    p_ev.add_argument("--horizon", type=int, default=60, help="保有分数")
    p_ev.add_argument("--target", type=float, default=None, help="利確 bps")
    p_ev.add_argument("--stop", type=float, default=None, help="損切 bps")
    p_ev.add_argument("--z-window", type=int, default=1440, help="z値の観測窓（分）")
    p_ev.add_argument("--taker-bps", type=float, default=4.5)
    p_ev.add_argument("--slippage-ticks", type=float, default=1.0)
    p_ev.add_argument("--tick-size", default=None)
    p_ev.set_defaults(func=cmd_events)

    p_cap = sub.add_parser("capture", help="現物と先物を同時に記録する")
    p_cap.add_argument("--symbol", default="BTCUSDT")
    p_cap.add_argument("--out", default="capture.jsonl")
    p_cap.add_argument("--duration", type=float, default=None, help="秒。省略で Ctrl-C まで")
    p_cap.add_argument("--max-events", type=int, default=None)
    p_cap.add_argument("--spot-depth-ms", type=int, default=100, choices=(100, 1000))
    p_cap.add_argument("--perp-depth-ms", type=int, default=100, choices=(100, 250, 500))
    p_cap.add_argument("--oi-interval", type=float, default=15.0, help="建玉残高の取得間隔（秒）")
    p_cap.add_argument("--spot-only", action="store_true")
    p_cap.add_argument("--perp-only", action="store_true")
    p_cap.add_argument(
        "--perp-fallback",
        default="auto",
        choices=FALLBACK_MODES,
        help="先物の約定・マークをRESTで取るか。auto は無音を10秒待って切替",
    )
    p_cap.add_argument("--fallback-after", type=float, default=10.0, help="auto の待ち時間（秒）")
    p_cap.add_argument("--trade-poll", type=float, default=1.0, help="REST約定の取得間隔（秒）")
    p_cap.add_argument("--mark-poll", type=float, default=1.0, help="RESTマークの取得間隔（秒）")
    p_cap.set_defaults(func=cmd_capture)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # A headless backtest with no stop condition would never return.
    if (
        args.command == "backtest"
        and getattr(args, "max_events", None) is None
        and getattr(args, "duration", None) is None
    ):
        args.max_events = 5_000

    try:
        return asyncio.run(args.func(args))
    except ConfigError as exc:
        # A setting that would make the session do nothing. Say so plainly
        # rather than showing a traceback for what is a usage question.
        console.print(f"[red]{exc}[/red]")
        return 2
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
