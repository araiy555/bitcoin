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
import statistics
import sys
import time
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
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
from .feed.bybit import BybitFeed
from .feed.replay import JsonlRecorder, ReplayFeed, SyntheticFeed, iter_tagged, iter_tagged_timed
from .mm.fair_value import FairValueConfig, FairValueEstimator
from .mm.inventory import FeeSchedule, Position
from .mm.quoter import Quoter, QuoterConfig
from .mm.risk import RiskLimits, RiskManager
from .mm.strategy import MarketMaker, StrategyConfig
from .mm.toxicity import ToxicityConfig, ToxicityGate
from .net import describe_tls_error, make_session
from .research import dynamic, statarb, triage
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
from .sim.cross_exchange import CrossArbConfig, CrossExchangeArb
from .sim.dealer import BinanceDealer
from .sim.hedge import HedgeConfig, Hedger
from .sim.pair import CrossMarketFairValue, PairQuoteGate
from .sim.paper import PAPER_OWNER, PaperConfig, PaperVenue
from .sim.runner import attach_virtual_clock, run
from .sim.s3 import RotatingJsonlSink, S3Target
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


async def fetch_bybit_instrument(symbol: str, category: str = "linear") -> Instrument:
    """Read tick and quantity steps from Bybit V5 instrument metadata."""
    import aiohttp

    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {"category": category, "symbol": symbol.upper()}
    async with make_session() as session, session.get(
        url, params=params, timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    if payload.get("retCode") != 0:
        raise RuntimeError(payload.get("retMsg") or "Bybit instruments-info failed")
    rows = (payload.get("result") or {}).get("list") or []
    if not rows:
        raise RuntimeError(f"{symbol.upper()} is not listed on Bybit {category}")
    info = rows[0]
    tick = (info.get("priceFilter") or {}).get("tickSize")
    lot = (info.get("lotSizeFilter") or {}).get("qtyStep")
    if not tick or not lot:
        raise RuntimeError(f"Bybit instrument info for {symbol.upper()} had no tick/lot")
    return Instrument(
        symbol=info["symbol"],
        tick_size=Decimal(tick).normalize(),
        lot_size=Decimal(lot).normalize(),
        base=info.get("baseCoin", ""),
        quote=info.get("quoteCoin", ""),
    )


class ConfigError(Exception):
    """A setting that would make the session do nothing, caught at startup."""


DEFAULT_SIZE = "0.01"
DEFAULT_MAX_POSITION = "0.10"


def _adapt_generic_defaults(
    instrument: Instrument,
    args: argparse.Namespace,
    *,
    skip_report: frozenset[str] = frozenset(),
) -> list[str]:
    """Rescale the untouched BTC-shaped defaults to the instrument at hand.

    `sweep` and `hedge` already did this; every other command left the user to
    discover the lot size by hitting two refusals in a row. Nobody asked for
    0.01 units — it is what the parser fills in when nothing is said — so
    adapting it is honouring the request rather than overriding it.

    A value the user typed is left alone, including a wrong one: _check_sizes
    still refuses it, because silently trading a size nobody asked for is worse
    than stopping. That distinction is why the parser default is None rather
    than "0.01" — an explicit `--size 0.01` on a whole-token instrument must
    still fail, and a string default cannot tell the two apart.
    """
    changed: list[str] = []
    if args.size is None:
        if instrument.to_lots(DEFAULT_SIZE) > 0:
            args.size = DEFAULT_SIZE
        else:
            text = f"{instrument.lot_size * 100:f}"
            # Strip only a fractional tail: "100" must not become "1".
            args.size = text.rstrip("0").rstrip(".") if "." in text else text
            if "size" not in skip_report:
                changed.append(f"--size {args.size}")
    if args.max_position is None:
        if instrument.to_lots(DEFAULT_MAX_POSITION) >= instrument.to_lots(args.size):
            args.max_position = DEFAULT_MAX_POSITION
        else:
            args.max_position = str(Decimal(args.size) * 10)
            if "max_position" not in skip_report:
                changed.append(f"--max-position {args.max_position}")
    return changed


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
    adapted = _adapt_generic_defaults(instrument, args)
    if adapted:
        console.print(
            f"[dim]{instrument.symbol} の最小単位に合わせました: {' '.join(adapted)}[/dim]"
        )
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
        toxicity=ToxicityGate(
            ToxicityConfig(
                # Some programmatic callers build a minimal Namespace rather
                # than going through argparse.  Keep the pre-toxicity API
                # compatible by applying the CLI defaults here as well.
                threshold=getattr(args, "toxicity_threshold", 0.0),
                pull_threshold=getattr(args, "toxicity_pull_threshold", 0.90),
                depth_levels=getattr(args, "toxicity_depth", 5),
                flow_weight=getattr(args, "toxicity_flow_weight", 0.50),
                book_weight=getattr(args, "toxicity_book_weight", 0.30),
                microprice_weight=getattr(args, "toxicity_microprice_weight", 0.20),
            )
        ),
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


def _toxicity_line(s: dict) -> str | None:
    t = s.get("toxicity") or {}
    if not t or t.get("threshold", 0.0) <= 0:
        return None
    return (
        f"  毒性フィルター : threshold {t['threshold']:.2f}"
        f"  片側化 {t['one_sided_share'] * 100:.1f}%"
        f"  全取消 {t['pull_share'] * 100:.1f}%"
        f"  最終score {t['last_score']:+.2f}"
    )


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
        _toxicity_line(s),
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
        (
            args.sizes,
            getattr(args, "requotes", ""),
            getattr(args, "latencies", ""),
            getattr(args, "toxicity_thresholds", ""),
            args.axis,
        )
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
    if getattr(args, "toxicity_thresholds", ""):
        axes["toxicity_threshold"] = _grid(args.toxicity_thresholds, float)

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


def _prepare_sweep_defaults(
    instrument: Instrument, args: argparse.Namespace, axes: dict[str, list]
) -> list[str]:
    """Make generic BTC defaults executable for the recorded instrument.

    The common parser defaults to 0.01 units with a 0.10 position cap. That is
    valid for BTC but rounds to zero on instruments such as USUSDT whose lot is
    one whole token. A sweep should compare its requested axes, not skip every
    row because an unrelated generic default cannot be represented.

    Explicit size/max-position sweep axes remain untouched so invalid requested
    combinations are still rejected visibly by _check_sizes.
    """
    # A swept axis replaces the value on every run, so adapting the base is
    # invisible and reporting it would be noise.
    changed = _adapt_generic_defaults(instrument, args, skip_report=frozenset(axes))
    if "size" not in axes and instrument.to_lots(args.size) <= 0:
        args.size = str(instrument.lot_size * 100)
        changed.append(f"size={args.size}")

    if (
        "max_position" not in axes
        and instrument.to_lots(args.max_position) < instrument.to_lots(args.size)
    ):
        args.max_position = str(Decimal(args.size) * 10)
        changed.append(f"max_position={args.max_position}")
    return changed


def _prepare_pair_defaults(
    maker: Instrument, hedge: Instrument, args: argparse.Namespace
) -> list[str]:
    """Choose a size representable on both legs of a paired trade.

    A quantity valid on the maker venue can still round to zero on the hedge
    venue when their lot sizes differ.  Silently reporting zero opportunities
    in that case is a configuration error disguised as a market result.
    """
    changed = _adapt_generic_defaults(maker, args)
    if maker.to_lots(args.size) <= 0 or hedge.to_lots(args.size) <= 0:
        args.size = str(max(maker.lot_size, hedge.lot_size) * 100)
        changed.append(f"size={args.size}")

    size_lots = maker.to_lots(args.size)
    if maker.to_lots(args.max_position) < size_lots:
        args.max_position = str(Decimal(args.size) * 10)
        changed.append(f"max_position={args.max_position}")
    return changed


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
    tox = s.get("toxicity") or {}
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
        "tox_one_sided_pct": tox.get("one_sided_share", 0.0) * 100.0,
        "tox_pull_pct": tox.get("pull_share", 0.0) * 100.0,
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

    adjusted = _prepare_sweep_defaults(instrument, args, axes)
    if adjusted:
        console.print(
            "[dim]録画銘柄のlotに合わせて未指定値を自動調整: "
            + "  ".join(adjusted)
            + "[/dim]"
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
        ("毒性片側%", "tox_one_sided_pct", "{:.0f}"),
        ("毒性全取消%", "tox_pull_pct", "{:.0f}"),
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
    for source, instrument in (
        (args.maker_source, maker_inst),
        (args.hedge_source, hedge_inst),
    ):
        if not instrument.base or not instrument.quote:
            raise ConfigError(
                f"{source} の銘柄情報が不完全です（base/quoteなし）。"
                "exchangeInfo取得失敗時の代替値で録画された可能性があります。\n"
                "  この録画では相対価値を判定できません。現物と先物の両方に上場する"
                "銘柄で capture し直してください。"
            )

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


async def cmd_pair(args: argparse.Namespace) -> int:
    """Price the maker venue from a second book and hedge every fill.

    Unlike ``hedge``, this does not quote first and ask whether hedging helped
    afterwards.  A proposed order reaches the paper venue only when the
    visible hedge depth clears maker fee, taker fee and the requested safety
    margin at that moment.
    """
    path = Path(args.path)
    if not path.exists():
        console.print(f"[red]{path} がありません。[/red]")
        return 1

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ConfigError(f"{meta_path.name} がありません。capture で録った録画が要ります。")
    sources = json.loads(meta_path.read_text()).get("sources") or {}
    if args.maker_source == args.hedge_source:
        raise ConfigError("--maker-source と --hedge-source は別の市場を指定してください。")
    for option, name in (("maker", args.maker_source), ("hedge", args.hedge_source)):
        if name not in sources:
            raise ConfigError(
                f"--{option}-source {name} は録画にありません。"
                f"使えるのは: {', '.join(sorted(sources))}"
            )

    maker_inst = _instrument_from_spec(sources[args.maker_source])
    hedge_inst = _instrument_from_spec(sources[args.hedge_source])
    for source, instrument in (
        (args.maker_source, maker_inst),
        (args.hedge_source, hedge_inst),
    ):
        if not instrument.base or not instrument.quote:
            raise ConfigError(
                f"{source} の銘柄情報が不完全です（base/quoteなし）。"
                "exchangeInfo取得失敗時の代替値で録画された可能性があります。\n"
                "  この録画では相対価値を判定できません。現物と先物の両方に上場する"
                "銘柄で capture し直してください。"
            )
    if maker_inst.base and hedge_inst.base and maker_inst.base != hedge_inst.base:
        raise ConfigError(
            f"ベース資産が違います: {maker_inst.base} / {hedge_inst.base}。"
            "同じ数量をヘッジできません。"
        )
    if maker_inst.quote and hedge_inst.quote and maker_inst.quote != hedge_inst.quote:
        raise ConfigError(
            f"建て通貨が違います: {maker_inst.quote} / {hedge_inst.quote}。"
            "為替換算なしでは比較できません。"
        )

    adjusted = _prepare_pair_defaults(maker_inst, hedge_inst, args)
    if adjusted:
        console.print(
            "[dim]録画銘柄のlotに合わせて未指定値を自動調整: "
            + "  ".join(adjusted)
            + "[/dim]"
        )

    console.print(
        f"{path.name} — ヘッジ市場から価格を作る相対価値MM\n"
        f"  メイク: {args.maker_source} {maker_inst.symbol} "
        f"tick={maker_inst.tick_size} lot={maker_inst.lot_size}\n"
        f"  ヘッジ: {args.hedge_source} {hedge_inst.symbol} "
        f"tick={hedge_inst.tick_size} lot={hedge_inst.lot_size}\n"
        f"  条件  : maker + taker({args.taker_bps:g}bps) + 板歩き後に "
        f"{args.pair_edge_bps:g}bps 以上 / hedge age ≤ {args.max_hedge_age_ms:g}ms\n"
        f"  更新  : requote {args.requote_ms:g}ms / hedge latency {args.hedge_latency_ms:g}ms"
    )

    rows = []
    for maker_bps in _grid(args.maker_fees, float):
        run_args = argparse.Namespace(**vars(args))
        run_args.maker_bps = maker_bps
        _check_sizes(maker_inst, run_args)
        row = await _run_pair(path, maker_inst, hedge_inst, run_args)
        row["maker_bps"] = maker_bps
        rows.append(row)

    columns = [
        ("maker", "maker_bps", "{:g}"),
        ("候補%", "pass_share", "{:.1f}"),
        ("最良edge", "best_edge", "{:+.2f}"),
        ("約定", "fills", "{:,.0f}"),
        ("ヘッジ", "hedges", "{:,.0f}"),
        ("未ヘッジ", "skipped", "{:,.0f}"),
        ("make.spread", "maker_spread", "{:+.2f}"),
        ("make.在庫", "maker_inventory", "{:+.2f}"),
        ("hedge.cross", "hedge_cross", "{:+.2f}"),
        ("basis", "hedge_inventory", "{:+.2f}"),
        ("手数料", "fees", "{:+.2f}"),
        ("Net bps", "net_bps", "{:+.2f}"),
    ]

    def cell(row, key, fmt):
        value = row[key]
        return "—" if not math.isfinite(value) else fmt.format(value)

    if args.plain:
        console.print("\t".join(c[0] for c in columns), highlight=False)
        for row in rows:
            console.print("\t".join(cell(row, key, fmt) for _, key, fmt in columns), highlight=False)
    else:
        table = Table(title=f"{maker_inst.symbol} — 相対価値MM＋即時ヘッジ", padding=(0, 1))
        for header, _, _ in columns:
            table.add_column(header, justify="right")
        for row in rows:
            cells = [cell(row, key, fmt) for _, key, fmt in columns]
            net = row["net_bps"]
            colour = "dim" if not math.isfinite(net) else ("green" if net > 0 else "red")
            cells[-1] = f"[{colour}]{cells[-1]}[/{colour}]"
            table.add_row(*cells)
        console.print(table)

    if all(row["quotes_passed"] == 0 for row in rows):
        console.print(
            "\n[yellow][bold]判定: この録画には、指定した執行コスト後に出せる注文がありません。[/bold][/yellow]\n"
            "[dim]損したのではなく、条件外なので一度もリスクを取りません。"
            "別銘柄か別期間を capture して同じ判定へ回します。[/dim]"
        )
    elif all(row["fills"] == 0 for row in rows):
        console.print(
            "\n[yellow][bold]判定: 価格差はありましたが、注文の順番が回らず約定していません。[/bold][/yellow]\n"
            "[dim]録画を長くするか、候補%が高い別銘柄を比較してください。[/dim]"
        )
    else:
        profitable = [row for row in rows if math.isfinite(row["net_bps"]) and row["net_bps"] > 0]
        if profitable:
            console.print(
                "\n[green][bold]判定: 全執行コスト後でプラスの行があります。[/bold][/green]\n"
                "[dim]同じ設定を固定し、別時間の録画で out-of-sample 検証してください。[/dim]"
            )
        else:
            console.print(
                "\n[red][bold]判定: 価格差を事前選別しても、約定後の全コストで赤字です。[/bold][/red]\n"
                "[dim]この録画・銘柄では採用しません。[/dim]"
            )
    return 0


async def cmd_dealer(args: argparse.Namespace) -> int:
    """Automatically rank every Binance spot/perpetual maker route."""
    path = Path(args.path)
    if not path.exists():
        raise ConfigError(f"{path} がありません。先に capture でspot/perpを録画してください。")
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ConfigError(f"{meta_path.name} がありません。capture で録った録画が要ります。")
    sources = json.loads(meta_path.read_text()).get("sources") or {}
    if not {"spot", "perp"}.issubset(sources):
        raise ConfigError("dealerにはspotとperpを同時に含むcapture録画が必要です。")
    instruments = {
        source: _instrument_from_spec(sources[source]) for source in ("spot", "perp")
    }
    spot = instruments["spot"]
    perp = instruments["perp"]
    if not spot.base or not spot.quote or not perp.base or not perp.quote:
        raise ConfigError("spot/perpのbase・quote情報がありません。captureし直してください。")
    if spot.base != perp.base or spot.quote != perp.quote:
        raise ConfigError(
            f"同じ資産の市場ではありません: {spot.base}/{spot.quote} と "
            f"{perp.base}/{perp.quote}"
        )
    fees = (
        args.spot_maker_bps,
        args.spot_taker_bps,
        args.perp_maker_bps,
        args.perp_taker_bps,
    )
    if any(value < 0 for value in fees):
        raise ConfigError("maker/taker手数料は0以上で指定してください。")
    if args.size_base <= 0 or args.sample_ms < 0 or args.depth <= 0:
        raise ConfigError(
            "--size-baseと--depthは0より大きく、--sample-msは0以上にしてください。"
        )

    minimum = max(float(spot.lot_size), float(perp.lot_size))
    size_base = max(args.size_base, minimum)
    if size_base != args.size_base:
        console.print(
            f"[dim]両市場のlotに合わせてsize-baseを {size_base:g} へ調整しました。[/dim]"
        )
    dealer = BinanceDealer(
        instruments=instruments,
        maker_bps={"spot": args.spot_maker_bps, "perp": args.perp_maker_bps},
        taker_bps={"spot": args.spot_taker_bps, "perp": args.perp_taker_bps},
        size_base=size_base,
        min_net_bps=args.dealer_edge_bps,
        max_age_ms=args.max_age_ms,
        funding_horizon_h=args.funding_horizon_h,
        depth=args.depth,
    )

    console.print(
        f"{path.name} — Binance市場横断dealer\n"
        f"  資産   : {spot.base}/{spot.quote} / size {size_base:g} {spot.base}\n"
        f"  経路   : spot/perp × maker買い・売り = 4経路を毎時点で比較\n"
        f"  spot   : maker {args.spot_maker_bps:g}bps / taker {args.spot_taker_bps:g}bps\n"
        f"  perp   : maker {args.perp_maker_bps:g}bps / taker {args.perp_taker_bps:g}bps\n"
        f"  条件   : 板歩き＋手数料＋次回予想funding後 {args.dealer_edge_bps:g}bps以上\n"
        f"  鮮度   : 両市場 {args.max_age_ms:g}ms以内 / 観測間隔 {args.sample_ms:g}ms"
    )

    last_evaluation_ns = 0
    interval_ns = int(args.sample_ms * 1e6)
    for source, received_ns, event in iter_tagged_timed(path):
        if source not in instruments:
            continue
        observable_ns = received_ns or getattr(event, "ts_ns", 0)
        if received_ns:
            event = replace(event, ts_ns=received_ns)
        dealer.apply(source, event, observable_ns)
        if observable_ns and observable_ns - last_evaluation_ns < interval_ns:
            continue
        dealer.evaluate()
        last_evaluation_ns = observable_ns

    columns = [
        ("経路", None),
        ("観測", None),
        ("執行可能%", None),
        ("候補%", None),
        ("最良選択", None),
        ("中央値edge", None),
        ("最良edge", None),
    ]
    rows = []
    for route, stats in dealer.stats.items():
        median = stats.median_net_bps
        best = stats.best_net_bps
        rows.append(
            (
                route.label,
                f"{stats.observations:,}",
                f"{stats.executable_share * 100:.1f}",
                f"{stats.candidate_share * 100:.2f}",
                f"{stats.selected:,}",
                "—" if not math.isfinite(median) else f"{median:+.2f}",
                "—" if not math.isfinite(best) else f"{best:+.2f}",
            )
        )
    if args.plain:
        console.print("\t".join(header for header, _ in columns), highlight=False)
        for row in rows:
            console.print("\t".join(row), highlight=False)
    else:
        table = Table(title=f"{spot.symbol} — 自動経路選択", padding=(0, 1))
        for header, _ in columns:
            table.add_column(header, justify="right" if header != "経路" else "left")
        for row in rows:
            table.add_row(*row)
        console.print(table)

    total_candidates = sum(stats.candidates for stats in dealer.stats.values())
    total_selected = sum(stats.selected for stats in dealer.stats.values())
    if total_candidates == 0:
        console.print(
            "\n[yellow][bold]判定: 4経路のどこにも建玉時コスト後の提示候補がありません。[/bold][/yellow]\n"
            "[dim]この録画ではdealerは注文を出さず、在庫も取りません。[/dim]"
        )
    else:
        console.print(
            f"\n[green][bold]提示候補を{total_candidates:,}件、時点別の最良経路を"
            f"{total_selected:,}件検出しました。[/bold][/green]\n"
            "[yellow]これは利益ではありません。[/yellow] maker注文の順番・未約定・"
            "約定後basisを含むpair replayを通過して初めて収益候補になります。"
        )
    return 0


async def _run_pair(path: Path, maker_inst: Instrument, hedge_inst: Instrument, args) -> dict:
    maker = build_maker(maker_inst, args)
    hedge_view = MarketView(instrument=hedge_inst, depth=args.depth)
    hedger = Hedger(
        instrument=hedge_inst,
        market=hedge_view,
        config=HedgeConfig(ratio=1.0, taker_bps=args.taker_bps, max_levels=args.depth),
    )

    now_ns = [0]

    def clock() -> int:
        return now_ns[0] or time.time_ns()

    maker.clock = clock
    maker.venue.clock = clock
    maker.market.clock = clock
    hedge_view.clock = clock
    maker.fair_value = CrossMarketFairValue(maker_inst, hedge_inst, hedge_view)  # type: ignore[assignment]
    gate = PairQuoteGate(
        maker_instrument=maker_inst,
        hedge_instrument=hedge_inst,
        maker_market=maker.market,
        hedge_market=hedge_view,
        hedger=hedger,
        maker_bps=args.maker_bps,
        taker_bps=args.taker_bps,
        min_net_bps=args.pair_edge_bps,
        max_hedge_age_ms=args.max_hedge_age_ms,
        clock=clock,
    )
    maker.quote_filter = gate.filter

    maker_turnover = 0.0
    pending_hedges: list[tuple[int, int, float]] = []

    def flush_hedges() -> None:
        ready = [item for item in pending_hedges if item[0] <= clock()]
        if not ready:
            return
        pending_hedges[:] = [item for item in pending_hedges if item[0] > clock()]
        for _due_ns, sign, qty_base in ready:
            hedger.on_maker_fill(sign, qty_base)

    for src, received_ns, event in iter_tagged_timed(path):
        if received_ns:
            # Receive timestamps are monotonic in a capture.  Preserve that
            # property for old hand-built fixtures whose event clocks may not be.
            now_ns[0] = max(now_ns[0], received_ns)
            # Execution can react only when this process received an update.
            # Re-stamp the replay event so requote intervals, staleness and
            # inventory age all live on that same observable clock.
            event = replace(event, ts_ns=now_ns[0])
        if src == args.hedge_source:
            hedge_view.apply(event)
            hedger.on_market()
            flush_hedges()
            # Reference-market changes are precisely when a cross-market
            # maker must cancel or move a stale quote.
            maker.requote()
            continue
        if src != args.maker_source:
            continue

        flush_hedges()
        for fill in maker.on_event(event):
            if fill.maker_owner != PAPER_OWNER:
                continue
            maker_turnover += maker_inst.notional(fill.price, fill.qty)
            sign = fill.aggressor.opposite.sign
            due_ns = clock() + int(args.hedge_latency_ms * 1e6)
            pending_hedges.append((due_ns, sign, maker_inst.qty_f(fill.qty)))
            flush_hedges()
        maker.requote()

    maker.flatten()
    make_attr = maker.attribution
    hedge_attr = hedger.attribution
    scale = 10_000.0 / maker_turnover if maker_turnover > 0 else math.nan
    best_edge = gate.stats.best_net_bps
    if not math.isfinite(best_edge):
        best_edge = math.nan
    return {
        "fills": int(maker.position.fill_count),
        "hedges": int(hedger.hedges),
        "skipped": int(hedger.skipped_no_book + len(pending_hedges)),
        "quotes_tested": int(gate.stats.quotes_tested),
        "quotes_passed": int(gate.stats.quotes_passed),
        "pass_share": gate.stats.pass_share * 100.0,
        "best_edge": best_edge,
        "maker_spread": make_attr.spread_capture * scale,
        "maker_inventory": make_attr.inventory_pnl * scale,
        "hedge_cross": hedge_attr.spread_capture * scale,
        "hedge_inventory": hedge_attr.inventory_pnl * scale,
        "fees": -(make_attr.fees + hedge_attr.fees) * scale,
        "net_bps": (make_attr.total + hedge_attr.total) * scale,
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


async def cmd_xcapture(args: argparse.Namespace) -> int:
    """Record Binance and Bybit perpetual books on one receive-time clock."""
    try:
        binance = await fetch_futures_instrument(args.symbol)
        bybit = await fetch_bybit_instrument(args.symbol, "linear")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]取引所の銘柄仕様を取得できません: {type(exc).__name__}: {exc}[/red]")
        hint = describe_tls_error(exc)
        if hint:
            console.print(f"[yellow]{hint}[/yellow]")
        return 1
    if binance.base != bybit.base or binance.quote != bybit.quote:
        raise ConfigError(
            f"同じ契約ではありません: Binance {binance.base}/{binance.quote}, "
            f"Bybit {bybit.base}/{bybit.quote}"
        )

    sources = {
        "binance": BinanceFuturesFeed(
            binance,
            depth_ms=args.binance_depth_ms,
            open_interest_interval=60.0,
            rest_fallback="never",
        ),
        "bybit": BybitFeed(bybit, category="linear", depth=args.bybit_depth),
    }
    specs = {
        "binance": {**_spec_dict(binance, "perp"), "venue": "binance"},
        "bybit": {**_spec_dict(bybit, "perp"), "venue": "bybit"},
    }
    out = Path(args.out)
    capture = MultiCapture(sources, out)
    console.rule(f"[bold cyan]{args.symbol.upper()} 取引所間録画")
    console.print(
        f"  Binance: tick={binance.tick_size} lot={binance.lot_size}\n"
        f"  Bybit  : tick={bybit.tick_size} lot={bybit.lot_size}\n"
        f"  出力   : {out}\n"
        f"  停止   : {args.duration:,.0f}秒後\n"
        "  [dim]二つのWebSocketを同時に受信し、ローカル受信時刻順で保存します。[/dim]"
    )

    last_report = [time.monotonic()]

    def on_event(_name, _event) -> None:
        now = time.monotonic()
        if now - last_report[0] < 10.0:
            return
        last_report[0] = now
        parts = [
            f"{name}[{st.status}] {st.events:,}"
            for name, st in capture.stats.items()
        ]
        console.print(f"  [dim]{' | '.join(parts)}[/dim]")

    capture.on_event = on_event
    result = await capture.run(duration_s=args.duration, max_events=args.max_events)
    meta = write_meta(out, specs)
    console.rule("[bold cyan]取引所間録画完了")
    console.print(
        f"  時間     : {result.duration_s:,.1f}秒\n"
        f"  イベント : {result.total_events:,}件\n"
        f"  出力     : {out} ({out.stat().st_size / 1e6:,.1f} MB)\n"
        f"  メタ     : {meta.name}"
    )
    for name, st in result.stats.items():
        console.print(f"  {name}: {st.events:,}件 / 切断 {st.errors}回")
    console.rule()
    return 0


def _replay_two_market_capture(
    path: Path,
    engine: CrossExchangeArb,
    instruments: dict[str, Instrument],
    sample_ms: float,
) -> None:
    """Stream a tagged capture and show coarse progress for large recordings."""
    interval_ns = int(sample_ms * 1e6)
    last_eval_ns = 0
    event_count = 0
    for source, received_ns, event in iter_tagged_timed(path):
        if source not in instruments:
            continue
        event_count += 1
        if event_count % 100_000 == 0:
            console.print(f"[dim]解析中: {event_count:,}イベント[/dim]")
        observable_ns = received_ns or getattr(event, "ts_ns", 0)
        if received_ns:
            event = replace(event, ts_ns=received_ns)
        engine.apply(source, event, observable_ns)
        if observable_ns and observable_ns - last_eval_ns < interval_ns:
            continue
        engine.evaluate()
        last_eval_ns = observable_ns
    engine.finalize()


async def cmd_xarb(args: argparse.Namespace) -> int:
    """Replay a causal Binance/Bybit spread strategy with four real crosses."""
    path = Path(args.path)
    if not path.exists():
        raise ConfigError(f"{path} がありません。先に xcapture を実行してください。")
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ConfigError(f"{meta_path.name} がありません。xcaptureの録画が必要です。")
    sources = json.loads(meta_path.read_text()).get("sources") or {}
    if not {"binance", "bybit"}.issubset(sources):
        raise ConfigError("xarbにはbinanceとbybitを同時に含むxcapture録画が必要です。")
    instruments = {
        source: _instrument_from_spec(sources[source]) for source in ("binance", "bybit")
    }
    if any(value < 0 for value in (args.binance_taker_bps, args.bybit_taker_bps)):
        raise ConfigError("taker手数料は0以上で指定してください。")
    if args.sample_ms < 0:
        raise ConfigError("--sample-msは0以上で指定してください。")
    config = CrossArbConfig(
        size_base=args.size_base,
        lookback_s=args.lookback_minutes * 60.0,
        min_samples=args.min_samples,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_hold_s=args.max_hold_minutes * 60.0,
        min_expected_net_bps=args.min_expected_net_bps,
        max_age_ms=args.max_age_ms,
        max_skew_ms=args.max_skew_ms,
        depth=args.depth,
        hedge_latency_buffer_bps=args.hedge_latency_buffer_bps,
        fill_model_buffer_bps=args.fill_model_buffer_bps,
        safety_margin_bps=args.safety_margin_bps,
        taker_bps={"binance": args.binance_taker_bps, "bybit": args.bybit_taker_bps},
    )
    engine = CrossExchangeArb(instruments, config)
    _replay_two_market_capture(path, engine, instruments, args.sample_ms)
    summary = engine.summary()

    console.rule("[bold cyan]取引所間Zスコア — 実板・往復全コスト replay")
    console.print(
        f"  データ : {path.name}\n"
        f"  契約   : Binance/Bybit {instruments['binance'].symbol} perpetual\n"
        f"  数量   : {args.size_base:g} {instruments['binance'].base}\n"
        f"  信号   : 過去{args.lookback_minutes:g}分のlog価格差 / "
        f"|z|≥{args.entry_z:g}で建て、|z|≤{args.exit_z:g}で閉じる\n"
        f"  期限   : {args.max_hold_minutes:g}分\n"
        f"  費用   : Binance {args.binance_taker_bps:g}bps + "
        f"Bybit {args.bybit_taker_bps:g}bpsを建玉・手仕舞いの4脚すべてに計上\n"
        f"  執行   : depth {args.depth}段の実板歩き / 両板age≤{args.max_age_ms:g}ms / "
        f"skew≤{args.max_skew_ms:g}ms\n"
        f"  余白   : hedge遅延 {args.hedge_latency_buffer_bps:g}bps + "
        f"約定モデル {args.fill_model_buffer_bps:g}bps + "
        f"安全余白 {args.safety_margin_bps:g}bps\n"
        "  funding: 決済直前に観測できた予測率を、決済時刻通過時だけ計上"
    )

    stats = engine.stats
    console.print(
        f"\n  観測 {stats.observations:,} / 同時に新鮮 {stats.fresh:,} / "
        f"学習窓完成 {stats.warm:,}\n"
        f"  信号 {stats.signals:,} / コスト不足で拒否 {stats.rejected_cost:,} / "
        f"板不足で拒否 {stats.rejected_depth:,} / 品質で拒否 {stats.rejected_quality:,} / "
        f"建玉 {stats.entries:,}"
    )
    if stats.reject_codes:
        console.print(
            "  拒否理由: "
            + " / ".join(f"{code}={count:,}" for code, count in sorted(stats.reject_codes.items()))
        )
    if stats.best_net_bps is not None:
        # Zero trades is an answer, but not the same answer every time. Say how
        # far the best moment fell short, so a dead idea is distinguishable
        # from one that only needs cheaper fills.
        cost_bps = stats.best_gross_bps - stats.best_net_bps
        console.print(
            f"  最良の機会: 収束 {stats.best_gross_bps:.2f}bps / "
            f"全コスト {cost_bps:.2f}bps / 差 {stats.best_net_bps:+.2f}bps"
        )
    if engine.trades:
        rows = engine.trades[-20:]
        if args.plain:
            console.print("long\tshort\tentry_z\texit_z\thold_s\texpected\tnet_bps\treason")
            for trade in rows:
                console.print(
                    f"{trade.long_source}\t{trade.short_source}\t{trade.entry_z:+.2f}\t"
                    f"{trade.exit_z:+.2f}\t{trade.hold_s:.1f}\t"
                    f"{trade.expected_net_bps:+.2f}\t{trade.net_bps:+.2f}\t{trade.exit_reason}",
                    highlight=False,
                )
        else:
            table = Table(title="直近20取引")
            for label in ("long", "short", "entry z", "exit z", "秒", "予想bps", "実現bps", "理由"):
                table.add_column(label, justify="right")
            for trade in rows:
                table.add_row(
                    trade.long_source,
                    trade.short_source,
                    f"{trade.entry_z:+.2f}",
                    f"{trade.exit_z:+.2f}",
                    f"{trade.hold_s:.1f}",
                    f"{trade.expected_net_bps:+.2f}",
                    f"{trade.net_bps:+.2f}",
                    trade.exit_reason,
                )
            console.print(table)

    console.print(
        f"\n  完了取引 : {int(summary['trades']):,}\n"
        f"  Gross    : {summary['gross_quote']:+.4f} USDT\n"
        f"  Funding  : {summary['funding_quote']:+.4f} USDT\n"
        f"  手数料   : {summary['fees_quote']:.4f} USDT\n"
        f"  Net      : {summary['net_quote']:+.4f} USDT / {summary['net_bps']:+.2f}bps\n"
        f"  勝率     : {summary['win_rate'] * 100:.1f}%\n"
        f"  最大DD   : {summary['max_drawdown_quote']:.4f} USDT"
    )
    trades = int(summary["trades"])
    if stats.warm == 0:
        console.print(
            "\n[yellow][bold]判定: 学習窓が完成していません。[/bold][/yellow]\n"
            f"[dim]{args.lookback_minutes:g}分より十分長い録画が必要です。[/dim]"
        )
    elif trades == 0:
        console.print(
            "\n[yellow][bold]判定: Zスコアの歪みは往復全コストを超えませんでした。[/bold][/yellow]\n"
            "[dim]閾値を下げて約定を捏造せず、この期間は取引なしです。[/dim]"
        )
    elif trades < 30:
        console.print(
            "\n[yellow][bold]判定: サンプル不足です。[/bold][/yellow]\n"
            "[dim]黒字でも採用せず、別時間を含む長時間録画で30取引以上を確認します。[/dim]"
        )
    elif summary["net_quote"] <= 0:
        console.print("\n[red][bold]判定: 往復全コスト後で赤字のため不採用です。[/bold][/red]")
    else:
        console.print(
            "\n[green][bold]判定: この録画では往復全コスト後プラスです。[/bold][/green]\n"
            "[yellow]まだ同じ1録画の研究結果です。別期間で固定条件を再検証するまで実運用しません。[/yellow]"
        )
    console.rule()
    return 0


def _carry_timing_grid(study, args: argparse.Namespace) -> None:
    """Every combination tried, not the best one found.

    A single reported winner from a grid searched on the same two years is a
    number about this data, not about the trade. Printing the whole surface
    makes the difference visible: a broad band of similar results is worth
    something, one bright cell surrounded by losses is not.
    """
    from .research.carry import run_timed

    lookbacks = [int(v) for v in args.lookbacks.split(",") if v.strip()]
    enters = [float(v) for v in args.enters.split(",") if v.strip()]
    baseline = study.annual_pct - args.entry_cost_bps / study.span_days * 365 / 100.0

    table = Table(title="入場条件つき（同じ2年で探索した値なので、過学習として読むこと）")
    table.add_column("窓")
    for enter in enters:
        table.add_column(f"≥{enter:g}bps", justify="right")
    for lookback in lookbacks:
        row = [f"{lookback * study.interval_hours / 24:g}日"]
        for enter in enters:
            result = run_timed(
                study.points,
                lookback=lookback,
                enter_bps=enter,
                exit_bps=enter / 2.0,
                entry_cost_bps=args.entry_cost_bps,
            )
            row.append(
                f"{result.annual_pct_full:+.1f}% / {result.time_in_market * 100:.0f}%在"
                if result.trips
                else "—"
            )
        table.add_row(*row)
    console.print()
    console.print(table)
    console.print(
        f"  [dim]各セルは「常時保有に対する代替案の年率 / 建玉していた時間の割合」。"
        f"常時保有は {baseline:+.1f}%。[/dim]"
    )


async def cmd_carry(args: argparse.Namespace) -> int:
    """Funding history turned into the worst case of holding the carry."""
    from .research.carry import CarryStudy, fetch_funding, parse_funding

    if args.entry_cost_bps < 0:
        raise ConfigError("--entry-cost-bps は0以上で指定してください。")
    if args.from_file:
        source = Path(args.from_file)
        if not source.exists():
            raise ConfigError(f"{source} がありません。")
        points = parse_funding(json.loads(source.read_text()))
        origin = source.name
    else:
        console.print(f"[dim]{args.symbol.upper()} のFunding履歴を取得中…[/dim]")
        points = await fetch_funding(args.symbol, days=args.days)
        origin = f"Binance /fapi/v1/fundingRate（直近{args.days:g}日）"
        if args.save:
            Path(args.save).write_text(
                json.dumps(
                    [{"fundingTime": p.ts_ms, "fundingRate": p.rate} for p in points],
                    indent=2,
                )
            )
    if not points:
        raise ConfigError("Funding履歴が空です。銘柄名と期間を確認してください。")

    study = CarryStudy(
        points,
        entry_cost_bps=args.entry_cost_bps,
        drawdown_window_days=args.window_days,
    )
    payback = study.payback_summary(max_hold_days=args.max_hold_days)
    first, last = study.points[0].when, study.points[-1].when

    console.rule(f"[bold cyan]{args.symbol.upper()} Fundingキャリー — 現物買い・先物売りを持ち切る")
    console.print(
        f"  出典   : {origin}\n"
        f"  期間   : {first:%Y-%m-%d} 〜 {last:%Y-%m-%d}  "
        f"({study.span_days:,.0f}日 / {len(study):,}回 / {study.interval_hours:g}時間ごと)\n"
        f"  入場費 : {args.entry_cost_bps:g}bps（4脚の往復手数料）"
    )
    console.print(
        f"\n  平均   : {study.mean_bps:+.4f} bps/回  中央値 {study.median_bps:+.4f}\n"
        f"  年率   : {study.annual_pct:+.2f}%（単利・手数料前）\n"
        f"  マイナス回: {study.negative_share * 100:.1f}%  最悪の1回 {study.worst_settlement_bps:+.2f} bps"
    )
    console.print(
        f"\n  最大DD : {study.max_drawdown_bps:.2f} bps（累積曲線の高値からの落ち込み）\n"
        f"  水面下 : 最長 {study.longest_underwater_days:,.1f}日\n"
        f"  最悪{args.window_days}日: {study.worst_window_bps():+.2f} bps"
    )
    if payback["median_days"] is None:
        console.print(
            f"\n  [red]回収 : {args.max_hold_days:g}日以内に入場費を回収できた入場はありません[/red]"
        )
    else:
        console.print(
            f"\n  回収   : 中央値 {payback['median_days']:.1f}日 / "
            f"9割が {payback['p90_days']:.1f}日以内\n"
            f"  未回収 : {payback['never']:,} / {payback['judged']:,} 回 "
            f"({payback['never_share'] * 100:.1f}%、{args.max_hold_days:g}日で打ち切り)"
        )

    if args.timing:
        _carry_timing_grid(study, args)

    net_annual = study.annual_pct
    verdict = (
        "判定: Fundingは入場費を回収し、平均では正。拘束される証拠金と清算リスクに見合うかは別問題です。"
        if payback["never_share"] < 0.1 and net_annual > 0
        else "判定: この期間のFundingでは、入場費を安定して回収できていません。"
    )
    console.print(f"\n{verdict}")
    console.print(
        "[dim]  清算リスクと取引所リスクは含みません。Fundingの系列には現れないためです。[/dim]"
    )
    return 0


async def cmd_basis(args: argparse.Namespace) -> int:
    """Replay Binance spot/perpetual basis with executable four-leg prices."""
    path = Path(args.path)
    if not path.exists():
        raise ConfigError(f"{path} がありません。先に capture を実行してください。")
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not meta_path.exists():
        raise ConfigError(f"{meta_path.name} がありません。captureの録画が必要です。")
    sources = json.loads(meta_path.read_text()).get("sources") or {}
    if not {"spot", "perp"}.issubset(sources):
        raise ConfigError("basisにはspotとperpを同時に含むcapture録画が必要です。")
    # The first source is the spread numerator, so this is log(perp)-log(spot).
    instruments = {
        source: _instrument_from_spec(sources[source]) for source in ("perp", "spot")
    }
    if any(value < 0 for value in (args.spot_taker_bps, args.perp_taker_bps)):
        raise ConfigError("taker手数料は0以上で指定してください。")
    if args.sample_ms < 0:
        raise ConfigError("--sample-msは0以上で指定してください。")
    allowed_directions = None if args.allow_spot_short else (("spot", "perp"),)
    config = CrossArbConfig(
        size_base=args.size_base,
        lookback_s=args.lookback_minutes * 60.0,
        min_samples=args.min_samples,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        max_hold_s=args.max_hold_hours * 3600.0,
        min_expected_net_bps=args.min_expected_net_bps,
        max_age_ms=args.max_age_ms,
        max_skew_ms=args.max_skew_ms,
        depth=args.depth,
        hedge_latency_buffer_bps=args.hedge_latency_buffer_bps,
        fill_model_buffer_bps=args.fill_model_buffer_bps,
        safety_margin_bps=args.safety_margin_bps,
        taker_bps={"spot": args.spot_taker_bps, "perp": args.perp_taker_bps},
        allowed_directions=allowed_directions,
        require_funding_settlement=args.require_funding,
    )
    engine = CrossExchangeArb(instruments, config)
    _replay_two_market_capture(path, engine, instruments, args.sample_ms)
    summary = engine.summary()
    stats = engine.stats

    console.rule("[bold cyan]Binance現物–先物ベーシス — 実板・全コスト replay")
    console.print(
        f"  データ : {path.name}\n"
        f"  契約   : Binance {instruments['spot'].symbol} spot/perpetual\n"
        f"  数量   : {args.size_base:g} {instruments['spot'].base}\n"
        f"  信号   : 過去{args.lookback_minutes:g}分のlog(perp/spot) / "
        f"|z|≥{args.entry_z:g}で建て、|z|≤{args.exit_z:g}で閉じる\n"
        f"  方向   : {'現物ショートも許可' if args.allow_spot_short else '現物買い・先物売りのみ'}\n"
        f"  期限   : {args.max_hold_hours:g}時間"
        f"{' / Funding通過まで平均回帰決済を禁止' if args.require_funding else ''}\n"
        f"  費用   : 現物 {args.spot_taker_bps:g}bps + 先物 {args.perp_taker_bps:g}bpsを"
        "建玉・手仕舞いの4脚に計上\n"
        f"  執行   : depth {args.depth}段の実板歩き / 両板age≤{args.max_age_ms:g}ms / "
        f"skew≤{args.max_skew_ms:g}ms\n"
        f"  余白   : hedge遅延 {args.hedge_latency_buffer_bps:g}bps + "
        f"約定モデル {args.fill_model_buffer_bps:g}bps + "
        f"安全余白 {args.safety_margin_bps:g}bps\n"
        "  funding: 保有期限内に到来する予想率をEntry判定し、通過時は実績損益に計上"
    )
    console.print(
        f"\n  観測 {stats.observations:,} / 同時に新鮮 {stats.fresh:,} / "
        f"学習窓完成 {stats.warm:,}\n"
        f"  信号 {stats.signals:,} / コスト不足 {stats.rejected_cost:,} / "
        f"現物ショート不可 {stats.rejected_direction:,} / Fundingなし {stats.rejected_funding:,} / "
        f"板不足 {stats.rejected_depth:,} / "
        f"品質不良 {stats.rejected_quality:,} / 建玉 {stats.entries:,}\n"
        f"  Funding評価 {stats.funding_candidates:,} / 決済通過 {stats.funding_settlements:,}"
    )
    if stats.reject_codes:
        console.print(
            "  拒否理由: "
            + " / ".join(f"{code}={count:,}" for code, count in sorted(stats.reject_codes.items()))
        )
    if stats.best_net_bps is not None:
        # Zero trades is an answer, but not the same answer every time. Say how
        # far the best moment fell short, so a dead idea is distinguishable
        # from one that only needs cheaper fills.
        cost_bps = stats.best_gross_bps - stats.best_net_bps
        console.print(
            f"  最良の機会: 収束 {stats.best_gross_bps:.2f}bps / "
            f"全コスト {cost_bps:.2f}bps / 差 {stats.best_net_bps:+.2f}bps"
        )
    if engine.trades:
        console.print("long\tshort\tentry_z\texit_z\thold_s\texpected\tnet_bps\treason")
        for trade in engine.trades[-20:]:
            console.print(
                f"{trade.long_source}\t{trade.short_source}\t{trade.entry_z:+.2f}\t"
                f"{trade.exit_z:+.2f}\t{trade.hold_s:.1f}\t{trade.expected_net_bps:+.2f}\t"
                f"{trade.net_bps:+.2f}\t{trade.exit_reason}",
                highlight=False,
            )
    console.print(
        f"\n  完了取引 : {int(summary['trades']):,}\n"
        f"  Gross    : {summary['gross_quote']:+.4f} USDT\n"
        f"  Funding  : {summary['funding_quote']:+.4f} USDT\n"
        f"  手数料   : {summary['fees_quote']:.4f} USDT\n"
        f"  Net      : {summary['net_quote']:+.4f} USDT / {summary['net_bps']:+.2f}bps\n"
        f"  勝率     : {summary['win_rate'] * 100:.1f}%\n"
        f"  最大DD   : {summary['max_drawdown_quote']:.4f} USDT"
    )
    trades = int(summary["trades"])
    if stats.warm == 0:
        console.print("\n[yellow][bold]判定: 学習窓が完成していません。[/bold][/yellow]")
    elif trades == 0:
        if stats.funding_candidates == 0:
            console.print(
                "\n[yellow][bold]判定: 短期ベーシスは往復全コストを超えませんでした。"
                "[/bold][/yellow]\n"
                "[dim]保有期限内のFundingが0件なので、Funding戦略はまだ未判定です。[/dim]"
            )
        else:
            console.print(
                "\n[yellow][bold]判定: ベーシスと予想Fundingは往復全コストを"
                "超えませんでした。[/bold][/yellow]"
            )
    elif trades < 30:
        console.print("\n[yellow][bold]判定: 30取引未満のため未判定です。[/bold][/yellow]")
    elif summary["net_quote"] <= 0:
        console.print("\n[red][bold]判定: 全コスト後で赤字のため不採用です。[/bold][/red]")
    else:
        console.print(
            "\n[green][bold]判定: この録画では全コスト後プラスです。[/bold][/green]\n"
            "[yellow]別期間のOut-of-sampleで再検証するまで実運用しません。[/yellow]"
        )
    console.rule()
    return 0


async def cmd_capture(args: argparse.Namespace) -> int:
    """Record spot and perp together, onto one timeline."""
    if args.spot_only and args.perp_only:
        console.print("[red]--spot-only と --perp-only は同時に指定できません。[/red]")
        return 2
    if args.basis_sample_ms < 0 or args.basis_depth <= 0:
        raise ConfigError("--basis-sample-msは0以上、--basis-depthは1以上で指定してください。")

    sources: dict = {}
    specs: dict = {}

    if not args.perp_only:
        try:
            spot = await fetch_instrument(args.symbol)
        except Exception as exc:  # noqa: BLE001
            if args.symbol.upper() not in KNOWN_INSTRUMENTS:
                console.print(
                    f"[red]spot exchangeInfo failed: {exc}[/red]\n"
                    f"[yellow]{args.symbol.upper()} の正しい現物tick/lotを確認できないため、"
                    "BTC用の代替値では録画しません。現物未上場の可能性もあります。\n"
                    "  現物–先物pairを調べるなら両方に上場する銘柄を指定してください。"
                    "先物だけなら --perp-only を使えます。[/yellow]"
                )
                return 1
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
    basis_sample_ms = args.basis_sample_ms if args.basis_sample_ms > 0 else None
    sink = None
    if args.s3_bucket:
        try:
            sink = RotatingJsonlSink(
                path=out,
                target=S3Target(bucket=args.s3_bucket, prefix=args.s3_prefix),
                symbol=args.symbol.upper(),
                rotate_seconds=args.rotate_minutes * 60.0,
                keep_local=args.keep_local,
            )
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
    capture = MultiCapture(
        sources,
        out,
        sink=sink,
        basis_sample_ms=basis_sample_ms,
        basis_depth=args.basis_depth,
    )

    console.rule(f"[bold cyan]{args.symbol.upper()} を記録")
    console.print(
        f"  出力   : {out}\n"
        + (
            f"  S3     : s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}/"
            f"symbol={args.symbol.upper()}/...  "
            f"({args.rotate_minutes:g}分ごとに gzip 転送"
            + ("・ローカルも保持" if args.keep_local else "・転送後ローカル削除")
            + ")\n"
            if sink is not None
            else ""
        )
        + f"  対象   : {', '.join(sources)}\n"
        f"  停止   : "
        + (f"{args.duration:.0f}秒後" if args.duration else "Ctrl-C まで")
        + (
            f"\n  圧縮   : ベーシス用 {args.basis_sample_ms:g}ms間隔・depth {args.basis_depth}段"
            if basis_sample_ms is not None
            else ""
        )
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

    meta = write_meta(
        out,
        specs,
        capture_mode="basis_compact" if basis_sample_ms is not None else "full",
        basis_sample_ms=basis_sample_ms,
        basis_depth=args.basis_depth if basis_sample_ms is not None else None,
    )

    console.print()
    console.rule("[bold cyan]記録完了")
    console.print(
        f"  停止理由 : {result.stopped_because}\n"
        f"  時間     : {result.duration_s:,.1f}秒\n"
        f"  イベント : {result.total_events:,} 件\n"
        + (
            f"  出力     : {out}  ({out.stat().st_size / 1e6:,.1f} MB)\n"
            if sink is None and out.exists()
            else ""
        )
        + f"  メタ     : {meta.name}"
    )
    if sink is not None:
        s3 = sink.summary()
        console.print(
            f"  S3       : {len(s3['uploaded']):,} 個を s3://{s3['bucket']} へ転送"
        )
        for key in s3["uploaded"][-3:]:
            console.print(f"    [dim]{key}[/dim]")
        for failure in s3["failed"]:
            # A failed part keeps its local copy; say so rather than leaving
            # the impression the hour is safely in the bucket.
            console.print(f"    [red]転送失敗（ローカルに残置）: {failure}[/red]")
        if s3["failed"]:
            # The parts are on disk but the file the user was told to analyse
            # never existed: with a sink the lines went to the parts instead.
            # Without this line the hour looks lost when it is intact.
            console.print(
                f"  [yellow]分割ファイルは手元にあります。"
                f"1本にまとめれば解析できます:[/yellow]\n"
                f"    [dim]cat {out.stem}-*{out.suffix} > {out.name}[/dim]"
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


async def cmd_statarb_download(args: argparse.Namespace) -> int:
    """Download a compact, reproducible cross-sectional futures dataset."""
    last = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=2)
    symbols = [value.strip() for value in args.symbols.split(",") if value.strip()]
    root = Path(args.data_dir)
    console.rule("[bold cyan]中速・市場中立 statarb — データ取得")
    console.print(
        f"  期間 : {last - timedelta(days=args.days - 1)} 〜 {last} ({args.days}日)\n"
        f"  対象 : {'指定銘柄' if symbols else f'USDⓈ-M 流動性上位 {args.top}銘柄'}\n"
        f"  保存 : {root}\n"
        "  [dim]5分足は月次アーカイブを優先し、実現fundingは公式APIから取得します。[/dim]\n"
    )

    def progress(symbol: str, done: int, total: int) -> None:
        console.print(f"  [{done:>2}/{total}] {symbol}")

    try:
        manifest = await statarb.download_dataset(
            days=args.days,
            top=args.top,
            end=last,
            root=root,
            interval=args.interval,
            symbols=symbols or None,
            progress=progress,
        )
    except Exception as exc:  # noqa: BLE001 - turn network/data failures into a usable message
        tls = describe_tls_error(exc)
        if tls:
            raise ConfigError(tls) from exc
        raise ConfigError(f"statarbデータ取得に失敗しました: {type(exc).__name__}: {exc}") from exc

    available = len(manifest["universe"])
    failures = manifest.get("failures", [])
    console.print(
        f"\n  [bold green]完了[/bold green]: {available}/{len(manifest['files'])}銘柄\n"
    )
    if failures:
        console.print("  [yellow]取得失敗の銘柄は検証対象から除外しました:[/yellow]")
        for row in failures:
            if row.get("error"):
                console.print(
                    f"    {row['symbol']}: {row.get('error_phase', 'archive')} "
                    f"{row['error']}"
                )
            else:
                console.print(f"    {row['symbol']}: 期間内の価格アーカイブなし")
    console.print(
        "  [dim]途中まで取得済みのZIPは、同じコマンドを再実行しても再取得しません。[/dim]\n"
        f"  次: [bold].venv/bin/jsboard statarb backtest --data-dir {root} "
        "--lookbacks 6h,12h,24h,72h --holds 4h,8h,24h "
        "--fees 4 --funding --walk-forward[/bold]"
    )
    console.rule()
    return 0


def _parse_hour_list(value: str, option: str) -> list[int]:
    try:
        parsed = [statarb.parse_hours(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ConfigError(f"{option}: {exc}") from exc
    if not parsed:
        raise ConfigError(f"{option} に1つ以上指定してください")
    return list(dict.fromkeys(parsed))


def _parse_positive_float_list(value: str, option: str) -> list[float]:
    try:
        parsed = [float(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ConfigError(f"{option} は数値をカンマ区切りで指定してください") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise ConfigError(f"{option} は0より大きい値を1つ以上指定してください")
    return list(dict.fromkeys(parsed))


def _print_statarb_diagnostics(label: str, result: statarb.Performance) -> None:
    profitable = sorted(result.symbol_bps.items(), key=lambda item: item[1], reverse=True)
    losing = sorted(result.symbol_bps.items(), key=lambda item: item[1])
    top = ", ".join(f"{symbol} {value:+.1f}" for symbol, value in profitable[:5]) or "—"
    bottom = ", ".join(f"{symbol} {value:+.1f}" for symbol, value in losing[:5]) or "—"
    console.print(
        f"\n  [bold]{label}の耐久性[/bold]\n"
        f"    最終資産 : {result.final_equity:.3f}倍"
        f"{' / [red]資金枯渇[/red]' if result.bankrupt else ''}\n"
        f"    最大1回 : {result.largest_win_bps / 100:+.2f}% / "
        f"最小1回 {result.largest_loss_bps / 100:+.2f}%\n"
        f"    上位10取引 / 全利益 : {result.top10_profit_share:.1%}\n"
        f"    銘柄寄与 上位 : {top}\n"
        f"    銘柄寄与 下位 : {bottom}"
    )


async def cmd_statarb_backtest(args: argparse.Namespace) -> int:
    """Run residual-reversion portfolios with costs and chronological selection."""
    root = Path(args.data_dir)
    lookbacks = _parse_hour_list(args.lookbacks, "--lookbacks")
    holds = _parse_hour_list(args.holds, "--holds")
    if args.strategy == "loser-btc":
        entry_zs = _parse_positive_float_list(args.entry_zs, "--entry-zs")
        configs = [
            statarb.StrategyConfig(lb, hold, strategy="loser_btc", entry_z=entry_z)
            for lb in lookbacks
            for hold in holds
            for entry_z in entry_zs
        ]
        strategy_description = (
            "残差が指定σ以上下落した銘柄だけ買い、推定beta分のBTCを売る"
        )
    else:
        configs = [
            statarb.StrategyConfig(lb, hold, strategy="symmetric")
            for lb in lookbacks
            for hold in holds
        ]
        strategy_description = "BTC betaを除いた相対騰落率の下位20%買い・上位20%売り"
    if args.fees < 0 or args.slippage_bps < 0 or args.target_vol < 0:
        raise ConfigError("--fees、--slippage-bps、--target-vol は0以上で指定してください")
    if args.min_train_trades <= 0:
        raise ConfigError("--min-train-trades は1以上で指定してください")
    if args.universe_top <= 0 or args.min_history_days < 0:
        raise ConfigError("--universe-top は1以上、--min-history-days は0以上で指定してください")
    try:
        volume_window_h = statarb.parse_hours(args.volume_lookback)
    except ValueError as exc:
        raise ConfigError(f"--volume-lookback: {exc}") from exc

    console.rule("[bold cyan]中速・市場中立 statarb — バックテスト")
    try:
        manifest, prices, execution, quote_volume, funding = statarb.load_dataset(root)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ConfigError(str(exc)) from exc
    if len(prices) < 5:
        raise ConfigError(
            f"価格を読めたのが{len(prices)}銘柄だけです。downloadの完了状況を確認してください"
        )
    if len(execution) < 5:
        raise ConfigError("次の5分足openを読めません。downloadデータを確認してください")
    eligible_by_hour = None
    universe_description = "取得時の固定銘柄"
    if args.point_in_time_universe:
        try:
            eligible_by_hour = statarb.build_point_in_time_universe(
                prices,
                quote_volume,
                top=args.universe_top,
                volume_window_h=volume_window_h,
                min_history_h=args.min_history_days * 24,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        usable = [len(symbols) for symbols in eligible_by_hour.values() if len(symbols) >= 4]
        if not usable:
            raise ConfigError(
                "時点別ユニバースでBTC以外の候補が4銘柄以上になる時点がありません。"
                "--topを増やして再取得するか、期間を延ばしてください"
            )
        universe_description = (
            f"各時点の過去{args.volume_lookback} quote volume上位{args.universe_top} / "
            f"初回観測後{args.min_history_days}日未満除外 / "
            f"有効時点平均{statistics.fmean(usable):.1f}銘柄"
        )
    if args.funding:
        missing = sorted(set(prices) - set(funding))
        if missing:
            raise ConfigError(
                "funding履歴が不足しています: " + ", ".join(missing[:8])
                + (" …" if len(missing) > 8 else "")
            )

    console.print(
        f"  データ : {manifest['start']} 〜 {manifest['end']} / {len(prices)}銘柄\n"
        f"  戦略   : {strategy_description}\n"
        f"  コスト : 片道 fee {args.fees:g}bps + 滑り {args.slippage_bps:g}bps "
        f"→ 往復 {2 * (args.fees + args.slippage_bps):g}bps\n"
        f"  執行   : シグナル確定後、次の5分足openで建て・手仕舞い\n"
        f"  リスク : 年率vol {args.target_vol:g}%へ縮小（1倍を上限）/ 複利資産\n"
        f"  銘柄   : {universe_description}\n"
        f"  funding: {'実現履歴を計上' if args.funding else '計上しない'}\n"
    )
    results = statarb.run_backtests(
        prices,
        funding,
        configs,
        execution_prices=execution,
        fee_bps=args.fees,
        slippage_bps=args.slippage_bps,
        include_funding=args.funding,
        target_vol_pct=args.target_vol,
        eligible_by_hour=eligible_by_hour,
    )

    table = Table(box=None, header_style="bold dim", padding=(0, 1))
    table.add_column("lookback", justify="right")
    table.add_column("hold", justify="right")
    if args.strategy == "loser-btc":
        table.add_column("出動", justify="right")
    table.add_column("取引", justify="right")
    table.add_column("価格", justify="right")
    table.add_column("funding", justify="right")
    table.add_column("コスト", justify="right")
    table.add_column("Net", justify="right")
    table.add_column("年率CAGR", justify="right")
    table.add_column("勝率", justify="right")
    table.add_column("最大DD", justify="right")
    for result in sorted(results, key=lambda row: row.total_bps, reverse=True):
        colour = "green" if result.total_bps > 0 else "red"
        cells = [
            f"{result.config.lookback_h}h",
            f"{result.config.hold_h}h",
        ]
        if args.strategy == "loser-btc":
            cells.append(f"≤-{result.config.entry_z:g}σ")
        cells.extend([
            f"{result.trades:,}",
            f"{result.price_bps:+.1f}",
            f"{result.funding_bps:+.1f}",
            f"-{result.cost_bps:.1f}",
            f"[{colour}]{result.total_bps:+.1f}[/{colour}]",
            f"[{colour}]{result.annual_bps / 100:+.1f}%[/{colour}]",
            f"{result.win_rate:.1%}",
            f"-{result.max_drawdown_bps / 100:.1f}%",
        ])
        table.add_row(*cells)
    console.print(table)
    best = max(results, key=lambda row: row.total_bps)
    console.print(
        f"\n  全期間最良 {best.config.label}: long {best.long_price_bps:+.1f}bps / "
        f"short {best.short_price_bps:+.1f}bps / funding {best.funding_bps:+.1f}bps"
    )
    _print_statarb_diagnostics("全期間最良", best)

    if args.walk_forward:
        try:
            folds, out = statarb.walk_forward(
                results,
                min_train_bps=args.min_train_bps,
                min_train_trades=args.min_train_trades,
            )
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        console.print("\n  [bold]Walk-forward（過去だけで設定を選び、次期間で採点）[/bold]")
        for index, fold in enumerate(folds, 1):
            end = datetime.fromtimestamp(fold.test_end_hour * 3600, tz=UTC).date()
            if fold.selected is None:
                console.print(
                    f"    fold {index}: CASH（取引なし） / "
                    f"best train {fold.train_bps:+.1f}bps → test +0.0bps (〜{end})"
                )
            else:
                console.print(
                    f"    fold {index}: {fold.selected.label} / "
                    f"train {fold.train_bps:+.1f} → test {fold.test_bps:+.1f}bps "
                    f"({fold.test_trades}回, 〜{end})"
                )
        colour = "green" if out.total_bps > 0 else "red"
        console.print(
            f"  未使用期間合計: [{colour}][bold]{out.total_bps:+.1f}bps[/bold][/{colour}] "
            f"/ 年率換算 {out.annual_bps / 100:+.1f}% / 最大DD "
            f"-{out.max_drawdown_bps / 100:.1f}%"
        )
        if out.monthly_bps:
            monthly = "  ".join(
                f"{month} {value:+.1f}" for month, value in sorted(out.monthly_bps.items())
            )
            console.print(f"  未使用期間の月別Net(bps): {monthly}")
        _print_statarb_diagnostics("未使用期間", out)
        if (
            out.total_bps > 0
            and not out.bankrupt
            and out.max_drawdown_bps <= 2_000
            and all(fold.selected is not None and fold.test_bps > 0 for fold in folds)
        ):
            console.print(
                "\n  [green]全foldが全コスト後プラスです。[/green]\n"
                "  [green]複利最大DDも20%以内です。[/green]\n"
                "  [dim]まだ同じデータで候補を選んだ研究結果です。設定を固定して、\n"
                "  期間を後ろへずらした再検証を通るまで実運用しません。[/dim]"
            )
        else:
            console.print(
                "\n  [red]未使用期間で安定したプラスを確認できません。[/red]\n"
                "  [dim]この戦略ルールは採用しません。手数料を消した表示へ\n"
                "  変更して黒字に見せることもしません。[/dim]"
            )
    if args.point_in_time_universe:
        console.print(
            "\n  [yellow]注意:[/yellow] 各時点の銘柄順位と上場直後の混入は補正しました。"
            "ただし取得候補は現在上場中の銘柄なので、上場廃止銘柄がない残余の"
            "生存者バイアスはあります。"
        )
    else:
        console.print(
            "\n  [yellow]注意:[/yellow] 銘柄は取得時点の流動性で固定され、上場廃止銘柄も"
            "含まないため生存者バイアスがあります。結果は上限寄りに読んでください。"
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
    mm.add_argument("--size", default=None, help=f"base quote size per level (既定 {DEFAULT_SIZE}、最小単位に満たなければ自動調整)")
    mm.add_argument("--max-position", default=None, help=f"inventory limit (既定 {DEFAULT_MAX_POSITION}、同上)")
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

    toxicity = p.add_argument_group("selective MM toxicity gate")
    toxicity.add_argument(
        "--toxicity-threshold",
        type=float,
        default=0.0,
        help="片側を止める圧力score。0で無効、研究開始値は0.4",
    )
    toxicity.add_argument(
        "--toxicity-pull-threshold",
        type=float,
        default=0.90,
        help="両側を全取消する絶対score",
    )
    toxicity.add_argument("--toxicity-depth", type=int, default=5)
    toxicity.add_argument("--toxicity-flow-weight", type=float, default=0.50)
    toxicity.add_argument("--toxicity-book-weight", type=float, default=0.30)
    toxicity.add_argument("--toxicity-microprice-weight", type=float, default=0.20)

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
    # A backtest wants the answer, not the wait: a one-hour recording replayed
    # at 1.0 takes an hour to say what it can say in seconds. Real time is the
    # exception here (watching the board move), so it is the flag, not the
    # default.
    p_rep.add_argument(
        "--speed", type=float, default=0.0,
        help="再生速度。既定の0は最速、1.0は実時間と同じ速さ",
    )
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
        "--toxicity-thresholds",
        default="",
        help="選択的MMの片側停止score一覧（例: 0,0.2,0.4,0.6,0.8）",
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

    p_pair = sub.add_parser(
        "pair", help="ヘッジ市場で価格を作り、全コスト後プラスの注文だけ出す"
    )
    add_common(p_pair)
    p_pair.add_argument("path", help="spot と perp を同時に capture した .jsonl")
    p_pair.add_argument("--maker-source", default="spot", help="指値を置く市場")
    p_pair.add_argument("--hedge-source", default="perp", help="即時ヘッジする市場")
    p_pair.add_argument("--maker-fees", default="0,1,2", help="試すメイカー手数料 bps")
    p_pair.add_argument(
        "--pair-edge-bps", type=float, default=0.0,
        help="maker/taker/板歩きを引いた後に必要な安全余白",
    )
    p_pair.add_argument(
        "--max-hedge-age-ms", type=float, default=250.0,
        help="これより古いヘッジ板では注文を出さない",
    )
    p_pair.add_argument(
        "--hedge-latency-ms", type=float, default=2.0,
        help="maker約定からhedge注文が市場へ届くまでの遅延",
    )
    p_pair.add_argument("--plain", action="store_true", help="タブ区切りで出す")
    p_pair.set_defaults(
        func=cmd_pair,
        headless=True,
        levels=1,
        max_distance=0,
        requote_ms=100.0,
        min_edge_bps=0.0,
    )

    p_dealer = sub.add_parser(
        "dealer", help="Binance現物・先物の全maker/hedge経路を自動評価する"
    )
    p_dealer.add_argument("path", help="spotとperpを同時にcaptureした.jsonl")
    p_dealer.add_argument("--size-base", type=float, default=0.001, help="評価数量（ベース資産）")
    p_dealer.add_argument("--spot-maker-bps", type=float, default=10.0)
    p_dealer.add_argument("--spot-taker-bps", type=float, default=10.0)
    p_dealer.add_argument("--perp-maker-bps", type=float, default=2.0)
    p_dealer.add_argument("--perp-taker-bps", type=float, default=4.0)
    p_dealer.add_argument(
        "--dealer-edge-bps", type=float, default=1.0, help="全コスト後に必要な安全余白"
    )
    p_dealer.add_argument("--max-age-ms", type=float, default=250.0)
    p_dealer.add_argument(
        "--funding-horizon-h", type=float, default=8.0,
        help="この時間内の次回予想fundingだけをedgeへ含める",
    )
    p_dealer.add_argument("--sample-ms", type=float, default=100.0, help="経路を再評価する間隔")
    p_dealer.add_argument("--depth", type=int, default=20, help="ヘッジ時に歩ける板レベル数")
    p_dealer.add_argument("--plain", action="store_true")
    p_dealer.set_defaults(func=cmd_dealer)

    p_sa = sub.add_parser(
        "statarb", help="先物の中速・市場中立スタットアーブを取得・検証する"
    )
    sa_sub = p_sa.add_subparsers(dest="statarb_command", required=True)

    p_sa_dl = sa_sub.add_parser("download", help="上位銘柄の5分足とfunding履歴を保存する")
    p_sa_dl.add_argument("--days", type=int, default=365)
    p_sa_dl.add_argument("--top", type=int, default=30)
    p_sa_dl.add_argument("--end", default=None, help="最終日 YYYY-MM-DD。既定は公開済み直近日")
    p_sa_dl.add_argument("--interval", default="5m", choices=("1m", "5m", "15m"))
    p_sa_dl.add_argument(
        "--symbols", default="", help="テスト用の明示銘柄。例 BTCUSDT,ETHUSDT"
    )
    p_sa_dl.add_argument("--data-dir", default=str(statarb.DEFAULT_ROOT))
    p_sa_dl.set_defaults(func=cmd_statarb_download)

    p_sa_bt = sa_sub.add_parser("backtest", help="全コスト込みで相対反転を検証する")
    p_sa_bt.add_argument(
        "--strategy",
        default="symmetric",
        choices=("symmetric", "loser-btc"),
        help="対称反転、または下落異常だけを買ってBTCでヘッジ",
    )
    p_sa_bt.add_argument("--lookbacks", default="6h,12h,24h,72h")
    p_sa_bt.add_argument("--holds", default="4h,8h,24h")
    p_sa_bt.add_argument(
        "--entry-zs",
        default="1,1.5,2,2.5",
        help="loser-btcが出動する残差下落σの候補",
    )
    p_sa_bt.add_argument("--fees", type=float, default=4.0, help="片道手数料 bps")
    p_sa_bt.add_argument(
        "--slippage-bps", type=float, default=1.0, help="片道の想定スリッページ bps"
    )
    p_sa_bt.add_argument(
        "--target-vol",
        type=float,
        default=20.0,
        help="過去168時間から推定する年率vol目標。0で縮小なし",
    )
    p_sa_bt.add_argument("--funding", action="store_true", help="実現fundingを損益へ含める")
    p_sa_bt.add_argument(
        "--walk-forward", action="store_true", help="過去だけで設定を選び次期間で採点する"
    )
    p_sa_bt.add_argument(
        "--min-train-bps",
        type=float,
        default=0.0,
        help="学習Netがこの値以下なら次期間は取引しない",
    )
    p_sa_bt.add_argument(
        "--min-train-trades",
        type=int,
        default=30,
        help="設定選択に必要な過去取引数",
    )
    p_sa_bt.add_argument(
        "--point-in-time-universe",
        action="store_true",
        help="各時点の過去出来高だけで取引候補を選ぶ",
    )
    p_sa_bt.add_argument(
        "--universe-top", type=int, default=30, help="各時点で採用するBTC以外の銘柄数"
    )
    p_sa_bt.add_argument(
        "--volume-lookback", default="7d", help="銘柄順位に使う過去quote volume期間"
    )
    p_sa_bt.add_argument(
        "--min-history-days",
        type=int,
        default=30,
        help="初回観測後、この日数未満の銘柄を除外",
    )
    p_sa_bt.add_argument("--data-dir", default=str(statarb.DEFAULT_ROOT))
    p_sa_bt.set_defaults(func=cmd_statarb_backtest)

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

    p_xcap = sub.add_parser("xcapture", help="BinanceとBybitの先物板を同時に記録する")
    p_xcap.add_argument("--symbol", default="BTCUSDT")
    p_xcap.add_argument("--out", default="btc-xarb.jsonl")
    p_xcap.add_argument("--duration", type=float, default=7200.0, help="録画秒数")
    p_xcap.add_argument("--max-events", type=int, default=None)
    p_xcap.add_argument("--binance-depth-ms", type=int, default=100, choices=(100, 250, 500))
    p_xcap.add_argument("--bybit-depth", type=int, default=50, choices=(1, 50, 200, 1000))
    p_xcap.set_defaults(func=cmd_xcapture)

    p_xarb = sub.add_parser("xarb", help="Binance/Bybit価格差を実板・往復費用込みで再生する")
    p_xarb.add_argument("path", help="xcaptureで作った.jsonl")
    p_xarb.add_argument("--size-base", type=float, default=0.001)
    p_xarb.add_argument("--lookback-minutes", type=float, default=30.0)
    p_xarb.add_argument("--min-samples", type=int, default=300)
    p_xarb.add_argument("--entry-z", type=float, default=2.5)
    p_xarb.add_argument("--exit-z", type=float, default=0.25)
    p_xarb.add_argument("--max-hold-minutes", type=float, default=15.0)
    p_xarb.add_argument("--min-expected-net-bps", type=float, default=1.0)
    p_xarb.add_argument("--binance-taker-bps", type=float, default=4.0)
    p_xarb.add_argument("--bybit-taker-bps", type=float, default=5.5)
    p_xarb.add_argument("--max-age-ms", type=float, default=250.0)
    p_xarb.add_argument("--max-skew-ms", type=float, default=250.0)
    p_xarb.add_argument("--hedge-latency-buffer-bps", type=float, default=0.0)
    p_xarb.add_argument("--fill-model-buffer-bps", type=float, default=0.0)
    p_xarb.add_argument("--safety-margin-bps", type=float, default=0.0)
    p_xarb.add_argument("--sample-ms", type=float, default=100.0)
    p_xarb.add_argument("--depth", type=int, default=20)
    p_xarb.add_argument("--plain", action="store_true")
    p_xarb.set_defaults(func=cmd_xarb)

    p_basis = sub.add_parser(
        "basis", help="Binance現物・無期限先物ベーシスを実板・Funding込みで再生する"
    )
    p_basis.add_argument("path", help="captureで作ったspot/perp同時録画.jsonl")
    p_basis.add_argument("--size-base", type=float, default=0.001)
    p_basis.add_argument("--lookback-minutes", type=float, default=30.0)
    p_basis.add_argument("--min-samples", type=int, default=300)
    p_basis.add_argument("--entry-z", type=float, default=2.0)
    p_basis.add_argument("--exit-z", type=float, default=0.25)
    p_basis.add_argument("--max-hold-hours", type=float, default=8.0)
    p_basis.add_argument(
        "--require-funding",
        action="store_true",
        help="次回Fundingが期限内にある候補だけ入り、決済通過前の平均回帰Exitを禁止",
    )
    p_basis.add_argument("--min-expected-net-bps", type=float, default=1.0)
    p_basis.add_argument("--spot-taker-bps", type=float, default=10.0)
    p_basis.add_argument("--perp-taker-bps", type=float, default=4.0)
    p_basis.add_argument("--max-age-ms", type=float, default=250.0)
    p_basis.add_argument("--max-skew-ms", type=float, default=250.0)
    p_basis.add_argument("--hedge-latency-buffer-bps", type=float, default=1.0)
    p_basis.add_argument("--fill-model-buffer-bps", type=float, default=1.0)
    p_basis.add_argument("--safety-margin-bps", type=float, default=1.0)
    p_basis.add_argument("--sample-ms", type=float, default=100.0)
    p_basis.add_argument("--depth", type=int, default=20)
    p_basis.add_argument(
        "--allow-spot-short",
        action="store_true",
        help="現物の借入可能性を別途確認できる場合だけ逆方向を許可する",
    )
    p_basis.add_argument("--plain", action="store_true")
    p_basis.set_defaults(func=cmd_basis)

    p_carry = sub.add_parser(
        "carry", help="現物買い・無期限先物売りを持ち続けたときのFunding収支を調べる"
    )
    p_carry.add_argument("--symbol", default="BTCUSDT")
    p_carry.add_argument("--days", type=float, default=730.0, help="遡る日数")
    p_carry.add_argument(
        "--entry-cost-bps", type=float, default=28.0,
        help="建てて閉じるまでの往復手数料。既定は現物10bps+先物4bpsの4脚",
    )
    p_carry.add_argument(
        "--max-hold-days", type=float, default=180.0, help="回収を待つ上限日数",
    )
    p_carry.add_argument(
        "--window-days", type=int, default=30, help="最悪期間を測る窓の長さ",
    )
    p_carry.add_argument("--save", default=None, help="取得した履歴をJSONで保存する")
    p_carry.add_argument("--from-file", default=None, help="保存済みJSONを読む（取得しない）")
    p_carry.add_argument(
        "--timing", action="store_true",
        help="Fundingが厚い時期だけ建てる案を、常時保有と並べて出す",
    )
    p_carry.add_argument(
        "--lookbacks", default="3,9,21,63",
        help="--timing の判断に使う直近の決済回数（カンマ区切り）",
    )
    p_carry.add_argument(
        "--enters", default="0.3,0.5,0.7,1.0",
        help="--timing の建玉しきい値 bps/回（カンマ区切り）。手仕舞いはその半値",
    )
    p_carry.add_argument("--plain", action="store_true")
    p_carry.set_defaults(func=cmd_carry)

    p_cap = sub.add_parser("capture", help="現物と先物を同時に記録する")
    p_cap.add_argument("--symbol", default="BTCUSDT")
    p_cap.add_argument("--out", default="capture.jsonl")
    p_cap.add_argument("--duration", type=float, default=None, help="秒。省略で Ctrl-C まで")
    p_cap.add_argument("--max-events", type=int, default=None)
    p_cap.add_argument("--spot-depth-ms", type=int, default=100, choices=(100, 1000))
    p_cap.add_argument("--perp-depth-ms", type=int, default=100, choices=(100, 250, 500))
    p_cap.add_argument(
        "--basis-sample-ms",
        type=float,
        default=0.0,
        help="0より大きいと板を指定間隔のfull snapshotへ圧縮し、Funding用データだけ保存",
    )
    p_cap.add_argument("--basis-depth", type=int, default=20, help="圧縮snapshotの板段数")
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
    s3 = p_cap.add_argument_group("S3")
    s3.add_argument("--s3-bucket", default=None, help="指定すると分割して gzip 転送する")
    s3.add_argument("--s3-prefix", default="raw", help="バケット内の接頭辞")
    s3.add_argument("--rotate-minutes", type=float, default=15.0, help="何分ごとに転送するか")
    s3.add_argument(
        "--keep-local", action="store_true",
        help="転送後もローカルの分割ファイルを消さない",
    )
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
