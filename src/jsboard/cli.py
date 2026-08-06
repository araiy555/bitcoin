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
import json
import logging
import sys
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .core.market import MarketView
from .core.types import Instrument
from .feed.base import Feed
from .feed.binance import BinanceFeed
from .feed.binance_futures import FALLBACK_MODES, BinanceFuturesFeed
from .feed.replay import JsonlRecorder, ReplayFeed, SyntheticFeed
from .mm.fair_value import FairValueConfig, FairValueEstimator
from .mm.inventory import FeeSchedule, Position
from .mm.quoter import Quoter, QuoterConfig
from .mm.risk import RiskLimits, RiskManager
from .mm.strategy import MarketMaker, StrategyConfig
from .net import describe_tls_error, make_session
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
from .research.scan import ScanFilters, rank_persistence, scan, summarise, watch
from .sim.capture import MultiCapture, write_meta
from .sim.paper import PaperConfig, PaperVenue
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


def build_maker(instrument: Instrument, args: argparse.Namespace) -> MarketMaker:
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


def _print_report(mm: MarketMaker, result) -> None:
    s = mm.summary()
    inst = mm.instrument
    console.print()
    console.rule(f"[bold cyan]{inst.symbol} — session report")
    console.print(
        f"  stopped        : {result.stopped_because}\n"
        f"  elapsed        : {result.elapsed_s:,.1f}s over {result.events:,} events\n"
        f"  quote cycles   : {s['cycles']:,}  "
        f"(placed {s['placed']:,} / cancelled {s['cancelled']:,} / kept {s['kept']:,})\n"
        f"  fills          : {int(s['fills']):,}  volume {s['volume']:,.5f} {inst.base}\n"
        f"  position       : {s['position']:+,.5f} {inst.base} @ {s['avg_price']:,.2f}\n"
        f"  realized P&L   : {s['realized']:+,.2f} {inst.quote}\n"
        f"  unrealized P&L : {s['unrealized']:+,.2f} {inst.quote}\n"
        f"  fees paid      : {s['fees']:,.2f} {inst.quote}\n"
        f"  [bold]total P&L      : {s['total']:+,.2f} {inst.quote}[/bold]\n"
        f"  last decision  : {s['decision']}"
    )
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


async def cmd_live(args: argparse.Namespace) -> int:
    try:
        instrument = await fetch_instrument(args.symbol)
        console.print(
            f"[dim]exchangeInfo: tick={instrument.tick_size} lot={instrument.lot_size}[/dim]"
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]exchangeInfo unavailable ({exc}); using built-in spec[/yellow]")
        instrument = build_instrument(args.symbol, args.tick_size, args.lot_size)

    feed = BinanceFeed(instrument, depth_ms=args.depth_ms)
    mm = build_maker(instrument, args)
    await drive(feed, mm, args, headless=args.headless)
    return 0


async def cmd_record(args: argparse.Namespace) -> int:
    instrument = build_instrument(args.symbol, args.tick_size, args.lot_size)
    feed = BinanceFeed(instrument, depth_ms=args.depth_ms)
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
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        instrument = Instrument(
            symbol=meta["symbol"],
            tick_size=Decimal(meta["tick_size"]),
            lot_size=Decimal(meta["lot_size"]),
            base=meta.get("base", ""),
            quote=meta.get("quote", ""),
        )
    else:
        instrument = build_instrument(args.symbol, args.tick_size, args.lot_size)

    feed = ReplayFeed(instrument, path, speed=args.speed)
    mm = build_maker(instrument, args)
    # A recording carries the timestamps it was captured with. Judged against
    # the wall clock those are always in the past — a day-old capture reads as
    # a book that is a day stale, and the risk gate pulls every quote before
    # one is ever placed. "Now", during a replay, is the timestamp of the
    # event being replayed.
    attach_virtual_clock(mm)
    await drive(feed, mm, args, headless=args.headless)
    return 0


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
    p_live.add_argument("--depth-ms", type=int, default=100, choices=(100, 1000))
    p_live.set_defaults(func=cmd_live)

    p_rec = sub.add_parser("record", help="capture a live session to JSONL")
    add_common(p_rec)
    p_rec.add_argument("--out", required=True)
    p_rec.add_argument("--depth-ms", type=int, default=100, choices=(100, 1000))
    p_rec.set_defaults(func=cmd_record)

    p_rep = sub.add_parser("replay", help="replay a capture")
    add_common(p_rep)
    p_rep.add_argument("path")
    p_rep.add_argument("--speed", type=float, default=1.0, help="0 = as fast as possible")
    p_rep.set_defaults(func=cmd_replay)

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
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
