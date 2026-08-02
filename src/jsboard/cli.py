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
from decimal import Decimal
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from .core.market import MarketView
from .core.types import Instrument
from .feed.base import Feed
from .feed.binance import BinanceFeed
from .feed.replay import JsonlRecorder, ReplayFeed, SyntheticFeed
from .mm.fair_value import FairValueConfig, FairValueEstimator
from .mm.inventory import FeeSchedule, Position
from .mm.quoter import Quoter, QuoterConfig
from .mm.risk import RiskLimits, RiskManager
from .mm.strategy import MarketMaker, StrategyConfig
from .net import describe_tls_error, make_session
from .research.scan import ScanFilters, rank_persistence, scan, summarise, watch
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
