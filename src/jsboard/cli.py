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
from .research.scan import ScanFilters, scan, summarise
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
    async with aiohttp.ClientSession(trust_env=True) as session, session.get(
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


async def cmd_scan(args: argparse.Namespace) -> int:
    filters = ScanFilters(
        quote_asset=args.quote.upper(),
        maker_bps=args.maker_bps,
        min_quote_volume=args.min_volume,
        min_trades=args.min_trades,
        size_quote=args.size_quote,
    )

    console.print("[dim]Binance の全銘柄を取得中…[/dim]")
    try:
        results, considered = await scan(filters)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]取得に失敗しました: {exc}[/red]")
        console.print("[dim]ネットワークから api.binance.com に到達できるか確認してください。[/dim]")
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
    table.add_column("約定数", justify="right")
    table.add_column("板の厚み", justify="right")

    shown = results[: args.top]
    for s in shown:
        net = s.net_bps(filters.maker_bps)
        style = "green" if net > 0 else "red"
        table.add_row(
            s.symbol,
            f"{s.spread_bps:,.2f}",
            f"[{style}]{net:+,.2f}[/{style}]",
            f"[{style}]{s.profit_per_round_trip(filters.maker_bps, filters.size_quote):+,.4f}[/{style}]",
            f"{s.quote_volume:,.0f}",
            f"{s.trades:,}",
            f"{s.top_of_book_quote:,.0f}",
        )

    console.print()
    console.print(table)
    console.print(
        f"\n  {considered:,} 銘柄 → 流動性条件を満たす [bold]{stats['liquid']:,}[/bold] 件"
        f" → 手数料を超える [bold]{stats['viable']:,}[/bold] 件"
    )
    console.print(
        f"  スプレッドの中央値 [bold]{stats['median_spread_bps']:.2f} bps[/bold]"
        f"（損益分岐は {breakeven:.1f} bps）"
    )

    if stats["viable"] == 0:
        console.print(
            "\n  [yellow]この手数料でスプレッドを超える銘柄はありません。[/yellow]\n"
            "  [dim]手数料を下げる以外に、この戦略が成立する道はありません。[/dim]"
        )
    else:
        console.print(
            "\n  [yellow]数字の読み方[/yellow]\n"
            "  [dim]スプレッドが広い銘柄は、誰も建値を置きたがらないから広いのが普通です。\n"
            "  net が正でも、約定数が少なければ回転せず、板が薄ければ在庫を捌けません。\n"
            "  net・約定数・板の厚みの3つが揃って初めて意味があります。[/dim]"
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
    p_scan.add_argument("--quote", default="USDT", help="建て通貨")
    p_scan.add_argument("--maker-bps", type=float, default=10.0, help="自分のメイカー手数料")
    p_scan.add_argument("--min-volume", type=float, default=1_000_000.0, help="24h出来高の下限")
    p_scan.add_argument("--min-trades", type=int, default=1_000, help="24h約定数の下限")
    p_scan.add_argument("--size-quote", type=float, default=1_000.0, help="1回の注文金額")
    p_scan.add_argument("--top", type=int, default=25, help="表示件数")
    p_scan.set_defaults(func=cmd_scan)

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
