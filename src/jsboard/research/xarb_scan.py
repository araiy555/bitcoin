"""Live cross-venue BTC arbitrage scanner.

This module does not place orders.  It consumes public order-book streams from
Binance, Bybit and OKX, walks visible depth for a requested quote notional, and
measures the executable two-leg edge after explicit fees and safety buffers.

The key distinction from ``sim.cross_exchange`` is that this scanner does not
wait for mean reversion.  A candidate is complete at the moment a BTC buy can
be hedged by an equal BTC sell on another market.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import websockets

from ..net import ssl_context

BINANCE_SPOT_WS = "wss://stream.binance.com:9443/ws/btcusdt@depth20@100ms"
BINANCE_PERP_WS = "wss://fstream.binance.com/ws/btcusdt@depth20@100ms"
BYBIT_WS = {
    "bybit_spot": "wss://stream.bybit.com/v5/public/spot",
    "bybit_perp": "wss://stream.bybit.com/v5/public/linear",
}
OKX_WS = "wss://ws.okx.com:8443/ws/v5/public"

DEFAULT_FEES_BPS = {
    # Explicitly configurable from the CLI. These are research defaults, not
    # an assertion about the user's fee tier.
    "binance_spot": 10.0,
    "binance_perp": 4.0,
    "bybit_spot": 10.0,
    "bybit_perp": 5.5,
    "okx_spot": 10.0,
    "okx_perp": 5.0,
}


@dataclass(frozen=True, slots=True)
class Book:
    source: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    exchange_ts_ns: int
    receive_ts_ns: int

    def valid(self) -> bool:
        return (
            bool(self.bids)
            and bool(self.asks)
            and self.bids[0][0] > 0
            and self.asks[0][0] > 0
            and self.bids[0][0] <= self.asks[0][0]
        )


@dataclass(frozen=True, slots=True)
class Execution:
    qty_base: float
    quote: float
    avg_price: float
    levels_used: int
    complete: bool


@dataclass(frozen=True, slots=True)
class Opportunity:
    ts_ns: int
    buy_source: str
    sell_source: str
    requested_quote: float
    qty_base: float
    buy_avg: float
    sell_avg: float
    gross_edge_bps: float
    fee_bps: float
    latency_buffer_bps: float
    fill_buffer_bps: float
    rebalance_buffer_bps: float
    safety_buffer_bps: float
    net_edge_bps: float
    book_skew_ms: float
    buy_age_ms: float
    sell_age_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class EdgeRun:
    started_ns: int
    last_ns: int
    peak_net_bps: float
    samples: int = 1


@dataclass(slots=True)
class EdgeSummary:
    observations: int = 0
    positive_samples: int = 0
    completed_runs: list[tuple[float, float]] = field(default_factory=list)
    # (duration_ms, peak_net_bps)


def _clean_levels(rows, *, reverse: bool) -> tuple[tuple[float, float], ...]:
    levels: list[tuple[float, float]] = []
    for row in rows or ():
        try:
            px, qty = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if px > 0 and qty > 0 and math.isfinite(px) and math.isfinite(qty):
            levels.append((px, qty))
    levels.sort(key=lambda x: x[0], reverse=reverse)
    return tuple(levels)


def walk_buy_quote(asks: tuple[tuple[float, float], ...], quote_target: float) -> Execution:
    """Spend up to ``quote_target`` buying BTC from asks."""
    if quote_target <= 0:
        raise ValueError("quote_target must be positive")
    remaining = quote_target
    qty = spent = 0.0
    used = 0
    for price, available in asks:
        if remaining <= 1e-12:
            break
        max_quote = price * available
        take_quote = min(remaining, max_quote)
        take_qty = take_quote / price
        qty += take_qty
        spent += take_quote
        remaining -= take_quote
        used += 1
    complete = remaining <= max(1e-9, quote_target * 1e-9)
    avg = spent / qty if qty > 0 else 0.0
    return Execution(qty, spent, avg, used, complete)


def walk_sell_base(bids: tuple[tuple[float, float], ...], qty_target: float) -> Execution:
    """Sell ``qty_target`` BTC into bids."""
    if qty_target <= 0:
        raise ValueError("qty_target must be positive")
    remaining = qty_target
    sold = proceeds = 0.0
    used = 0
    for price, available in bids:
        if remaining <= 1e-15:
            break
        take = min(remaining, available)
        sold += take
        proceeds += take * price
        remaining -= take
        used += 1
    complete = remaining <= max(1e-12, qty_target * 1e-9)
    avg = proceeds / sold if sold > 0 else 0.0
    return Execution(sold, proceeds, avg, used, complete)


def evaluate_cross(
    buy_book: Book,
    sell_book: Book,
    *,
    quote_notional: float,
    fees_bps: dict[str, float],
    now_ns: int,
    max_age_ms: float,
    max_skew_ms: float,
    latency_buffer_bps: float,
    fill_buffer_bps: float,
    rebalance_buffer_bps: float,
    safety_buffer_bps: float,
) -> Opportunity | None:
    """Return an executable two-leg opportunity or ``None``.

    The gross edge uses average prices after walking visible depth, so depth
    slippage is already embedded and must not be deducted a second time.
    """
    if buy_book.source == sell_book.source or not (buy_book.valid() and sell_book.valid()):
        return None
    buy_age = max(0.0, (now_ns - buy_book.receive_ts_ns) / 1e6)
    sell_age = max(0.0, (now_ns - sell_book.receive_ts_ns) / 1e6)
    skew = abs(buy_book.receive_ts_ns - sell_book.receive_ts_ns) / 1e6
    if buy_age > max_age_ms or sell_age > max_age_ms or skew > max_skew_ms:
        return None

    buy = walk_buy_quote(buy_book.asks, quote_notional)
    if not buy.complete or buy.qty_base <= 0:
        return None
    sell = walk_sell_base(sell_book.bids, buy.qty_base)
    if not sell.complete or sell.qty_base <= 0:
        return None

    gross = (sell.avg_price / buy.avg_price - 1.0) * 10_000.0
    fee = fees_bps[buy_book.source] + fees_bps[sell_book.source]
    net = (
        gross
        - fee
        - latency_buffer_bps
        - fill_buffer_bps
        - rebalance_buffer_bps
        - safety_buffer_bps
    )
    return Opportunity(
        ts_ns=now_ns,
        buy_source=buy_book.source,
        sell_source=sell_book.source,
        requested_quote=quote_notional,
        qty_base=buy.qty_base,
        buy_avg=buy.avg_price,
        sell_avg=sell.avg_price,
        gross_edge_bps=gross,
        fee_bps=fee,
        latency_buffer_bps=latency_buffer_bps,
        fill_buffer_bps=fill_buffer_bps,
        rebalance_buffer_bps=rebalance_buffer_bps,
        safety_buffer_bps=safety_buffer_bps,
        net_edge_bps=net,
        book_skew_ms=skew,
        buy_age_ms=buy_age,
        sell_age_ms=sell_age,
    )


def _apply_absolute_delta(book: dict[float, float], rows) -> None:
    for row in rows or ():
        try:
            px, qty = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if qty <= 0:
            book.pop(px, None)
        elif px > 0:
            book[px] = qty


async def _binance_book(source: str, url: str, out: asyncio.Queue) -> None:
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**14,
                ssl=ssl_context(),
            ) as ws:
                async for raw in ws:
                    recv = time.monotonic_ns()
                    data = json.loads(raw)
                    bids = _clean_levels(data.get("b"), reverse=True)
                    asks = _clean_levels(data.get("a"), reverse=False)
                    ts = int(data.get("E") or 0) * 1_000_000
                    book = Book(source, bids, asks, ts, recv)
                    if book.valid():
                        await out.put(book)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1.0)


async def _bybit_book(source: str, out: asyncio.Queue) -> None:
    url = BYBIT_WS[source]
    topic = "orderbook.50.BTCUSDT"
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**14,
                ssl=ssl_context(),
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
                async for raw in ws:
                    recv = time.monotonic_ns()
                    payload = json.loads(raw)
                    if payload.get("success") is False:
                        raise RuntimeError(payload.get("ret_msg") or "Bybit subscribe failed")
                    if payload.get("topic") != topic:
                        continue
                    data = payload.get("data") or {}
                    if payload.get("type") == "snapshot" or int(data.get("u", 0)) == 1:
                        bids.clear()
                        asks.clear()
                    _apply_absolute_delta(bids, data.get("b"))
                    _apply_absolute_delta(asks, data.get("a"))
                    book = Book(
                        source=source,
                        bids=tuple(sorted(bids.items(), reverse=True)[:50]),
                        asks=tuple(sorted(asks.items())[:50]),
                        exchange_ts_ns=int(data.get("cts") or payload.get("ts") or 0) * 1_000_000,
                        receive_ts_ns=recv,
                    )
                    if book.valid():
                        await out.put(book)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1.0)


async def _okx_books(out: asyncio.Queue) -> None:
    args = [
        {"channel": "books5", "instId": "BTC-USDT"},
        {"channel": "books5", "instId": "BTC-USDT-SWAP"},
    ]
    source_for = {
        "BTC-USDT": "okx_spot",
        "BTC-USDT-SWAP": "okx_perp",
    }
    while True:
        try:
            async with websockets.connect(
                OKX_WS,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**14,
                ssl=ssl_context(),
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": args}))
                async for raw in ws:
                    recv = time.monotonic_ns()
                    payload = json.loads(raw)
                    if payload.get("event") == "error":
                        raise RuntimeError(payload.get("msg") or "OKX subscribe failed")
                    arg = payload.get("arg") or {}
                    if arg.get("channel") != "books5":
                        continue
                    source = source_for.get(arg.get("instId"))
                    if not source:
                        continue
                    for row in payload.get("data") or ():
                        book = Book(
                            source=source,
                            bids=_clean_levels(row.get("bids"), reverse=True),
                            asks=_clean_levels(row.get("asks"), reverse=False),
                            exchange_ts_ns=int(row.get("ts") or 0) * 1_000_000,
                            receive_ts_ns=recv,
                        )
                        if book.valid():
                            await out.put(book)
        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(1.0)


def _finish_run(
    key: tuple[str, str, float],
    runs: dict[tuple[str, str, float], EdgeRun],
    summary: EdgeSummary,
) -> None:
    run = runs.pop(key, None)
    if run is None:
        return
    duration_ms = max(0.0, (run.last_ns - run.started_ns) / 1e6)
    summary.completed_runs.append((duration_ms, run.peak_net_bps))


async def run_live(
    *,
    duration_s: float,
    notionals: tuple[float, ...],
    fees_bps: dict[str, float] | None = None,
    min_net_bps: float = 2.0,
    max_age_ms: float = 250.0,
    max_skew_ms: float = 250.0,
    latency_buffer_bps: float = 1.0,
    fill_buffer_bps: float = 1.0,
    rebalance_buffer_bps: float = 1.0,
    safety_buffer_bps: float = 1.0,
    out_path: str | Path | None = None,
    status_every_s: float = 5.0,
) -> EdgeSummary:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if not notionals or any(n <= 0 for n in notionals):
        raise ValueError("notionals must all be positive")
    fees = dict(DEFAULT_FEES_BPS if fees_bps is None else fees_bps)
    missing = set(DEFAULT_FEES_BPS) - set(fees)
    if missing:
        raise ValueError(f"missing fees for: {sorted(missing)}")

    queue: asyncio.Queue[Book] = asyncio.Queue(maxsize=100_000)
    tasks = [
        asyncio.create_task(_binance_book("binance_spot", BINANCE_SPOT_WS, queue)),
        asyncio.create_task(_binance_book("binance_perp", BINANCE_PERP_WS, queue)),
        asyncio.create_task(_bybit_book("bybit_spot", queue)),
        asyncio.create_task(_bybit_book("bybit_perp", queue)),
        asyncio.create_task(_okx_books(queue)),
    ]

    books: dict[str, Book] = {}
    runs: dict[tuple[str, str, float], EdgeRun] = {}
    summary = EdgeSummary()
    started = time.monotonic()
    deadline = started + duration_s
    last_status = started
    best_seen: Opportunity | None = None

    fh = None
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("a", encoding="utf-8")

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            try:
                book = await asyncio.wait_for(queue.get(), timeout=min(1.0, deadline - now))
            except TimeoutError:
                continue
            books[book.source] = book
            now_ns = time.monotonic_ns()

            for buy_source, buy_book in tuple(books.items()):
                for sell_source, sell_book in tuple(books.items()):
                    if buy_source == sell_source:
                        continue
                    for notional in notionals:
                        summary.observations += 1
                        opp = evaluate_cross(
                            buy_book,
                            sell_book,
                            quote_notional=notional,
                            fees_bps=fees,
                            now_ns=now_ns,
                            max_age_ms=max_age_ms,
                            max_skew_ms=max_skew_ms,
                            latency_buffer_bps=latency_buffer_bps,
                            fill_buffer_bps=fill_buffer_bps,
                            rebalance_buffer_bps=rebalance_buffer_bps,
                            safety_buffer_bps=safety_buffer_bps,
                        )
                        key = (buy_source, sell_source, notional)
                        if opp is None or opp.net_edge_bps < min_net_bps:
                            _finish_run(key, runs, summary)
                            continue

                        summary.positive_samples += 1
                        current = runs.get(key)
                        if current is None:
                            runs[key] = EdgeRun(now_ns, now_ns, opp.net_edge_bps)
                        else:
                            current.last_ns = now_ns
                            current.peak_net_bps = max(current.peak_net_bps, opp.net_edge_bps)
                            current.samples += 1
                        if best_seen is None or opp.net_edge_bps > best_seen.net_edge_bps:
                            best_seen = opp
                        if fh is not None:
                            fh.write(json.dumps(opp.as_dict(), ensure_ascii=False) + "\n")

            if now - last_status >= status_every_s:
                live = len(runs)
                best = (
                    f"{best_seen.buy_source}->{best_seen.sell_source} "
                    f"${best_seen.requested_quote:g} net={best_seen.net_edge_bps:+.2f}bps"
                    if best_seen
                    else "none"
                )
                print(
                    f"[xarb-scan] {now-started:7.1f}s "
                    f"books={len(books)}/6 live_edges={live} "
                    f"positive_samples={summary.positive_samples} best={best}",
                    flush=True,
                )
                if fh is not None:
                    fh.flush()
                last_status = now
    finally:
        end_ns = time.monotonic_ns()
        for key, run in list(runs.items()):
            run.last_ns = max(run.last_ns, end_ns)
            _finish_run(key, runs, summary)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if fh is not None:
            fh.flush()
            fh.close()

    return summary


def format_summary(summary: EdgeSummary) -> str:
    runs = summary.completed_runs
    if not runs:
        return (
            "No executable Net-positive runs met the threshold.\n"
            f"observations={summary.observations:,} "
            f"positive_samples={summary.positive_samples:,}"
        )
    durations = [d for d, _ in runs]
    peaks = [p for _, p in runs]
    return "\n".join(
        [
            f"completed edge runs : {len(runs):,}",
            f"positive samples    : {summary.positive_samples:,}",
            f"duration median     : {statistics.median(durations):.1f} ms",
            f"duration p90        : {_percentile(durations, 0.90):.1f} ms",
            f"peak net median     : {statistics.median(peaks):+.2f} bps",
            f"peak net p90        : {_percentile(peaks, 0.90):+.2f} bps",
            f"peak net max        : {max(peaks):+.2f} bps",
        ]
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight
