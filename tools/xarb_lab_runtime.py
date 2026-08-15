#!/usr/bin/env python3
"""Memory-safe runtime for the one-shot six-market xarb lab.

The long capture is started only after synthetic self-tests and live precheck
pass. Offline analysis streams book events instead of retaining millions of
per-pair dictionaries, and keeps trade tape in compact typed arrays.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import contextlib
import heapq
import json
import sys
import time
from array import array
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import websockets

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsboard.net import ssl_context
from jsboard.research import xarb_scan as scan
from tools.xarb_compat import binance_book_compatible
from tools.xarb_scan import okx_books_normalised, okx_contract_value_btc

SOURCES = (
    "binance_spot",
    "binance_perp",
    "bybit_spot",
    "bybit_perp",
    "okx_spot",
    "okx_perp",
)
SOURCE_SET = set(SOURCES)
BINANCE_SPOT_TRADE_WS = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
BINANCE_PERP_TRADE_WS = "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade"
OKX_PUBLIC_WS = scan.OKX_WS


@dataclass(frozen=True, slots=True)
class Trade:
    source: str
    price: float
    qty_base: float
    aggressor: str
    exchange_ts_ns: int
    receive_ts_ns: int


@dataclass(frozen=True, slots=True)
class SampleBook:
    source: str
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    receive_ts_ns: int

    @property
    def bid(self) -> tuple[float, float]:
        return self.bids[0]

    @property
    def ask(self) -> tuple[float, float]:
        return self.asks[0]


@dataclass(slots=True)
class TradeSeries:
    times: array = field(default_factory=lambda: array("q"))
    prices: array = field(default_factory=lambda: array("d"))
    qty: array = field(default_factory=lambda: array("d"))
    sides: array = field(default_factory=lambda: array("b"))

    def append(self, receive_ts_ns: int, price: float, qty_base: float, aggressor: str) -> None:
        self.times.append(receive_ts_ns)
        self.prices.append(price)
        self.qty.append(qty_base)
        self.sides.append(1 if aggressor == "buy" else -1)

    def __len__(self) -> int:
        return len(self.times)


def book_to_record(book: scan.Book) -> dict[str, Any]:
    return {
        "kind": "book",
        "source": book.source,
        "exchange_ts_ns": book.exchange_ts_ns,
        "receive_ts_ns": book.receive_ts_ns,
        "bids": book.bids,
        "asks": book.asks,
    }


def trade_to_record(trade: Trade) -> dict[str, Any]:
    return {
        "kind": "trade",
        "source": trade.source,
        "exchange_ts_ns": trade.exchange_ts_ns,
        "receive_ts_ns": trade.receive_ts_ns,
        "price": trade.price,
        "qty_base": trade.qty_base,
        "aggressor": trade.aggressor,
    }


async def _binance_trade(source: str, url: str, out: asyncio.Queue) -> None:
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**16,
                ssl=ssl_context(),
            ) as ws:
                async for raw in ws:
                    recv = time.monotonic_ns()
                    payload = json.loads(raw)
                    data = payload.get("data", payload)
                    if data.get("e") != "aggTrade":
                        continue
                    price = float(data.get("p") or 0)
                    qty = float(data.get("q") or 0)
                    if price <= 0 or qty <= 0:
                        continue
                    await out.put(("trade", Trade(
                        source=source,
                        price=price,
                        qty_base=qty,
                        aggressor="sell" if bool(data.get("m")) else "buy",
                        exchange_ts_ns=int(data.get("T") or data.get("E") or 0) * 1_000_000,
                        receive_ts_ns=recv,
                    )))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[xarb-lab] {source} trade reconnect: {type(exc).__name__}: {exc}")
            await asyncio.sleep(1.0)


async def _bybit_trade(source: str, out: asyncio.Queue) -> None:
    url = scan.BYBIT_WS[source]
    topic = "publicTrade.BTCUSDT"
    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**16,
                ssl=ssl_context(),
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
                async for raw in ws:
                    recv = time.monotonic_ns()
                    payload = json.loads(raw)
                    if payload.get("success") is False:
                        raise RuntimeError(payload.get("ret_msg") or "Bybit trade subscription failed")
                    if payload.get("topic") != topic:
                        continue
                    for row in payload.get("data") or ():
                        price = float(row.get("p") or 0)
                        qty = float(row.get("v") or 0)
                        side = str(row.get("S") or "").lower()
                        if price <= 0 or qty <= 0 or side not in {"buy", "sell"}:
                            continue
                        await out.put(("trade", Trade(
                            source=source,
                            price=price,
                            qty_base=qty,
                            aggressor=side,
                            exchange_ts_ns=int(row.get("T") or payload.get("ts") or 0) * 1_000_000,
                            receive_ts_ns=recv,
                        )))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[xarb-lab] {source} trade reconnect: {type(exc).__name__}: {exc}")
            await asyncio.sleep(1.0)


async def _okx_trades(out: asyncio.Queue, contract_btc: float) -> None:
    subscriptions = [
        {"channel": "trades", "instId": "BTC-USDT"},
        {"channel": "trades", "instId": "BTC-USDT-SWAP"},
    ]
    source_for = {"BTC-USDT": "okx_spot", "BTC-USDT-SWAP": "okx_perp"}
    while True:
        try:
            async with websockets.connect(
                OKX_PUBLIC_WS,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**16,
                ssl=ssl_context(),
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": subscriptions}))
                async for raw in ws:
                    recv = time.monotonic_ns()
                    payload = json.loads(raw)
                    if payload.get("event") == "error":
                        raise RuntimeError(payload.get("msg") or "OKX trade subscribe failed")
                    arg = payload.get("arg") or {}
                    if arg.get("channel") != "trades":
                        continue
                    inst = arg.get("instId")
                    source = source_for.get(inst)
                    if source is None:
                        continue
                    multiplier = contract_btc if inst == "BTC-USDT-SWAP" else 1.0
                    for row in payload.get("data") or ():
                        price = float(row.get("px") or 0)
                        qty = float(row.get("sz") or 0) * multiplier
                        side = str(row.get("side") or "").lower()
                        if price <= 0 or qty <= 0 or side not in {"buy", "sell"}:
                            continue
                        await out.put(("trade", Trade(
                            source=source,
                            price=price,
                            qty_base=qty,
                            aggressor=side,
                            exchange_ts_ns=int(row.get("ts") or 0) * 1_000_000,
                            receive_ts_ns=recv,
                        )))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[xarb-lab] OKX trade reconnect: {type(exc).__name__}: {exc}")
            await asyncio.sleep(1.0)


async def _relay_books(book_queue: asyncio.Queue, out: asyncio.Queue) -> None:
    while True:
        await out.put(("book", await book_queue.get()))


async def start_streams(out: asyncio.Queue) -> list[asyncio.Task]:
    book_q: asyncio.Queue[scan.Book] = asyncio.Queue(maxsize=200_000)
    contract_btc = await okx_contract_value_btc()
    print(f"[xarb-lab] OKX BTC-USDT-SWAP ctVal={contract_btc:g} BTC/contract")
    return [
        asyncio.create_task(binance_book_compatible("binance_spot", scan.BINANCE_SPOT_WS, book_q)),
        asyncio.create_task(binance_book_compatible("binance_perp", scan.BINANCE_PERP_WS, book_q)),
        asyncio.create_task(scan._bybit_book("bybit_spot", book_q)),
        asyncio.create_task(scan._bybit_book("bybit_perp", book_q)),
        asyncio.create_task(okx_books_normalised(book_q)),
        asyncio.create_task(_relay_books(book_q, out)),
        asyncio.create_task(_binance_trade("binance_spot", BINANCE_SPOT_TRADE_WS, out)),
        asyncio.create_task(_binance_trade("binance_perp", BINANCE_PERP_TRADE_WS, out)),
        asyncio.create_task(_bybit_trade("bybit_spot", out)),
        asyncio.create_task(_bybit_trade("bybit_perp", out)),
        asyncio.create_task(_okx_trades(out, contract_btc)),
    ]


async def stop_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


def selftest() -> None:
    now = 1_000_000_000
    buy = SampleBook("a", ((99.0, 5.0),), ((100.0, 5.0),), now)
    sell = SampleBook("b", ((101.0, 5.0),), ((102.0, 5.0),), now)
    got = _tt_gross(buy, sell, 100.0)
    assert got is not None and got[0] > 99.0
    series = TradeSeries()
    series.append(now + 10_000_000, 99.0, 0.4, "sell")
    series.append(now + 20_000_000, 99.0, 0.8, "sell")
    fill = _fill_proxy(series, ts_ns=now, window_ms=100.0, side="buy",
                       price=99.0, queue_ahead=0.5, order_qty=0.1)
    assert fill["filled"]
    print("[xarb-lab] selftest PASS")


async def precheck(duration_s: float = 20.0) -> dict[str, Any]:
    out: asyncio.Queue = asyncio.Queue(maxsize=300_000)
    tasks = await start_streams(out)
    book_counts, trade_counts, bad_books = Counter(), Counter(), Counter()
    last_books: dict[str, scan.Book] = {}
    deadline = time.monotonic() + duration_s
    try:
        while time.monotonic() < deadline:
            left = max(0.01, deadline - time.monotonic())
            try:
                kind, obj = await asyncio.wait_for(out.get(), timeout=min(1.0, left))
            except TimeoutError:
                continue
            if kind == "book":
                book_counts[obj.source] += 1
                if not obj.valid():
                    bad_books[obj.source] += 1
                else:
                    last_books[obj.source] = obj
            else:
                trade_counts[obj.source] += 1
    finally:
        await stop_tasks(tasks)

    missing_books = sorted(SOURCE_SET - set(book_counts))
    missing_trades = sorted(SOURCE_SET - set(trade_counts))
    now_ns = time.monotonic_ns()
    stale = sorted(s for s, b in last_books.items() if (now_ns - b.receive_ts_ns) / 1e6 > 1_000.0)
    passed = not missing_books and not missing_trades and not bad_books and not stale
    result = {
        "passed": passed,
        "duration_s": duration_s,
        "book_counts": {s: book_counts[s] for s in SOURCES},
        "trade_counts": {s: trade_counts[s] for s in SOURCES},
        "missing_books": missing_books,
        "missing_trades": missing_trades,
        "bad_books": dict(bad_books),
        "stale_at_end": stale,
    }
    print("\n=== xarb-lab precheck ===")
    for source in SOURCES:
        print(f"{source:14s} books={book_counts[source]:6d} trades={trade_counts[source]:7d}")
    print("missing books :", ", ".join(missing_books) if missing_books else "none")
    print("missing trades:", ", ".join(missing_trades) if missing_trades else "none")
    print("bad books     :", dict(bad_books) if bad_books else "none")
    print("stale at end  :", ", ".join(stale) if stale else "none")
    print("PRECHECK      :", "PASS" if passed else "FAIL")
    return result


async def capture(path: Path, duration_s: float) -> dict[str, Any]:
    out: asyncio.Queue = asyncio.Queue(maxsize=500_000)
    tasks = await start_streams(out)
    counts = Counter()
    started = time.monotonic()
    deadline = started + duration_s
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "kind": "meta",
            "version": 2,
            "started_wall_ns": time.time_ns(),
            "duration_s": duration_s,
            "sources": SOURCES,
        }, separators=(",", ":")) + "\n")
        pending_flush = 0
        try:
            while time.monotonic() < deadline:
                left = max(0.01, deadline - time.monotonic())
                try:
                    kind, obj = await asyncio.wait_for(out.get(), timeout=min(1.0, left))
                except TimeoutError:
                    continue
                if kind == "book":
                    record = book_to_record(obj)
                    counts[f"book:{obj.source}"] += 1
                else:
                    record = trade_to_record(obj)
                    counts[f"trade:{obj.source}"] += 1
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                pending_flush += 1
                if pending_flush >= 2_000:
                    fh.flush()
                    pending_flush = 0
        finally:
            await stop_tasks(tasks)
            fh.flush()

    meta = {
        "path": str(path),
        "duration_s": time.monotonic() - started,
        "counts": dict(counts),
        "bytes": path.stat().st_size,
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\n=== capture complete ===")
    print(f"path    : {path}")
    print(f"size    : {meta['bytes'] / 1024 / 1024:.1f} MB")
    for source in SOURCES:
        print(f"{source:14s} books={counts[f'book:{source}']:7d} trades={counts[f'trade:{source}']:8d}")
    return meta


def _book_from_record(row: dict[str, Any]) -> SampleBook:
    return SampleBook(
        source=row["source"],
        bids=tuple((float(p), float(q)) for p, q in row["bids"]),
        asks=tuple((float(p), float(q)) for p, q in row["asks"]),
        receive_ts_ns=int(row["receive_ts_ns"]),
    )


def _fresh_pair(a: SampleBook, b: SampleBook, now_ns: int, max_age_ms: float, max_skew_ms: float) -> bool:
    age_a = max(0.0, (now_ns - a.receive_ts_ns) / 1e6)
    age_b = max(0.0, (now_ns - b.receive_ts_ns) / 1e6)
    skew = abs(a.receive_ts_ns - b.receive_ts_ns) / 1e6
    return age_a <= max_age_ms and age_b <= max_age_ms and skew <= max_skew_ms


def _tt_gross(buy_book: SampleBook, sell_book: SampleBook, notional: float):
    buy = scan.walk_buy_quote(buy_book.asks, notional)
    if not buy.complete or buy.qty_base <= 0:
        return None
    sell = scan.walk_sell_base(sell_book.bids, buy.qty_base)
    if not sell.complete:
        return None
    return (sell.avg_price / buy.avg_price - 1.0) * 10_000.0, buy.qty_base


def _mt_buy_gross(maker_buy: SampleBook, taker_sell: SampleBook, notional: float):
    maker_px, queue_ahead = maker_buy.bid
    qty = notional / maker_px
    hedge = scan.walk_sell_base(taker_sell.bids, qty)
    if not hedge.complete:
        return None
    return (hedge.avg_price / maker_px - 1.0) * 10_000.0, qty, queue_ahead


def _mt_sell_gross(taker_buy: SampleBook, maker_sell: SampleBook, notional: float):
    maker_px, queue_ahead = maker_sell.ask
    qty = notional / maker_px
    spent = 0.0
    remaining = qty
    for px, avail in taker_buy.asks:
        take = min(remaining, avail)
        spent += take * px
        remaining -= take
        if remaining <= qty * 1e-9:
            break
    if remaining > qty * 1e-9:
        return None
    avg_buy = spent / qty
    return (maker_px / avg_buy - 1.0) * 10_000.0, qty, queue_ahead


def _heap_push_top(heap: list, score: float, seq: int, item: dict[str, Any], limit: int) -> None:
    node = (score, seq, item)
    if len(heap) < limit:
        heapq.heappush(heap, node)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, node)


def _load_trade_series(path: Path) -> tuple[dict[str, TradeSeries], int]:
    series = {s: TradeSeries() for s in SOURCES}
    book_events = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "book":
                book_events += 1
            elif kind == "trade" and row.get("source") in series:
                series[row["source"]].append(
                    int(row["receive_ts_ns"]),
                    float(row["price"]),
                    float(row["qty_base"]),
                    row["aggressor"],
                )
    return series, book_events


def _iter_books(path: Path) -> Iterator[SampleBook]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") == "book":
                yield _book_from_record(row)


def _fill_proxy(series: TradeSeries, *, ts_ns: int, window_ms: float,
                side: str, price: float, queue_ahead: float, order_qty: float) -> dict[str, Any]:
    lo = bisect.bisect_left(series.times, ts_ns)
    hi = bisect.bisect_right(series.times, ts_ns + int(window_ms * 1e6))
    required = queue_ahead + order_qty
    consumed = 0.0
    fill_ts = None
    want_side = -1 if side == "buy" else 1
    for i in range(lo, hi):
        if series.sides[i] != want_side:
            continue
        trade_px = series.prices[i]
        if side == "buy" and trade_px > price:
            continue
        if side == "sell" and trade_px < price:
            continue
        consumed += series.qty[i]
        if consumed >= required:
            fill_ts = int(series.times[i])
            break
    return {
        "filled": fill_ts is not None,
        "fill_ts_ns": fill_ts,
        "required_qty": required,
        "aggressive_qty": consumed,
        "fill_ratio": min(1.0, consumed / required) if required > 0 else 0.0,
    }


def analyze(path: Path, *, notionals: tuple[float, ...], sample_ms: float,
            max_age_ms: float, max_skew_ms: float, buffers_bps: tuple[float, ...],
            fee_grid_bps: tuple[float, ...], maker_window_ms: float,
            maker_top_per_size: int, maker_fee_bps: float) -> dict[str, Any]:
    print("\n[xarb-lab] pass 1/2: compacting trade tape...")
    trades, book_events = _load_trade_series(path)
    print("[xarb-lab] pass 2/2: sweeping books...")

    latest: dict[str, SampleBook] = {}
    next_sample_ns: int | None = None
    sample_step_ns = int(sample_ms * 1e6)
    gross_values = {n: array("d") for n in notionals}
    tt_best: dict[float, dict[str, Any]] = {}
    tt_top_heap: list = []
    mt_heaps = {n: [] for n in notionals}
    mm_heaps = {n: [] for n in notionals}
    tt_observations = 0
    fresh_samples = 0
    seq = 0

    for book in _iter_books(path):
        ts_ns = book.receive_ts_ns
        latest[book.source] = book
        if next_sample_ns is None:
            next_sample_ns = ts_ns
        if ts_ns < next_sample_ns:
            continue
        next_sample_ns = ts_ns + sample_step_ns
        if len(latest) < len(SOURCES):
            continue
        fresh_samples += 1

        for notional in notionals:
            for buy_source, buy_book in latest.items():
                for sell_source, sell_book in latest.items():
                    if buy_source == sell_source:
                        continue
                    if not _fresh_pair(buy_book, sell_book, ts_ns, max_age_ms, max_skew_ms):
                        continue
                    got = _tt_gross(buy_book, sell_book, notional)
                    if got is None:
                        continue
                    gross, qty = got
                    tt_observations += 1
                    gross_values[notional].append(gross)
                    fees = scan.DEFAULT_FEES_BPS[buy_source] + scan.DEFAULT_FEES_BPS[sell_source]
                    row = {
                        "ts_ns": ts_ns,
                        "notional": notional,
                        "buy": buy_source,
                        "sell": sell_source,
                        "gross_bps": gross,
                        "actual_fee_bps": fees,
                        "actual_net_bps_at_buffer4": gross - fees - 4.0,
                        "qty_base": qty,
                    }
                    current = tt_best.get(notional)
                    if current is None or gross > current["gross_bps"]:
                        tt_best[notional] = row
                    seq += 1
                    _heap_push_top(tt_top_heap, gross, seq, row, 100)

                    mt = _mt_buy_gross(buy_book, sell_book, notional)
                    if mt is not None:
                        g, q, queue_ahead = mt
                        seq += 1
                        _heap_push_top(mt_heaps[notional], g, seq, {
                            "kind": "maker_buy_taker_sell",
                            "ts_ns": ts_ns,
                            "notional": notional,
                            "maker_source": buy_source,
                            "hedge_source": sell_source,
                            "maker_side": "buy",
                            "maker_price": buy_book.bid[0],
                            "gross_bps_at_quote": g,
                            "qty_base": q,
                            "queue_ahead": queue_ahead,
                        }, maker_top_per_size)

                    mt = _mt_sell_gross(buy_book, sell_book, notional)
                    if mt is not None:
                        g, q, queue_ahead = mt
                        seq += 1
                        _heap_push_top(mt_heaps[notional], g, seq, {
                            "kind": "taker_buy_maker_sell",
                            "ts_ns": ts_ns,
                            "notional": notional,
                            "maker_source": sell_source,
                            "hedge_source": buy_source,
                            "maker_side": "sell",
                            "maker_price": sell_book.ask[0],
                            "gross_bps_at_quote": g,
                            "qty_base": q,
                            "queue_ahead": queue_ahead,
                        }, maker_top_per_size)

                    buy_px, buy_q = buy_book.bid
                    sell_px, sell_q = sell_book.ask
                    mm_gross = (sell_px / buy_px - 1.0) * 10_000.0
                    seq += 1
                    _heap_push_top(mm_heaps[notional], mm_gross, seq, {
                        "kind": "maker_maker",
                        "ts_ns": ts_ns,
                        "notional": notional,
                        "buy_source": buy_source,
                        "sell_source": sell_source,
                        "buy_price": buy_px,
                        "sell_price": sell_px,
                        "gross_bps_at_quote": mm_gross,
                        "qty_base": notional / buy_px,
                        "buy_queue_ahead": buy_q,
                        "sell_queue_ahead": sell_q,
                    }, maker_top_per_size)

    sensitivity: list[dict[str, Any]] = []
    for notional in notionals:
        ordered = sorted(gross_values[notional])
        best = ordered[-1] if ordered else None
        for buffer in buffers_bps:
            for total_fee in fee_grid_bps:
                threshold = buffer + total_fee
                positive = 0 if not ordered else len(ordered) - bisect.bisect_right(ordered, threshold)
                sensitivity.append({
                    "notional": notional,
                    "buffer_bps": buffer,
                    "combined_fee_bps": total_fee,
                    "positive_samples": positive,
                    "best_net_bps": None if best is None else best - threshold,
                })
        del ordered
        gross_values[notional] = array("d")

    mt_candidates = [item for n in notionals for _, _, item in sorted(mt_heaps[n], reverse=True)]
    mm_candidates = [item for n in notionals for _, _, item in sorted(mm_heaps[n], reverse=True)]

    mt_validated = []
    for item0 in mt_candidates:
        item = dict(item0)
        source = item["maker_source"]
        fill = _fill_proxy(
            trades[source],
            ts_ns=item["ts_ns"],
            window_ms=maker_window_ms,
            side=item["maker_side"],
            price=item["maker_price"],
            queue_ahead=item["queue_ahead"],
            order_qty=item["qty_base"],
        )
        taker_fee = scan.DEFAULT_FEES_BPS[item["hedge_source"]]
        item.update({
            "fill_proxy": fill,
            "assumed_maker_fee_bps": maker_fee_bps,
            "taker_fee_bps": taker_fee,
            "net_bps_before_hedge_reprice": item["gross_bps_at_quote"] - maker_fee_bps - taker_fee,
            "break_even_maker_fee_bps_before_buffers": item["gross_bps_at_quote"] - taker_fee,
        })
        mt_validated.append(item)

    mm_validated = []
    for item0 in mm_candidates:
        item = dict(item0)
        buy_fill = _fill_proxy(
            trades[item["buy_source"]],
            ts_ns=item["ts_ns"],
            window_ms=maker_window_ms,
            side="buy",
            price=item["buy_price"],
            queue_ahead=item["buy_queue_ahead"],
            order_qty=item["qty_base"],
        )
        sell_fill = _fill_proxy(
            trades[item["sell_source"]],
            ts_ns=item["ts_ns"],
            window_ms=maker_window_ms,
            side="sell",
            price=item["sell_price"],
            queue_ahead=item["sell_queue_ahead"],
            order_qty=item["qty_base"],
        )
        item.update({
            "buy_fill_proxy": buy_fill,
            "sell_fill_proxy": sell_fill,
            "both_fill_proxy": bool(buy_fill["filled"] and sell_fill["filled"]),
            "leg_gap_ms": None if not (buy_fill["fill_ts_ns"] and sell_fill["fill_ts_ns"]) else abs(buy_fill["fill_ts_ns"] - sell_fill["fill_ts_ns"]) / 1e6,
            "assumed_maker_fee_each_bps": maker_fee_bps,
            "net_bps_before_leg_risk": item["gross_bps_at_quote"] - 2.0 * maker_fee_bps,
            "break_even_combined_maker_fee_bps": item["gross_bps_at_quote"],
        })
        mm_validated.append(item)

    def top(rows: list[dict[str, Any]], key: str, n: int = 30) -> list[dict[str, Any]]:
        return sorted(rows, key=lambda r: r[key], reverse=True)[:n]

    return {
        "capture": str(path),
        "config": {
            "notionals": notionals,
            "sample_ms": sample_ms,
            "max_age_ms": max_age_ms,
            "max_skew_ms": max_skew_ms,
            "buffers_bps": buffers_bps,
            "fee_grid_bps": fee_grid_bps,
            "maker_window_ms": maker_window_ms,
            "maker_top_per_size": maker_top_per_size,
            "maker_fee_bps_assumption": maker_fee_bps,
        },
        "data": {
            "book_events": book_events,
            "trade_events": {s: len(trades[s]) for s in SOURCES},
            "fresh_sample_ticks": fresh_samples,
        },
        "tt": {
            "observations": tt_observations,
            "best_by_notional": tt_best,
            "top": [item for _, _, item in sorted(tt_top_heap, reverse=True)[:30]],
            "sensitivity": sensitivity,
        },
        "mt": {
            "economic_candidates_checked": len(mt_validated),
            "fill_proxy_passed": sum(1 for r in mt_validated if r["fill_proxy"]["filled"]),
            "top_fill_proxy_passed": top([r for r in mt_validated if r["fill_proxy"]["filled"]], "gross_bps_at_quote"),
            "note": "Displayed quantity at the maker quote is treated as queue ahead; observed aggressor volume must consume queue+order inside the window. Conservative proxy, not order-level replay. Hedge repricing after the maker fill remains a separate stress buffer.",
        },
        "mm": {
            "economic_candidates_checked": len(mm_validated),
            "both_fill_proxy_passed": sum(1 for r in mm_validated if r["both_fill_proxy"]),
            "top_both_fill_proxy_passed": top([r for r in mm_validated if r["both_fill_proxy"]], "gross_bps_at_quote"),
            "note": "Both maker legs must pass the queue/tape proxy in the same window; leg_gap_ms is reported and interim inventory risk is not removed by assumption.",
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n================ XARB LAB REPORT ================")
    print(f"book events={report['data']['book_events']:,} fresh samples={report['data']['fresh_sample_ticks']:,}")
    print("trades:", ", ".join(f"{s}={report['data']['trade_events'][s]:,}" for s in SOURCES))
    print("\n-- Taker/Taker best by size --")
    for notional, row in sorted(report["tt"]["best_by_notional"].items()):
        gross = row["gross_bps"]
        fees = row["actual_fee_bps"]
        print(f"${notional:8g} {row['buy']:14s}->{row['sell']:14s} gross={gross:+7.3f}bps actual_fees={fees:5.2f} net@4buf={gross-fees-4:+7.3f}")
    print("\n-- Fee break-even from best observed Taker/Taker gross --")
    for notional, row in sorted(report["tt"]["best_by_notional"].items()):
        gross = row["gross_bps"]
        print(f"${notional:8g} combined fee ceiling: {gross:+.3f}bps @buf0 / {gross-1:+.3f} @buf1 / {gross-2:+.3f} @buf2 / {gross-4:+.3f} @buf4")
    mt = report["mt"]
    print("\n-- Maker/Taker queue+tape proxy --")
    print(f"checked={mt['economic_candidates_checked']:,} fill_proxy_passed={mt['fill_proxy_passed']:,}")
    for row in mt["top_fill_proxy_passed"][:10]:
        print(f"${row['notional']:g} {row['kind']} {row['maker_source']} / {row['hedge_source']} gross={row['gross_bps_at_quote']:+.3f} break_even_maker_fee={row['break_even_maker_fee_bps_before_buffers']:+.3f}bps")
    mm = report["mm"]
    print("\n-- Maker/Maker queue+tape proxy --")
    print(f"checked={mm['economic_candidates_checked']:,} both_fill_proxy_passed={mm['both_fill_proxy_passed']:,}")
    for row in mm["top_both_fill_proxy_passed"][:10]:
        print(f"${row['notional']:g} {row['buy_source']}->{row['sell_source']} gross={row['gross_bps_at_quote']:+.3f}bps leg_gap={row['leg_gap_ms']:.1f}ms break_even_combined_maker_fee={row['break_even_combined_maker_fee_bps']:+.3f}bps")
    print("=================================================")


def add_analysis_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--notional", type=float, nargs="+", default=[100, 500, 1000, 5000, 10000, 50000])
    p.add_argument("--sample-ms", type=float, default=100.0)
    p.add_argument("--max-age-ms", type=float, default=250.0)
    p.add_argument("--max-skew-ms", type=float, default=250.0)
    p.add_argument("--fee-grid", type=float, nargs="+", default=[0, 0.5, 1, 2, 5, 10, 15, 20])
    p.add_argument("--buffer-grid", type=float, nargs="+", default=[0, 1, 2, 4, 6])
    p.add_argument("--maker-window-ms", type=float, default=500.0)
    p.add_argument("--maker-top-per-size", type=int, default=2000)
    p.add_argument("--maker-fee-bps", type=float, default=0.0, help="maker fee assumption for summary; break-even is always reported")
    p.add_argument("--report", default="xarb-lab-report.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-shot BTC six-market cross-venue research harness")
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("precheck")
    pre.add_argument("--duration", type=float, default=20.0)
    cap = sub.add_parser("capture")
    cap.add_argument("--duration", type=float, default=3600.0)
    cap.add_argument("--out", default="xarb-lab.jsonl")
    cap.add_argument("--precheck", type=float, default=20.0)
    ana = sub.add_parser("analyze")
    ana.add_argument("path")
    add_analysis_args(ana)
    allp = sub.add_parser("all")
    allp.add_argument("--duration", type=float, default=3600.0)
    allp.add_argument("--precheck", type=float, default=20.0)
    allp.add_argument("--out", default="xarb-lab.jsonl")
    add_analysis_args(allp)
    sub.add_parser("selftest")
    return p


def run_analysis(args: argparse.Namespace, path: Path) -> dict[str, Any]:
    return analyze(
        path,
        notionals=tuple(args.notional),
        sample_ms=args.sample_ms,
        max_age_ms=args.max_age_ms,
        max_skew_ms=args.max_skew_ms,
        buffers_bps=tuple(args.buffer_grid),
        fee_grid_bps=tuple(args.fee_grid),
        maker_window_ms=args.maker_window_ms,
        maker_top_per_size=args.maker_top_per_size,
        maker_fee_bps=args.maker_fee_bps,
    )


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "precheck":
        selftest()
        result = await precheck(args.duration)
        return 0 if result["passed"] else 2
    if args.command == "capture":
        selftest()
        result = await precheck(args.precheck)
        if not result["passed"]:
            print("[xarb-lab] PRECHECK failed. Long capture was NOT started.")
            return 2
        await capture(Path(args.out), args.duration)
        return 0
    if args.command == "all":
        selftest()
        check = await precheck(args.precheck)
        if not check["passed"]:
            print("[xarb-lab] PRECHECK failed. Long capture was NOT started.")
            return 2
        path = Path(args.out)
        await capture(path, args.duration)
        report = run_analysis(args, path)
        report["precheck"] = check
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_report(report)
        print(f"\nreport: {args.report}")
        return 0
    raise AssertionError(args.command)


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "selftest":
        selftest()
        return 0
    if args.command == "analyze":
        selftest()
        report = run_analysis(args, Path(args.path))
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_report(report)
        print(f"\nreport: {args.report}")
        return 0
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
