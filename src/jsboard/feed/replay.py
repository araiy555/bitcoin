"""Offline feeds: a synthetic market and a recorded-session replayer.

`SyntheticFeed` exists so the engine, the quoter and the UI can be exercised
without a network — and, being seeded, so tests get the same market twice.
It is a caricature of a real book, not a calibrated model: a random-walk mid,
depth that decays with distance from the touch, and Poisson trade arrivals
skewed by book imbalance.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections.abc import AsyncIterator
from pathlib import Path

from ..core.types import Instrument, Side
from .base import (
    DepthDelta,
    DepthSnapshot,
    Feed,
    FeedEvent,
    FeedStatus,
    Liquidation,
    MarkPrice,
    OpenInterest,
    TradeTick,
)


class SyntheticFeed(Feed):
    """A self-contained fake market."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        start_price: float = 64000.0,
        levels: int = 25,
        tick_interval: float = 0.1,
        volatility_bps: float = 0.1,
        base_depth_lots: int = 20_000,
        spread_ticks: int = 8,
        trade_rate: float = 15.0,
        seed: int | None = None,
        max_events: int | None = None,
    ) -> None:
        super().__init__(instrument)
        self.levels = levels
        self.tick_interval = tick_interval
        # Volatility is per emitted tick, and these defaults are chosen to be
        # *mutually* consistent rather than individually realistic. 0.1bps per
        # 100ms scales to roughly 1%/day; the 8-tick spread and 20k-lot levels
        # then leave a book a maker can actually work — wide enough to quote
        # inside, deep enough that queue position still costs something.
        self.volatility_bps = volatility_bps
        self.base_depth_lots = base_depth_lots
        self.spread_ticks = max(1, spread_ticks)
        self.trade_rate = trade_rate
        self.max_events = max_events
        self._rng = random.Random(seed)
        self._mid = instrument.to_ticks(start_price)
        self._update_id = 1
        self._bids: dict[int, int] = {}
        self._asks: dict[int, int] = {}
        # Virtual clock. Advances by `tick_interval` whether or not we actually
        # sleep, so a zero-delay backtest still has a coherent timeline for
        # requote intervals, latency, and staleness checks.
        self._now_ns = time.time_ns()
        self._step_ns = int((tick_interval or 0.1) * 1e9)

    @property
    def now_ns(self) -> int:
        return self._now_ns

    # ------------------------------------------------------------ mechanics

    def _depth_for(self, distance: int) -> int:
        """Level size: thin at the touch, fatter behind it, plus noise."""
        shape = 1.0 - math.exp(-0.45 * (distance + 1))
        noise = self._rng.uniform(0.55, 1.45)
        return max(1, int(self.base_depth_lots * shape * noise))

    def _build_book(self) -> tuple[dict[int, int], dict[int, int]]:
        half = self.spread_ticks // 2
        best_bid = self._mid - half - (self.spread_ticks % 2)
        best_ask = self._mid + half + 1
        bids = {best_bid - i: self._depth_for(i) for i in range(self.levels)}
        asks = {best_ask + i: self._depth_for(i) for i in range(self.levels)}
        return bids, asks

    def _step_mid(self) -> None:
        sigma = self._mid * self.volatility_bps / 10_000.0
        self._mid = max(1, int(round(self._mid + self._rng.gauss(0.0, sigma))))

    def _diff(self, old: dict[int, int], new: dict[int, int]) -> tuple[tuple[int, int], ...]:
        """Absolute quantities for every level that moved; 0 means delete."""
        changed = [(p, q) for p, q in new.items() if old.get(p) != q]
        changed += [(p, 0) for p in old if p not in new]
        return tuple(sorted(changed))

    def _maybe_trades(self) -> list[TradeTick]:
        """Poisson arrivals, leaning toward the heavier side of the book."""
        # Rate is per second of *simulated* time. Using the wall-clock sleep
        # interval instead would silence the tape entirely in a zero-delay
        # backtest, where that interval is 0.
        n = self._poisson(self.trade_rate * self._step_ns / 1e9)
        if not n:
            return []
        best_bid = max(self._bids) if self._bids else None
        best_ask = min(self._asks) if self._asks else None
        if best_bid is None or best_ask is None:
            return []

        bid_qty = self._bids[best_bid]
        ask_qty = self._asks[best_ask]
        # A heavy bid means queued buyers, so the next print more often lifts
        # the offer. Same intuition as the microprice.
        p_buy = bid_qty / (bid_qty + ask_qty) if (bid_qty + ask_qty) else 0.5

        out = []
        for _ in range(n):
            if self._rng.random() < p_buy:
                side, price = Side.BUY, best_ask
            else:
                side, price = Side.SELL, best_bid
            qty = max(1, int(self._rng.expovariate(1.0 / max(1.0, self.base_depth_lots * 0.08))))
            out.append(
                TradeTick(
                    price=price,
                    qty=qty,
                    aggressor=side,
                    trade_id=self._update_id,
                    ts_ns=self._now_ns,
                )
            )
        return out

    def _poisson(self, lam: float) -> int:
        if lam <= 0:
            return 0
        threshold, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self._rng.random()
            if p <= threshold:
                return k
            k += 1
            if k > 50:  # guard against pathological lambda
                return k

    # --------------------------------------------------------------- stream

    async def stream(self) -> AsyncIterator[FeedEvent]:
        yield FeedStatus("connecting", "synthetic market")
        self._bids, self._asks = self._build_book()
        yield DepthSnapshot(
            bids=tuple(sorted(self._bids.items(), reverse=True)),
            asks=tuple(sorted(self._asks.items())),
            last_update_id=self._update_id,
            ts_ns=self._now_ns,
        )
        yield FeedStatus("live", "synthetic market")

        emitted = 0
        while self.max_events is None or emitted < self.max_events:
            if self.tick_interval:
                # Real-time mode: the data is generated on demand, so it *is*
                # current. Read the wall clock rather than accumulating the
                # nominal interval — each loop really takes `tick_interval`
                # plus however long the consumer spent rendering, and adding
                # only the nominal amount makes the stamp fall further behind
                # every tick until the staleness gate pulls quotes for good.
                await asyncio.sleep(self.tick_interval)
                self._now_ns = time.time_ns()
            else:
                # No sleeping, so there is no wall clock worth reading; advance
                # the simulated timeline instead.
                self._now_ns += self._step_ns

            self._step_mid()
            new_bids, new_asks = self._build_book()
            bid_diff = self._diff(self._bids, new_bids)
            ask_diff = self._diff(self._asks, new_asks)
            self._bids, self._asks = new_bids, new_asks

            first = self._update_id + 1
            self._update_id += 1
            if bid_diff or ask_diff:
                yield DepthDelta(
                    bids=bid_diff,
                    asks=ask_diff,
                    first_id=first,
                    final_id=self._update_id,
                    ts_ns=self._now_ns,
                )

            for trade in self._maybe_trades():
                yield trade

            emitted += 1


class JsonlRecorder:
    """Append raw feed events to a JSONL file for later replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> JsonlRecorder:
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def write(self, event: FeedEvent) -> None:
        if self._fh is None:
            raise RuntimeError("recorder used outside its context manager")
        self._fh.write(json.dumps(_encode(event)) + "\n")


def iter_tagged(path: str | Path):
    """Yield (source, event) for every line, in recorded order.

    `ReplayFeed` deliberately serves one venue, because feeding two into one
    book builds a book that never existed. Cross-venue work needs the
    opposite: both streams, still interleaved, each routed to its own book.
    """
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            src = raw.pop(SOURCE_KEY, None)
            for key in ENVELOPE_KEYS:
                raw.pop(key, None)
            yield src, _decode(raw)


def iter_tagged_timed(path: str | Path):
    """Yield ``(source, receive_time_ns, event)`` in observable order.

    Cross-market decisions must use the time this process knew each update,
    not compare exchange clocks from two different products.  Old recordings
    without ``rx_ns`` remain usable by falling back to the event timestamp.
    """
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            src = raw.pop(SOURCE_KEY, None)
            received_ns = raw.pop(RX_KEY, None)
            for key in ENVELOPE_KEYS:
                raw.pop(key, None)
            event = _decode(raw)
            if received_ns is None:
                received_ns = getattr(event, "ts_ns", 0)
            yield src, int(received_ns or 0), event


SOURCE_KEY = "src"
"""Which venue a captured line came from."""

RX_KEY = "rx_ns"
"""When this process received the line, as distinct from the exchange time."""

ENVELOPE_KEYS = (
    SOURCE_KEY,
    RX_KEY,
    "schema_version",
    "capture_id",
    "event_seq",
    "venue",
    "market_type",
    "symbol_native",
    "instrument_id",
    "event_type",
    "ts_exchange_ns",
    "ts_receive_ns",
    "ts_wall_ns",
    "connection_id",
    "source_mode",
)
"""Keys `capture` adds around an encoded event, which are not event fields.

They describe the recording rather than the market. Passing them to an event
constructor is a TypeError, so every reader strips the complete versioned
envelope. ``src`` and ``rx_ns`` remain for old recordings and old commands;
the longer names are the canonical CMR-001 schema.
"""


class ReplayFeed(Feed):
    """Replay a JSONL recording, optionally in original wall-clock time."""

    def __init__(
        self,
        instrument: Instrument,
        path: str | Path,
        *,
        speed: float = 1.0,
        source: str | None = None,
    ) -> None:
        super().__init__(instrument)
        self.path = Path(path)
        self.speed = speed
        self.source = source
        """Which `src` tag to replay. A `capture` recording interleaves spot
        and perp on one timeline, and feeding both into a single book would
        build a book that never existed. None replays every line, which is
        what a single-venue `record` file wants."""

    async def stream(self) -> AsyncIterator[FeedEvent]:
        yield FeedStatus("connecting", str(self.path))
        prev_ts: int | None = None
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                src = raw.pop(SOURCE_KEY, None)
                for key in ENVELOPE_KEYS:
                    raw.pop(key, None)
                if self.source is not None and src is not None and src != self.source:
                    continue
                event = _decode(raw)
                ts = getattr(event, "ts_ns", None)
                if self.speed > 0 and prev_ts is not None and ts is not None:
                    gap = (ts - prev_ts) / 1e9 / self.speed
                    if 0 < gap < 5.0:
                        await asyncio.sleep(gap)
                if ts is not None:
                    prev_ts = ts
                yield event
        yield FeedStatus("disconnected", "replay exhausted")


# ------------------------------------------------------------ (de)serialising

_KINDS = {
    "snapshot": DepthSnapshot,
    "delta": DepthDelta,
    "trade": TradeTick,
    "status": FeedStatus,
    "mark": MarkPrice,
    "oi": OpenInterest,
    "liq": Liquidation,
}
_NAMES = {v: k for k, v in _KINDS.items()}


def _encode(event: FeedEvent) -> dict:
    kind = _NAMES[type(event)]
    if isinstance(event, TradeTick):
        body = {
            "price": event.price,
            "qty": event.qty,
            "aggressor": int(event.aggressor),
            "trade_id": event.trade_id,
            "ts_ns": event.ts_ns,
        }
    elif isinstance(event, DepthSnapshot):
        body = {
            "bids": [list(x) for x in event.bids],
            "asks": [list(x) for x in event.asks],
            "last_update_id": event.last_update_id,
            "ts_ns": event.ts_ns,
        }
    elif isinstance(event, DepthDelta):
        body = {
            "bids": [list(x) for x in event.bids],
            "asks": [list(x) for x in event.asks],
            "first_id": event.first_id,
            "final_id": event.final_id,
            "ts_ns": event.ts_ns,
        }
    elif isinstance(event, MarkPrice):
        body = {
            "mark": event.mark,
            "index": event.index,
            "funding_rate": event.funding_rate,
            "next_funding_ns": event.next_funding_ns,
            "ts_ns": event.ts_ns,
        }
    elif isinstance(event, OpenInterest):
        body = {"lots": event.lots, "ts_ns": event.ts_ns}
    elif isinstance(event, Liquidation):
        body = {
            "price": event.price,
            "qty": event.qty,
            "side": int(event.side),
            "ts_ns": event.ts_ns,
        }
    else:
        body = {"state": event.state, "detail": event.detail, "ts_ns": event.ts_ns}
    return {"k": kind, **body}


def _decode(row: dict) -> FeedEvent:
    # Be defensive at the codec boundary.  Most readers strip the capture
    # envelope before calling us, but direct callers and older tools may hand
    # over the complete row.
    row = dict(row)
    for key in ENVELOPE_KEYS:
        row.pop(key, None)
    kind = row.pop("k")
    cls = _KINDS[kind]
    if cls is TradeTick:
        row["aggressor"] = Side(row["aggressor"])
    elif cls is Liquidation:
        row["side"] = Side(row["side"])
    elif cls in (DepthSnapshot, DepthDelta):
        row["bids"] = tuple(tuple(x) for x in row["bids"])
        row["asks"] = tuple(tuple(x) for x in row["asks"])
    return cls(**row)
