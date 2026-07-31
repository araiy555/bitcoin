"""Live view of one instrument: the book, the tape, and derived signals.

`MarketView` is the single object the quoter and the UI read from. It owns an
`OrderBook` kept in sync with the feed, a bounded trade tape, and the rolling
statistics (realised volatility, trade-flow imbalance) that the quoter needs.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from ..feed.base import DepthDelta, DepthSnapshot, FeedEvent, FeedStatus, TradeTick
from .book import OrderBook
from .types import BookSnapshot, Instrument, Side

MARKET_OWNER = "market"
"""Owner tag for synthetic orders standing in for public depth."""


@dataclass(slots=True)
class RollingVol:
    """EWMA of squared log returns, reported in basis points.

    `halflife` is in samples, not seconds — the caller decides how often to
    feed it. Seeded on the first two observations so it does not spend its
    early life reading zero.
    """

    halflife: float = 60.0
    _var: float = 0.0
    _last: float | None = None
    _n: int = 0

    @property
    def _alpha(self) -> float:
        return 1.0 - math.exp(-math.log(2.0) / max(1e-9, self.halflife))

    def update(self, price: float) -> None:
        if price <= 0:
            return
        if self._last is None:
            self._last = price
            return
        ret = math.log(price / self._last)
        self._last = price
        self._n += 1
        a = self._alpha if self._n > 1 else 1.0
        self._var = (1 - a) * self._var + a * ret * ret

    @property
    def sigma(self) -> float:
        """Per-sample stdev of log returns."""
        return math.sqrt(max(0.0, self._var))

    @property
    def bps(self) -> float:
        return self.sigma * 10_000.0


@dataclass(slots=True)
class FlowImbalance:
    """Exponentially decayed signed trade volume — who is leaning on the book."""

    halflife: float = 40.0
    _buy: float = 0.0
    _sell: float = 0.0

    @property
    def _decay(self) -> float:
        return math.exp(-math.log(2.0) / max(1e-9, self.halflife))

    def update(self, side: Side, qty: float) -> None:
        d = self._decay
        self._buy *= d
        self._sell *= d
        if side is Side.BUY:
            self._buy += qty
        else:
            self._sell += qty

    @property
    def value(self) -> float:
        """In [-1, 1]; positive means buyers are lifting offers."""
        total = self._buy + self._sell
        return 0.0 if total <= 0 else (self._buy - self._sell) / total


@dataclass(slots=True)
class MarketView:
    instrument: Instrument
    depth: int = 20
    tape_size: int = 500
    clock: object = time.time_ns
    """Source of "now" for staleness. Swapped for a virtual clock in backtests,
    where wall-clock time races far ahead of the simulated timeline."""
    book: OrderBook = field(init=False)
    tape: deque[TradeTick] = field(init=False)
    vol: RollingVol = field(default_factory=RollingVol)
    flow: FlowImbalance = field(default_factory=FlowImbalance)
    status: str = "connecting"
    status_detail: str = ""
    last_update_ns: int = 0
    events_seen: int = 0
    resync_count: int = 0

    def __post_init__(self) -> None:
        self.book = OrderBook()
        self.tape = deque(maxlen=self.tape_size)

    # ----------------------------------------------------------- ingestion

    def apply(self, event: FeedEvent) -> None:
        self.events_seen += 1
        if isinstance(event, DepthSnapshot):
            self.book.replace_l2(list(event.bids), list(event.asks), owner=MARKET_OWNER)
            self.last_update_ns = event.ts_ns
        elif isinstance(event, DepthDelta):
            for price, qty in event.bids:
                self.book.apply_l2_delta(Side.BUY, price, qty, owner=MARKET_OWNER)
            for price, qty in event.asks:
                self.book.apply_l2_delta(Side.SELL, price, qty, owner=MARKET_OWNER)
            self.last_update_ns = event.ts_ns
            mid = self.book.mid
            if mid:
                self.vol.update(mid)
        elif isinstance(event, TradeTick):
            self.tape.append(event)
            self.flow.update(event.aggressor, self.instrument.qty_f(event.qty))
            self.last_update_ns = event.ts_ns
        elif isinstance(event, FeedStatus):
            if event.state == "resyncing":
                self.resync_count += 1
            self.status = event.state
            self.status_detail = event.detail

    # -------------------------------------------------------------- reads

    def snapshot(self, depth: int | None = None) -> BookSnapshot:
        return self.book.snapshot(depth or self.depth)

    @property
    def is_live(self) -> bool:
        return self.status == "live" and self.book.best_bid() is not None

    @property
    def age_ms(self) -> float:
        if not self.last_update_ns:
            return float("inf")
        return max(0.0, (self.clock() - self.last_update_ns) / 1e6)

    @property
    def mid(self) -> float | None:
        return self.book.mid

    @property
    def mid_price(self) -> float | None:
        """Mid in human units rather than ticks."""
        mid = self.book.mid
        return None if mid is None else mid * float(self.instrument.tick_size)

    @property
    def microprice(self) -> float | None:
        return self.snapshot(1).microprice

    def imbalance(self, levels: int = 5) -> float:
        return self.snapshot(levels).imbalance(levels)

    @property
    def spread_ticks(self) -> int | None:
        return self.book.spread

    def recent_trades(self, n: int = 20) -> list[TradeTick]:
        return list(self.tape)[-n:]

    def vwap(self, n: int = 50) -> float | None:
        trades = self.recent_trades(n)
        if not trades:
            return None
        notional = sum(t.price * t.qty for t in trades)
        volume = sum(t.qty for t in trades)
        return notional / volume if volume else None
