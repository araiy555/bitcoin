"""Paper-trading venue: fills our orders against the real tape.

The interesting part is queue position. A resting order does not fill because
the market traded at its price — it fills because the market traded *through
everyone who was already standing there*. So each simulated order remembers
how much size was ahead of it when it arrived, and public prints eat that
backlog before they touch us.

What is modelled:
  * FIFO queue position, seeded from visible depth at placement
  * price improvement — quoting inside the touch puts us at the front
  * partial fills, and level depletion carrying through to better prices
  * order-entry latency, so quotes are not live the instant we decide them

What is not, and where results will flatter us:
  * market impact — our size never scares anyone off
  * hidden/iceberg liquidity sitting invisibly ahead of us
  * queue jumping by faster participants after we arrive
  * gap-throughs — if the book jumps past a resting quote with no print in
    between, nothing here fills it, though in reality someone would have
    taken it on the way past. This is the largest single optimism in the
    model: it under-counts exactly the adverse selection a maker most fears,
    so treat the P&L as an upper bound rather than an estimate.

`cancel_ahead_ratio` is the one honest knob for the biggest unknown: when
depth at our level shrinks without a print, we cannot see whether those
cancels were in front of us or behind. 0.0 is pessimistic (all behind), 1.0
optimistic (all ahead); the default splits the difference.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

from ..core.types import Fill, Instrument, Side
from ..feed.base import TradeTick
from ..mm.quoter import Quote

PAPER_OWNER = "mm"


@dataclass(slots=True)
class PaperOrder:
    order_id: int
    side: Side
    price: int
    qty: int
    remaining: int
    queue_ahead: int
    placed_ns: int
    active_ns: int
    level_depth: int
    filled: int = 0

    @property
    def is_live(self) -> bool:
        return self.remaining > 0

    def is_active(self, now_ns: int) -> bool:
        return self.is_live and now_ns >= self.active_ns


@dataclass(slots=True)
class PaperConfig:
    latency_ms: float = 5.0
    """Delay between deciding a quote and it resting on the book."""

    cancel_ahead_ratio: float = 0.5
    """Share of unexplained depth reduction assumed to be ahead of us."""

    allow_price_improvement: bool = True
    """Quoting inside the touch starts us at the front of a fresh level."""


@dataclass(slots=True)
class PaperVenue:
    """A simulated exchange connection for one instrument."""

    instrument: Instrument
    config: PaperConfig = field(default_factory=PaperConfig)
    clock: object = time.time_ns
    orders: dict[int, PaperOrder] = field(default_factory=dict)
    _ids: object = field(default_factory=lambda: itertools.count(1))
    _last_depth: dict[tuple[int, int], int] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    rejected: int = 0

    # Why we did or did not fill. A session with no fills has two very
    # different explanations — the tape never came to our price, or it came
    # and the queue in front of us swallowed it — and only counting both
    # tells them apart.
    prints_seen: int = 0
    prints_at_our_price: int = 0
    queue_absorbed_lots: int = 0
    filled_lots: int = 0

    def _now(self) -> int:
        return self.clock()

    # ---------------------------------------------------------------- entry

    def place(self, quote: Quote, visible_depth: int, best_opposite: int | None) -> PaperOrder | None:
        """Rest a quote. `visible_depth` is public size already at that price."""
        if quote.qty <= 0:
            return None

        # A quote that crosses is a taker order; this venue only makes.
        if best_opposite is not None:
            crosses = (
                quote.price >= best_opposite if quote.side is Side.BUY else quote.price <= best_opposite
            )
            if crosses:
                self.rejected += 1
                return None

        now = self._now()
        order = PaperOrder(
            order_id=next(self._ids),
            side=quote.side,
            price=quote.price,
            qty=quote.qty,
            remaining=quote.qty,
            queue_ahead=max(0, visible_depth),
            placed_ns=now,
            active_ns=now + int(self.config.latency_ms * 1e6),
            level_depth=max(0, visible_depth),
        )
        self.orders[order.order_id] = order
        return order

    def cancel(self, order_id: int) -> bool:
        order = self.orders.pop(order_id, None)
        return order is not None

    def cancel_all(self) -> int:
        n = len(self.orders)
        self.orders.clear()
        return n

    def open_orders(self) -> list[PaperOrder]:
        return [o for o in self.orders.values() if o.is_live]

    # -------------------------------------------------------------- updates

    def on_depth(self, side: Side, price: int, depth: int) -> None:
        """Track visible depth so we can infer cancels ahead of our orders."""
        key = (int(side), price)
        previous = self._last_depth.get(key)
        self._last_depth[key] = depth

        if previous is None or depth >= previous:
            return

        # Depth fell. Prints are handled in `on_trade`; whatever is left here
        # is cancellations, split by the configured ratio.
        reduction = previous - depth
        ahead = int(reduction * self.config.cancel_ahead_ratio)
        if ahead <= 0:
            return
        for order in self.orders.values():
            if order.side is side and order.price == price and order.queue_ahead > 0:
                order.queue_ahead = max(0, order.queue_ahead - ahead)

    def on_trade(self, trade: TradeTick) -> list[Fill]:
        """Apply a public print; returns the fills it generated for us."""
        now = self._now()
        # An aggressive buy consumes resting *sell* orders, and vice versa.
        our_side = trade.aggressor.opposite
        remaining = trade.qty
        produced: list[Fill] = []

        candidates = [
            o
            for o in self.orders.values()
            if o.side is our_side and o.is_active(now) and self._price_hit(o, trade)
        ]
        # Best-priced first; the tape sweeps in price order.
        candidates.sort(key=lambda o: o.price, reverse=our_side is Side.BUY)

        self.prints_seen += 1
        if candidates:
            self.prints_at_our_price += 1

        for order in candidates:
            if remaining <= 0:
                break

            # The queue in front of us absorbs the print first.
            if order.queue_ahead > 0:
                eaten = min(order.queue_ahead, remaining)
                order.queue_ahead -= eaten
                remaining -= eaten
                self.queue_absorbed_lots += eaten
                if remaining <= 0:
                    break

            fill_qty = min(order.remaining, remaining)
            if fill_qty <= 0:
                continue

            order.remaining -= fill_qty
            order.filled += fill_qty
            remaining -= fill_qty
            self.filled_lots += fill_qty

            produced.append(
                Fill(
                    price=order.price,
                    qty=fill_qty,
                    maker_id=order.order_id,
                    taker_id=trade.trade_id,
                    maker_owner=PAPER_OWNER,
                    taker_owner="market",
                    aggressor=trade.aggressor,
                    ts_ns=trade.ts_ns or now,
                )
            )

            if order.remaining <= 0:
                self.orders.pop(order.order_id, None)

        self.fills.extend(produced)
        return produced

    def _price_hit(self, order: PaperOrder, trade: TradeTick) -> bool:
        """Would this print have reached our price?"""
        if order.side is Side.BUY:
            # We bid; an aggressive sell at or below our price hits us.
            return trade.price <= order.price
        return trade.price >= order.price

    # --------------------------------------------------------------- stats

    @property
    def resting_lots(self) -> int:
        return sum(o.remaining for o in self.orders.values())

    def queue_report(self) -> list[tuple[int, int, int, int]]:
        """(side, price, remaining, queue_ahead) for the UI."""
        return sorted(
            ((int(o.side), o.price, o.remaining, o.queue_ahead) for o in self.orders.values()),
            key=lambda r: (-r[0], -r[1]),
        )
