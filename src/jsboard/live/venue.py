"""The maker's venue interface, backed by real orders.

`MarketMaker` talks to its venue synchronously: place, cancel, list what is
resting. A real exchange answers in tens of milliseconds, so this venue
records each decision as an intent and returns at once; an executor sends
the intents in the background (see `runner.py`), and fills found by polling
are handed back to the maker through `on_book`, the same call the paper
venue fills through. The strategy code does not change between paper and
live, which is the point: what was measured is what runs.

Spot only. A buy needs the yen and a sell needs the coin, so every quote is
checked against the balance left after what is already resting.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from ..core.types import Fill, Instrument, Side
from ..mm.quoter import Quote

NEW, OPEN, CANCELLING, DONE = "new", "open", "cancelling", "done"


@dataclass(slots=True)
class LiveOrder:
    order_id: int
    side: Side
    price: int
    qty: int
    remaining: int
    state: str = NEW
    exchange_id: int | None = None
    filled: int = 0
    cancel_tries: int = 0

    @property
    def is_live(self) -> bool:
        return self.remaining > 0 and self.state != DONE


@dataclass(slots=True)
class LiveConfig:
    cancel_latency_ms: float = 0.0
    """Unused: the real latency is the venue's. Present because the maker's
    summary reports it."""


@dataclass
class LiveVenue:
    instrument: Instrument
    quote_balance: Decimal = Decimal(0)
    """Yen free at the start, before any order rests."""
    base_balance: Decimal = Decimal(0)
    """Coin free at the start."""
    config: LiveConfig = field(default_factory=LiveConfig)
    orders: dict[int, LiveOrder] = field(default_factory=dict)
    intents: deque = field(default_factory=deque)
    pending_fills: list[Fill] = field(default_factory=list)
    blocked: str = ""
    """Non-empty while a guard forbids new orders; cancels still go out."""
    rejected: int = 0
    _ids: object = field(default_factory=lambda: itertools.count(1))

    # Read by the maker's summary; a live venue has no simulated queue.
    prints_seen: int = 0
    prints_at_our_price: int = 0
    queue_absorbed_lots: int = 0
    gap_fills: int = 0
    gap_filled_lots: int = 0
    doomed_fills: int = 0
    doomed_lots: int = 0
    filled_lots: int = 0

    # ------------------------------------------------------------ balances

    def price_of(self, ticks: int) -> Decimal:
        return self.instrument.tick_size * ticks

    def amount_of(self, lots: int) -> Decimal:
        return self.instrument.lot_size * lots

    def _reserved(self, side: Side) -> Decimal:
        live = [o for o in self.orders.values() if o.side is side and o.is_live]
        if side is Side.BUY:
            return sum((self.price_of(o.price) * self.amount_of(o.remaining) for o in live),
                       Decimal(0))
        return sum((self.amount_of(o.remaining) for o in live), Decimal(0))

    def can_afford(self, side: Side, price: int, qty: int) -> bool:
        if side is Side.BUY:
            need = self.price_of(price) * self.amount_of(qty)
            return need <= self.quote_balance - self._reserved(Side.BUY)
        return self.amount_of(qty) <= self.base_balance - self._reserved(Side.SELL)

    # -------------------------------------------------- the maker's calls

    def place(self, quote: Quote, visible_depth: int, best_opposite: int | None):
        if quote.qty <= 0 or self.blocked:
            return None
        if best_opposite is not None and (
            quote.price >= best_opposite if quote.side is Side.BUY else quote.price <= best_opposite
        ):
            self.rejected += 1  # would take: the venue's post-only would refuse it too
            return None
        if not self.can_afford(quote.side, quote.price, quote.qty):
            self.rejected += 1
            return None
        order = LiveOrder(next(self._ids), quote.side, quote.price, quote.qty, quote.qty)
        self.orders[order.order_id] = order
        self.intents.append(("new", order.order_id))
        return order

    def cancel(self, order_id: int) -> bool:
        order = self.orders.get(order_id)
        if order is None or order.state in (CANCELLING, DONE):
            return False
        order.state = CANCELLING
        self.intents.append(("cancel", order_id))
        return True

    def cancel_all(self) -> int:
        return sum(self.cancel(o.order_id) for o in list(self.orders.values()))

    def open_orders(self) -> list[LiveOrder]:
        """Orders the strategy may still act on, including ones in flight."""
        return [o for o in self.orders.values() if o.is_live and o.state in (NEW, OPEN)]

    def on_depth(self, side: Side, price: int, depth: int) -> None:
        return None

    def on_trade(self, trade) -> list[Fill]:
        return []

    def on_book(self, best_bid, best_ask, ts_ns: int = 0) -> list[Fill]:
        fills, self.pending_fills = self.pending_fills, []
        return fills

    # ------------------------------------------------ executor's callbacks

    def by_exchange_id(self, exchange_id: int) -> LiveOrder | None:
        return next((o for o in self.orders.values() if o.exchange_id == exchange_id), None)

    def record_fill(self, order: LiveOrder, fill: Fill) -> None:
        """A fill the venue reported: adjust the order, the balances, the maker."""
        order.remaining = max(0, order.remaining - fill.qty)
        order.filled += fill.qty
        self.filled_lots += fill.qty
        price, amount = self.price_of(fill.price), self.amount_of(fill.qty)
        if order.side is Side.BUY:
            self.quote_balance -= price * amount
            self.base_balance += amount
        else:
            self.quote_balance += price * amount
            self.base_balance -= amount
        if order.remaining == 0:
            order.state = DONE
        self.pending_fills.append(fill)

    def forget_done(self) -> None:
        for order_id, order in list(self.orders.items()):
            if order.state == DONE:
                del self.orders[order_id]
