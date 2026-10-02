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

import dataclasses
import itertools
import statistics
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from ..core.types import Fill, Instrument, Side
from ..feed.base import DepthDelta, DepthSnapshot
from ..mm.quoter import Quote
from . import clock

NEW, OPEN, CANCELLING, DONE = "new", "open", "cancelling", "done"
RELEASE_GRACE_S = 1.0
"""How long a cancelled order's yen or coin is still treated as locked."""


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
    retry_at: float = 0.0
    """Monotonic time before which a retried cancel is not sent again."""
    cancel_asked: float | None = None
    """Wall-clock seconds when the maker asked to cancel; a fill executed
    after this is one the guard meant to avoid."""

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
    releasing: deque = field(default_factory=deque)
    """(until, order): cancelled orders whose balance is still held back."""
    sent: dict = field(default_factory=dict)
    """Every order the venue accepted, by its exchange id, kept after it is
    done: a fill is read from the trade history a second or more after it
    happens, often after the order was cancelled or forgotten, and one that
    finds no order is a fill the maker never books."""
    rejected: int = 0
    post_only_refused: int = 0
    insufficient: int = 0
    requests_sent: int = 0
    """Calls that went to the venue (a batch of cancels counts once)."""
    _ids: object = field(default_factory=lambda: itertools.count(1))

    # Read by the maker's summary; a live venue has no simulated queue.
    prints_seen: int = 0
    prints_at_our_price: int = 0
    queue_absorbed_lots: int = 0
    gap_fills: int = 0
    gap_filled_lots: int = 0
    doomed_fills: int = 0
    doomed_lots: int = 0
    fills_seen: int = 0
    cancel_ms: list = field(default_factory=list)
    filled_lots: int = 0

    # ------------------------------------------------------------ balances

    def price_of(self, ticks: int) -> Decimal:
        return self.instrument.tick_size * ticks

    def amount_of(self, lots: int) -> Decimal:
        return self.instrument.lot_size * lots

    def _reserved(self, side: Side) -> Decimal:
        now = clock.monotonic()
        while self.releasing and self.releasing[0][0] <= now:
            self.releasing.popleft()
        live = [o for o in self.orders.values() if o.side is side and o.is_live]
        live += [o for _, o in self.releasing if o.side is side]
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
        order.cancel_asked = clock.wall()
        self.intents.append(("cancel", order_id))
        return True

    def cancelled(self, order: LiveOrder) -> None:
        """The venue confirmed the cancel: note how long getting out took."""
        if order.cancel_asked is not None and order.exchange_id is not None:
            self.cancel_ms.append((clock.wall() - order.cancel_asked) * 1000)
            del self.cancel_ms[:-500]
            # The venue frees what the order locked a little after it confirms
            # the cancel. Reusing it at once was what drew runs of 60001.
            self.releasing.append((clock.monotonic() + RELEASE_GRACE_S, order))
        order.state = DONE

    def cancel_report(self) -> str:
        """How often and how slowly we got out of the way, for the log."""
        share = self.doomed_fills / self.fills_seen * 100 if self.fills_seen else 0.0
        typical = f"{statistics.median(self.cancel_ms):.0f}ms" if self.cancel_ms else "—"
        return (f"取消し中の約定 {self.doomed_fills}/{self.fills_seen}回（{share:.0f}%）"
                f"  取消し 中央値 {typical}  指値拒否 {self.post_only_refused}回"
                f"  残高不足 {self.insufficient}回")

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

    def remember(self, order: LiveOrder) -> None:
        self.sent[order.exchange_id] = order
        if len(self.sent) > 5000:
            del self.sent[next(iter(self.sent))]

    def strip_own(self, event):
        """The public book without our own resting orders in it.

        On paper our orders never reach the public feed. Live they do, and a
        maker that reads its own order as somebody else's chases itself: the
        unwind quote goes "one tick inside the touch", the touch is our own
        order, so every requote steps a tick further until it meets the other
        side; and our size at the touch tilts the microprice toward us.
        Depth updates carry each level's total, so our share is taken out.
        """
        if not isinstance(event, (DepthSnapshot, DepthDelta)):
            return event
        mine = {Side.BUY: {}, Side.SELL: {}}
        for o in self.orders.values():
            if o.exchange_id is not None and o.state in (OPEN, CANCELLING) and o.remaining > 0:
                book = mine[o.side]
                book[o.price] = book.get(o.price, 0) + o.remaining
        if not mine[Side.BUY] and not mine[Side.SELL]:
            return event
        keep_empty = isinstance(event, DepthDelta)  # a zero in a delta deletes the level

        def strip(levels, own):
            out = []
            for price, qty in levels:
                qty = max(0, qty - own.get(price, 0))
                if qty > 0 or keep_empty:
                    out.append((price, qty))
            return tuple(out)

        return dataclasses.replace(
            event,
            bids=strip(event.bids, mine[Side.BUY]),
            asks=strip(event.asks, mine[Side.SELL]),
        )

    def idle(self) -> bool:
        """Nothing of ours rests, waits to be sent, or still holds balance."""
        return (not self.intents and self._reserved(Side.BUY) == 0
                and self._reserved(Side.SELL) == 0)

    def track_sale(self, exchange_id: int, side: Side, lots: int, price: int) -> LiveOrder:
        """A market order sent to get out, kept so its fills are booked.

        It never rests, so it stays out of `orders`: nothing cancels it and
        the public book is not stripped of it.
        """
        order = LiveOrder(next(self._ids), side, price, lots, lots, state=OPEN,
                          exchange_id=exchange_id)
        self.remember(order)
        return order

    def by_exchange_id(self, exchange_id: int) -> LiveOrder | None:
        return self.sent.get(exchange_id)

    def record_fill(self, order: LiveOrder, fill: Fill) -> None:
        """A fill the venue reported: adjust the order, the balances, the maker."""
        order.remaining = max(0, order.remaining - fill.qty)
        order.filled += fill.qty
        self.filled_lots += fill.qty
        self.fills_seen += 1
        if order.cancel_asked is not None and fill.ts_ns / 1e9 >= order.cancel_asked:
            # We had already decided to pull this order and the venue filled
            # it anyway: the cost of a slow or refused cancel.
            self.doomed_fills += 1
            self.doomed_lots += fill.qty
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
