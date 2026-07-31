"""Price–time priority limit order book with a matching engine.

Structure: one `SortedDict[price_ticks -> PriceLevel]` per side. Best bid is
the largest bid key, best ask the smallest ask key, both O(1) via `peekitem`.

Within a level, orders sit in a `deque` in arrival order. Cancels mark the
order dead and decrement the level total rather than splicing it out of the
middle; dead orders are skipped and discarded when the queue front reaches
them. That keeps cancel O(1) at the cost of holding a dead object until it
surfaces, which is the usual trade for this shape of book.
"""

from __future__ import annotations

import enum
import itertools
import time
from collections import deque
from dataclasses import dataclass, field

from sortedcontainers import SortedDict

from .types import (
    MARKET,
    BookSnapshot,
    Fill,
    Level,
    Order,
    OrderStatus,
    Side,
    TimeInForce,
)


class STPPolicy(enum.Enum):
    """What to do when an order would match against its own owner."""

    NONE = "NONE"
    """Allow the self-trade. Useful in tests, wrong in production."""

    CANCEL_MAKER = "CANCEL_MAKER"
    """Cancel the resting order and keep aggressing. The common default."""

    CANCEL_TAKER = "CANCEL_TAKER"
    """Stop the incoming order dead."""


class Reject(Exception):
    """An order was refused before it ever reached the book."""


@dataclass(slots=True)
class PriceLevel:
    price: int
    queue: deque[Order] = field(default_factory=deque)
    total_qty: int = 0
    live_orders: int = 0

    def push(self, order: Order) -> None:
        self.queue.append(order)
        self.total_qty += order.remaining
        self.live_orders += 1

    def _drop_dead_front(self) -> None:
        while self.queue and not self.queue[0].is_live:
            self.queue.popleft()

    def peek(self) -> Order | None:
        self._drop_dead_front()
        return self.queue[0] if self.queue else None

    def consume(self, order: Order, qty: int) -> None:
        """Record `qty` lots taken out of `order`, which must be at the front."""
        order.remaining -= qty
        self.total_qty -= qty
        if order.remaining == 0:
            order.status = OrderStatus.FILLED
            self.live_orders -= 1
            self.queue.popleft()
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

    def remove(self, order: Order) -> None:
        """Cancel `order` in place; it stays in the deque until it surfaces."""
        self.total_qty -= order.remaining
        self.live_orders -= 1
        order.remaining = 0
        order.status = OrderStatus.CANCELLED

    @property
    def empty(self) -> bool:
        return self.live_orders <= 0


@dataclass(slots=True)
class MatchResult:
    order: Order
    fills: list[Fill] = field(default_factory=list)
    resting: bool = False
    reject_reason: str | None = None

    @property
    def filled_qty(self) -> int:
        return sum(f.qty for f in self.fills)

    @property
    def avg_price(self) -> float | None:
        if not self.fills:
            return None
        notional = sum(f.price * f.qty for f in self.fills)
        return notional / self.filled_qty


class OrderBook:
    """A single-symbol matching engine.

    Not thread-safe; drive it from one task/thread. All prices are ticks and
    all quantities are lots (see `types.Instrument`).
    """

    __slots__ = ("bids", "asks", "_orders", "_seq", "stp", "_clock")

    def __init__(self, stp: STPPolicy = STPPolicy.CANCEL_MAKER, clock=time.time_ns) -> None:
        # price -> PriceLevel. Bids read from the top (peekitem(-1)),
        # asks from the bottom (peekitem(0)).
        self.bids: SortedDict[int, PriceLevel] = SortedDict()
        self.asks: SortedDict[int, PriceLevel] = SortedDict()
        self._orders: dict[int, Order] = {}
        self._seq = 0
        self.stp = stp
        self._clock = clock

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ---------------------------------------------------------------- reads

    def _side_levels(self, side: Side) -> SortedDict:
        return self.bids if side is Side.BUY else self.asks

    def best_bid(self) -> int | None:
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> int | None:
        return self.asks.peekitem(0)[0] if self.asks else None

    def best(self, side: Side) -> int | None:
        return self.best_bid() if side is Side.BUY else self.best_ask()

    @property
    def spread(self) -> int | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else ask - bid

    @property
    def mid(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else (bid + ask) / 2.0

    def get(self, order_id: int) -> Order | None:
        return self._orders.get(order_id)

    def open_orders(self, owner: str | None = None) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if o.is_live and (owner is None or o.owner == owner)
        ]

    def depth_at(self, side: Side, price: int) -> int:
        level = self._side_levels(side).get(price)
        return level.total_qty if level else 0

    def snapshot(self, depth: int = 10) -> BookSnapshot:
        bids = tuple(
            Level(price, lvl.total_qty, lvl.live_orders)
            for price, lvl in itertools.islice(reversed(self.bids.items()), depth)
        )
        asks = tuple(
            Level(price, lvl.total_qty, lvl.live_orders)
            for price, lvl in itertools.islice(self.asks.items(), depth)
        )
        return BookSnapshot(bids=bids, asks=asks, ts_ns=self._clock(), seq=self._seq)

    # --------------------------------------------------------------- writes

    def submit(self, order: Order) -> MatchResult:
        """Run `order` through the engine: match, then rest or discard."""
        order.seq = self._next_seq()
        order.ts_ns = order.ts_ns or self._clock()
        result = MatchResult(order=order)

        if order.tif is TimeInForce.POST_ONLY and self._would_cross(order):
            order.status = OrderStatus.REJECTED
            result.reject_reason = "post-only order would take liquidity"
            return result

        if order.tif is TimeInForce.FOK and self._crossable_qty(order) < order.qty:
            order.status = OrderStatus.REJECTED
            result.reject_reason = "fill-or-kill could not be filled in full"
            return result

        self._match(order, result)

        if order.status is OrderStatus.CANCELLED:
            # Self-trade prevention killed the aggressor part-way through.
            return result

        if order.remaining == 0:
            order.status = OrderStatus.FILLED
            return result

        if order.tif in (TimeInForce.IOC, TimeInForce.FOK) or order.is_market:
            # Whatever did not trade immediately is dropped, filled or not.
            order.remaining = 0
            order.status = OrderStatus.CANCELLED
            return result

        self._rest(order)
        result.resting = True
        return result

    def cancel(self, order_id: int) -> Order | None:
        """Cancel a resting order. Returns it, or None if it was already gone."""
        order = self._orders.get(order_id)
        if order is None or not order.is_live:
            return None
        levels = self._side_levels(order.side)
        level = levels.get(order.price)
        if level is None:
            return None
        level.remove(order)
        if level.empty:
            del levels[order.price]
        del self._orders[order_id]
        return order

    def cancel_all(self, owner: str | None = None) -> list[Order]:
        victims = [o.order_id for o in self.open_orders(owner)]
        return [o for oid in victims if (o := self.cancel(oid)) is not None]

    def amend(self, order_id: int, *, price: int | None = None, qty: int | None = None) -> MatchResult | None:
        """Modify a resting order.

        A pure size *reduction* keeps queue position. Anything else — price
        change or size increase — is a cancel/replace and goes to the back of
        the new level, which is what every real venue does.
        """
        order = self._orders.get(order_id)
        if order is None or not order.is_live:
            return None

        new_price = order.price if price is None else price
        new_remaining = order.remaining if qty is None else qty

        if new_remaining <= 0:
            self.cancel(order_id)
            return None

        if new_price == order.price and new_remaining < order.remaining:
            level = self._side_levels(order.side)[order.price]
            level.total_qty -= order.remaining - new_remaining
            order.remaining = new_remaining
            order.qty = order.filled + new_remaining
            return MatchResult(order=order, resting=True)

        self.cancel(order_id)
        replacement = Order(
            side=order.side,
            qty=new_remaining,
            price=new_price,
            tif=order.tif,
            owner=order.owner,
        )
        return self.submit(replacement)

    # -------------------------------------------------------------- helpers

    def _crosses(self, side: Side, limit: int | None, resting_price: int) -> bool:
        if limit is MARKET:
            return True
        return limit >= resting_price if side is Side.BUY else limit <= resting_price

    def _would_cross(self, order: Order) -> bool:
        opposite_best = self.best(order.side.opposite)
        if opposite_best is None:
            return False
        return self._crosses(order.side, order.price, opposite_best)

    def _crossable_qty(self, order: Order) -> int:
        """How much of `order` the book could fill right now."""
        levels = self._side_levels(order.side.opposite)
        items = reversed(levels.items()) if order.side is Side.SELL else levels.items()
        total = 0
        for price, level in items:
            if not self._crosses(order.side, order.price, price):
                break
            total += level.total_qty
            if total >= order.qty:
                return total
        return total

    def _match(self, taker: Order, result: MatchResult) -> None:
        levels = self._side_levels(taker.side.opposite)
        # Asks ascend from best; bids descend from best.
        take_best = (lambda: levels.peekitem(0)) if taker.side is Side.BUY else (lambda: levels.peekitem(-1))

        while taker.remaining > 0 and levels:
            price, level = take_best()
            if not self._crosses(taker.side, taker.price, price):
                break

            while taker.remaining > 0:
                maker = level.peek()
                if maker is None:
                    break

                if maker.owner == taker.owner and self.stp is not STPPolicy.NONE:
                    if self.stp is STPPolicy.CANCEL_TAKER:
                        taker.status = OrderStatus.CANCELLED
                        taker.remaining = 0
                        result.reject_reason = "self-trade prevention: taker cancelled"
                        return
                    level.remove(maker)
                    self._orders.pop(maker.order_id, None)
                    continue

                qty = min(taker.remaining, maker.remaining)
                maker_id, maker_owner = maker.order_id, maker.owner
                level.consume(maker, qty)
                if maker.remaining == 0:
                    self._orders.pop(maker_id, None)

                taker.remaining -= qty
                taker.status = (
                    OrderStatus.FILLED if taker.remaining == 0 else OrderStatus.PARTIALLY_FILLED
                )
                result.fills.append(
                    Fill(
                        price=price,
                        qty=qty,
                        maker_id=maker_id,
                        taker_id=taker.order_id,
                        maker_owner=maker_owner,
                        taker_owner=taker.owner,
                        aggressor=taker.side,
                        ts_ns=self._clock(),
                        seq=self._next_seq(),
                    )
                )

            if level.empty:
                del levels[price]

    def _rest(self, order: Order) -> None:
        levels = self._side_levels(order.side)
        level = levels.get(order.price)
        if level is None:
            level = PriceLevel(price=order.price)
            levels[order.price] = level
        level.push(order)
        self._orders[order.order_id] = order

    # ------------------------------------------------------------ bulk load

    def replace_l2(self, bids: list[tuple[int, int]], asks: list[tuple[int, int]], owner: str = "market") -> None:
        """Rebuild the book from an aggregated L2 image.

        Used by the market-data path, where we only ever see level totals and
        not individual orders. Every level becomes one synthetic order owned
        by `owner`, so our own resting orders (a different owner) are left
        untouched.
        """
        for side, levels, rows in (
            (Side.BUY, self.bids, bids),
            (Side.SELL, self.asks, asks),
        ):
            for price in [p for p, lvl in levels.items() if any(o.owner == owner for o in lvl.queue)]:
                lvl = levels[price]
                for o in list(lvl.queue):
                    if o.owner == owner and o.is_live:
                        lvl.remove(o)
                        self._orders.pop(o.order_id, None)
                if lvl.empty:
                    del levels[price]

            for price, qty in rows:
                if qty <= 0:
                    continue
                self._rest(Order(side=side, qty=qty, price=price, owner=owner))

    def apply_l2_delta(self, side: Side, price: int, qty: int, owner: str = "market") -> None:
        """Set the market-owned quantity at one level (`qty == 0` clears it)."""
        levels = self._side_levels(side)
        level = levels.get(price)
        if level is not None:
            for o in list(level.queue):
                if o.owner == owner and o.is_live:
                    level.remove(o)
                    self._orders.pop(o.order_id, None)
            if level.empty:
                del levels[price]
        if qty > 0:
            self._rest(Order(side=side, qty=qty, price=price, owner=owner))

    def __len__(self) -> int:
        return len(self._orders)

    def __repr__(self) -> str:
        return (
            f"<OrderBook bid={self.best_bid()} ask={self.best_ask()} "
            f"levels={len(self.bids)}/{len(self.asks)} orders={len(self._orders)}>"
        )
