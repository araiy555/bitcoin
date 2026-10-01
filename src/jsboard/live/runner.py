"""Sending the maker's decisions to bitbank, and the guards around it.

Three background jobs run beside the maker's event loop:

- the executor sends each intent the venue recorded, within a request
  budget, and drops a new order that was cancelled before it went out;
- the fill poller reads our executions and hands them to the venue;
- the watchdog cancels everything when the feed goes quiet, when the lead
  market lurches, or when the venue keeps refusing calls.

`DryRunApi` stands in for the exchange: it accepts every order and never
fills, so the whole loop can run on the live feed without a yen at risk.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from ..core.types import Fill, Side
from ..sim.paper import PAPER_OWNER
from .bitbank import DONE as FINISHED
from .bitbank import NOT_FOUND, BitbankError
from .venue import CANCELLING, DONE, NEW, OPEN, LiveVenue

log = logging.getLogger(__name__)

MAX_CANCEL_TRIES = 40
"""Retries of a cancel the venue says it cannot find (about two seconds);
after that the order's status decides, and the final sweep is the backstop."""


class DryRunApi:
    """Accepts orders, never fills them, and sends nothing anywhere."""

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self.open: set[int] = set()
        self.orders_sent = 0
        self.cancels_sent = 0

    async def order(self, pair, side, price, amount) -> int:
        self.orders_sent += 1
        order_id = next(self._ids)
        self.open.add(order_id)
        return order_id

    async def cancel(self, pair, order_id) -> None:
        self.cancels_sent += 1
        if order_id not in self.open:
            raise BitbankError(NOT_FOUND, "dry-run cancel")
        self.open.discard(order_id)

    async def cancel_many(self, pair, order_ids) -> None:
        self.cancels_sent += 1
        missing = [i for i in order_ids if i not in self.open]
        if missing:
            raise BitbankError(NOT_FOUND, "dry-run cancel_orders")
        self.open.difference_update(order_ids)

    async def status(self, pair, order_id) -> str:
        return "UNFILLED" if order_id in self.open else "CANCELED_UNFILLED"

    async def active_orders(self, pair) -> list[int]:
        return sorted(self.open)

    async def trade_history(self, pair, since_ms) -> list[dict]:
        return []


@dataclass
class RequestBudget:
    """At most `per_s` order-changing calls in any one-second window."""

    per_s: float = 4.0
    _sent: deque = field(default_factory=deque)

    async def take(self) -> None:
        while True:
            now = time.monotonic()
            while self._sent and now - self._sent[0] >= 1.0:
                self._sent.popleft()
            if len(self._sent) < self.per_s:
                self._sent.append(now)
                return
            await asyncio.sleep(1.0 - (now - self._sent[0]))


@dataclass
class Health:
    """Why quoting is paused, and the tally that ends the run."""

    consecutive_errors: int = 0
    max_errors: int = 5
    fatal: str = ""

    def ok(self) -> None:
        self.consecutive_errors = 0

    def error(self, exc: Exception) -> None:
        self.consecutive_errors += 1
        log.warning("bitbank call failed (%s)", exc)
        if self.consecutive_errors >= self.max_errors:
            self.fatal = f"bitbank への呼び出しが{self.max_errors}回続けて失敗: {exc}"


async def execute_once(venue: LiveVenue, api, pair: str, health: Health) -> bool:
    """Send the oldest intent. False when there was nothing to send."""
    if not venue.intents:
        return False
    kind, order_id = venue.intents.popleft()
    order = venue.orders.get(order_id)
    if order is None or order.state == DONE:
        return True
    if kind == "new":
        if order.state == CANCELLING:
            order.state = DONE  # cancelled before it ever went out
            return True
        try:
            order.exchange_id = await api.order(
                pair, "buy" if order.side is Side.BUY else "sell",
                venue.price_of(order.price), venue.amount_of(order.remaining),
            )
        except BitbankError as exc:
            order.state = DONE
            venue.rejected += 1
            health.error(exc)
            return True
        health.ok()
        if order.state == NEW:
            order.state = OPEN
        return True
    # A cancel. One for an order whose placement is still queued behind it
    # cannot happen (intents are FIFO), so the order has an id or never sent.
    if order.exchange_id is None:
        order.state = DONE
        return True
    if order.cancel_tries == 0 and await _cancel_batch(venue, api, pair, order, health):
        return True
    try:
        await api.cancel(pair, order.exchange_id)
        order.state = DONE
        health.ok()
    except BitbankError as exc:
        order.cancel_tries += 1
        if exc.code == NOT_FOUND and order.cancel_tries < MAX_CANCEL_TRIES:
            # Accepted but not yet on the book: try again after the queue.
            venue.intents.append(("cancel", order_id))
            await asyncio.sleep(0.05)
            return True
        try:
            if await api.status(pair, order.exchange_id) in FINISHED:
                order.state = DONE
                return True
        except BitbankError as missing:
            if missing.code == NOT_FOUND:
                # Still unknown after two seconds: the venue never took it.
                order.state = DONE
                log.warning("order %s never reached the book", order.exchange_id)
                return True
        venue.intents.append(("cancel", order_id))
        health.error(exc)
    return True


BATCH_MAX = 30


async def _cancel_batch(venue: LiveVenue, api, pair: str, first, health: Health) -> bool:
    """Cancel `first` together with every other first-try cancel queued.

    True when the batch went through. On any error the orders go back to
    the one-at-a-time path, which knows how to wait for an order that has
    not reached the book yet.
    """
    batch = [first]
    rest = deque()
    while venue.intents:
        kind, order_id = venue.intents.popleft()
        order = venue.orders.get(order_id)
        if (kind == "cancel" and len(batch) < BATCH_MAX and order is not None
                and order.state != DONE and order.exchange_id is not None
                and order.cancel_tries == 0 and order not in batch):
            batch.append(order)
        else:
            rest.append((kind, order_id))
    venue.intents.extendleft(reversed(rest))
    if len(batch) == 1:
        return False
    try:
        await api.cancel_many(pair, [o.exchange_id for o in batch])
    except BitbankError:
        for order in batch:
            order.cancel_tries = 1  # retry singly
        venue.intents.extendleft(("cancel", o.order_id) for o in reversed(batch[1:]))
        return False
    for order in batch:
        order.state = DONE
    health.ok()
    return True


def drop_noops(venue: LiveVenue) -> None:
    """Clear intents at the head of the queue that would send nothing.

    An order cancelled before it went out leaves a new-then-cancel pair that
    resolves without a request. Each used to wait for a slot in the request
    budget anyway, and with the lead gate pulling and restoring a side every
    few seconds those pairs kept the queue 20-30 deep.
    """
    while venue.intents:
        kind, order_id = venue.intents[0]
        order = venue.orders.get(order_id)
        if order is None or order.state == DONE:
            pass
        elif kind == "new" and order.state == CANCELLING:
            order.state = DONE
        elif kind == "cancel" and order.exchange_id is None:
            order.state = DONE  # its placement was dropped or refused
        else:
            return
        venue.intents.popleft()


async def run_executor(
    venue: LiveVenue, api, pair: str, health: Health, budget: RequestBudget
) -> None:
    while True:
        drop_noops(venue)
        if venue.intents:
            await budget.take()
            drop_noops(venue)  # the wait may have made the head moot
            if venue.intents:
                venue.requests_sent += 1
                await execute_once(venue, api, pair, health)
        else:
            await asyncio.sleep(0.01)


def fill_from_trade(venue: LiveVenue, trade: dict) -> tuple | None:
    """A venue execution as (our order, Fill), or None if it is not ours."""
    order = venue.by_exchange_id(int(trade["order_id"]))
    if order is None:
        return None
    inst = venue.instrument
    price = int((Decimal(str(trade["price"])) / inst.tick_size).to_integral_value())
    qty = inst.to_lots(str(trade["amount"]))
    ts_ns = int(trade.get("executed_at", 0)) * 1_000_000
    if trade.get("maker_taker") == "taker":
        fill = Fill(price, qty, 0, order.order_id, "market", PAPER_OWNER, order.side, ts_ns)
    else:
        fill = Fill(price, qty, order.order_id, 0, PAPER_OWNER, "market",
                    order.side.opposite, ts_ns)
    return order, fill


async def poll_fills_once(venue: LiveVenue, api, pair: str, state: dict) -> int:
    trades = await api.trade_history(pair, state.setdefault("since_ms", 0))
    found = 0
    for trade in trades:
        trade_id = int(trade["trade_id"])
        if trade_id in state.setdefault("seen", set()):
            continue
        state["seen"].add(trade_id)
        state["since_ms"] = max(state["since_ms"], int(trade.get("executed_at", 0)))
        mapped = fill_from_trade(venue, trade)
        if mapped is not None:
            venue.record_fill(*mapped)
            found += 1
    return found


async def run_fill_poller(
    venue: LiveVenue, api, pair: str, health: Health, interval_s: float = 1.0
) -> None:
    state = {"since_ms": int(time.time() * 1000)}
    while True:
        try:
            await poll_fills_once(venue, api, pair, state)
            health.ok()
        except BitbankError as exc:
            health.error(exc)
        await asyncio.sleep(interval_s)


@dataclass
class Breaker:
    """Stops quoting when the lead market moves too far too fast."""

    move_pct: float = 1.5
    window_s: float = 60.0
    cooldown_s: float = 300.0
    _mids: deque = field(default_factory=deque)
    until: float = 0.0

    def observe(self, mid: float | None, now: float) -> bool:
        """True when this observation trips the breaker."""
        if mid is None or mid <= 0:
            return False
        self._mids.append((now, mid))
        while self._mids and now - self._mids[0][0] > self.window_s:
            self._mids.popleft()
        lo = min(m for _, m in self._mids)
        hi = max(m for _, m in self._mids)
        if (hi - lo) / lo * 100 >= self.move_pct and now >= self.until:
            self.until = now + self.cooldown_s
            return True
        return False

    def active(self, now: float) -> bool:
        return now < self.until


async def run_watchdog(
    venue: LiveVenue,
    health: Health,
    last_event: Callable[[], float],
    lead_mid: Callable[[], float | None],
    breaker: Breaker,
    notify: Callable[[str], None],
    stale_s: float = 5.0,
    tick_s: float = 0.5,
) -> None:
    """Pull every order on a quiet feed, a lurching lead, or a sick venue."""
    while True:
        now = time.monotonic()
        reason = ""
        if health.fatal:
            reason = health.fatal
        elif now - last_event() > stale_s:
            reason = f"板のデータが{stale_s:.0f}秒止まっています"
        else:
            if breaker.observe(lead_mid(), now):
                notify(f"先行市場が{breaker.window_s:.0f}秒で{breaker.move_pct:g}%以上動いたので、"
                       f"{breaker.cooldown_s / 60:.0f}分止めます")
            if breaker.active(now):
                reason = "急変ブレーカー作動中"
        if reason and venue.blocked != reason:
            venue.blocked = reason
            venue.cancel_all()
            log.warning("quoting paused: %s", reason)
        elif not reason and venue.blocked:
            venue.blocked = ""
        await asyncio.sleep(tick_s)


async def cancel_everything(api, pair: str, venue: LiveVenue, timeout_s: float = 10.0) -> list[int]:
    """Last act of every run: nothing of ours may be left resting.

    Cancels every open order on the pair, ours or not — the account is the
    program's while it runs — and returns the ids it could not clear.
    """
    from .bitbank import cancel_until_gone

    venue.blocked = venue.blocked or "終了"
    stuck = []
    try:
        ids = await api.active_orders(pair)
    except BitbankError:
        ids = [o.exchange_id for o in venue.orders.values() if o.exchange_id is not None]
    for order_id in ids:
        try:
            await cancel_until_gone(api, pair, order_id, timeout_s=timeout_s)
        except BitbankError:
            stuck.append(order_id)
    for order in venue.orders.values():
        if order.state != DONE:
            order.state = DONE
    return stuck
