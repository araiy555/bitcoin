"""A bitbank that behaves as bitbank did, for replaying the live trader.

The replay that said the ADA maker made money ran a different program from
the one that traded: a paper venue that filled whenever the book moved
through a price, rested orders the instant they were placed and honoured
every cancel. Live, bitbank filled only on executions, took 0.2 seconds
(sometimes tens of seconds) to rest an order, refused some cancels, and
half the live fills landed on orders already being pulled.

So this runs the live code itself — the `LiveVenue`, the executor and the
fill poller from `jsboard.live` — against `SimBitbank`, which answers the
same calls the way the venue does:

- an order rests after `land_ms`, or now and then after many seconds;
  until it rests a cancel or status call answers 50009;
- a post-only order that would take is refused (REJECTED);
- a resting order fills only on recorded executions that reach its price,
  behind the size that was queued at that price when it landed;
- a cancel lands `cancel_ms` after it is sent, and the order can fill in
  between; now and then a cancel is refused with 50010;
- an order locks its yen or coin, and one that does not fit is refused
  with 60001;
- the executor holds the one connection for a round trip per call, within
  the per-second budget, and fills are seen only when the poller reads
  the trade history.

Every parameter is something the live log measured, so the replay can be
checked against the live result before it is trusted with a new idea.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal

from ..core.types import Instrument, Side
from ..feed.base import DepthDelta, DepthSnapshot, TradeTick
from ..live.bitbank import NOT_FOUND, BitbankError, OrderRefused
from ..mm.quoter import Quote
from .paper import PaperConfig, PaperVenue

CANNOT_CANCEL = 50010
INSUFFICIENT = 60001
MS = 1_000_000


@dataclass
class VenueBehaviour:
    """How bitbank answered, as measured on the live run of 2026-10-02."""

    ack_ms: float = 80.0
    """Round trip of an order or cancel call."""
    land_ms: float = 200.0
    """From placing an order to it resting on the book (probe: 199ms)."""
    slow_land_share: float = 0.005
    """Orders that took many seconds to rest ('never reached the book')."""
    slow_land_mean_s: float = 20.0
    cancel_ms: float = 250.0
    """From a cancel being sent to the order leaving the book."""
    refuse_share: float = 0.05
    """Cancels answered 50010 while the venue was busy with the order."""
    query_ms: float = 40.0
    """Round trip of a read (trade history, status, open orders)."""
    cancel_ahead: float = 0.5
    """Share of unexplained depth drops at our price taken from in front."""
    gap_fill_share: float = 0.0
    """Book moves through our price that fill us without an execution; 0 is
    bitbank's rule (only executions fill a resting order)."""


@dataclass
class SimBitbank:
    """The private API of bitbank, answered from a replayed market."""

    instrument: Instrument
    behaviour: VenueBehaviour = field(default_factory=VenueBehaviour)
    jpy: Decimal = Decimal(0)
    coin: Decimal = Decimal(0)
    seed: int = 7
    now_ns: int = 0
    best_bid: int | None = None
    best_ask: int | None = None
    counts: dict = field(default_factory=lambda: {
        "orders": 0, "refused": 0, "insufficient": 0, "not_found": 0,
        "cannot_cancel": 0, "slow_land": 0, "fills": 0, "own_prints": 0,
    })

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        b = self.behaviour
        self.book = PaperVenue(
            self.instrument,
            config=PaperConfig(
                latency_ms=0.0,
                cancel_ahead_ratio=b.cancel_ahead,
                cancel_latency_ms=b.cancel_ms,
                gap_through_fills=b.gap_fill_share > 0,
                gap_fill_share=b.gap_fill_share,
            ),
            clock=lambda: self.now_ns,
        )
        self._ids = iter(range(10_000_001, 10**12))
        self.by_id: dict[int, object] = {}
        self.paper_to_id: dict[int, int] = {}
        self.history: deque = deque()
        self._trade_ids = iter(range(1, 10**12))
        self._cancelled: set[int] = set()
        self._live: dict[int, object] = {}
        self.all_trades: list[dict] = []
        """Every execution, kept whole for comparing with a live run."""
        self.own_prints: list[tuple] = []
        """(maker side, price ticks, lots, exchange ms) of executions that
        hit the live trader's own orders on the day the recording was made;
        see `mark_own_prints`."""

    # ---------------------------------------------------------- the market

    def on_market(self, event) -> None:
        """Feed one recorded bitbank event: depth for the queue, prints for fills."""
        if isinstance(event, (DepthSnapshot, DepthDelta)):
            for side, levels in ((Side.BUY, event.bids), (Side.SELL, event.asks)):
                for price, qty in levels:
                    self.book.on_depth(side, price, qty)
        elif isinstance(event, TradeTick):
            self._front_if_ours(event)
            self._book(self.book.on_trade(event))

    def mark_own_prints(self, trades: list[dict]) -> None:
        """Tell the model which recorded prints filled our live orders.

        A recording made while the live trader ran has our own orders in
        its depth. A replayed order at the same price is queued behind
        that ghost of itself, and the print that filled us live is eaten
        by the ghost: the replay skipped exactly the fills that cost the
        most. When such a print comes, a replayed order resting at that
        price is put at the front, where the live order stood.
        """
        inst = self.instrument
        self.own_prints = sorted(
            (Side.BUY if t["side"] == "buy" else Side.SELL,
             int((Decimal(str(t["price"])) / inst.tick_size).to_integral_value()),
             inst.to_lots(str(t["amount"])), int(t["executed_at"]))
            for t in trades
        )

    def _front_if_ours(self, trade: TradeTick, window_ms: int = 1500) -> None:
        if not self.own_prints:
            return
        maker = trade.aggressor.opposite
        ms = (trade.ts_ns or self.now_ns) // MS
        for i, (side, price, lots, at) in enumerate(self.own_prints):
            if side is maker and price == trade.price and lots == trade.qty \
                    and abs(at - ms) <= window_ms:
                del self.own_prints[i]
                self.counts["own_prints"] += 1
                for order in self.book.orders.values():
                    if order.side is maker and order.price == price:
                        order.queue_ahead = 0
                return

    def touch(self, best_bid: int | None, best_ask: int | None) -> None:
        """The public touch after an event, for post-only checks (and gap fills)."""
        self.best_bid, self.best_ask = best_bid, best_ask
        if self.behaviour.gap_fill_share > 0:
            self._book(self.book.on_book(best_bid, best_ask, self.now_ns))

    def _book(self, fills) -> None:
        for fill in fills:
            exchange_id = self.paper_to_id.get(fill.maker_id)
            if exchange_id is None:
                continue
            order = self.by_id[exchange_id]
            price = self.instrument.tick_size * fill.price
            amount = self.instrument.lot_size * fill.qty
            if order.side is Side.BUY:
                self.jpy -= price * amount
                self.coin += amount
            else:
                self.jpy += price * amount
                self.coin -= amount
            self.counts["fills"] += 1
            trade = {
                "trade_id": next(self._trade_ids), "order_id": exchange_id,
                "side": "buy" if order.side is Side.BUY else "sell",
                "price": str(price), "amount": str(amount), "maker_taker": "maker",
                "executed_at": self.now_ns // MS,
            }
            self.history.append(trade)
            self.all_trades.append(trade)

    # ------------------------------------------------------ order helpers

    def _landed(self, order) -> bool:
        return self.now_ns >= order.active_ns

    def _open(self, order) -> bool:
        return order.remaining > 0 and not order.is_gone(self.now_ns)

    def _still_open(self) -> list:
        """(exchange id, order) for orders not yet filled or cancelled.

        Every order ever placed stays in `by_id` for status calls; walking
        all of them on each order made a day's replay take hours.
        """
        for exchange_id in [i for i, o in self._live.items() if not self._open(o)]:
            del self._live[exchange_id]
        return list(self._live.items())

    def _locked(self, side: Side) -> Decimal:
        total = Decimal(0)
        for _, order in self._still_open():
            if order.side is side:
                amount = self.instrument.lot_size * order.remaining
                if side is Side.BUY:
                    total += self.instrument.tick_size * order.price * amount
                else:
                    total += amount
        return total

    # --------------------------------------------- the calls the trader makes

    async def order(self, pair, side, price: Decimal, amount: Decimal) -> int:
        self.counts["orders"] += 1
        our = Side.BUY if side == "buy" else Side.SELL
        ticks = int((price / self.instrument.tick_size).to_integral_value())
        lots = self.instrument.to_lots(amount)
        opposite = self.best_ask if our is Side.BUY else self.best_bid
        if opposite is not None and (ticks >= opposite if our is Side.BUY else ticks <= opposite):
            self.counts["refused"] += 1
            raise OrderRefused(next(self._ids), "REJECTED")
        free = (self.jpy - self._locked(Side.BUY) if our is Side.BUY
                else self.coin - self._locked(Side.SELL))
        need = price * amount if our is Side.BUY else amount
        if need > free:
            self.counts["insufficient"] += 1
            raise BitbankError(INSUFFICIENT, "/v1/user/spot/order")
        visible = self.book._last_depth.get((int(our), ticks), 0)
        paper = self.book.place(Quote(our, ticks, lots), visible_depth=visible, best_opposite=None)
        land = self.behaviour.land_ms * MS
        if self.rng.random() < self.behaviour.slow_land_share:
            self.counts["slow_land"] += 1
            land = self.rng.expovariate(1 / self.behaviour.slow_land_mean_s) * 1e9
        paper.active_ns = self.now_ns + int(land)
        exchange_id = next(self._ids)
        self.by_id[exchange_id] = paper
        self._live[exchange_id] = paper
        self.paper_to_id[paper.order_id] = exchange_id
        return exchange_id

    def _cancel_one(self, order_id: int) -> str:
        order = self.by_id.get(order_id)
        if order is None or not self._landed(order):
            return "not_found"
        if not self._open(order) or order.cancel_at_ns is not None:
            return "cannot"
        self.book.cancel(order.order_id)
        self._cancelled.add(order_id)
        return "ok"

    async def cancel(self, pair, order_id: int) -> None:
        if self.rng.random() < self.behaviour.refuse_share:
            self.counts["cannot_cancel"] += 1
            raise BitbankError(CANNOT_CANCEL, "/v1/user/spot/cancel_order")
        result = self._cancel_one(order_id)
        if result == "not_found":
            self.counts["not_found"] += 1
            raise BitbankError(NOT_FOUND, "/v1/user/spot/cancel_order")
        if result == "cannot":
            self.counts["cannot_cancel"] += 1
            raise BitbankError(CANNOT_CANCEL, "/v1/user/spot/cancel_order")

    async def cancel_many(self, pair, order_ids) -> set[int]:
        # Orders not resting yet are skipped without an error, as bitbank does.
        return {i for i in order_ids if self._cancel_one(i) == "ok"}

    async def status(self, pair, order_id: int) -> str:
        order = self.by_id.get(order_id)
        if order is None or not self._landed(order):
            self.counts["not_found"] += 1
            raise BitbankError(NOT_FOUND, "/v1/user/spot/order")
        if order.remaining == 0:
            return "FULLY_FILLED"
        if order.is_gone(self.now_ns):
            return "CANCELED_PARTIALLY_FILLED" if order.filled else "CANCELED_UNFILLED"
        return "PARTIALLY_FILLED" if order.filled else "UNFILLED"

    async def active_orders(self, pair) -> list[int]:
        return [i for i, o in self._still_open() if self._landed(o)]

    async def trade_history(self, pair, since_ms: int) -> list[dict]:
        while self.history and self.history[0]["executed_at"] < since_ms - 60_000:
            self.history.popleft()
        return [t for t in self.history if t["executed_at"] >= since_ms]

    async def onhand(self) -> dict[str, Decimal]:
        return {self.instrument.quote.lower(): self.jpy, self.instrument.base.lower(): self.coin}


# ---------------------------------------------------------------- the replay


@dataclass
class LiveSimResult:
    pnl: float
    fills: int
    doomed: int
    refused: int
    volume: float
    counts: dict
    stopped: str = ""
    start_jpy: Decimal = Decimal(0)
    end_value: float = 0.0

    @property
    def doomed_pct(self) -> float:
        return self.doomed / self.fills * 100 if self.fills else float("nan")


async def run_livesim(
    rows,
    mm,
    lead_view,
    sim: SimBitbank,
    *,
    pair: str,
    per_s: float = 4.0,
    poll_ms: float = 500.0,
    check_s: float = 30.0,
    breaker_pct: float = 1.5,
    mids=None,
) -> LiveSimResult:
    """Run the live venue and executor over recorded rows on replay time.

    `rows` yields (source, receive_ns, event) in order; "lead" rows feed the
    lead view, the rest are the bitbank book. Time moves only with the rows,
    and between them the executor and poller act at the instants they would
    have: one call at a time, each holding the connection for its round
    trip, no more than `per_s` order changes in any second. A `mids`
    line (`research.fillcheck.MidLine`) is filled with the book's mid.
    """
    from ..live import clock
    from ..live.runner import (
        Breaker,
        Health,
        drop_noops,
        execute_once,
        head_due,
        poll_fills_once,
    )
    from ..live.venue import DONE, LiveVenue

    venue = LiveVenue(mm.instrument, quote_balance=sim.jpy, base_balance=sim.coin)
    mm.venue = venue
    start_coin = sim.coin
    health, breaker = Health(), Breaker(move_pct=breaker_pct)
    clock.use(lambda: sim.now_ns / 1e9)
    mm.clock = mm.market.clock = lambda: sim.now_ns
    lead_view.clock = mm.market.clock

    sent: deque = deque()
    busy_until = 0
    next_poll = next_check = None
    poll_state: dict = {}
    stopped = ""
    b = sim.behaviour

    async def act_until(t: int) -> None:
        nonlocal busy_until, next_poll, next_check
        while True:
            drop_noops(venue)
            want = venue.intents and head_due(venue)
            t_exec = None
            if want:
                while sent and sent[0] <= busy_until - 1_000_000_000:
                    sent.popleft()
                t_exec = busy_until
                if len(sent) >= per_s:
                    t_exec = max(t_exec, sent[0] + 1_000_000_000)
            options = [x for x in (t_exec, max(next_poll, busy_until)) if x is not None]
            when = min(options)
            if when > t:
                return
            sim.now_ns = when
            if t_exec is not None and when == t_exec:
                sent.append(when)
                venue.requests_sent += 1
                await execute_once(venue, sim, pair, health)
                busy_until = sim.now_ns + int(b.ack_ms * MS)
            else:
                await poll_fills_once(venue, sim, pair, poll_state)
                busy_until = sim.now_ns + int(b.query_ms * MS)
                next_poll = sim.now_ns + int(poll_ms * MS)
                if sim.now_ns >= next_check:
                    next_check = sim.now_ns + int(check_s * 1e9)
                    tracked = {o.exchange_id for o in venue.orders.values()
                               if o.state != DONE and o.exchange_id is not None}
                    for stray in await sim.active_orders(pair):
                        if stray not in tracked:
                            sim._cancel_one(stray)

    async for src, rx, event in rows:
        if next_poll is None:
            next_poll = next_check = rx
            poll_state["since_ms"] = rx // MS
        await act_until(rx)
        sim.now_ns = rx
        if src == "lead":
            lead_view.apply(event)
            if breaker.observe(lead_view.mid, rx / 1e9):
                venue.blocked = "急変ブレーカー"
                venue.cancel_all()
            elif venue.blocked and not breaker.active(rx / 1e9):
                venue.blocked = ""
            continue
        sim.on_market(event)
        mm.on_event(event)
        if mids is not None and mm.market.mid is not None:
            mids.add(rx, mm.market.mid * float(mm.instrument.tick_size))
        sim.touch(mm.market.book.best_bid(), mm.market.book.best_ask())
        if not venue.blocked:
            mm.requote()
        venue.forget_done()
        if health.fatal:
            stopped = health.fatal
            break
        if mm.risk.halted:
            stopped = f"損失上限: {mm.risk.halt_reason}"
            break
    venue.cancel_all()
    clock.reset()
    s = mm.summary()
    mid = mm.market.mid
    held = sim.coin - start_coin
    end_value = float(held) * (mid or 0) * float(mm.instrument.tick_size)
    return LiveSimResult(
        pnl=s["total"], fills=int(s["fills"]), doomed=venue.doomed_fills,
        refused=venue.post_only_refused, volume=s.get("volume", 0.0),
        counts=dict(sim.counts), stopped=stopped, end_value=end_value,
    )
