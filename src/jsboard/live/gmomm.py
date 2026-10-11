"""Quote GMO's leverage book around the fair price with limit orders, and
stop cleanly.

The quoting rule is the one the replay (research.fairmm) tested: fair =
mid x (1 + shrink x prediction), leaned against the inventory; a bid
`width_bps` under it and an ask as far over, post-only; a resting quote
is left alone until its target has moved `requote_bps` from it. The side
that would shrink the position closes it (closeBulkOrder), so the account
does not pile up positions both ways.

What the replay could only estimate, this run measures: how long GMO takes
to accept an order and to confirm a cancel, which fills come after a
cancel was asked for, and how the price moves after each fill.

Guards:

- **Inventory.** A new order goes out only if the position plus every
  order of ours on that side GMO has not confirmed finished (being sent,
  resting, or with a cancel on its way) plus the new one stays within
  `max_inv`. An order counts until GMO reports it cancelled or executed;
  a fill GMO has reported in the order's status but not yet in the
  executions still counts.
- **Loss.** Two counts, either stops the run: the fills as we read them,
  with the position marked where it could be closed (the bid for a long,
  the ask for a short); and GMO's own valuation of the account
  (actualProfitLoss), read every few seconds against its value at the
  start, which holds whatever our count misses. At `stop_jpy` the run stops: no new orders;
  every order cancelled, and GMO asked until none is left; the position
  closed at market, and GMO asked until none is left. The stop starts
  short of `max_loss_jpy` so the close has room; a gap in the market can
  still carry the loss past it.
- **Anything unknown** stops the same way: an order whose fate is unknown
  (the request failed without GMO's answer), the position GMO reports not
  matching the fills, a fill on an order that is not ours, three errors in
  a row, or being told to stop.

Without `--live` the same code runs against `DryApi`, which sends nothing
and fills orders on the live book by the replay's cautious rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .gmo import GmoError

NS = 1_000_000_000
AFTER_S = (0.1, 0.5, 1.0, 5.0)
FINAL = {"EXECUTED", "CANCELED", "EXPIRED"}
FATAL = ("ERR-201", "ERR-5010", "ERR-5011", "ERR-5012")
"""Not enough margin, or the key is refused: retrying will not help."""


def ts_ns(text: str) -> int | None:
    """GMO's ISO timestamp ("2026-10-11T00:00:00.123Z") in ns."""
    try:
        return int(datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp() * NS)
    except ValueError:
        return None


@dataclass
class Quote:
    side: int            # +1 bid, -1 ask
    price: float
    size: float
    closing: bool        # sent as a close of the opposite position
    sent_ns: int
    order_id: str | None = None
    ack_ns: int | None = None
    cancel_ns: int | None = None       # cancel wanted
    cancel_sent_ns: int | None = None
    cancel_ack_ns: int | None = None
    dead_ns: int | None = None         # GMO says it is finished
    filled: float = 0.0                # from the executions we have seen
    executed: float | None = None      # GMO's executed size once finished
    status: str = "SENDING"

    @property
    def live(self) -> bool:
        return self.dead_ns is None

    def exposure(self) -> float:
        """Coins this order may still add to the position as we know it."""
        if self.live:
            return max(0.0, self.size - self.filled)
        return max(0.0, (self.executed or 0.0) - self.filled)


@dataclass
class Stats:
    placed: int = 0
    acked: int = 0
    refused: int = 0
    """Orders GMO turned down outright."""
    unlanded: int = 0
    """Orders GMO cancelled before any fill without our asking (post-only)."""
    cancels: int = 0
    held: int = 0
    fills: int = 0
    fill_qty: float = 0.0
    notional: float = 0.0
    edge_bps: float = 0.0
    late: int = 0
    late_qty: float = 0.0
    errors: int = 0
    max_position: float = 0.0
    ack_ms: list = field(default_factory=list)
    cancel_ack_ms: list = field(default_factory=list)
    cancel_dead_ms: list = field(default_factory=list)
    after: dict = field(default_factory=lambda: {s: [0.0, 0.0] for s in AFTER_S})


def _median(xs: list) -> str:
    return f"{statistics.median(xs):.0f}" if xs else "-"


@dataclass
class Maker:
    api: object
    symbol: str
    tick: float
    size: float = 10.0
    width_bps: float = 2.0
    requote_bps: float = 0.5
    max_inv: float = 30.0
    skew_bps: float = 0.5
    max_loss_jpy: float = 500.0
    stop_jpy: float = 400.0
    time_in_force: str = "SOK"
    where: str = "GMO"
    """Who confirms the stop: GMO, or the dry run standing in for it."""
    max_posts_per_s: float = 5.0
    clock: object = time.time_ns
    sleep: object = asyncio.sleep
    log: object = None
    notify: object = None
    quotes: list = field(default_factory=list)
    position: float = 0.0
    long_qty: float = 0.0
    short_qty: float = 0.0
    cash: float = 0.0
    seen: set = field(default_factory=set)
    own_ids: set = field(default_factory=set)
    stopping: bool = False
    stopped: str | None = None
    clean: bool | None = None
    equity0: float | None = None
    """GMO's valuation of the account when the run started."""
    equity: float | None = None
    """After a stop: True once GMO shows no order and no position of ours."""
    stats: Stats = field(default_factory=Stats)
    errors_in_row: int = 0
    mismatches: int = 0
    _later: list = field(default_factory=list)
    _mids: deque = field(default_factory=deque)
    """(ns, GMO mid) every grid step for the last half minute: a fill is
    seen up to a poll late, and is measured from when GMO made it."""
    _seq: object = field(default_factory=itertools.count)
    _tasks: set = field(default_factory=set)
    _sending: set = field(default_factory=set)
    _posts: deque = field(default_factory=deque)
    _flatten: object = None

    # ---------------------------------------------------------------- state

    def exposure(self, side: int) -> float:
        return sum(q.exposure() for q in self.quotes if q.side == side)

    def pnl(self, bid: float | None, ask: float | None) -> float | None:
        """Cash plus the position marked where it could be closed now."""
        if self.position > 1e-12:
            return None if bid is None else self.cash + self.position * bid
        if self.position < -1e-12:
            return None if ask is None else self.cash + self.position * ask
        return self.cash

    def _record(self, rec: dict) -> None:
        if self.log:
            self.log({"t_ns": self.clock(), **rec})

    async def _say(self, text: str) -> None:
        if self.notify:
            await self.notify(text)

    def _spawn(self, coro, sending: bool = False) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        if sending:
            self._sending.add(task)
            task.add_done_callback(self._sending.discard)

    async def _budget(self) -> None:
        """At most `max_posts_per_s` orders and cancels a second."""
        while True:
            now = self.clock()
            while self._posts and self._posts[0] <= now - NS:
                self._posts.popleft()
            if len(self._posts) < self.max_posts_per_s:
                self._posts.append(now)
                return
            await self.sleep(max(0.01, (self._posts[0] + NS - now) / NS))

    def _fmt(self, price: float) -> str:
        decimals = max(0, -math.floor(math.log10(self.tick) + 1e-9))
        return f"{price:.{decimals}f}"

    # --------------------------------------------------------------- quoting

    def plan(self, fair: float | None, bid: float | None, ask: float | None) -> list:
        """What to cancel and what to send so each side sits where `fair`
        puts it: [("cancel", quote) | ("place", side, price)]."""
        acts: list = []
        for side in (1, -1):
            work = next((q for q in self.quotes
                         if q.side == side and q.live and q.cancel_ns is None), None)
            want = None
            if fair is not None and not self.stopping and side * self.position < self.max_inv - 1e-9:
                raw = fair * (1 - side * self.width_bps / 1e4)
                ticks = (math.floor(raw / self.tick + 1e-9) if side > 0
                         else math.ceil(raw / self.tick - 1e-9))
                price = round(ticks * self.tick, 10)
                # Post-only here as well: never send a price that would take.
                if bid is not None and ask is not None and (price < ask if side > 0 else price > bid):
                    want = price
            if work is not None and want is not None and (
                    abs(want - work.price) / fair * 1e4 < self.requote_bps - 1e-9):
                continue
            if work is not None:
                acts.append(("cancel", work))
            if want is None:
                continue
            if side * self.position + self.exposure(side) + self.size > self.max_inv + 1e-9:
                self.stats.held += 1   # wait for an old order to finish
                continue
            acts.append(("place", side, want))
        return acts

    def requote(self, fair: float | None, bid: float | None, ask: float | None) -> None:
        """Lean `fair` against the position, plan, and send."""
        if fair is not None:
            fair *= 1 - self.skew_bps * self.position / self.size / 1e4
        for act in self.plan(fair, bid, ask):
            if act[0] == "cancel":
                self._ask_cancel(act[1])
            else:
                self._send(act[1], act[2])

    def _ask_cancel(self, q: Quote) -> None:
        if q.cancel_ns is None:
            q.cancel_ns = self.clock()
            self.stats.cancels += 1
        if q.order_id is not None and q.cancel_sent_ns is None and q.live:
            q.cancel_sent_ns = self.clock()
            self._spawn(self._cancel(q))

    def _send(self, side: int, price: float) -> None:
        # Close the opposite position when there is enough of it not
        # already being closed; otherwise open.
        other = self.short_qty if side > 0 else self.long_qty
        closing_out = sum(q.exposure() for q in self.quotes if q.side == side and q.closing)
        q = Quote(side, price, self.size, other - closing_out >= self.size - 1e-9, self.clock())
        self.quotes.append(q)   # counted against the cap from now on
        self.stats.placed += 1
        self._spawn(self._place(q), sending=True)

    async def _place(self, q: Quote) -> None:
        await self._budget()
        word = "BUY" if q.side > 0 else "SELL"
        send = self.api.limit_close if q.closing else self.api.limit_open
        q.sent_ns = self.clock()
        try:
            oid = await send(self.symbol, word, self._fmt(q.price), f"{q.size:g}", self.time_in_force)
        except GmoError as exc:
            # GMO answered no: there is no order.
            q.status, q.executed, q.dead_ns = "REFUSED", 0.0, self.clock()
            self.stats.refused += 1
            self._record({"ev": "refused", "side": word, "price": q.price, "closing": q.closing,
                          "codes": exc.codes, "error": str(exc)})
            await self._error(exc)
            return
        except Exception as exc:  # noqa: BLE001 - the order may exist without our knowing
            q.status = "UNKNOWN"
            self._record({"ev": "unknown", "side": word, "price": q.price,
                          "error": f"{type(exc).__name__}: {exc}"})
            self.stats.errors += 1
            if not self.stopping:
                self._spawn(self.flatten(f"注文の結果が分かりません（{type(exc).__name__}）"))
            return
        q.order_id, q.ack_ns, q.status = oid, self.clock(), "ORDERED"
        self.own_ids.add(oid)
        self.errors_in_row = 0
        self.stats.acked += 1
        self.stats.ack_ms.append((q.ack_ns - q.sent_ns) / 1e6)
        self._record({"ev": "placed", "order_id": oid, "side": word, "price": q.price,
                      "size": q.size, "closing": q.closing, "sent_ns": q.sent_ns,
                      "ack_ms": round((q.ack_ns - q.sent_ns) / 1e6, 1)})
        if q.cancel_ns is not None or self.stopping:
            self._ask_cancel(q)

    async def _cancel(self, q: Quote) -> None:
        await self._budget()
        t = self.clock()
        q.cancel_sent_ns = t
        try:
            await self.api.cancel_order(q.order_id)
        except GmoError as exc:
            # Most often it already finished; its status will say.
            self._record({"ev": "cancel_refused", "order_id": q.order_id, "codes": exc.codes})
            q.cancel_ack_ns = self.clock()
            return
        except Exception as exc:  # noqa: BLE001 - tried again from the poll
            q.cancel_sent_ns = None
            self.stats.errors += 1
            self._record({"ev": "cancel_failed", "order_id": q.order_id,
                          "error": f"{type(exc).__name__}: {exc}"})
            return
        q.cancel_ack_ns = self.clock()
        ms = (q.cancel_ack_ns - t) / 1e6
        self.stats.cancel_ack_ms.append(ms)
        self._record({"ev": "cancelled", "order_id": q.order_id, "asked_ns": q.cancel_ns,
                      "sent_ns": t, "ack_ms": round(ms, 1)})

    async def _error(self, exc: Exception) -> None:
        self.stats.errors += 1
        self.errors_in_row += 1
        codes = getattr(exc, "codes", []) or []
        if self.stopping:
            return   # already stopping; the stop keeps asking on its own
        if any(c in FATAL for c in codes) or self.errors_in_row >= 3:
            self._spawn(self.flatten(f"GMO がエラーを返しました: {exc}"))

    # ------------------------------------------------------------ reading GMO

    def apply(self, e: dict, mid: float | None = None) -> None:
        """One execution from GMO: position, cash, and the order it filled."""
        eid = str(e.get("executionId"))
        if eid in self.seen:
            return
        self.seen.add(eid)
        oid = str(e.get("orderId"))
        side = 1 if e.get("side") == "BUY" else -1
        qty, price = float(e["size"]), float(e["price"])
        fee = float(e.get("fee") or 0)
        now = self.clock()
        self.position += side * qty
        self.cash -= side * qty * price + fee
        if e.get("settleType") == "CLOSE":
            if side > 0:
                self.short_qty = max(0.0, self.short_qty - qty)
            else:
                self.long_qty = max(0.0, self.long_qty - qty)
        elif side > 0:
            self.long_qty += qty
        else:
            self.short_qty += qty
        s = self.stats
        s.max_position = max(s.max_position, abs(self.position))
        q = next((x for x in self.quotes if x.order_id == oid), None)
        when = ts_ns(e.get("timestamp", "")) or now
        rec = {"ev": "fill", "order_id": oid, "execution_id": eid, "side": e.get("side"),
               "settle": e.get("settleType"), "size": qty, "price": price, "fee": fee,
               "exchange_ns": when, "seen_ns": now, "position": self.position}
        if q is not None:
            q.filled += qty
            late = q.cancel_ns is not None and when >= q.cancel_ns
            s.fills += 1
            s.fill_qty += qty
            s.notional += qty * price
            if late:
                s.late += 1
                s.late_qty += qty
            then = self.mid_at(when) or mid
            if then:
                s.edge_bps += side * (then - price) / price * 1e4 * qty
            for a in AFTER_S:
                heapq.heappush(self._later, (when + int(a * NS), next(self._seq), side, price, qty, a))
            rec.update(late=late, cancel_asked_ns=q.cancel_ns, quote_price=q.price, mid=then)
        elif oid not in self.own_ids:
            rec["foreign"] = True
        self._record(rec)
        if rec.get("foreign") and not self.stopping:
            self._spawn(self.flatten("自分のものではない約定がありました（手動の注文？）"))

    def mid_at(self, t_ns: int) -> float | None:
        """GMO's mid as it stood at `t_ns`, from the grid steps kept."""
        for t, m in reversed(self._mids):
            if t <= t_ns:
                return m
        return None

    def settle_after(self, mid: float | None) -> None:
        """Keep this step's mid; then how the mid moved 0.1/0.5/1/5 s after
        each fill (counted from GMO's time of the fill), in its favour."""
        now = self.clock()
        if mid:
            self._mids.append((now, mid))
        while self._mids and self._mids[0][0] < now - 30 * NS:
            self._mids.popleft()
        while self._later and self._later[0][0] <= now:
            due, _, side, price, qty, a = heapq.heappop(self._later)
            m = self.mid_at(due) or mid
            if m:
                cell = self.stats.after[a]
                cell[0] += qty
                cell[1] += side * (m - price) / price * 1e4 * qty

    def _status(self, o: dict) -> None:
        oid = str(o.get("orderId"))
        q = next((x for x in self.quotes if x.order_id == oid), None)
        if q is None or not q.live:
            return
        q.status = str(o.get("status", q.status))
        if q.status not in FINAL:
            return
        q.executed = float(o.get("executedSize") or 0)
        q.dead_ns = self.clock()
        if q.cancel_ns is None and q.executed < 1e-12 and q.status == "CANCELED":
            self.stats.unlanded += 1   # post-only: it would have taken
        if q.cancel_sent_ns is not None:
            self.stats.cancel_dead_ms.append((q.dead_ns - q.cancel_sent_ns) / 1e6)
        self._record({"ev": "finished", "order_id": oid, "status": q.status,
                      "executed": q.executed, "cancel_asked_ns": q.cancel_ns})

    async def poll(self, mid: float | None = None, reconcile: bool = False) -> None:
        """Read GMO: new executions, the state of our live orders, and now
        and then the position, which must match the fills."""
        try:
            rows = await self.api.latest_executions(self.symbol)
            for e in sorted(rows, key=lambda r: (str(r.get("timestamp")), str(r.get("executionId")))):
                self.apply(e, mid)
            ids = [q.order_id for q in self.quotes if q.order_id and q.live]
            for i in range(0, len(ids), 10):
                for o in await self.api.orders(ids[i:i + 10]):
                    self._status(o)
            if reconcile:
                await self._reconcile()
            self.errors_in_row = 0
        except GmoError as exc:
            await self._error(exc)
        except Exception as exc:  # noqa: BLE001
            self._record({"ev": "poll_failed", "error": f"{type(exc).__name__}: {exc}"})
            await self._error(exc)
        now = self.clock()
        for q in self.quotes:
            # A cancel that went nowhere: ask again.
            if q.live and q.order_id and q.cancel_ns is not None and q.cancel_sent_ns is None:
                self._ask_cancel(q)
            elif (q.live and q.order_id and q.cancel_sent_ns is not None and q.cancel_ack_ns is not None
                  and now - q.cancel_ack_ns > 3 * NS):
                q.cancel_sent_ns = q.cancel_ack_ns = None
                self._ask_cancel(q)
        self.quotes = [q for q in self.quotes if q.live or q.exposure() > 1e-12]

    async def _reconcile(self) -> None:
        rows = await self.api.position_summary(self.symbol)
        long_ = sum(float(r.get("sumPositionQuantity") or 0) for r in rows if r.get("side") == "BUY")
        short = sum(float(r.get("sumPositionQuantity") or 0) for r in rows if r.get("side") == "SELL")
        if abs(long_ - self.long_qty) > 1e-9 or abs(short - self.short_qty) > 1e-9:
            self.mismatches += 1
            self._record({"ev": "mismatch", "gmo_long": long_, "gmo_short": short,
                          "ours_long": self.long_qty, "ours_short": self.short_qty})
            if self.mismatches >= 3 and not self.stopping:
                self._spawn(self.flatten(
                    f"GMO のポジション（買 {long_:g}・売 {short:g}）が記録"
                    f"（買 {self.long_qty:g}・売 {self.short_qty:g}）と合いません"))
        else:
            self.mismatches = 0
        await self.read_equity()

    async def read_equity(self) -> None:
        """GMO's own valuation of the account: the loss as GMO counts it."""
        m = await self.api.margin()
        eq = float(m.get("actualProfitLoss") or 0)
        if self.equity0 is None:
            self.equity0 = eq
        self.equity = eq
        loss = self.equity0 - eq
        if loss >= self.stop_jpy and not self.stopping:
            self._spawn(self.flatten(
                f"GMO の口座で損失が {loss:,.0f}円 になりました（{self.stop_jpy:,.0f}円で止める設定、"
                f"上限 {self.max_loss_jpy:,.0f}円）"))

    def check_loss(self, bid: float | None, ask: float | None) -> None:
        pnl = self.pnl(bid, ask)
        if pnl is not None and pnl <= -self.stop_jpy and not self.stopping:
            self._spawn(self.flatten(
                f"損失が {-pnl:,.0f}円 になりました（{self.stop_jpy:,.0f}円で止める設定、上限 {self.max_loss_jpy:,.0f}円）"))

    # ------------------------------------------------------------------ stop

    async def flatten(self, reason: str) -> None:
        """Stop for good: no new orders, cancel everything until GMO shows
        none, close the position at market until GMO shows none."""
        if self._flatten is None:
            self.stopping = True
            self.stopped = reason
            self._flatten = asyncio.ensure_future(self._do_flatten(reason))
        await asyncio.shield(self._flatten)

    async def _do_flatten(self, reason: str) -> None:
        await self._say(f":octagonal_sign: GMO 指値を止めます: {reason}\n"
                        "  注文を全部取り消して、ポジションを閉じます。")
        pending = [t for t in self._sending if not t.done()]
        if pending:   # let orders in flight get their ids, so they can be cancelled
            await asyncio.wait(pending, timeout=5)
        problems = []
        gone = False
        for _ in range(30):
            with contextlib.suppress(Exception):
                await self.api.cancel_all(self.symbol)
            await self.sleep(1)
            try:
                if not await self.api.active_orders(self.symbol):
                    gone = True
                    break
            except Exception:  # noqa: BLE001, S112 - keep asking
                continue
        if not gone:
            problems.append("取り消せない注文が残っています")
        with contextlib.suppress(Exception):
            await self.poll()
        closed = False
        for attempt in range(10):
            try:
                rows = await self.api.position_summary(self.symbol)
            except Exception:  # noqa: BLE001
                await self.sleep(1)
                continue
            held = {r.get("side"): float(r.get("sumPositionQuantity") or 0) for r in rows}
            held = {k: v for k, v in held.items() if v > 1e-12}
            if not held:
                closed = True
                break
            for side, qty in held.items():
                back = "SELL" if side == "BUY" else "BUY"
                try:
                    oid = await self.api.market_close_all(self.symbol, back, f"{qty:g}")
                    self.own_ids.add(oid)
                    self._record({"ev": "close", "order_id": oid, "side": back, "size": qty})
                except Exception as exc:  # noqa: BLE001
                    self._record({"ev": "close_failed", "side": back, "size": qty,
                                  "error": f"{type(exc).__name__}: {exc}"})
            await self.sleep(1 + attempt)
            with contextlib.suppress(Exception):
                await self.poll()
        if not closed:
            problems.append("閉じられないポジションが残っています")
        if gone:
            now = self.clock()
            for q in self.quotes:
                q.dead_ns = q.dead_ns or now
        self.clean = not problems
        self._record({"ev": "stopped", "reason": reason, "clean": self.clean, "problems": problems,
                      "position": self.position, "cash": self.cash})
        if problems:
            await self._say(":rotating_light: GMO 指値: " + "・".join(problems)
                            + "。GMO の画面で確認してください。\n  " + self.text())
        else:
            await self._say(f":white_check_mark: GMO 指値を止めました。注文 0件・ポジション 0 を{self.where}で確認しました。\n  "
                            + self.text())

    # ---------------------------------------------------------------- report

    def text(self, bid: float | None = None, ask: float | None = None) -> str:
        s = self.stats
        pnl = self.pnl(bid, ask) if (bid or ask or not self.position) else None
        per = f"{pnl / s.notional * 1e4:+.2f}bps" if pnl is not None and s.notional else "-"
        after = "  ".join(
            f"{a:g}秒後 {c[1] / c[0]:+.2f}" if c[0] else f"{a:g}秒後 -"
            for a, c in ((a, s.after[a]) for a in AFTER_S))
        edge = f"{s.edge_bps / s.fill_qty:+.2f}" if s.fill_qty else "-"
        money = f"{pnl:+,.1f}円（約定1回あたり {per}）" if pnl is not None else "-"
        gmo = (f"GMO の口座で {self.equity - self.equity0:+,.1f}円  "
               if self.equity is not None and self.equity0 is not None else "")
        return (f"{gmo}損益（約定から計算）{money}  建玉 {self.position:+g}枚（最大 {s.max_position:g}）\n"
                f"  指値 {s.placed}回（受付 {s.acked}・拒否 {s.refused}・板に載らず {s.unlanded}）  "
                f"取消 {s.cancels}回  在庫枠で見送り {s.held}回  エラー {s.errors}回\n"
                f"  約定 {s.fills}回 {s.fill_qty:g}枚（うち取消を出した後 {s.late}回 {s.late_qty:g}枚）\n"
                f"  約定時の取り分 {edge}bps  約定後の値動き（bps、プラスが有利）{after}\n"
                f"  注文→受付 {_median(s.ack_ms)}ms  取消→受付 {_median(s.cancel_ack_ms)}ms  "
                f"取消→消えた確認 {_median(s.cancel_dead_ms)}ms（中央値）")


class DryApi:
    """Sends nothing. Orders land `place_ms` after they are sent, at the
    back of their price level (refused if they would take: post-only), and
    fill only on GMO's own prints by the replay's cautious rule; a cancel
    takes `cancel_ms`. A close at market fills at the live book."""

    def __init__(self, tick: float, touch, level, walk, *, place_ms: float = 200.0,
                 cancel_ms: float = 200.0, clock=time.time_ns) -> None:
        self.tick = tick
        self.touch = touch    # () -> (best bid, best ask) yen, or (None, None)
        self.level = level    # (side, price yen) -> coins resting there
        self.walk = walk      # (side, qty) -> average price yen, or None
        self.place = int(place_ms * 1e6)
        self.cancel = int(cancel_ms * 1e6)
        self.clock = clock
        self.by_id: dict[str, dict] = {}
        self.execs: list[dict] = []
        self.long = self.short = 0.0
        self.cash = 0.0
        self._ids = itertools.count(1)
        self.sent = 0

    def _t(self, ns: int) -> str:
        return datetime.fromtimestamp(ns / NS, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _settle(self) -> None:
        now = self.clock()
        for o in self.by_id.values():
            if o["status"] == "WAITING" and now >= o["active_at"]:
                bid, ask = self.touch()
                if bid is None or ask is None or (
                        o["price"] >= ask - 1e-12 if o["side"] > 0 else o["price"] <= bid + 1e-12):
                    o["status"] = "CANCELED"
                    continue
                o["status"] = "ORDERED"
                o["queue"] = self.level(o["side"], o["price"])
            if o["status"] in ("WAITING", "ORDERED") and o["cancel_at"] is not None and now >= o["cancel_at"]:
                o["status"] = "CANCELED"

    def _fill(self, o: dict, qty: float, price: float) -> None:
        qty = min(qty, o["size"] - o["executed"])
        if qty <= 1e-12:
            return
        o["executed"] += qty
        if o["executed"] >= o["size"] - 1e-12:
            o["status"] = "EXECUTED"
        self._book(o["id"], o["side"], o["settle"], qty, price)

    def _book(self, oid: str, side: int, settle: str, qty: float, price: float) -> None:
        self.cash -= side * qty * price
        if settle == "CLOSE":
            if side > 0:
                self.short -= qty
            else:
                self.long -= qty
        elif side > 0:
            self.long += qty
        else:
            self.short += qty
        self.execs.insert(0, {
            "executionId": f"dry-e{next(self._ids)}", "orderId": oid,
            "side": "BUY" if side > 0 else "SELL", "settleType": settle, "size": f"{qty:g}",
            "price": f"{price}", "fee": "0", "lossGain": "0", "timestamp": self._t(self.clock())})
        del self.execs[200:]

    def on_trade(self, aggressor: int, price: float, qty: float) -> None:
        """A print on GMO: a taker buy (+1) lifts asks, a taker sell hits bids."""
        self._settle()
        hit = -aggressor
        for o in self.by_id.values():
            if o["side"] != hit or o["status"] != "ORDERED":
                continue
            if (hit > 0 and price < o["price"] - 1e-12) or (hit < 0 and price > o["price"] + 1e-12):
                self._fill(o, qty, o["price"])          # swept through our level
            elif abs(price - o["price"]) < 1e-12:
                o["queue"] -= qty
                if o["queue"] < 0:
                    self._fill(o, -o["queue"], o["price"])
                    o["queue"] = 0.0

    def _new(self, side: str, price: str, size: str, settle: str) -> str:
        self._settle()
        self.sent += 1
        oid = f"dry-{next(self._ids)}"
        self.by_id[oid] = {"id": oid, "side": 1 if side == "BUY" else -1, "price": float(price),
                            "size": float(size), "executed": 0.0, "settle": settle,
                            "status": "WAITING", "active_at": self.clock() + self.place,
                            "cancel_at": None, "queue": 0.0}
        return oid

    async def limit_open(self, symbol, side, price, size, time_in_force="SOK") -> str:
        return self._new(side, price, size, "OPEN")

    async def limit_close(self, symbol, side, price, size, time_in_force="SOK") -> str:
        held = self.short if side == "BUY" else self.long
        closing = sum(o["size"] - o["executed"] for o in self.by_id.values()
                      if o["settle"] == "CLOSE" and o["side"] == (1 if side == "BUY" else -1)
                      and o["status"] in ("WAITING", "ORDERED"))
        if held - closing < float(size) - 1e-9:
            raise GmoError(["DRY-NO-POSITION"], "/v1/closeBulkOrder")
        return self._new(side, price, size, "CLOSE")

    async def market_close_all(self, symbol, side, size) -> str:
        s = 1 if side == "BUY" else -1
        price = self.walk(s, float(size))
        if price is None:
            raise GmoError(["DRY-NO-BOOK"], "/v1/closeBulkOrder")
        oid = f"dry-{next(self._ids)}"
        self._book(oid, s, "CLOSE", float(size), price)
        return oid

    async def cancel_order(self, order_id: str) -> None:
        self._settle()
        o = self.by_id.get(str(order_id))
        if o is None or o["status"] in FINAL:
            raise GmoError(["ERR-5122"], "/v1/cancelOrder")
        if o["cancel_at"] is None:
            o["cancel_at"] = self.clock() + self.cancel

    async def cancel_all(self, symbol) -> list[str]:
        self._settle()
        out = []
        for o in self.by_id.values():
            if o["status"] in ("WAITING", "ORDERED") and o["cancel_at"] is None:
                o["cancel_at"] = self.clock() + self.cancel
                out.append(o["id"])
        return out

    async def orders(self, order_ids) -> list[dict]:
        self._settle()
        return [{"orderId": o["id"], "status": o["status"], "executedSize": f"{o['executed']:g}",
                 "size": f"{o['size']:g}"} for i in order_ids if (o := self.by_id.get(str(i)))]

    async def active_orders(self, symbol) -> list[dict]:
        self._settle()
        return [{"orderId": o["id"]} for o in self.by_id.values() if o["status"] in ("WAITING", "ORDERED")]

    async def latest_executions(self, symbol, count: int = 100) -> list[dict]:
        self._settle()
        return list(self.execs[:count])

    async def position_summary(self, symbol) -> list[dict]:
        return [{"side": k, "sumPositionQuantity": f"{v:g}"}
                for k, v in (("BUY", self.long), ("SELL", self.short)) if v > 1e-12]

    async def margin(self) -> dict:
        """The account valued at the mid, from a notional 10,000 yen."""
        bid, ask = self.touch()
        mid = (bid + ask) / 2 if bid is not None and ask is not None else 0.0
        return {"actualProfitLoss": f"{10_000 + self.cash + (self.long - self.short) * mid}"}
