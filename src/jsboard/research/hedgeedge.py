"""A bitbank maker fill hedged on GMO, priced from GMO's real book.

The bitbank maker lost because nothing took its inventory off its hands:
it held each buy until a seller came, and the price moved meanwhile. A
hedge removes that, but only if hedging costs less than the edge the fill
carried. This measures exactly that, from simultaneous recordings of the
two books, and nothing in it is assumed rather than looked up:

1. each bitbank execution is taken as a maker fill of ours at its price
   (the ceiling: as if we were first in the queue for every print), up to
   the order size;
2. `latency_ms` later, the same quantity is sold (or bought) at market on
   GMO by walking GMO's recorded book level by level — not its mid, not
   its last price — and GMO's taker fee is charged;
3. `h` seconds on, the hedge is bought back by walking GMO's book again,
   and the bitbank side is valued at bitbank's mid then (the price a
   later maker fill of ours would close it near), with bitbank's maker
   rebate on the opening fill.

What is left is the maker's edge minus every cost of holding no position.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.types import Instrument, Side
from ..feed.base import DepthDelta, DepthSnapshot, TradeTick

NS = 1_000_000_000
HORIZONS_S = (10, 60, 300)


@dataclass
class Leg:
    n: int = 0
    notional: float = 0.0
    pnl: float = 0.0
    unhedged: float = 0.0
    """The same fills held unhedged to the same instant, for comparison."""

    def bps(self, value: float | None = None) -> float:
        v = self.pnl if value is None else value
        return v / self.notional * 1e4 if self.notional else float("nan")


@dataclass
class HedgeResult:
    label: str
    hours: float = 0.0
    prints: int = 0
    too_small: int = 0
    no_book: int = 0
    too_thin: int = 0
    hedged: int = 0
    open_cost_bps: float = 0.0
    """GMO's half spread and depth walked when hedging, notional-weighted."""
    close_cost_bps: float = 0.0
    horizons: dict = field(default_factory=lambda: {h: Leg() for h in HORIZONS_S})
    _open_notional: float = 0.0
    _close_notional: float = 0.0

    def row(self) -> str:
        def f(x: float) -> str:
            return "-" if x != x else f"{x:+.1f}"

        cols = [
            self.label, f"{self.hours:.1f}", f"{self.prints:,}", f"{self.hedged:,}",
            f"{self.too_small + self.no_book + self.too_thin:,}",
            f(self.open_cost_bps), f(self.close_cost_bps),
        ]
        for h in HORIZONS_S:
            leg = self.horizons[h]
            cols += [f(leg.bps()), f"{leg.pnl:+,.0f}"]
        cols.append(f(self.horizons[60].bps(self.horizons[60].unhedged)))
        return "\t".join(cols)

    @property
    def passed(self) -> bool:
        return self.hedged >= 100 and all(self.horizons[h].pnl > 0 for h in HORIZONS_S)


HEADER = "\t".join([
    "銘柄", "時間", "bitbank約定", "ヘッジ数", "ヘッジ不可",
    "GMO片道コストbps(建て)", "GMO片道コストbps(決済)",
    "10秒bps", "10秒円", "60秒bps", "60秒円", "5分bps", "5分円", "参考:ヘッジなし60秒bps",
])
NOTE = (
    "bitbank の全約定で自分がメイカーだった場合の上限（並び順・速さは最高）。\n"
    "ヘッジは GMO の録画した実際の板を、指定量ぶん上から食った値段（中値・最終値は使わない）。\n"
    "bps と円は、bitbank リベート・GMO 手数料・GMO の板食い込み・ヘッジまでの遅れ込みの最終損益。\n「板連動」は GMO の1秒前の板から bitbank の値段を決めた場合（古い値段で当たる分も込み）。"
)


def walk(levels, qty: float, inst: Instrument) -> float | None:
    """Average price (yen) for taking `qty` coins off `levels`, best first;
    None if the book holds less than that."""
    left, cost = qty, 0.0
    for level in levels:
        take = min(left, inst.qty_f(level.qty))
        cost += take * inst.price_f(level.price)
        left -= take
        if left <= 1e-12:
            return cost / qty
    return None


@dataclass
class _Fill:
    sign: int
    price: float
    qty: float
    hedge: float = 0.0


def analyse(rows, maker: Instrument, hedge: Instrument, *, label: str,
            rebate_bps: float, hedge_fee_bps: float, size_jpy: float,
            hedge_min: float = 0.0, latency_ms: float = 200.0, depth: int = 100,
            quote_margin_bps: float | None = None, quote_delay_ms: float = 1000.0) -> HedgeResult:
    """`rows` yields (source, receive_ns, event), source "bitbank" or "gmo".

    With `quote_margin_bps` the maker no longer takes every print at its
    price. It quotes on bitbank from GMO's book: a bid `margin` below GMO's
    best bid, an ask `margin` above GMO's best ask, so a fill can be sold
    back on GMO at a known profit. The quote follows GMO only as fast as
    orders can be replaced, so it is set from GMO's book `quote_delay_ms`
    earlier (bitbank's cancels took about a second): when GMO moves, the
    old quote stays out and gets picked off, as it would. A bid at or
    above bitbank's best ask could not rest (post-only), so it is pulled
    back one tick inside. A print fills the quote when it trades at or
    through its price, at the quote's price.
    """
    bb, gm = MarketView(instrument=maker, depth=depth), MarketView(instrument=hedge, depth=depth)
    out = HedgeResult(label)
    tasks: list = []
    seq = itertools.count()
    first = last = None
    lat = int(latency_ms * 1e6)
    delay = int(quote_delay_ms * 1e6)
    gmo_tops: deque = deque()  # (receive_ns, best bid yen, best ask yen)
    bb_tops: deque = deque()   # (receive_ns, best bid tick, best ask tick)
    tick = float(maker.tick_size)

    def as_of(tops: deque, t: int):
        while len(tops) >= 2 and tops[1][0] <= t:
            tops.popleft()
        return tops[0] if tops and tops[0][0] <= t else None

    def quote_at(t: int):
        """Our bitbank (bid, ask) in yen, as set `delay` before `t` from
        GMO's book and kept post-only against bitbank's book of then."""
        g, b = as_of(gmo_tops, t - delay), as_of(bb_tops, t - delay)
        if g is None or b is None:
            return None
        _, gb, ga = g
        _, best_bid, best_ask = b
        m = quote_margin_bps / 1e4
        bid = math.floor(gb * (1 - m) / tick + 1e-9) * tick
        ask = math.ceil(ga * (1 + m) / tick - 1e-9) * tick
        bid = min(bid, (best_ask - 1) * tick)
        ask = max(ask, (best_bid + 1) * tick)
        return bid, ask

    def run_due(now: int) -> None:
        while tasks and tasks[0][0] <= now:
            due, _, kind, fill, h = heapq.heappop(tasks)
            if kind == "open":
                _open(due, fill)
            else:
                _close(fill, h)

    def side_levels(sign: int):
        snap = gm.snapshot(depth)
        # Our bitbank buy is hedged by a GMO sell: it takes GMO's bids.
        return snap, (snap.bids if sign > 0 else snap.asks)

    def _open(due: int, fill: _Fill) -> None:
        snap, levels = side_levels(fill.sign)
        if snap.mid is None:
            out.no_book += 1
            return
        price = walk(levels, fill.qty, hedge)
        if price is None:
            out.too_thin += 1
            return
        fill.hedge = price
        mid = snap.mid * float(hedge.tick_size)
        notional = price * fill.qty
        out.open_cost_bps += fill.sign * (mid - price) / mid * 1e4 * notional
        out._open_notional += notional
        out.hedged += 1
        for h in HORIZONS_S:
            heapq.heappush(tasks, (due + h * NS, next(seq), "close", fill, h))

    def _close(fill: _Fill, h: int) -> None:
        snap, _ = side_levels(fill.sign)
        levels = snap.asks if fill.sign > 0 else snap.bids  # buying the hedge back
        back = walk(levels, fill.qty, hedge) if snap.mid is not None else None
        bb_mid = bb.mid
        if back is None or bb_mid is None:
            return
        bb_mid *= float(maker.tick_size)
        q, s = fill.qty, fill.sign
        maker_leg = s * q * (bb_mid - fill.price) + fill.price * q * rebate_bps / 1e4
        hedge_leg = -s * q * (back - fill.hedge)
        fees = (fill.hedge + back) * q * hedge_fee_bps / 1e4
        leg = out.horizons[h]
        leg.n += 1
        leg.notional += fill.price * q
        leg.pnl += maker_leg + hedge_leg - fees
        leg.unhedged += maker_leg
        if h == HORIZONS_S[0]:
            mid = snap.mid * float(hedge.tick_size)
            out.close_cost_bps += s * (back - mid) / mid * 1e4 * back * q
            out._close_notional += back * q

    for src, rx, event in rows:
        run_due(rx - 1)
        first = rx if first is None else first
        last = rx
        if src == "gmo":
            if isinstance(event, (DepthSnapshot, DepthDelta)):
                gm.apply(event)
                gb, ga = gm.book.best_bid(), gm.book.best_ask()
                if quote_margin_bps is not None and gb is not None and ga is not None:
                    htick = float(hedge.tick_size)
                    gmo_tops.append((rx, gb * htick, ga * htick))
            continue
        if isinstance(event, TradeTick):
            out.prints += 1
            price = maker.price_f(event.price)
            sign = 1 if event.aggressor is Side.SELL else -1  # a seller hit our bid
            if quote_margin_bps is not None:
                quote = quote_at(rx)
                if quote is None:
                    out.no_book += 1
                    continue
                ours = quote[0] if sign > 0 else quote[1]
                if (price > ours + 1e-12) if sign > 0 else (price < ours - 1e-12):
                    continue  # the print did not reach our quote
                price = ours
            qty = min(maker.qty_f(event.qty), size_jpy / price)
            step = float(hedge.lot_size)
            qty = int(qty / step + 1e-9) * step
            if qty <= 0 or qty < hedge_min:
                out.too_small += 1
                continue
            heapq.heappush(tasks, (rx + lat, next(seq), "open", _Fill(sign, price, qty), 0))
        elif isinstance(event, (DepthSnapshot, DepthDelta)):
            bb.apply(event)
            bb_bid, bb_ask = bb.book.best_bid(), bb.book.best_ask()
            if quote_margin_bps is not None and bb_bid is not None and bb_ask is not None:
                bb_tops.append((rx, bb_bid, bb_ask))
    if out._open_notional:
        out.open_cost_bps /= out._open_notional
    if out._close_notional:
        out.close_cost_bps /= out._close_notional
    if first is not None:
        out.hours = (last - first) / NS / 3600
    return out


def slack_summary(results: list[HedgeResult], interim: bool = False) -> str:
    title = "途中経過" if interim else "判定"
    lines = [f":shield: ヘッジ込みの{title}（bitbank でメイカー → GMO で即ヘッジ）"]
    for r in results:
        h60 = r.horizons[60]
        verdict = "仮合格" if r.passed else ("判定保留（データ不足）" if r.hedged < 100 else "不合格")
        lines.append(
            f"  • {r.label}  {verdict}  60秒 {h60.bps():+.1f}bps（{h60.pnl:+,.0f}円）"
            f"  ヘッジ {r.hedged:,}回  GMO片道 {r.open_cost_bps:.1f}bps  録画 {r.hours:.1f}時間"
        )
    lines.append("  （上限の計算です。本番で勝てるとは限りません）")
    return "\n".join(lines)
