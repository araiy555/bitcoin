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
    "bps と円は、bitbank リベート・GMO 手数料・GMO の板食い込み・ヘッジまでの遅れ込みの最終損益。"
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
            hedge_min: float = 0.0, latency_ms: float = 200.0, depth: int = 100) -> HedgeResult:
    """`rows` yields (source, receive_ns, event), source "bitbank" or "gmo"."""
    bb, gm = MarketView(instrument=maker, depth=depth), MarketView(instrument=hedge, depth=depth)
    out = HedgeResult(label)
    tasks: list = []
    seq = itertools.count()
    first = last = None
    lat = int(latency_ms * 1e6)

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
            continue
        if isinstance(event, TradeTick):
            out.prints += 1
            price = maker.price_f(event.price)
            qty = min(maker.qty_f(event.qty), size_jpy / price)
            step = float(hedge.lot_size)
            qty = int(qty / step + 1e-9) * step
            if qty <= 0 or qty < hedge_min:
                out.too_small += 1
                continue
            sign = 1 if event.aggressor is Side.SELL else -1  # a seller hit our bid
            heapq.heappush(tasks, (rx + lat, next(seq), "open", _Fill(sign, price, qty), 0))
        elif isinstance(event, (DepthSnapshot, DepthDelta)):
            bb.apply(event)
    if out._open_notional:
        out.open_cost_bps /= out._open_notional
    if out._close_notional:
        out.close_cost_bps /= out._close_notional
    if first is not None:
        out.hours = (last - first) / NS / 3600
    return out


def slack_summary(results: list[HedgeResult]) -> str:
    lines = [":shield: ヘッジ込みの判定（bitbank でメイカー → GMO で即ヘッジ）"]
    for r in results:
        h60 = r.horizons[60]
        verdict = "仮合格" if r.passed else ("判定保留（データ不足）" if r.hedged < 100 else "不合格")
        lines.append(
            f"  • {r.label}  {verdict}  60秒 {h60.bps():+.1f}bps（{h60.pnl:+,.0f}円）"
            f"  ヘッジ {r.hedged:,}回  GMO片道 {r.open_cost_bps:.1f}bps  録画 {r.hours:.1f}時間"
        )
    lines.append("  （上限の計算です。本番で勝てるとは限りません）")
    return "\n".join(lines)
