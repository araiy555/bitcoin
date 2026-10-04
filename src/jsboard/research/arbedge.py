"""bitbank against GMO: trade only when their prices part by more than it costs.

Both legs are market orders priced by walking each venue's recorded book
for the order size, each venue's taker fee charged. Holding coin on one
venue and its short on the other cancels the price risk; what is left is
the gap between the two venues, which must open far enough to pay for two
round trips through two books.

A trade opens one way (buy the cheap venue, sell the dear one) when that
leaves at least `open_bps` after costs, and closes the other way once the
pair, opening and closing together, has made `target_bps`; after
`max_hold_s` it closes at whatever the books offer. One position at a time
per threshold. Results are kept per UTC day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.types import Instrument
from ..feed.base import DepthDelta, DepthSnapshot

NS = 1_000_000_000


class _Book:
    """Levels of one L2 book in plain dicts, best prices kept up to date."""

    __slots__ = ("asks", "best_ask", "best_bid", "bids")

    def __init__(self) -> None:
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.best_bid: int | None = None
        self.best_ask: int | None = None

    def apply(self, event) -> None:
        if isinstance(event, DepthSnapshot):
            self.bids = {p: q for p, q in event.bids if q > 0}
            self.asks = {p: q for p, q in event.asks if q > 0}
            self.best_bid = max(self.bids) if self.bids else None
            self.best_ask = min(self.asks) if self.asks else None
            return
        for price, qty in event.bids:
            if qty > 0:
                self.bids[price] = qty
                if self.best_bid is None or price > self.best_bid:
                    self.best_bid = price
            elif self.bids.pop(price, None) is not None and price == self.best_bid:
                self.best_bid = max(self.bids) if self.bids else None
        for price, qty in event.asks:
            if qty > 0:
                self.asks[price] = qty
                if self.best_ask is None or price < self.best_ask:
                    self.best_ask = price
            elif self.asks.pop(price, None) is not None and price == self.best_ask:
                self.best_ask = min(self.asks) if self.asks else None

    def take(self, side: int, qty: float, inst: Instrument) -> float | None:
        """Average yen price to buy (+1, walking asks) or sell (-1, walking
        bids) `qty` coins; None if the book is too thin."""
        levels = sorted(self.asks.items()) if side > 0 else sorted(self.bids.items(), reverse=True)
        left, cost = qty, 0.0
        for price, lots in levels:
            take = min(left, inst.qty_f(lots))
            cost += take * inst.price_f(price)
            left -= take
            if left <= 1e-12:
                return cost / qty
        return None


@dataclass
class Leg:
    n: int = 0
    wins: int = 0
    notional: float = 0.0
    pnl: float = 0.0
    forced: int = 0

    def bps(self) -> float:
        return self.pnl / self.notional * 1e4 if self.notional else float("nan")


@dataclass
class ArbResult:
    label: str
    thresholds: tuple = ()
    days: dict = field(default_factory=dict)
    """(day, threshold) -> Leg."""
    best_gap_bps: dict = field(default_factory=dict)
    """day -> the widest after-cost gap seen either way (bps)."""
    open_seconds: dict = field(default_factory=dict)
    """day -> seconds during which some direction paid after costs (> 0bps)."""

    def rows(self) -> list[str]:
        out = []
        seen = sorted(set(self.best_gap_bps) | {d for d, _ in self.days})
        for day, t in ((d, t) for d in seen for t in self.thresholds):
            leg = self.days.get((day, t), Leg())
            win = f"{leg.wins / leg.n:.0%}" if leg.n else "-"
            out.append("\t".join([
                self.label, day, f"{t:g}", f"{leg.n:,}", win,
                "-" if not leg.n else f"{leg.bps():+.1f}", f"{leg.pnl:+,.0f}", f"{leg.forced:,}",
                f"{self.best_gap_bps.get(day, float('nan')):+.1f}",
                f"{self.open_seconds.get(day, 0.0):,.0f}",
            ]))
        return out


HEADER = "\t".join([
    "銘柄", "日(UTC)", "開くしきいbps", "往復回数", "勝率", "1往復bps", "合計円",
    "時間切れ", "その日一番開いた差bps", "差がプラスだった秒数",
])
NOTE = (
    "両方とも成行。値段は録画した実際の板を注文の量ぶん食った値段、手数料は両取引所の実際の値。\n"
    "「差」は手数料と板の食い込みを引いたあとの、片方で買って片方で売ったときの取り分。\n"
    "片方に現金、もう片方にコイン（または売り建て）を置いておく前提。送金はしない。"
)


def analyse(rows, bb: Instrument, gm: Instrument, *, label: str, size_jpy: float,
            bb_fee_bps: float, gm_fee_bps: float, gm_min: float, thresholds,
            target_bps: float = 2.0, max_hold_s: float = 4 * 3600) -> ArbResult:
    """`rows` yields (source, receive_ns, event), source "bitbank" or "gmo"."""
    books = {"bitbank": _Book(), "gmo": _Book()}
    out = ArbResult(label, tuple(thresholds))
    # threshold -> None or the open trade
    held: dict = {t: None for t in thresholds}
    last_rx, last_positive = None, False

    def day_of(ns: int) -> str:
        return datetime.fromtimestamp(ns / NS, UTC).strftime("%Y-%m-%d")

    def quote(direction: int, qty: float):
        """direction +1: buy bitbank, sell GMO; -1: buy GMO, sell bitbank.
        Returns (bps after costs, buy price, sell price) or None."""
        b, g = books["bitbank"], books["gmo"]
        if direction > 0:
            buy, sell = b.take(1, qty, bb), g.take(-1, qty, gm)
            fee = bb_fee_bps * (buy or 0) + gm_fee_bps * (sell or 0)
        else:
            buy, sell = g.take(1, qty, gm), b.take(-1, qty, bb)
            fee = gm_fee_bps * (buy or 0) + bb_fee_bps * (sell or 0)
        if buy is None or sell is None:
            return None
        return ((sell - buy) - fee / 1e4) / buy * 1e4, buy, sell

    def top_bound(direction: int) -> float | None:
        """The after-fee gap at the touch: walking deeper only lowers it."""
        b, g = books["bitbank"], books["gmo"]
        if None in (b.best_bid, b.best_ask, g.best_bid, g.best_ask):
            return None
        if direction > 0:
            buy, sell = bb.price_f(b.best_ask), gm.price_f(g.best_bid)
            fees = bb_fee_bps + gm_fee_bps
        else:
            buy, sell = gm.price_f(g.best_ask), bb.price_f(b.best_bid)
            fees = bb_fee_bps + gm_fee_bps
        return (sell - buy) / buy * 1e4 - fees

    for src, rx, event in rows:
        if not isinstance(event, (DepthSnapshot, DepthDelta)) or src not in books:
            continue
        books[src].apply(event)
        day = day_of(rx)
        if last_rx is not None and last_positive:
            out.open_seconds[day] = out.open_seconds.get(day, 0.0) + (rx - last_rx) / NS
        last_rx = rx
        bounds = {d: top_bound(d) for d in (1, -1)}
        if None in bounds.values():
            last_positive = False
            continue
        last_positive = max(bounds.values()) > 0
        out.best_gap_bps[day] = max(out.best_gap_bps.get(day, -1e9), *bounds.values())

        b = books["bitbank"]
        mid = (b.best_bid + b.best_ask) / 2 * float(bb.tick_size)
        step = float(gm.lot_size)
        qty = int(size_jpy / mid / step + 1e-9) * step
        if qty <= 0 or qty < gm_min:
            continue

        for t in thresholds:
            trade = held[t]
            if trade is None:
                for d in (1, -1):
                    if bounds[d] < t:
                        continue
                    q = quote(d, qty)
                    if q is not None and q[0] >= t:
                        held[t] = {"dir": d, "edge": q[0], "notional": q[1] * qty,
                                   "at": rx, "day": day}
                        break
                continue
            back = -trade["dir"]
            need = target_bps - trade["edge"]
            timed_out = rx - trade["at"] >= max_hold_s * NS
            if bounds[back] < need and not timed_out:
                continue
            q = quote(back, qty)
            if q is None or (q[0] < need and not timed_out):
                continue
            total = trade["edge"] + q[0]
            leg = out.days.setdefault((trade["day"], t), Leg())
            leg.n += 1
            leg.wins += total > 0
            leg.notional += trade["notional"]
            leg.pnl += total / 1e4 * trade["notional"]
            leg.forced += timed_out and q[0] < need
            held[t] = None
    return out
