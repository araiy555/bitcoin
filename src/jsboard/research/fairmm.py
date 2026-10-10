"""Quote GMO around the fair price, replayed on a recording, conservatively.

The fair-price model (fairprice) predicts GMO's mid a few seconds ahead.
Taking on it at market came out at about zero; this asks the other
question, the one market makers answer: rest a bid and an ask around the
fair price and earn the spread. The answer depends on things a book
replay cannot see, so every one of them is estimated on the cautious side
and reported separately from anything measured:

- **Fills.** An order joins the back of its price level as it stood when
  the order landed. Only prints at that price move it forward; orders
  cancelled ahead of it do not (they may have, but that cannot be known).
  A print at a price worse than ours (a sweep through our level) fills us
  up to the print's size.
- **Latency.** A new order lands `place_ms` after it was sent; a cancel
  takes effect `cancel_ms` after it was asked for, and until then the old
  order can still be filled. Fills inside that window are counted apart:
  they are the ones a slow cancel gives away.
- **Adverse selection.** After each fill, GMO's mid 0.1, 0.5, 1 and 5 s
  later, against the fill price, in the fill's favour.
- **Profit.** Cash plus inventory marked at the mid at the end, and again
  with the inventory closed by crossing half the spread.

The quotes: fair = mid x (1 + shrink x prediction), less a skew per lot of
inventory; bid and ask `width_bps` either side of it, kept post-only, and
no quote on the side that would take inventory past `max_inv`. Every
combination of cancel latency and width runs in one pass.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

from ..core.types import Instrument, Side
from ..feed.base import DepthDelta, DepthSnapshot, TradeTick
from .fairprice import Grid

NS = 1_000_000_000
AFTER_S = (0.1, 0.5, 1.0, 5.0)


@dataclass
class Order:
    side: int            # +1 bid, -1 ask
    price: int           # ticks
    qty: float           # coins left
    active_at: int       # ns when it lands on the book
    cancel_at: int | None = None   # ns when a cancel was asked for
    dead_at: int | None = None     # ns when the cancel takes effect
    queue: float | None = None     # coins ahead of us at our price; None until it lands
    rejected: bool = False


@dataclass
class Book:
    """One quoting variant's account and its measurements."""
    cancel_ms: float
    width_bps: float
    orders: list = field(default_factory=list)
    inv: float = 0.0
    cash: float = 0.0
    placed: int = 0
    landed: int = 0
    rejected: int = 0
    fills: int = 0
    fill_qty: float = 0.0
    notional: float = 0.0
    edge_bps: float = 0.0
    """Sum over fills of how far inside the mid the fill was (bps, x qty)."""
    after: dict = field(default_factory=lambda: {s: [0, 0.0] for s in AFTER_S})
    late: int = 0
    """Fills that came while a cancel was on its way."""
    late_after: list = field(default_factory=lambda: [0, 0.0])
    max_inv: float = 0.0
    rebate: float = 0.0

    def working(self, side: int, now: int) -> Order | None:
        for o in self.orders:
            if o.side == side and o.cancel_at is None and o.qty > 1e-12 and not o.rejected:
                return o
        return None


@dataclass
class MMResult:
    label: str
    books: list
    hours: float = 0.0
    end_mid: float | None = None
    half_spread: float = 0.0


def run(rows, gmo: Instrument, leads: dict, model: dict, *, label: str, test: tuple[int, int],
        hold: str = "2", shrink: float = 0.5, widths=(0.5, 1.0, 2.0),
        cancel_ms=(200.0, 500.0, 1000.0), place_ms: float = 200.0, size: float = 10.0,
        max_inv: float = 30.0, skew_bps: float = 0.5, rebate_bps: float = 0.0,
        step_ms: int = 100, stale_s: float = 10.0) -> MMResult:
    """`rows` yields (source, receive_ns, event): "gmo" for GMO's book and
    prints, "bybit"/"binance"/"x:<name>" for the leads. `model` is a saved
    fairprice model; its weights for `hold` seconds price GMO."""
    weights = model["weights"][hold]
    grid = Grid(gmo, leads, model.get("cross", []), step_ms, float(model.get("hinge_bps", 8.0)))
    books = [Book(c, w) for c in cancel_ms for w in widths]
    out = MMResult(label, books)
    step = step_ms * 1_000_000
    tick = float(gmo.tick_size)
    place = int(place_ms * 1e6)
    stale = int(stale_s * NS)
    last_rx: dict = {}
    later: list = []   # (due ns, seq, book, side, fill price, qty, after s, late)
    seq = itertools.count()
    grid_t = None
    first = last = None

    def mid() -> float | None:
        return grid.mid("gmo")

    def level_qty(side: int, price: int) -> float:
        b = grid.books["gmo"]
        lots = (b.bids if side > 0 else b.asks).get(price, 0)
        return gmo.qty_f(lots)

    def land(bk: Book, o: Order) -> None:
        """The order reaches the book: refuse it if it would take (post-only),
        else join the back of its level."""
        b = grid.books["gmo"]
        no_book = b.best_bid is None or b.best_ask is None
        o.rejected = no_book or (o.price >= b.best_ask if o.side > 0 else o.price <= b.best_bid)
        if o.rejected:
            bk.rejected += 1
            return
        o.queue = level_qty(o.side, o.price)
        bk.landed += 1

    def fill(bk: Book, o: Order, qty: float, now: int) -> None:
        qty = min(qty, o.qty)
        if qty <= 1e-12:
            return
        o.qty -= qty
        price = gmo.price_f(o.price)
        bk.inv += o.side * qty
        bk.cash -= o.side * qty * price
        bk.rebate += qty * price * rebate_bps / 1e4
        bk.fills += 1
        bk.fill_qty += qty
        bk.notional += qty * price
        bk.max_inv = max(bk.max_inv, abs(bk.inv))
        m = mid()
        if m:
            bk.edge_bps += o.side * (m - price) / price * 1e4 * qty
        is_late = o.cancel_at is not None and now >= o.cancel_at
        bk.late += is_late
        for s in AFTER_S:
            heapq.heappush(later, (now + int(s * NS), next(seq), bk, o.side, price, qty, s, is_late))

    def settle_due(now: int) -> None:
        m = mid()
        while later and later[0][0] <= now:
            _, _, bk, side, price, qty, s, is_late = heapq.heappop(later)
            if not m:
                continue
            move = side * (m - price) / price * 1e4
            cell = bk.after[s]
            cell[0] += qty
            cell[1] += move * qty
            if is_late and s == 1.0:
                bk.late_after[0] += qty
                bk.late_after[1] += move * qty

    def tidy(bk: Book, now: int) -> None:
        for o in bk.orders:
            if o.queue is None and not o.rejected and now >= o.active_at:
                land(bk, o)
        bk.orders = [o for o in bk.orders
                     if not o.rejected and o.qty > 1e-12 and (o.dead_at is None or now < o.dead_at)]

    def on_trade(t: TradeTick, now: int) -> None:
        # A taker sell hits bids (ours if at or above the print), a taker
        # buy lifts asks.
        hit = 1 if t.aggressor is Side.SELL else -1
        qty = gmo.qty_f(t.qty)
        for bk in books:
            tidy(bk, now)
            for o in bk.orders:
                if o.side != hit or o.queue is None:
                    continue
                if (hit > 0 and t.price < o.price) or (hit < 0 and t.price > o.price):
                    fill(bk, o, qty, now)            # swept through our level
                elif t.price == o.price:
                    o.queue -= qty
                    if o.queue < 0:
                        fill(bk, o, -o.queue, now)
                        o.queue = 0.0

    def on_grid(t: int) -> None:
        settle_due(t)
        if t - last_rx.get("gmo", -stale - 1) > stale or not any(
                t - last_rx.get(k, -stale - 1) <= stale for k in ("bybit", "binance")):
            for ser in grid.series.values():
                ser.push(None)
            if test[0] <= t < test[1]:
                for bk in books:   # a feed is down: take the quotes away
                    requote_one(bk, t, None)
            return
        grid.push()
        gm = mid()
        if gm is None or not grid.ready():
            return
        x = grid.features(gm)
        if not test[0] <= t < test[1]:
            return
        p = sum(w * v for w, v in zip(weights, x, strict=True))
        fair = gm * (1 + shrink * p / 1e4)
        for bk in books:
            # Lean against the inventory: a long book quotes lower to sell it.
            requote_one(bk, t, fair * (1 - skew_bps * bk.inv / size / 1e4))

    def requote_one(bk: Book, now: int, fair: float | None) -> None:
        """Move each side's quote to where `fair` puts it: cancel the old
        one (it lives on until the cancel lands) and send the new one.
        No fair price: take both sides away."""
        tidy(bk, now)
        cancel = int(bk.cancel_ms * 1e6)
        for side in (1, -1):
            o = bk.working(side, now)
            want = None
            if fair is not None and not (side * bk.inv >= max_inv - 1e-9):
                raw = fair * (1 - side * bk.width_bps / 1e4)
                want = math.floor(raw / tick + 1e-9) if side > 0 else math.ceil(raw / tick - 1e-9)
            if o is not None and o.price == want:
                continue
            if o is not None:
                o.cancel_at, o.dead_at = now, now + cancel
            if want is not None:
                bk.orders.append(Order(side, want, size, now + place))
                bk.placed += 1

    for src, rx, event in rows:
        if src not in grid.books:
            continue
        if grid_t is None:
            grid_t = (rx // step + 1) * step
        if rx - grid_t > 60 * NS:
            skip = (rx - grid_t) // step
            grid_t += skip * step
            grid.basis_avg.clear()
            for ser in grid.series.values():
                ser.vals.clear()
            for bk in books:   # nothing rests through a gap in the recording
                bk.orders.clear()
        while grid_t <= rx:
            on_grid(grid_t)
            grid_t += step
        if isinstance(event, TradeTick):
            if src == "gmo" and test[0] <= rx < test[1]:
                on_trade(event, rx)
            continue
        if not isinstance(event, (DepthSnapshot, DepthDelta)):
            continue
        grid.apply(src, event)
        last_rx[src] = rx
        if src == "gmo" and test[0] <= rx < test[1]:
            first = rx if first is None else first
            last = rx
    if first is not None:
        out.hours = (last - first) / NS / 3600
    out.end_mid = mid()
    b = grid.books["gmo"]
    if b.best_bid is not None and b.best_ask is not None:
        out.half_spread = (gmo.price_f(b.best_ask) - gmo.price_f(b.best_bid)) / 2
    return out


HEADER = "\t".join([
    "取消の遅れ", "指値の幅", "出した指値", "板に載った", "約定（推定）", "1時間あたり", "最大在庫",
    "約定時の取り分 bps", "0.1秒後", "0.5秒後", "1秒後", "5秒後",
    "取消中の約定", "その1秒後", "損益 円", "1約定あたり bps", "在庫を閉じたら 円",
])


def rows_of(r: MMResult) -> list[str]:
    def f(x: float) -> str:
        return "-" if x != x else f"{x:+.2f}"

    lines = []
    for bk in r.books:
        mark = bk.cash + bk.rebate + (bk.inv * r.end_mid if r.end_mid else 0.0)
        flat = mark - abs(bk.inv) * r.half_spread
        per = mark / bk.notional * 1e4 if bk.notional else float("nan")
        after = [f(c[1] / c[0]) if c[0] else "-" for c in (bk.after[s] for s in AFTER_S)]
        late = bk.late_after
        lines.append("\t".join([
            f"{bk.cancel_ms:g}ms", f"{bk.width_bps:g}bps", f"{bk.placed:,}", f"{bk.landed:,}",
            f"{bk.fills:,}", f"{bk.fills / r.hours:.1f}" if r.hours else "-", f"{bk.max_inv:g}",
            f(bk.edge_bps / bk.fill_qty) if bk.fill_qty else "-", *after,
            f"{bk.late:,}", f(late[1] / late[0]) if late[0] else "-",
            f"{mark:+,.1f}", f(per), f"{flat:+,.1f}",
        ]))
    return lines


NOTE = (
    "約定は推定です（本物の注文ではありません）。指値は板に載った時点の同じ値段の行列の最後に並び、"
    "その値段の約定でだけ前に進みます（前の注文の取消では進めない＝保守的）。\n"
    "0.1〜5秒後 = 約定後の GMO の中値の動きを、約定値段から自分に有利な向きをプラスで（マイナスが逆選択）。\n"
    "取消中の約定 = 取消を出してから実際に消えるまでに約定した回数。損益は現金＋在庫を最後の中値で評価。"
)


def slack_summary(r: MMResult) -> str:
    lines = [f":scales: フェア価格で指値（{r.label}、{r.hours:.1f}時間、約定は推定・注文なし）"]
    for bk in r.books:
        mark = bk.cash + bk.rebate + (bk.inv * r.end_mid if r.end_mid else 0.0)
        per = mark / bk.notional * 1e4 if bk.notional else float("nan")
        one = bk.after[1.0]
        adverse = f"{one[1] / one[0]:+.2f}" if one[0] else "-"
        lines.append(f"  • 取消 {bk.cancel_ms:g}ms・幅 {bk.width_bps:g}bps: 約定 {bk.fills}回  "
                     f"損益 {mark:+,.1f}円（{per:+.2f}bps）  1秒後 {adverse}bps  取消中の約定 {bk.late}回")
    lines.append("  （約定は保守的な推定。本物の約定率・取消の速さは実注文でしか分かりません）")
    return "\n".join(lines)
