"""GMO's fair price from many inputs at once, against the one-signal rule.

The follow rule trades GMO when one lead market has moved and GMO has not.
This puts everything recorded into one linear model instead, and asks it
for GMO's move over the next few seconds:

- the same coin's perp on Bybit and on Binance: returns over 0.1, 0.5 and
  1 s, and how far GMO's price sits from each of them against its own
  last minute (GMO's premium over the perp drifts with the yen, so only
  its deviation from that drift says anything about the next seconds);
- other coins' Binance perps (BTC, ETH): returns over 0.5 and 1 s;
- GMO itself: returns over 0.1, 0.5 and 1 s, the imbalance of size at
  the touch and over five levels, and the spread.

Everything is sampled on a fixed grid (every `step_ms`). The target is
GMO's mid from `latency_ms` after the sample (when an order would land)
to `h` seconds after that, so a move GMO makes before our order arrives
is not counted as ours.

The model is fitted on one period and traded on a later one it has never
seen. A trade is taken only when the predicted move is larger than the
whole round trip as it stands at that moment: both sides of GMO's book
walked for the order size (spread and depth), plus the taker fee each
way, plus a margin. The trade itself is then priced on the book as it is
when the order lands and when it leaves, so what the latency costs is in
the result, not assumed.

The one-signal rule is run on the same later period with the same
execution, so the two are compared on the same footing.

Memory stays flat: the fit keeps only XᵀX and Xᵀy, and the trading pass
holds a few seconds of grid.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field

from ..core.types import Instrument
from ..feed.base import DepthDelta, DepthSnapshot
from .arbedge import _Book

NS = 1_000_000_000
LAGS = (1, 5, 10)  # grid steps: 0.1, 0.5, 1 s at the default step


def feature_names(cross) -> list[str]:
    names = ["定数"]
    for lead in ("bybit", "binance"):
        names += [f"{lead} {x / 10:g}秒" for x in LAGS] + [f"{lead} との乖離"]
    names += [f"GMO {x / 10:g}秒" for x in LAGS]
    names += ["GMO 板の偏り(最良)", "GMO 板の偏り(5段)", "GMO スプレッド"]
    for c in cross:
        names += [f"{c} 0.5秒", f"{c} 1秒"]
    return names


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """a x = b by Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            continue
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col and m[r][col]:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


@dataclass
class Book:
    """Trades taken by one rule at one hold."""
    n: int = 0
    wins: int = 0
    notional: float = 0.0
    pnl: float = 0.0

    def bps(self) -> float:
        return self.pnl / self.notional * 1e4 if self.notional else float("nan")


@dataclass
class Fit:
    k: int
    holds: tuple
    xtx: list = field(default_factory=list)
    xty: dict = field(default_factory=dict)
    yy: dict = field(default_factory=dict)
    ysum: dict = field(default_factory=dict)
    n: int = 0

    def __post_init__(self) -> None:
        self.xtx = [[0.0] * self.k for _ in range(self.k)]
        self.xty = {h: [0.0] * self.k for h in self.holds}
        self.yy = {h: 0.0 for h in self.holds}
        self.ysum = {h: 0.0 for h in self.holds}

    def add(self, x: list[float], ys: dict) -> None:
        self.n += 1
        for i, xi in enumerate(x):
            if xi:
                row = self.xtx[i]
                for j, xj in enumerate(x):
                    row[j] += xi * xj
        for h, y in ys.items():
            v = self.xty[h]
            for i, xi in enumerate(x):
                v[i] += xi * y
            self.yy[h] += y * y
            self.ysum[h] += y

    def weights(self, ridge: float = 1e-3) -> dict:
        lam = ridge * max(self.n, 1)
        a = [[v + (lam if i == j and i else 0.0) for j, v in enumerate(row)]
             for i, row in enumerate(self.xtx)]
        return {h: solve(a, self.xty[h]) for h in self.holds}


@dataclass
class Score:
    """Out-of-sample fit of one hold."""
    n: int = 0
    sse: float = 0.0
    sst: float = 0.0
    ysum: float = 0.0
    hit: int = 0
    called: int = 0

    def r2(self) -> float:
        if not self.n:
            return float("nan")
        mean = self.ysum / self.n
        sst = self.sst - self.n * mean * mean
        return 1 - self.sse / sst if sst > 0 else float("nan")


@dataclass
class FairResult:
    label: str
    names: list
    holds: tuple
    weights: dict = field(default_factory=dict)
    train_n: int = 0
    test_n: int = 0
    score: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    rule: dict = field(default_factory=dict)
    cal: dict = field(default_factory=dict)
    """(margin, hold) -> Book, traded on the last part of the training period."""
    chosen: dict = field(default_factory=dict)
    """hold -> the margin picked on that part and used on the test."""
    cal_n: int = 0


class _Series:
    """One price on the grid: the last `keep` grid values."""

    def __init__(self, keep: int) -> None:
        self.vals: deque = deque(maxlen=keep)

    def push(self, v: float | None) -> None:
        self.vals.append(v)

    def ret(self, lag: int) -> float:
        if len(self.vals) <= lag:
            return 0.0
        now, then = self.vals[-1], self.vals[-1 - lag]
        if not now or not then:
            return 0.0
        return math.log(now / then) * 1e4


class Grid:
    """The model's inputs, kept current from the books and read every grid
    step. Shared by the replay and the live run, so both compute the very
    same features."""

    def __init__(self, gmo: Instrument, leads: dict, cross=(), step_ms: int = 100) -> None:
        self.gmo = gmo
        self.cross = tuple(cross)
        self.books = {"gmo": _Book(), "bybit": _Book(), "binance": _Book()}
        self.books.update({f"x:{c}": _Book() for c in self.cross})
        self.series = {k: _Series(max(LAGS) + 1) for k in self.books}
        self.insts = {"gmo": gmo, **leads}
        self.basis_avg: dict = {}
        self.alpha = step_ms / 60_000  # one-minute average of the premium

    def apply(self, src: str, event) -> None:
        if src in self.books:
            self.books[src].apply(event)

    def mid(self, key: str) -> float | None:
        b, inst = self.books[key], self.insts.get(key)
        if b.best_bid is None or b.best_ask is None or inst is None:
            return None
        return (b.best_bid + b.best_ask) / 2 * float(inst.tick_size)

    def imbalance(self, levels: int) -> float:
        b = self.books["gmo"]
        bids = heapq.nlargest(levels, b.bids.items())
        asks = heapq.nsmallest(levels, b.asks.items())
        bq, aq = sum(q for _, q in bids), sum(q for _, q in asks)
        return (bq - aq) / (bq + aq) if bq + aq else 0.0

    def push(self) -> None:
        """Take this grid step's mids."""
        for key, s in self.series.items():
            s.push(self.mid(key))

    def ready(self) -> bool:
        return self.mid("gmo") is not None and (
            self.mid("bybit") is not None or self.mid("binance") is not None)

    def features(self, gmo_mid: float) -> list[float]:
        x = [1.0]
        for lead in ("bybit", "binance"):
            s = self.series[lead]
            x += [s.ret(lag) for lag in LAGS]
            lm = self.mid(lead)
            if lm:
                basis = math.log(gmo_mid / lm) * 1e4
                avg = self.basis_avg.get(lead)
                self.basis_avg[lead] = basis if avg is None else avg + self.alpha * (basis - avg)
                x.append(basis - self.basis_avg[lead])
            else:
                x.append(0.0)
        x += [self.series["gmo"].ret(lag) for lag in LAGS]
        b = self.books["gmo"]
        gmo = self.gmo
        spread = (gmo.price_f(b.best_ask) - gmo.price_f(b.best_bid)) / gmo_mid * 1e4
        x += [self.imbalance(1), self.imbalance(5), spread]
        for c in self.cross:
            s = self.series[f"x:{c}"]
            x += [s.ret(5), s.ret(10)]
        return x


def run(rows, gmo: Instrument, leads: dict, *, label: str, train: tuple[int, int],
        test: tuple[int, int],
        holds=(2.0, 5.0, 10.0), cross=(), step_ms: int = 100, latency_ms: int = 200,
        size_jpy: float = 10_000.0, fee_bps: float = 0.0, min_order: float = 0.0,
        margin_bps: float | None = None, rule_bps: float = 8.0, ridge: float = 1e-3,
        margins=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0), cal_frac: float = 0.3,
        min_cal_trades: int = 20) -> FairResult:
    """`rows` yields (source, receive_ns, event) in time order. Sources:
    "gmo" (GMO's book), "bybit" and "binance" (the same coin's perps), and
    "x:<name>" for another coin's perp; `leads` maps each of those sources
    to its Instrument. `train` and `test` are
    [start, end) receive times in ns; test must start after train ends.

    How far the prediction must clear the round trip before a trade is
    taken (the margin) is chosen inside `train`, never on `test`: the
    model is fitted on the first part of `train`, traded at every margin
    in `margins` on its last `cal_frac`, and the margin that made the most
    there (with at least `min_cal_trades` trades; else the largest) is the
    one used on `test`. A fixed `margin_bps` skips that and fits on all
    of `train`."""
    names = feature_names(cross)
    out = FairResult(label, names, tuple(holds))
    step = step_ms * 1_000_000
    lat = max(1, round(latency_ms / step_ms))
    hsteps = {h: max(1, round(h * 1000 / step_ms)) for h in holds}
    longest = lat + max(hsteps.values())
    grid = Grid(gmo, leads, cross, step_ms)
    books, series, mid = grid.books, grid.series, grid.mid

    fit = Fit(len(names), tuple(holds))
    weights: dict | None = None
    pending: deque = deque()   # (grid index, features, phase) awaiting their target
    gmo_mids: deque = deque()  # GMO's mid at each grid index, oldest first
    tasks: dict = {}           # grid index -> list of actions to run then
    busy = {(kind, h): -1 for kind in ("model", "rule") for h in holds}
    armed = True
    out.score = {h: Score() for h in holds}
    out.model = {h: Book() for h in holds}
    out.rule = {h: Book() for h in holds}
    calibrate = margin_bps is None
    # Set from the first sample actually inside `train`: a window opened
    # before the recording starts would otherwise leave nothing to fit on.
    cal_start = train[1]
    out.cal = {(m, h): Book() for m in margins for h in holds} if calibrate else {}
    out.chosen = {h: margin_bps for h in holds} if not calibrate else {}
    busy.update({("cal", m, h): -1 for m in margins for h in holds})
    grid_t = None
    g = 0

    def qty_for(price: float) -> float:
        lot = float(gmo.lot_size)
        q = math.floor(size_jpy / price / lot + 1e-9) * lot
        if q < min_order:
            q = math.ceil(min_order / lot - 1e-9) * lot
        return q

    def round_trip_bps(gmo_mid: float, qty: float) -> float | None:
        b = books["gmo"]
        buy, sell = b.take(1, qty, gmo), b.take(-1, qty, gmo)
        if buy is None or sell is None:
            return None
        return (buy - sell) / gmo_mid * 1e4 + 2 * fee_bps

    def schedule(at: int, action) -> None:
        tasks.setdefault(at, []).append(action)

    def open_trade(book: Book, h: float, side: int, qty: float) -> None:
        def enter() -> None:
            price = books["gmo"].take(side, qty, gmo)
            if price is None:
                return

            def leave() -> None:
                back = books["gmo"].take(-side, qty, gmo)
                if back is None:
                    return
                pnl = side * qty * (back - price) - (price + back) * qty * fee_bps / 1e4
                book.n += 1
                book.wins += pnl > 0
                book.notional += price * qty
                book.pnl += pnl

            schedule(g + hsteps[h], leave)
        schedule(g + lat, enter)

    def trade(x: list[float], gm: float, slots) -> None:
        """Take each (margin, hold) slot whose prediction clears the round
        trip by the margin; `slots` are (margin, hold, busy key, book)."""
        qty = qty_for(gm)
        cost = None
        preds = {h: sum(w * v for w, v in zip(weights[h], x, strict=True)) for h in holds}
        for m, h, key, book in slots:
            p = preds[h]
            if g < busy[key] or abs(p) <= m:
                continue
            # Walking the book costs a sort, so only when a trade is in reach.
            cost = round_trip_bps(gm, qty) if cost is None else cost
            if cost is not None and abs(p) > cost + m:
                busy[key] = g + lat + hsteps[h]
                open_trade(book, h, 1 if p > 0 else -1, qty)

    def on_grid(t: int) -> None:
        nonlocal weights, armed, cal_start
        for action in tasks.pop(g, ()):
            action()
        grid.push()
        gm = mid("gmo")
        gmo_mids.append(gm)
        while len(gmo_mids) > longest + 2:
            gmo_mids.popleft()
        base = g - len(gmo_mids) + 1

        def mid_at(i: int) -> float | None:
            return gmo_mids[i - base] if 0 <= i - base < len(gmo_mids) else None

        # Targets that have come due, from the grid kept so far.
        while pending and pending[0][0] + longest <= g:
            g0, x, phase = pending.popleft()
            start = mid_at(g0 + lat)
            ys = {}
            for h, hs in hsteps.items():
                end = mid_at(g0 + lat + hs)
                if start and end:
                    ys[h] = math.log(end / start) * 1e4
            if len(ys) != len(hsteps):
                continue
            if phase == "train":
                fit.add(x, ys)
            elif phase == "cal":
                continue
            elif weights is not None:
                for h, y in ys.items():
                    p = sum(w * v for w, v in zip(weights[h], x, strict=True))
                    sc = out.score[h]
                    sc.n += 1
                    sc.sse += (y - p) ** 2
                    sc.sst += y * y
                    sc.ysum += y
                    if abs(p) > 1e-9 and abs(y) > 1e-9:
                        sc.called += 1
                        sc.hit += (p > 0) == (y > 0)
        if gm is None or mid("bybit") is None and mid("binance") is None:
            return
        x = grid.features(gm)
        if calibrate and not out.train_n and train[0] <= t < train[1]:
            cal_start = t + int((train[1] - t) * (1 - cal_frac))
        if train[0] <= t < cal_start:
            pending.append((g, x, "train"))
            out.train_n += 1
            return
        if weights is None and (cal_start <= t < train[1] or test[0] <= t < test[1]):
            weights = fit.weights(ridge)
            out.weights = weights
        if cal_start <= t < train[1]:
            out.cal_n += 1
            trade(x, gm, [(m, h, ("cal", m, h), out.cal[(m, h)]) for m in margins for h in holds])
            return
        if not test[0] <= t < test[1]:
            return
        if not out.chosen:
            for h in holds:
                tried = [(m, out.cal[(m, h)]) for m in margins]
                enough = [(m, b) for m, b in tried if b.n >= min_cal_trades]
                # A tie goes to the larger margin: fewer trades for the same money.
                out.chosen[h] = (max(enough, key=lambda mb: (mb[1].pnl, mb[0]))[0]
                                 if enough else max(margins))
        pending.append((g, x, "test"))
        out.test_n += 1
        trade(x, gm, [(out.chosen[h], h, ("model", h), out.model[h]) for h in holds])
        qty = qty_for(gm)
        # The one-signal rule on the same grid: a lead moved `rule_bps`
        # more than GMO over the last second, the lead itself that far.
        leads = [series[k].ret(10) for k in ("bybit", "binance") if mid(k)]
        if not leads:
            return
        lead_move = sum(leads) / len(leads)
        gap = lead_move - series["gmo"].ret(10)
        if abs(gap) < rule_bps or abs(lead_move) < rule_bps or lead_move * gap <= 0:
            armed = True
            return
        if not armed:
            return
        armed = False
        for h in holds:
            if g >= busy[("rule", h)]:
                busy[("rule", h)] = g + lat + hsteps[h]
                open_trade(out.rule[h], h, 1 if gap > 0 else -1, qty)

    for src, rx, event in rows:
        if src not in books or not isinstance(event, (DepthSnapshot, DepthDelta)):
            continue
        if grid_t is None:
            grid_t = (rx // step + 1) * step
        while grid_t <= rx:
            on_grid(grid_t)
            grid_t += step
            g += 1
        books[src].apply(event)
    if weights is None:
        out.weights = fit.weights(ridge)
    if calibrate and not out.chosen and out.cal:
        # No test period seen (fitting a model to use live): pick the margin
        # from the training period all the same.
        for h in holds:
            enough = [(m, out.cal[(m, h)]) for m in margins if out.cal[(m, h)].n >= min_cal_trades]
            out.chosen[h] = (max(enough, key=lambda mb: (mb[1].pnl, mb[0]))[0]
                             if enough else max(margins))
    return out


def model_dict(r: FairResult, *, cross, step_ms: int, latency_ms: int, train) -> dict:
    """What the live run needs to price GMO exactly as the replay did."""
    return {
        "symbol": r.label, "names": r.names, "cross": list(cross), "step_ms": step_ms,
        "latency_ms": latency_ms, "train": list(train),
        "weights": {f"{h:g}": w for h, w in r.weights.items()},
        "margin_bps": {f"{h:g}": m for h, m in r.chosen.items()},
        "calibration": {f"{m:g}/{h:g}": [b.n, round(b.pnl, 2)] for (m, h), b in r.cal.items()},
    }


HEADER = "\t".join(["銘柄", "持つ秒", "未知データでの説明力 R²", "向きの的中", "やり方",
                    "取引数", "1回あたり bps", "勝率", "合計円"])


def report(r: FairResult) -> list[str]:
    lines = []
    for h in r.holds:
        sc = r.score[h]
        hit = f"{sc.hit / sc.called:.0%}" if sc.called else "-"
        for kind, books in (("フェア価格モデル", r.model), ("単純な後追い", r.rule)):
            b = books[h]
            lines.append("\t".join([
                r.label, f"{h:g}", f"{sc.r2():+.3f}", hit, kind, f"{b.n:,}",
                "-" if not b.n else f"{b.bps():+.2f}",
                "-" if not b.n else f"{b.wins / b.n:.0%}", f"{b.pnl:+,.0f}",
            ]))
    return lines


def calibration_text(r: FairResult) -> list[str]:
    """What each margin made on the end of the training period, and the pick."""
    if not r.cal:
        return [f"  入る基準（固定）: 往復コスト + {r.chosen.get(h, 0):g}bps" for h in r.holds[:1]]
    margins = sorted({m for m, _ in r.cal})
    out = [f"  入る基準の決め方: 学習期間の最後の部分（{r.cal_n:,} 点）で、上乗せ幅ごとに売買した結果"]
    for h in r.holds:
        cells = []
        for m in margins:
            b = r.cal[(m, h)]
            cells.append(f"+{m:g}bps: {b.n}回 {b.pnl:+,.0f}円" if b.n else f"+{m:g}bps: 0回")
        out.append(f"   {h:g}秒  " + " / ".join(cells) + f"  → 選んだのは +{r.chosen.get(h, 0):g}bps")
    return out


def weights_text(r: FairResult, h: float) -> list[str]:
    w = r.weights.get(h) or []
    return [f"  {name}: {v:+.4f}" for name, v in zip(r.names, w, strict=False)]


def slack_summary(results: list[FairResult]) -> str:
    lines = [":brain: フェア価格モデル vs 単純な後追い（学習に使っていない時間帯で、実際の板・手数料込み）"]
    for r in results:
        for h in r.holds:
            m, s, sc = r.model[h], r.rule[h], r.score[h]

            def cell(b: Book) -> str:
                return "取引なし" if not b.n else f"{b.bps():+.1f}bps {b.n}回 勝率{b.wins / b.n:.0%} {b.pnl:+,.0f}円"

            lines.append(f"  • {r.label} {h:g}秒  モデル（コスト+{r.chosen.get(h, 0):g}bps で入る）: {cell(m)}"
                         f"  /  後追い: {cell(s)}  （R² {sc.r2():+.3f}）")
    lines.append("  （モデルと入る基準は前の時間帯だけで決め、この時間帯では一度も調整していません）")
    return "\n".join(lines)
