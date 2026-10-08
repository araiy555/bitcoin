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


def feature_names(cross, hinge_bps: float = 8.0) -> list[str]:
    names = ["定数"]
    for lead in ("bybit", "binance"):
        names += [f"{lead} {x / 10:g}秒" for x in LAGS] + [f"{lead} との乖離"]
        names.append(f"{lead} が GMO より1秒で先に動いた分の {hinge_bps:g}bps 超え")
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
    """hold -> the margin picked on that part and used on the test; None
    when no margin made money there, and then the model does not trade."""
    cal_n: int = 0
    min_gap: float = 0.0
    """Fitted and traded only where a lead was this far ahead of GMO."""
    variants: dict = field(default_factory=dict)
    """min_gap -> FairResult, for every subset fitted; this one is the pick."""


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

    def __init__(self, gmo: Instrument, leads: dict, cross=(), step_ms: int = 100,
                 hinge_bps: float = 8.0) -> None:
        self.gmo = gmo
        self.hinge = hinge_bps
        self.gap = 0.0
        """The largest |lead 1 s move - GMO 1 s move| at the last features() call."""
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
        gmo_1s = self.series["gmo"].ret(10)
        self.gap = 0.0
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
            # How far the lead got ahead of GMO over the last second is
            # already in the two 1 s returns, linearly. What a straight line
            # cannot say is "only beyond 8bps does it really pay": this is
            # the part of the lead beyond the hinge, signed, zero below it.
            gap = s.ret(10) - gmo_1s if lm else 0.0
            self.gap = max(self.gap, abs(gap))
            x.append(math.copysign(max(abs(gap) - self.hinge, 0.0), gap))
        x += [self.series["gmo"].ret(lag) for lag in LAGS]
        b = self.books["gmo"]
        gmo = self.gmo
        spread = (gmo.price_f(b.best_ask) - gmo.price_f(b.best_bid)) / gmo_mid * 1e4
        x += [self.imbalance(1), self.imbalance(5), spread]
        for c in self.cross:
            s = self.series[f"x:{c}"]
            x += [s.ret(5), s.ret(10)]
        return x


class _Variant:
    """One training subset: its fit, its weights, its trades."""

    def __init__(self, gap: float, out: FairResult, k: int, holds: tuple) -> None:
        self.gap = gap
        self.out = out
        self.fit = Fit(k, holds)
        self.weights: dict | None = None


def _pick_margins(cal: dict, margins, holds, min_trades: int) -> dict:
    """hold -> the margin that made the most on the calibration part, with
    at least `min_trades` trades (a tie to the larger margin); None when
    none made money, so the model stands aside rather than lose least."""
    chosen = {}
    for h in holds:
        enough = [(m, cal[(m, h)]) for m in margins if cal[(m, h)].n >= min_trades]
        best = max(enough, key=lambda mb: (mb[1].pnl, mb[0])) if enough else None
        chosen[h] = best[0] if best and best[1].pnl > 0 else None
    return chosen


def run(rows, gmo: Instrument, leads: dict, *, label: str, train: tuple[int, int],
        test: tuple[int, int],
        holds=(2.0, 5.0, 10.0), cross=(), step_ms: int = 100, latency_ms: int = 200,
        size_jpy: float = 10_000.0, fee_bps: float = 0.0, min_order: float = 0.0,
        margin_bps: float | None = None, rule_bps: float = 8.0, ridge: float = 1e-3,
        margins=(0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0), cal_frac: float = 0.3,
        min_cal_trades: int = 20, cal_start_ns: int | None = None,
        stale_s: float = 10.0, min_gaps=(0.0,), hinge_bps: float = 8.0) -> FairResult:
    """`rows` yields (source, receive_ns, event) in time order. Sources:
    "gmo" (GMO's book), "bybit" and "binance" (the same coin's perps), and
    "x:<name>" for another coin's perp; `leads` maps each of those sources
    to its Instrument. `train` and `test` are
    [start, end) receive times in ns; test must start after train ends.

    How far the prediction must clear the round trip before a trade is
    taken (the margin) is chosen inside `train`, never on `test`: the
    model is fitted on the first part of `train`, traded at every margin
    in `margins` on its last `cal_frac`, and the margin that made the most
    there (with at least `min_cal_trades` trades) is the one used on
    `test`; if none made money, the model does not trade at that hold. A
    fixed `margin_bps` skips that and fits on all of `train`.
    `cal_start_ns` sets where that last part begins; the caller knows when
    the recording was actually running, and a split by clock time alone put
    the whole calibration inside a two-day gap once.

    `min_gaps` fits one model per value, each on (and trading only at) the
    moments a lead was at least that many bps ahead of GMO over the last
    second; 0 is every moment. Which one is returned is decided on the
    calibration part alone: the most money there at the longest hold. The
    others are in `variants`.

    Grid steps where GMO, or every lead, has sent nothing for `stale_s`
    are not sampled: a stopped recording is not a still market. Across a
    gap longer than a minute the grid jumps ahead and starts afresh."""
    names = feature_names(cross, hinge_bps)
    rule_out = {h: Book() for h in holds}
    variants = []
    for v in min_gaps:
        o = FairResult(label, names, tuple(holds), min_gap=float(v))
        o.score = {h: Score() for h in holds}
        o.model = {h: Book() for h in holds}
        o.rule = rule_out
        variants.append(_Variant(float(v), o, len(names), tuple(holds)))
    step = step_ms * 1_000_000
    lat = max(1, round(latency_ms / step_ms))
    hsteps = {h: max(1, round(h * 1000 / step_ms)) for h in holds}
    longest = lat + max(hsteps.values())
    grid = Grid(gmo, leads, cross, step_ms, hinge_bps)
    books, series, mid = grid.books, grid.series, grid.mid

    pending: deque = deque()   # (grid index, features, gap, phase) awaiting their target
    gmo_mids: deque = deque()  # GMO's mid at each grid index, oldest first
    tasks: dict = {}           # grid index -> list of actions to run then
    busy: dict = {("rule", h): -1 for h in holds}
    armed = True
    calibrate = margin_bps is None
    cal_start = cal_start_ns if (calibrate and cal_start_ns is not None) else train[1]
    last_rx: dict = {}
    stale = int(stale_s * NS)
    for var in variants:
        var.out.cal = {(m, h): Book() for m in margins for h in holds} if calibrate else {}
        var.out.chosen = {h: margin_bps for h in holds} if not calibrate else {}
        busy.update({(var.gap, "cal", m, h): -1 for m in margins for h in holds})
        busy.update({(var.gap, "model", h): -1 for h in holds})
    train_n = 0
    picked = False
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

    def trade(x: list[float], gm: float, var: _Variant, slots, cost: list) -> None:
        """Take each (margin, hold) slot whose prediction clears the round
        trip by the margin; `slots` are (margin, hold, busy key, book)."""
        qty = qty_for(gm)
        preds = {h: sum(w * v for w, v in zip(var.weights[h], x, strict=True)) for h in holds}
        for m, h, key, book in slots:
            p = preds[h]
            if m is None or g < busy[key] or abs(p) <= m:
                continue
            # Walking the book costs a sort, so only when a trade is in reach.
            if not cost:
                cost.append(round_trip_bps(gm, qty))
            if cost[0] is not None and abs(p) > cost[0] + m:
                busy[key] = g + lat + hsteps[h]
                open_trade(book, h, 1 if p > 0 else -1, qty)

    def pick_variant() -> None:
        nonlocal picked
        picked = True
        for var in variants:
            if not var.out.chosen:
                var.out.chosen = _pick_margins(var.out.cal, margins, holds, min_cal_trades)

    def on_grid(t: int) -> None:
        nonlocal armed, cal_start, train_n
        for action in tasks.pop(g, ()):
            action()
        live_leads = [k for k in ("bybit", "binance") if t - last_rx.get(k, -stale - 1) <= stale]
        if t - last_rx.get("gmo", -stale - 1) > stale or not live_leads:
            # The recording stopped, or a feed did: nothing to learn here.
            for ser in series.values():
                ser.push(None)
            gmo_mids.append(None)
            while len(gmo_mids) > longest + 2:
                gmo_mids.popleft()
            return
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
            g0, x, gap, phase = pending.popleft()
            start = mid_at(g0 + lat)
            ys = {}
            for h, hs in hsteps.items():
                end = mid_at(g0 + lat + hs)
                if start and end:
                    ys[h] = math.log(end / start) * 1e4
            if len(ys) != len(hsteps):
                continue
            for var in variants:
                if gap < var.gap:
                    continue
                if phase == "train":
                    var.fit.add(x, ys)
                elif phase == "test" and var.weights is not None:
                    for h, y in ys.items():
                        p = sum(w * v for w, v in zip(var.weights[h], x, strict=True))
                        sc = var.out.score[h]
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
        gap = grid.gap
        if calibrate and cal_start_ns is None and not train_n and train[0] <= t < train[1]:
            cal_start = t + int((train[1] - t) * (1 - cal_frac))
        if train[0] <= t < cal_start:
            pending.append((g, x, gap, "train"))
            train_n += 1
            for var in variants:
                var.out.train_n += gap >= var.gap
            return
        in_cal = cal_start <= t < train[1]
        in_test = test[0] <= t < test[1]
        if not (in_cal or in_test):
            return
        for var in variants:
            if var.weights is None:
                var.weights = var.fit.weights(ridge)
                var.out.weights = var.weights
        cost: list = []
        if in_cal:
            for var in variants:
                var.out.cal_n += 1
                if gap >= var.gap:
                    trade(x, gm, var, [(m, h, (var.gap, "cal", m, h), var.out.cal[(m, h)])
                                       for m in margins for h in holds], cost)
            return
        if not picked:
            pick_variant()
        pending.append((g, x, gap, "test"))
        for var in variants:
            var.out.test_n += 1
            if gap >= var.gap:
                trade(x, gm, var, [(var.out.chosen[h], h, (var.gap, "model", h), var.out.model[h])
                                   for h in holds], cost)
        qty = qty_for(gm)
        # The one-signal rule on the same grid: a lead moved `rule_bps`
        # more than GMO over the last second, the lead itself that far.
        lead_moves = [series[k].ret(10) for k in ("bybit", "binance") if mid(k)]
        if not lead_moves:
            return
        lead_move = sum(lead_moves) / len(lead_moves)
        rgap = lead_move - series["gmo"].ret(10)
        if abs(rgap) < rule_bps or abs(lead_move) < rule_bps or lead_move * rgap <= 0:
            armed = True
            return
        if not armed:
            return
        armed = False
        for h in holds:
            if g >= busy[("rule", h)]:
                busy[("rule", h)] = g + lat + hsteps[h]
                open_trade(rule_out[h], h, 1 if rgap > 0 else -1, qty)

    for src, rx, event in rows:
        if src not in books or not isinstance(event, (DepthSnapshot, DepthDelta)):
            continue
        if grid_t is None:
            grid_t = (rx // step + 1) * step
        if rx - grid_t > 60 * NS:
            # A gap in the recording: jump to it rather than step through
            # hours of nothing, and forget what came before.
            skip = (rx - grid_t) // step
            grid_t += skip * step
            g += skip
            pending.clear()
            gmo_mids.clear()
            grid.basis_avg.clear()
            for ser in series.values():
                ser.vals.clear()
        while grid_t <= rx:
            on_grid(grid_t)
            grid_t += step
            g += 1
        books[src].apply(event)
        last_rx[src] = rx
    for var in variants:
        if var.weights is None:
            var.weights = var.fit.weights(ridge)
            var.out.weights = var.weights
    if not picked and calibrate:
        # No test period seen (fitting a model to use live): pick all the same.
        pick_variant()

    def cal_money(var: _Variant) -> float:
        h = max(holds)
        m = var.out.chosen.get(h)
        return var.out.cal[(m, h)].pnl if calibrate and m is not None else float("-inf")

    best = max(variants, key=cal_money) if calibrate else variants[0]
    best.out.variants = {var.gap: var.out for var in variants}
    return best.out


def active_split(starts_ns: list[int], window: tuple[int, int], frac: float = 0.7,
                 part_s: float = 3600.0) -> tuple[int | None, float]:
    """Where `frac` of the time actually recorded inside `window` has passed,
    and how many hours were recorded there.

    `starts_ns` are the start times of the recording's parts (five minutes
    each as recorded); a part is taken to run until the next one starts, or
    for one and a half times the usual spacing between parts (at most
    `part_s`), so a gap between recordings counts for almost nothing."""
    starts = sorted(t for t in starts_ns if window[0] <= t < window[1])
    # A part lasts about as long as the usual spacing between parts; that,
    # not the gap after the last one before a stop, is its length.
    diffs = sorted(b - a for a, b in zip(starts, starts[1:], strict=False) if b > a)
    longest = min(part_s * NS, 1.5 * diffs[len(diffs) // 2]) if diffs else part_s * NS
    spans = []
    for i, t in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else t + int(longest)
        end = min(nxt, t + int(longest), window[1])
        if end > t:
            spans.append((t, end))
    total = sum(e - b for b, e in spans)
    if not total:
        return None, 0.0
    goal, run = total * frac, 0
    for b, e in spans:
        if run + (e - b) >= goal:
            return b + int(goal - run), total / NS / 3600
        run += e - b
    return spans[-1][1], total / NS / 3600


def _margin_text(m) -> str:
    return "入らない（どの基準でも儲からなかった）" if m is None else f"+{m:g}bps"


def model_dict(r: FairResult, *, cross, step_ms: int, latency_ms: int, train,
               hinge_bps: float = 8.0) -> dict:
    """What the live run needs to price GMO exactly as the replay did."""
    return {
        "symbol": r.label, "names": r.names, "cross": list(cross), "step_ms": step_ms,
        "latency_ms": latency_ms, "train": list(train),
        "hinge_bps": hinge_bps, "min_gap_bps": r.min_gap,
        "weights": {f"{h:g}": w for h, w in r.weights.items()},
        "margin_bps": {f"{h:g}": m for h, m in r.chosen.items()},
        "calibration": {f"{m:g}/{h:g}": [b.n, round(b.pnl, 2)] for (m, h), b in r.cal.items()},
    }


HEADER = "\t".join(["銘柄", "学習した場面", "持つ秒", "未知データでの説明力 R²", "向きの的中", "やり方",
                    "取引数", "1回あたり bps", "勝率", "合計円"])


def _scene(r: FairResult) -> str:
    return "全部" if not r.min_gap else f"乖離{r.min_gap:g}bps以上"


def report(r: FairResult) -> list[str]:
    """Every fitted subset on the test, the chosen one marked, then the rule."""
    lines = []
    subsets = list(r.variants.values()) or [r]
    for h in r.holds:
        for v in subsets:
            sc, b = v.score[h], v.model[h]
            hit = f"{sc.hit / sc.called:.0%}" if sc.called else "-"
            kind = "フェア価格モデル" + ("（採用）" if v is r else "")
            lines.append("\t".join([
                r.label, _scene(v), f"{h:g}", f"{sc.r2():+.3f}", hit, kind, f"{b.n:,}",
                "-" if not b.n else f"{b.bps():+.2f}",
                "-" if not b.n else f"{b.wins / b.n:.0%}", f"{b.pnl:+,.0f}",
            ]))
        b = r.rule[h]
        lines.append("\t".join([
            r.label, "-", f"{h:g}", "-", "-", "単純な後追い", f"{b.n:,}",
            "-" if not b.n else f"{b.bps():+.2f}",
            "-" if not b.n else f"{b.wins / b.n:.0%}", f"{b.pnl:+,.0f}",
        ]))
    return lines


def calibration_text(r: FairResult) -> list[str]:
    """For each fitted subset: how many samples it learned from, what each
    margin made on the end of the training period, and the pick."""
    if not r.cal:
        return [f"  入る基準（固定）: 往復コスト {_margin_text(r.chosen.get(h))}" for h in r.holds[:1]]
    out = []
    for v in (list(r.variants.values()) or [r]):
        margins = sorted({m for m, _ in v.cal})
        mark = "  ← 採用（入る基準を決める部分で一番儲かった）" if v is r else ""
        out.append(f"  学習した場面: {_scene(v)}（学習 {v.train_n:,} 点）{mark}")
        for h in v.holds:
            cells = []
            for m in margins:
                b = v.cal[(m, h)]
                cells.append(f"+{m:g}bps: {b.n}回 {b.pnl:+,.0f}円" if b.n else f"+{m:g}bps: 0回")
            out.append(f"   {h:g}秒  " + " / ".join(cells) + f"  → 選んだのは {_margin_text(v.chosen.get(h))}")
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

            lines.append(f"  • {r.label} {h:g}秒  モデル（{_scene(r)}・コスト{_margin_text(r.chosen.get(h))}）: "
                         f"{cell(m)}  /  後追い: {cell(s)}  （R² {sc.r2():+.3f}）")
    lines.append("  （モデル・学習する場面・入る基準は前の時間帯だけで決め、この時間帯では一度も調整していません）")
    return "\n".join(lines)
