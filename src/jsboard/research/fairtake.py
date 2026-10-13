"""Buy GMO below its fair value, sell it back when it gets there: replayed.

The fair value is the one a market maker on a follower venue uses
(Annanay Kapila, "Fair Value Is Not a Prediction"): the deeper market's
price now times the smoothed ratio between the two (Grid.leader_fair).
It is not a forecast; nothing is fitted, so every recorded hour is a test.

The trade takes, it does not wait in the book to be picked off:

- **In.** When GMO's best ask is at least `gap_bps` under the fair value,
  buy `size` at market (or sell when the best bid is that far over). The
  order lands `latency_ms` later and walks the book as it stands then.
- **Out.** When GMO's best bid has come back to the fair value as it
  stands then (the best ask, for a short), close at market, again
  `latency_ms` later on the book of that moment; after `max_hold_s` close
  whatever the price.
- **Cost.** GMO's leverage book charges no fee, but it rounds each
  closed position's profit down to the yen (measured live: 0.47 yen a
  close on average). Every trade is booked both ways: as priced, and as
  GMO would pay it (floored).

One position at a time per gap. The gaps are fixed before the run and all
reported: picking the best one afterwards is not a result. Each is also
split into the first and second half of the recorded time, and shown
without its best and worst 5% of trades, so a result that rests on one
stretch or a few trades shows as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.types import Instrument
from ..feed.base import DepthDelta, DepthSnapshot
from .fairprice import Grid

NS = 1_000_000_000


@dataclass
class Trade:
    side: int
    signal_ns: int
    entry: float
    exit: float = 0.0
    exit_ns: int = 0
    held_s: float = 0.0
    forced: bool = False
    size: float = 0.0

    @property
    def pnl(self) -> float:
        return self.side * (self.exit - self.entry) * self.size

    @property
    def gmo_pnl(self) -> float:
        """As GMO pays it: the profit of the closed position floored to the yen."""
        return float(math.floor(self.pnl + 1e-9))

    @property
    def bps(self) -> float:
        return self.side * (self.exit - self.entry) / self.entry * 1e4


@dataclass
class Gap:
    gap_bps: float
    trades: list = field(default_factory=list)
    open: Trade | None = None
    pending: bool = False


@dataclass
class TakeResult:
    label: str
    gaps: list
    size: float
    hours: float = 0.0
    mid_ns: int = 0
    """Halfway through the recorded time, for the first/second-half split."""


def run(rows, gmo: Instrument, leads: dict, *, label: str, test: tuple[int, int],
        gaps=(4.0, 6.0, 8.0), size: float = 50.0, latency_ms: float = 200.0,
        max_hold_s: float = 60.0, step_ms: int = 100, stale_s: float = 10.0) -> TakeResult:
    """`rows` yields (source, receive_ns, event): "gmo", "bybit", "binance"."""
    grid = Grid(gmo, leads, (), step_ms)
    books = grid.books
    step = step_ms * 1_000_000
    lat = max(1, round(latency_ms / step_ms))
    hold = int(max_hold_s * NS)
    stale = int(stale_s * NS)
    out = TakeResult(label, [Gap(float(g)) for g in gaps], size)
    tasks: dict = {}
    last_rx: dict = {}
    covered = 0
    marks: list = []   # (t, covered so far) to find the halfway point afterwards
    grid_t = None
    g = 0
    prev_t = None

    def take(side: int) -> float | None:
        b = books["gmo"]
        if b.best_bid is None or b.best_ask is None:
            return None
        return b.take(side, size, gmo)

    def schedule(at: int, action) -> None:
        tasks.setdefault(at, []).append(action)

    def enter(gp: Gap, side: int, t: int) -> None:
        def act() -> None:
            gp.pending = False
            price = take(side)
            if price is not None:
                gp.open = Trade(side, t, price, size=size)
        gp.pending = True
        schedule(g + lat, act)

    def leave(gp: Gap, now: int, forced: bool) -> None:
        tr = gp.open

        def act() -> None:
            gp.pending = False
            price = take(-tr.side)
            if price is None:
                return
            tr.exit, tr.exit_ns, tr.forced = price, grid_t, forced
            tr.held_s = (grid_t - tr.signal_ns) / NS
            gp.trades.append(tr)
            gp.open = None
        gp.pending = True
        schedule(g + lat, act)

    def on_grid(t: int) -> None:
        nonlocal covered, prev_t
        for action in tasks.pop(g, ()):
            action()
        fresh = (t - last_rx.get("gmo", -stale - 1) <= stale
                 and any(t - last_rx.get(k, -stale - 1) <= stale for k in ("bybit", "binance")))
        if not fresh:
            for ser in grid.series.values():
                ser.push(None)
            prev_t = None
            return
        grid.push(t)
        gm = grid.mid("gmo")
        if gm is None or not grid.ready():
            return
        grid.features(gm)   # keeps the premium's average current
        if not test[0] <= t < test[1]:
            return
        if prev_t is not None:
            covered += t - prev_t
        prev_t = t
        if len(marks) == 0 or t - marks[-1][0] >= 60 * NS:
            marks.append((t, covered))
        fair = grid.leader_fair()
        b = books["gmo"]
        if fair is None or b.best_bid is None or b.best_ask is None:
            return
        bid, ask = gmo.price_f(b.best_bid), gmo.price_f(b.best_ask)
        for gp in out.gaps:
            if gp.pending:
                continue
            tr = gp.open
            if tr is None:
                if (fair - ask) / fair * 1e4 >= gp.gap_bps:
                    enter(gp, 1, t)
                elif (bid - fair) / fair * 1e4 >= gp.gap_bps:
                    enter(gp, -1, t)
                continue
            back = bid >= fair if tr.side > 0 else ask <= fair
            if back or t - tr.signal_ns >= hold:
                leave(gp, t, forced=not back)

    for src, rx, event in rows:
        if src not in books or not isinstance(event, (DepthSnapshot, DepthDelta)):
            continue
        if grid_t is None:
            grid_t = (rx // step + 1) * step
        if rx - grid_t > 60 * NS:
            # A gap in the recording: close nothing on prices that are not
            # there; drop what was open (counted as nothing) and start afresh.
            skip = (rx - grid_t) // step
            grid_t += skip * step
            g += skip
            tasks.clear()
            grid.forget()
            prev_t = None
            for gp in out.gaps:
                gp.open, gp.pending = None, False
        while grid_t <= rx:
            on_grid(grid_t)
            grid_t += step
            g += 1
        books[src].apply(event)
        last_rx[src] = rx
    out.hours = covered / NS / 3600
    half = covered / 2
    out.mid_ns = next((t for t, c in marks if c >= half), test[1])
    return out


HEADER = "\t".join([
    "ずれ", "取引", "勝率", "1回あたり bps", "損益 円（値段どおり）", "損益 円（GMO の切り捨て後）",
    "平均の保有秒", "時間切れ", "前半 円", "後半 円", "良い5%除く bps", "両端5%除く bps",
])


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def rows_of(r: TakeResult) -> list[str]:
    def f(x: float) -> str:
        return "-" if x != x else f"{x:+.2f}"

    lines = []
    for gp in r.gaps:
        ts = gp.trades
        n = len(ts)
        bps = sorted(t.bps for t in ts)
        k = max(1, n // 20)
        first = sum(t.gmo_pnl for t in ts if t.signal_ns < r.mid_ns)
        second = sum(t.gmo_pnl for t in ts if t.signal_ns >= r.mid_ns)
        lines.append("\t".join([
            f"{gp.gap_bps:g}bps", f"{n}", f"{sum(t.pnl > 0 for t in ts) / n:.0%}" if n else "-",
            f(_mean(bps)), f"{sum(t.pnl for t in ts):+,.1f}", f"{sum(t.gmo_pnl for t in ts):+,.0f}",
            f"{_mean([t.held_s for t in ts]):.1f}" if n else "-", f"{sum(t.forced for t in ts)}",
            f"{first:+,.0f}", f"{second:+,.0f}",
            f(_mean(bps[:-k])) if n > 2 * k else "-", f(_mean(bps[k:-k])) if n > 2 * k else "-",
        ]))
    return lines


NOTE = (
    "フェア価格 = Binance（なければ Bybit）の今の値段 × GMO との比率の1分平均（予想ではない、学習もしない）。\n"
    "ずれ = GMO の売り値（買うとき）がフェア価格よりこれだけ安い / 買い値（売るとき）がこれだけ高いと成行で入る。"
    "GMO の値段がフェア価格に戻ったら成行で出る（時間切れなら値段に関係なく出る）。注文は 0.2秒後の板で約定として計算。\n"
    "GMO の切り捨て後 = 閉じた建玉ごとに損益の1円未満を切り捨てた額（実際の口座はこちら）。"
)


def slack_summary(r: TakeResult) -> str:
    lines = [f":dart: フェア価格からのずれを取りに行く（{r.label}、1回 {r.size:g}枚、録画 {r.hours:.1f}時間、注文なし）"]
    for gp in r.gaps:
        ts = gp.trades
        n = len(ts)
        if not n:
            lines.append(f"  • ずれ {gp.gap_bps:g}bps: 取引なし")
            continue
        first = sum(t.gmo_pnl for t in ts if t.signal_ns < r.mid_ns)
        second = sum(t.gmo_pnl for t in ts if t.signal_ns >= r.mid_ns)
        lines.append(f"  • ずれ {gp.gap_bps:g}bps: {n}回  勝率 {sum(t.pnl > 0 for t in ts) / n:.0%}  "
                     f"1回 {_mean([t.bps for t in ts]):+.2f}bps  GMO の切り捨て後 {sum(t.gmo_pnl for t in ts):+,.0f}円"
                     f"（前半 {first:+,.0f}・後半 {second:+,.0f}）")
    lines.append("  （録画で再現した成行の売り買い。0.2秒の遅れと板の厚さ、GMO の1円未満切り捨てを入れています）")
    return "\n".join(lines)
