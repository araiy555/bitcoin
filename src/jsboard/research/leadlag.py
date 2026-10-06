"""Follow a fast market on a slow one: trade GMO only when the lead jumps.

When Bybit's (or Binance's) perp moves and GMO's leverage book has not
followed yet, buy (or sell) on GMO and get out a little later. GMO charges
no taker fee on XRP and ETH, so the only costs are the book itself, both
ways. Every price here is taken from the recorded books:

- the signal: over the last `window_ms`, the lead's mid moved `gap` bps
  more than GMO's mid did, in one direction, by at least the threshold;
- entry `latency_ms` after the signal, at the price got by walking GMO's
  recorded book for the order size (never its mid);
- exit `h` seconds after entry, walking the other side of the book;
- GMO's taker fee on both legs.

One move of the lead is one signal: a threshold re-arms only after the
gap has closed below it. One trade at a time per threshold and hold: a
10 s trade frees its slot after 10 s, not after the longest hold, so short
holds are counted as often as they could really be traded.
Results are kept per UTC day, so a threshold chosen on one day can be
checked on the next.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime

from ..core.types import Instrument
from ..feed.base import DepthDelta, DepthSnapshot
from .arbedge import _Book

NS = 1_000_000_000


@dataclass
class Cell:
    n: int = 0
    wins: int = 0
    notional: float = 0.0
    pnl: float = 0.0
    mid_n: int = 0
    mid_bps: float = 0.0
    """GMO's mid `h` after the signal, in the signal's direction, before any
    cost: whether GMO follows at all, apart from whether the spread is paid."""

    def mid(self) -> float:
        return self.mid_bps / self.mid_n if self.mid_n else float("nan")

    def bps(self) -> float:
        return self.pnl / self.notional * 1e4 if self.notional else float("nan")


@dataclass
class LeadLagResult:
    label: str
    signals: dict = field(default_factory=dict)
    """threshold -> signals seen (before the one-at-a-time rule)."""
    cells: dict = field(default_factory=dict)
    """(day, threshold, hold) -> Cell."""
    no_book: int = 0

    def cell(self, day: str, threshold: float, hold: int) -> Cell:
        return self.cells.setdefault((day, threshold, hold), Cell())

    def rows(self, holds) -> list[str]:
        out = []
        for day, threshold in sorted({(d, t) for d, t, _ in self.cells}):
            cols = [self.label, day, f"{threshold:g}"]
            n = self.cells.get((day, threshold, holds[0]), Cell()).n
            cols.append(f"{n:,}")
            for h in holds:
                c = self.cells.get((day, threshold, h), Cell())
                win = f"{c.wins / c.n:.0%}" if c.n else "-"
                bps = "-" if c.n == 0 else f"{c.bps():+.1f}"
                cols += [bps, f"{c.pnl:+,.0f}", win]
            out.append("\t".join(cols))
        return out


def header(holds) -> str:
    cols = ["銘柄(先行)", "日(UTC)", "しきいbps", "取引数"]
    for h in holds:
        cols += [f"{h:g}秒bps", f"{h:g}秒円", f"{h:g}秒勝率"]
    return "\t".join(cols)


NOTE = (
    "しきい = 先行市場が遅い側より何bps 先に動いたら入るか（直前の窓の中で）。\n"
    "入りも出も遅い側の録画した実際の板を、注文の量ぶん上から食った値段。手数料込み。\n"
    "1つのしきいにつき同時に1取引だけ。日ごとに分けてあるので、1日目で選んだしきいを2日目で確かめる。"
)


def analyse(rows, gmo: Instrument, lead: Instrument, *, label: str, thresholds, holds,
            window_ms: float = 1000.0, latency_ms: float = 200.0, size_jpy: float = 10_000.0,
            fee_bps: float = 0.0, min_order: float = 0.0, depth: int = 100) -> LeadLagResult:
    """`rows` yields (source, receive_ns, event), source "gmo" or "lead"."""
    # Plain books: only the touch is needed on every update, and the depth
    # only when a trade is priced. A full MarketView per update made a day
    # take many minutes.
    gm, ld = _Book(), _Book()

    def mid_of(book: _Book) -> float | None:
        if book.best_bid is None or book.best_ask is None:
            return None
        return (book.best_bid + book.best_ask) / 2.0
    out = LeadLagResult(label)
    window, lat = int(window_ms * 1e6), int(latency_ms * 1e6)
    lead_hist: deque = deque()   # (rx, mid)
    gmo_hist: deque = deque()
    busy_until = {(t, h): -1 for t in thresholds for h in holds}
    # A threshold re-arms only once the gap has closed below it, so one move
    # of the lead is one signal, however long it takes GMO to catch up.
    armed = {t: True for t in thresholds}
    tasks: list = []
    seq = itertools.count()

    def ago(hist: deque, now: int):
        while len(hist) >= 2 and hist[1][0] <= now - window:
            hist.popleft()
        return hist[0][1] if hist and hist[0][0] <= now - window else None

    def day_of(ns: int) -> str:
        return datetime.fromtimestamp(ns / NS, UTC).strftime("%Y-%m-%d")

    def gmo_price(side: int, qty: float) -> float | None:
        if mid_of(gm) is None:
            return None
        return gm.take(side, qty, gmo)

    def run_due(now: int) -> None:
        while tasks and tasks[0][0] <= now:
            due, _, kind, trade = heapq.heappop(tasks)
            if isinstance(kind, tuple) and kind[0] == "mid":
                m = mid_of(gm)
                if m is not None:
                    c = out.cell(trade["day0"], trade["threshold"], kind[1])
                    c.mid_n += 1
                    c.mid_bps += trade["side"] * math.log(m / trade["mid0"]) * 1e4
                continue
            if kind == "enter":
                m = mid_of(gm)
                if m is None:
                    out.no_book += 1
                    continue
                mid = m * float(gmo.tick_size)
                step = float(gmo.lot_size)
                qty = math.floor(size_jpy / mid / step + 1e-9) * step
                if qty < min_order:
                    # BTC's minimum (0.01) is far above a 10,000 yen order:
                    # trade the minimum rather than skip the coin entirely.
                    qty = math.ceil(min_order / step - 1e-9) * step
                if qty <= 0:
                    out.no_book += 1
                    continue
                price = gmo_price(trade["side"], qty)
                if price is None:
                    out.no_book += 1
                    continue
                trade.update(qty=qty, entry=price, day=day_of(due))
                for h in trade["holds"]:
                    heapq.heappush(tasks, (due + int(h * NS), next(seq), ("exit", h), trade))
            else:
                h = kind[1]
                back = gmo_price(-trade["side"], trade["qty"])
                if back is None:
                    continue
                q, s = trade["qty"], trade["side"]
                pnl = s * q * (back - trade["entry"]) - (trade["entry"] + back) * q * fee_bps / 1e4
                c = out.cell(trade["day"], trade["threshold"], h)
                c.n += 1
                c.wins += pnl > 0
                c.notional += trade["entry"] * q
                c.pnl += pnl

    for src, rx, event in rows:
        run_due(rx - 1)
        if not isinstance(event, (DepthSnapshot, DepthDelta)):
            continue
        view, hist = (gm, gmo_hist) if src == "gmo" else (ld, lead_hist)
        view.apply(event)
        m = mid_of(view)
        if m is not None and (not hist or hist[-1][1] != m):
            hist.append((rx, m))
        lead_mid, gmo_mid = mid_of(ld), mid_of(gm)
        if src != "lead" or lead_mid is None or gmo_mid is None:
            continue
        lead_then, gmo_then = ago(lead_hist, rx), ago(gmo_hist, rx)
        if not lead_then or not gmo_then:
            continue
        lead_move = math.log(lead_mid / lead_then) * 1e4
        gap = lead_move - math.log(gmo_mid / gmo_then) * 1e4
        for t in thresholds:
            # The lead itself must have moved that far, that way: when GMO
            # catches up after the lead's move has left the window, the gap
            # flips sign with no move of the lead at all.
            if abs(gap) < t or lead_move * gap <= 0 or abs(lead_move) < t:
                armed[t] = True
                continue
            if not armed[t]:
                continue
            armed[t] = False
            out.signals[t] = out.signals.get(t, 0) + 1
            free = [h for h in holds if rx >= busy_until[(t, h)]]
            if not free:
                continue
            for h in free:
                busy_until[(t, h)] = rx + lat + int(h * NS)
            side = 1 if gap > 0 else -1  # the lead rose further: buy GMO
            trade = {"side": side, "threshold": t, "mid0": gmo_mid, "day0": day_of(rx),
                     "holds": free}
            heapq.heappush(tasks, (rx + lat, next(seq), "enter", trade))
            for h in free:
                heapq.heappush(tasks, (rx + int(h * NS), next(seq), ("mid", h), trade))
    return out


def slack_summary(results: list[LeadLagResult], holds, interim: bool = False,
                  min_trades: int = 20, since: str | None = None) -> str:
    title = "途中経過" if interim else "結果"
    lines = [f":zap: 後追い取引の{title}（先行市場が動いた → GMO で成行、実際の板・手数料込み）"]
    if since:
        lines.append(f"  {since}（UTC）より後の録画だけで計算")
    for r in results:
        # The best threshold and hold over the whole recording, every day pooled.
        pooled: dict = {}
        for (_, t, h), c in r.cells.items():
            p = pooled.setdefault((t, h), Cell())
            p.n += c.n
            p.wins += c.wins
            p.notional += c.notional
            p.pnl += c.pnl
            p.mid_n += c.mid_n
            p.mid_bps += c.mid_bps
        ranked = sorted(((k, c) for k, c in pooled.items() if c.n >= min_trades), key=lambda kc: -kc[1].bps())
        if not ranked:
            lines.append(f"  • {r.label}  取引がまだ少なく判定できません")
            continue
        (t, h), c = ranked[0]
        by_hold = " / ".join(
            f"{x:g}秒 {pooled[(t, x)].bps():+.1f}" for x in holds if pooled.get((t, x), Cell()).n
        )
        moved = " / ".join(
            f"{x:g}秒 {pooled[(t, x)].mid():+.1f}" for x in holds if pooled.get((t, x), Cell()).mid_n
        )
        lines.append(
            f"  • {r.label}  一番良い: しきい {t:g}bps・{h:g}秒持つ → {c.bps():+.1f}bps"
            f"  {c.pnl:+,.0f}円  {c.n}回  勝率 {c.wins / c.n:.0%}"
            f"\n      損益（入ってから）: {by_hold}"
            f"\n      GMO の値段の動き（合図から、手数料・スプレッド前）: {moved}"
        )
    lines.append("  （一番良いものを選んだ数字です。別の日で確かめるまで信用しないでください）")
    return "\n".join(lines)


DIRECTION_HORIZONS = (1, 5, 10, 30, 60)


@dataclass
class Direction:
    same: int = 0
    opposite: int = 0
    flat: int = 0
    total_bps: float = 0.0

    @property
    def n(self) -> int:
        return self.same + self.opposite + self.flat


@dataclass
class DirectionResult:
    label: str
    signals: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)
    """(threshold, horizon) -> Direction."""

    def rows(self, thresholds) -> list[str]:
        out = []
        for t in thresholds:
            cols = [self.label, f"{t:g}", f"{self.signals.get(t, 0):,}"]
            for h in DIRECTION_HORIZONS:
                d = self.cells.get((t, h), Direction())
                if not d.n:
                    cols += ["-", "-", "-", "-"]
                    continue
                cols += [f"{d.same / d.n:.0%}", f"{d.opposite / d.n:.0%}",
                         f"{d.flat / d.n:.0%}", f"{d.total_bps / d.n:+.1f}"]
            out.append("\t".join(cols))
        return out


def direction_header() -> str:
    cols = ["銘柄(先行)", "しきいbps", "合図の数"]
    for h in DIRECTION_HORIZONS:
        cols += [f"{h}秒 同じ向き", f"{h}秒 逆向き", f"{h}秒 変わらず", f"{h}秒 平均bps"]
    return "\t".join(cols)


DIRECTION_NOTE = (
    "合図 = 先行市場が直前1秒で、遅い側より しきいbps 以上先に動いた瞬間（1秒に1回まで）。\n"
    "その後、遅い側の中値が合図と同じ向きに動いたか、逆か、変わらないか。手数料・約定値段は入れない。\n"
    "平均bps は合図の向きを正にした中値の動き（プラスなら追随、マイナスなら逆戻り）。"
)


def direction(rows, follower: Instrument, lead: Instrument, *, label: str, thresholds,
              window_ms: float = 1000.0) -> DirectionResult:
    """Which way the slow market goes after the fast one moves first.

    `rows` yields (source, receive_ns, event), source "gmo" (the slow
    market, whichever venue) or "lead"."""
    fo, ld = _Book(), _Book()
    out = DirectionResult(label)
    window = int(window_ms * 1e6)
    lead_hist: deque = deque()
    fo_hist: deque = deque()
    next_ok = {t: -1 for t in thresholds}
    tasks: list = []
    seq = itertools.count()

    def mid_of(book: _Book) -> float | None:
        if book.best_bid is None or book.best_ask is None:
            return None
        return (book.best_bid + book.best_ask) / 2.0

    def ago(hist: deque, now: int):
        while len(hist) >= 2 and hist[1][0] <= now - window:
            hist.popleft()
        return hist[0][1] if hist and hist[0][0] <= now - window else None

    def run_due(now: int) -> None:
        while tasks and tasks[0][0] <= now:
            _, _, t, h, sign, start = heapq.heappop(tasks)
            m = mid_of(fo)
            if m is None:
                continue
            move = sign * math.log(m / start) * 1e4
            d = out.cells.setdefault((t, h), Direction())
            if move > 1e-9:
                d.same += 1
            elif move < -1e-9:
                d.opposite += 1
            else:
                d.flat += 1
            d.total_bps += move

    for src, rx, event in rows:
        run_due(rx - 1)
        if not isinstance(event, (DepthSnapshot, DepthDelta)):
            continue
        book, hist = (fo, fo_hist) if src == "gmo" else (ld, lead_hist)
        book.apply(event)
        m = mid_of(book)
        if m is not None and (not hist or hist[-1][1] != m):
            hist.append((rx, m))
        lead_mid, fo_mid = mid_of(ld), mid_of(fo)
        if src != "lead" or lead_mid is None or fo_mid is None:
            continue
        lead_then, fo_then = ago(lead_hist, rx), ago(fo_hist, rx)
        if not lead_then or not fo_then:
            continue
        gap = (math.log(lead_mid / lead_then) - math.log(fo_mid / fo_then)) * 1e4
        for t in thresholds:
            if abs(gap) < t or rx < next_ok[t]:
                continue
            next_ok[t] = rx + window
            out.signals[t] = out.signals.get(t, 0) + 1
            sign = 1 if gap > 0 else -1
            for h in DIRECTION_HORIZONS:
                heapq.heappush(tasks, (rx + h * NS, next(seq), t, h, sign, fo_mid))
    return out
