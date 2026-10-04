"""What standing on the other side of every print would have earned.

Our own fills depend on our latency and our place in the queue. The
prints do not: each one had a maker, and what that maker made — the half
spread and the rebate, less how far the price ran afterwards — is the most
any maker could make on this book. If that ceiling is below zero, no
setting of ours wins; if it is above zero only for some prints (small
ones, quiet moments, a calm lead market), those are the only ones worth
standing in front of.

All figures are basis points of the trade's value (1bps = 0.01%), from the
maker's side, with bitbank's 2bps maker rebate added.
"""

from __future__ import annotations

import bisect
import statistics
from collections import deque
from dataclasses import dataclass, field

from ..core.types import Side
from ..feed.base import DepthDelta, DepthSnapshot, TradeTick
from .fillcheck import NS, MidLine

REBATE_BPS = 2.0
HORIZONS_S = (1, 10, 60)


@dataclass(slots=True)
class Print:
    ts_ns: int
    sign: int
    """+1 when the maker bought (a seller crossed), -1 when it sold."""
    price: float
    yen: float
    mid: float
    spread_bps: float
    lead_bps: float
    """How far the lead market moved in the taker's direction in the second
    before the print (positive: the taker was following it)."""


@dataclass
class Bucket:
    name: str
    n: int = 0
    yen: float = 0.0
    edge: float = 0.0
    after: dict = field(default_factory=lambda: {h: 0.0 for h in HORIZONS_S})
    rebate: float = REBATE_BPS

    def bps(self, horizon: int) -> float:
        """Maker's result at `horizon` seconds, rebate included, per yen traded."""
        return self.after[horizon] / self.yen + self.rebate if self.yen else float("nan")

    def merge(self, other: Bucket) -> None:
        self.n += other.n
        self.yen += other.yen
        self.edge += other.edge
        for h in HORIZONS_S:
            self.after[h] += other.after[h]

    def add(self, p: Print, moves: dict) -> None:
        w = p.yen
        self.n += 1
        self.yen += w
        self.edge += w * p.sign * (p.mid - p.price) / p.price * 1e4
        for h, later in moves.items():
            self.after[h] += w * p.sign * (later - p.price) / p.price * 1e4

    def row(self) -> str:
        if not self.n:
            return f"{self.name}\t0\t-\t-\t-\t-\t-"
        avg = self.yen / self.n
        edge = self.edge / self.yen
        cols = [f"{self.bps(h):+.1f}" for h in HORIZONS_S]
        return f"{self.name}\t{self.n:,}\t{avg:,.0f}\t{edge:+.1f}\t" + "\t".join(cols)


class _Top:
    """The best bid and ask of an L2 book, and nothing else.

    `collect` needs only the touch of each book, and keeping a full
    `MarketView` (sorted levels, volatility, flow) for every update of two
    books made a day of one book take over ten minutes.
    """

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

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread_ticks(self) -> int | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid


def collect(rows, instrument, lead_instrument,
            tops: list | None = None) -> tuple[list[Print], MidLine, list[float]]:
    """Every bitbank print with the book and lead state it met.

    `rows` yields (source, receive_ns, event) with source "bitbank" or
    "lead". Returns the prints, the bitbank mid line and the spread (bps)
    sampled at each book update.
    """
    tick = float(instrument.tick_size)
    book, lead = _Top(), _Top()
    lead_hist: deque = deque()
    mids, prints, spreads = MidLine(), [], []
    for src, rx, event in rows:
        if src == "lead":
            if not isinstance(event, (DepthSnapshot, DepthDelta)):
                continue
            lead.apply(event)
            if lead.mid is not None:
                lead_hist.append((rx, lead.mid))
                while lead_hist and rx - lead_hist[0][0] > 2 * NS:
                    lead_hist.popleft()
            continue
        if isinstance(event, TradeTick):
            mid, spread = book.mid, book.spread_ticks
            if mid is None or spread is None or spread <= 0:
                continue
            taker = 1 if event.aggressor is Side.BUY else -1
            lead_bps = 0.0
            ago = next((m for t, m in lead_hist if rx - t <= NS), None)
            if ago and lead.mid:
                lead_bps = taker * (lead.mid - ago) / ago * 1e4
            price = event.price * tick
            prints.append(Print(
                ts_ns=rx, sign=-taker, price=price, yen=price * instrument.qty_f(event.qty),
                mid=mid * tick, spread_bps=spread / mid * 1e4, lead_bps=lead_bps,
            ))
        elif isinstance(event, (DepthSnapshot, DepthDelta)):
            book.apply(event)
            if book.mid is not None:
                mids.add(rx, book.mid * tick)
                if tops is not None and (not tops or tops[-1][1:] != (
                        book.best_bid * tick, book.best_ask * tick)):
                    tops.append((rx, book.best_bid * tick, book.best_ask * tick))
                if book.spread_ticks:
                    spreads.append(book.spread_ticks / book.mid * 1e4)
    return prints, mids, spreads


def group(prints: list[Print], mids: MidLine, rebate: float = REBATE_BPS) -> dict:
    """The prints sorted into the report's buckets."""
    end = mids.ts[-1] if mids.ts else 0

    def b(name: str) -> Bucket:
        return Bucket(name, rebate=rebate)

    groups: dict[str, list[Bucket]] = {
        "全部": [b("すべての約定")],
        "大きさ": [b("〜5千円"), b("5千〜2万円"), b("2万〜10万円"), b("10万円〜")],
        "スプレッド": [b("2bps未満"), b("2〜5bps"), b("5bps以上")],
        "先物": [b("先物が同じ向きに2bps以上"), b("先物が静か"), b("先物が逆向きに2bps以上")],
    }
    for p in prints:
        moves = {}
        for h in HORIZONS_S:
            later = mids.at(min(p.ts_ns + h * NS, end))
            if later is None:
                break
            moves[h] = later
        if len(moves) < len(HORIZONS_S):
            continue
        groups["全部"][0].add(p, moves)
        size = 0 if p.yen < 5e3 else 1 if p.yen < 2e4 else 2 if p.yen < 1e5 else 3
        groups["大きさ"][size].add(p, moves)
        spread = 0 if p.spread_bps < 2 else 1 if p.spread_bps < 5 else 2
        groups["スプレッド"][spread].add(p, moves)
        lead = 0 if p.lead_bps >= 2 else 2 if p.lead_bps <= -2 else 1
        groups["先物"][lead].add(p, moves)
    return groups


def report(prints: list[Print], mids: MidLine, spreads: list[float], our_yen: float) -> str:
    end = mids.ts[-1] if mids.ts else 0
    groups = group(prints, mids)
    hours = (end - mids.ts[0]) / NS / 3600 if mids.ts else 0
    lines = []
    if spreads:
        tight = sum(s < 2 for s in spreads) / len(spreads) * 100
        lines += [
            f"時間 {hours:.1f}時間  約定(全員) {len(prints):,}件  "
            f"1時間あたり {len(prints) / hours if hours else 0:,.0f}件",
            f"スプレッド 中央値 {statistics.median(spreads):.1f}bps  "
            f"2bps未満の時間 {tight:.0f}%  (1bps = 0.01%)",
            f"自分の注文1回 約{our_yen:,.0f}円",
            "",
        ]
    head = "区分\t件数\t平均額(円)\t取れる幅bps\t1秒後bps\t10秒後bps\t60秒後bps"
    for title, buckets in groups.items():
        lines += [f"[{title}]", head, *(b.row() for b in buckets), ""]
    lines += [
        "見方: 「その約定の反対側に自分がいたら」のもうけ（リベート2bps込み、約定額あたり）。",
        "取れる幅 = 約定した瞬間の中値との差。1秒後〜60秒後 = その時点の中値で手仕舞った場合。",
        "マイナスなら、どれだけ速く・うまく並んでも、その区分の約定は損になる。",
    ]
    return "\n".join(lines)


@dataclass
class BookSummary:
    """One book over several days, for the table that ranks every book."""

    label: str
    rebate: float
    has_lead: bool = True
    days: int = 0
    hours: float = 0.0
    prints: int = 0
    spreads: list = field(default_factory=list)
    good_days: int = 0
    groups: dict | None = None

    def add_day(self, prints: list[Print], mids: MidLine, spreads: list[float]) -> None:
        if not prints or not mids.ts:
            return
        day = group(prints, mids, self.rebate)
        self.days += 1
        self.hours += (mids.ts[-1] - mids.ts[0]) / NS / 3600
        self.prints += len(prints)
        self.spreads += spreads[:: max(1, len(spreads) // 5000)]
        self.good_days += day["全部"][0].bps(60) > 0
        if self.groups is None:
            self.groups = day
        else:
            for name, buckets in day.items():
                for mine, theirs in zip(self.groups[name], buckets, strict=True):
                    mine.merge(theirs)

    def row(self) -> str:
        if not self.groups:
            return f"{self.label}\t0\t-"
        every, quiet = self.groups["全部"][0], self.groups["先物"][1]
        chase = self.groups["先物"][0]

        def f(x: float) -> str:
            return "-" if x != x else f"{x:+.1f}"

        cols = [
            self.label, str(self.days), f"{self.prints / self.hours if self.hours else 0:,.0f}",
            f"{every.yen / every.n if every.n else 0:,.0f}",
            f"{statistics.median(self.spreads):.1f}" if self.spreads else "-",
            f(every.bps(1)), f(every.bps(10)), f(every.bps(60)),
            *((f(quiet.bps(1)), f(quiet.bps(10)), f(quiet.bps(60)), f(chase.bps(10)))
              if self.has_lead else ("先物なし", "-", "-", "-")),
            f"{self.good_days}/{self.days}",
        ]
        return "\t".join(cols)


SUMMARY_HEADER = "\t".join([
    "銘柄", "日数", "約定/時", "平均額(円)", "スプレッド中央bps",
    "全部1秒", "全部10秒", "全部60秒", "先物静か1秒", "先物静か10秒", "先物静か60秒",
    "先物追い10秒", "60秒がプラスの日",
])
SUMMARY_NOTE = (
    "数字は「その約定の反対側に自分がいたら」のもうけ（bps=0.01%、メイカーリベート込み）。\n"
    "先物静か = 直前1秒に先物が2bps以上動いていない約定。先物追い = 先物と同じ向きの約定。\n"
    "どれも速さ・並び順を最高に置いた上限。ここがマイナスの銘柄は、どうやっても勝てない。"
)


def _moves(p: Print, mids: MidLine, end: int) -> dict | None:
    moves = {}
    for h in HORIZONS_S:
        later = mids.at(min(p.ts_ns + h * NS, end))
        if later is None:
            return None
        moves[h] = later
    return moves


def chance_day(prints: list[Print], mids: MidLine, rebate: float, levels) -> dict:
    """The maker's side of the prints that came just after the lead moved the
    other way by at least each level: the lead rose and a seller hit the bid
    anyway, so the maker bought ahead of the move. Quoting only then, and
    only that side, is the "trade the chances only" maker."""
    end = mids.ts[-1] if mids.ts else 0
    out = {x: Bucket(f"先物が逆に{x:g}bps以上", rebate=rebate) for x in levels}
    for p in prints:
        if p.lead_bps > -min(levels):
            continue
        moves = _moves(p, mids, end)
        if moves is None:
            continue
        for x in levels:
            if p.lead_bps <= -x:
                out[x].add(p, moves)
    return out


@dataclass
class ChanceSummary:
    label: str
    rebate: float
    levels: tuple
    days: int = 0
    totals: dict = field(default_factory=dict)
    good: dict = field(default_factory=dict)
    counted: dict = field(default_factory=dict)

    def add_day(self, prints: list[Print], mids: MidLine) -> None:
        if not prints or not mids.ts:
            return
        self.days += 1
        for x, b in chance_day(prints, mids, self.rebate, self.levels).items():
            total = self.totals.setdefault(x, Bucket(b.name, rebate=self.rebate))
            total.merge(b)
            if b.n:
                self.counted[x] = self.counted.get(x, 0) + 1
                self.good[x] = self.good.get(x, 0) + (b.bps(60) > 0)

    def rows(self) -> list[str]:
        out = []
        for x in self.levels:
            b = self.totals.get(x)
            if b is None or not b.n:
                out.append(f"{self.label}\t{x:g}\t{self.days}\t0")
                continue
            out.append("\t".join([
                self.label, f"{x:g}", str(self.days), f"{b.n:,}",
                f"{b.n / self.days:,.1f}" if self.days else "-",
                f"{b.yen / b.n:,.0f}",
                *(f"{b.bps(h):+.1f}" for h in HORIZONS_S),
                f"{self.good.get(x, 0)}/{self.counted.get(x, 0)}",
            ]))
        return out


CHANCE_HEADER = "\t".join([
    "銘柄", "先物の逆向きbps以上", "日数", "約定", "1日あたり", "平均額(円)",
    "1秒後bps", "10秒後bps", "60秒後bps", "60秒がプラスの日",
])
CHANCE_NOTE = (
    "先物（Binance）が上がった直後に、bitbank で売ってきた人から買えた約定だけ（下がったときは逆）。\n"
    "＝ 先物が動いた直後に、有利な側だけ注文を出す「チャンスだけ」のマーケットメイクの上限。\n"
    "bps はリベート込み、約定額あたり。並び順と速さは最高の場合。"
)


@dataclass
class RoundTrip:
    """Chance fills taken all the way out, at prices the book offered."""

    level: float
    wait_s: float
    n: int = 0
    maker_exits: int = 0
    bps_sum: float = 0.0
    days: int = 0
    good_days: int = 0
    _day_sum: float = 0.0
    _day_n: int = 0

    def end_day(self) -> None:
        if self._day_n:
            self.days += 1
            self.good_days += self._day_sum > 0
        self._day_sum, self._day_n = 0.0, 0

    def row(self, label: str, order_jpy: float, day_count: int) -> str:
        if not self.n:
            return f"{label}\t{self.level:g}\t{self.wait_s:g}\t0"
        avg = self.bps_sum / self.n
        per_day = self.bps_sum * order_jpy / 1e4 / max(1, day_count)
        return "\t".join([
            label, f"{self.level:g}", f"{self.wait_s:g}", f"{self.n:,}",
            f"{self.maker_exits / self.n:.0%}", f"{avg:+.1f}", f"{per_day:+,.0f}",
            f"{self.good_days}/{self.days}",
        ])


ROUNDTRIP_HEADER = "\t".join([
    "銘柄", "先物の逆向きbps以上", "待つ秒", "取引", "板で売れた割合", "1回あたりbps",
    "1日あたり円(1回1万円)", "プラスの日",
])
ROUNDTRIP_NOTE = (
    "買えたら（売れたら）0.2秒後に反対側の一番良い値段へ注文を置き、待つ秒以内にそこで約定すれば\n"
    "リベート2回込みの差益。来なければ成行で一番良い値段に逃げて、スプレッドとテイカー手数料を払う。\n"
    "中値では売れない前提。並び順は先頭（上限）。"
)


def roundtrips(prints: list[Print], tops: list, results: list[RoundTrip], *,
               rebate_bps: float, taker_bps: float, latency_ms: float = 200.0) -> None:
    """Add one day's chance fills, each taken out of the market, to `results`.

    Bought on a chance: 200ms later an ask goes up at the best ask then. A
    buyer who trades at or above it within the wait takes it (maker exit,
    a second rebate). Otherwise the coin is sold at the best bid when the
    wait ends, paying the taker fee. A chance sale is the mirror image.
    """
    if not tops:
        for r in results:
            r.end_day()
        return
    times = [t for t, _, _ in tops]
    lat = int(latency_ms * 1e6)

    def top_at(t: int):
        i = bisect.bisect_right(times, t) - 1
        return tops[i] if i >= 0 else None

    order = sorted(range(len(prints)), key=lambda i: prints[i].ts_ns)
    ordered = [prints[i] for i in order]
    for i, p in enumerate(ordered):
        for r in results:
            if p.lead_bps > -r.level:
                continue
            start = top_at(p.ts_ns + lat)
            end_ns = p.ts_ns + int(r.wait_s * NS)
            end = top_at(end_ns)
            if start is None or end is None:
                continue
            exit_price = start[2] if p.sign > 0 else start[1]
            filled = False
            for q in ordered[i + 1:]:
                if q.ts_ns > end_ns:
                    break
                if q.ts_ns <= p.ts_ns + lat or q.sign == p.sign:
                    continue
                # A buyer took the offer (maker sold) at or above our ask.
                if (q.price >= exit_price) if p.sign > 0 else (q.price <= exit_price):
                    filled = True
                    break
            if filled:
                bps = p.sign * (exit_price - p.price) / p.price * 1e4 + 2 * rebate_bps
                r.maker_exits += 1
            else:
                out = end[1] if p.sign > 0 else end[2]
                bps = p.sign * (out - p.price) / p.price * 1e4 + rebate_bps - taker_bps
            r.n += 1
            r.bps_sum += bps
            r._day_sum += bps
            r._day_n += 1
    for r in results:
        r.end_day()
