"""Every bitbank book, weeks of its executions, without recording anything.

The first question for any book is whether its makers get paid at all:
take the maker's side of every execution, add the rebate, and see where
the price stood some seconds later. `printedge` answers it from our own
recordings, which cover four books. bitbank publishes each pair's full
execution list per day, so the same answer is available for every pair
and many days.

With no order book the later price is the last execution at or before
that instant. Executions land on the bid or the ask about evenly, so on
average that is the mid; a single print carries half a spread of noise
either way. One second is left out: the prints of one sweep land within
it at falling prices and would read as an adverse move.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

HORIZONS_S = (10, 60, 300)
PASS_SHARE = 0.8
"""Share of days positive at a minute for a book to pass."""


@dataclass(frozen=True, slots=True)
class Exec:
    ts_ms: int
    taker: int
    """+1 a buyer crossed, -1 a seller."""
    price: float
    amount: float


def parse(rows: list[dict]) -> list[Exec]:
    """bitbank's `/transactions/YYYYMMDD` rows, oldest first."""
    out = [
        Exec(int(r["executed_at"]), 1 if r["side"] == "buy" else -1,
             float(r["price"]), float(r["amount"]))
        for r in rows
    ]
    return sorted(out, key=lambda e: e.ts_ms)


@dataclass
class DayEdge:
    n: int = 0
    yen: float = 0.0
    after: dict = field(default_factory=lambda: {h: 0.0 for h in HORIZONS_S})
    hours: float = 0.0


def day_edge(day: list[Exec], following: list[Exec]) -> DayEdge:
    """The maker's side of every execution of `day`, priced `h` seconds on.

    `following` is the next day's executions, so the day's last prints have
    a later price; prints with nothing that far ahead are left out.
    """
    tape = day + following
    times = [e.ts_ms for e in tape]
    last = times[-1] if times else 0
    out = DayEdge()
    if day:
        out.hours = (day[-1].ts_ms - day[0].ts_ms) / 3_600_000
    for e in day:
        if e.ts_ms + max(HORIZONS_S) * 1000 > last:
            continue
        yen = e.price * e.amount
        out.n += 1
        out.yen += yen
        for h in HORIZONS_S:
            later = tape[bisect.bisect_right(times, e.ts_ms + h * 1000) - 1].price
            out.after[h] += yen * -e.taker * (later - e.price) / e.price * 1e4
    return out


@dataclass
class PairEdge:
    pair: str
    rebate_bps: float
    days: list[DayEdge] = field(default_factory=list)

    def bps(self, h: int, day: DayEdge | None = None) -> float:
        rows = [day] if day else self.days
        yen = sum(d.yen for d in rows)
        return sum(d.after[h] for d in rows) / yen + self.rebate_bps if yen else float("nan")

    @property
    def good_days(self) -> int:
        return sum(self.bps(60, d) > 0 for d in self.days if d.yen)

    @property
    def counted_days(self) -> int:
        return sum(1 for d in self.days if d.yen)

    @property
    def passed(self) -> bool:
        n = self.counted_days
        return n > 0 and self.bps(60) > 0 and self.good_days >= PASS_SHARE * n

    def row(self) -> str:
        n = sum(d.n for d in self.days)
        hours = sum(d.hours for d in self.days)
        yen = sum(d.yen for d in self.days)

        def f(x: float) -> str:
            return "-" if x != x else f"{x:+.1f}"

        return "\t".join([
            self.pair, str(self.counted_days),
            f"{n / hours:,.0f}" if hours else "-",
            f"{yen / n:,.0f}" if n else "-",
            f"{self.rebate_bps:+.1f}",
            *(f(self.bps(h)) for h in HORIZONS_S),
            f"{self.good_days}/{self.counted_days}",
            "合格" if self.passed else "不合格",
        ])


HEADER = "\t".join([
    "銘柄", "日数", "約定/時", "平均額(円)", "リベートbps",
    "10秒後bps", "60秒後bps", "5分後bps", "60秒がプラスの日", "判定",
])
NOTE = (
    "数字は「その約定の反対側に自分がいたら」のもうけ（bps=0.01%、リベート込み、約定額あたり）。\n"
    "速さも並び順も最高だった場合の上限。ここがマイナスなら、どうやっても勝てない。\n"
    f"合格 = 60秒後がプラス、かつ {PASS_SHARE:.0%} 以上の日でプラス。"
)
