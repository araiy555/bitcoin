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
        cols = [f"{self.after[h] / self.yen + REBATE_BPS:+.1f}" for h in HORIZONS_S]
        return f"{self.name}\t{self.n:,}\t{avg:,.0f}\t{edge:+.1f}\t" + "\t".join(cols)


def collect(rows, instrument, lead_instrument) -> tuple[list[Print], MidLine, list[float]]:
    """Every bitbank print with the book and lead state it met.

    `rows` yields (source, receive_ns, event) with source "bitbank" or
    "lead". Returns the prints, the bitbank mid line and the spread (bps)
    sampled at each book update.
    """
    from ..core.market import MarketView

    tick = float(instrument.tick_size)
    book = MarketView(instrument=instrument)
    lead = MarketView(instrument=lead_instrument)
    lead_hist: deque = deque()
    mids, prints, spreads = MidLine(), [], []
    for src, rx, event in rows:
        if src == "lead":
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
                if book.spread_ticks:
                    spreads.append(book.spread_ticks / book.mid * 1e4)
    return prints, mids, spreads


def report(prints: list[Print], mids: MidLine, spreads: list[float], our_yen: float) -> str:
    end = mids.ts[-1] if mids.ts else 0
    groups: dict[str, list[Bucket]] = {
        "全部": [Bucket("すべての約定")],
        "大きさ": [Bucket("〜5千円"), Bucket("5千〜2万円"), Bucket("2万〜10万円"), Bucket("10万円〜")],
        "スプレッド": [Bucket("2bps未満"), Bucket("2〜5bps"), Bucket("5bps以上")],
        "先物": [Bucket("先物が同じ向きに2bps以上"), Bucket("先物が静か"),
                 Bucket("先物が逆向きに2bps以上")],
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
