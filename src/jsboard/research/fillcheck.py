"""The live run's fills beside the replay's, over the same hours.

The replay of 2026-10-02 08:33-15:14 matched the live fill count but lost
a fifth of what live lost (-87 against -422 yen). A total says nothing
about why, so this takes both sets of fills apart the same way:

- 板の中値との差: what each fill earned against the mid when it happened
  (the half spread a maker is paid for standing there);
- 10秒後 / 60秒後: how far the mid then moved against the fill (what the
  traders who hit us knew);
- 在庫の値動き: the rest of the result, from holding coin while the price
  moved;
- the result in half-hour slices, so a gap between the two shows when.

Fees use bitbank's schedule for both sides (maker -0.02%, taker 0.12%),
so the comparison is about prices, not about how each side reports fees.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

MAKER_BPS = -2.0
TAKER_BPS = 12.0
NS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class FillRow:
    ts_ns: int
    sign: int
    """+1 bought, -1 sold."""
    price: float
    amount: float
    maker: bool = True

    @property
    def fee(self) -> float:
        bps = MAKER_BPS if self.maker else TAKER_BPS
        return self.price * self.amount * bps / 10_000


def from_trade(trade: dict) -> FillRow:
    """One execution as bitbank's trade history (or the model) reports it."""
    return FillRow(
        ts_ns=int(trade["executed_at"]) * 1_000_000,
        sign=1 if trade["side"] == "buy" else -1,
        price=float(trade["price"]),
        amount=float(trade["amount"]),
        maker=trade.get("maker_taker", "maker") != "taker",
    )


class MidLine:
    """The book's mid in yen through time, read at any instant."""

    def __init__(self) -> None:
        self.ts: list[int] = []
        self.mid: list[float] = []

    def add(self, ts_ns: int, mid: float | None) -> None:
        if mid is None or (self.mid and self.mid[-1] == mid):
            return
        if self.ts and ts_ns < self.ts[-1]:
            ts_ns = self.ts[-1]
        self.ts.append(ts_ns)
        self.mid.append(mid)

    def at(self, ts_ns: int) -> float | None:
        i = bisect.bisect_right(self.ts, ts_ns) - 1
        return self.mid[i] if i >= 0 else None


@dataclass
class Breakdown:
    fills: int = 0
    buys: int = 0
    volume: float = 0.0
    edge: float = 0.0
    move_10s: float = 0.0
    move_60s: float = 0.0
    fees: float = 0.0
    pnl: float = 0.0

    @property
    def carry(self) -> float:
        """What holding the coin made or lost: the result less the edge and fees."""
        return self.pnl - self.edge + self.fees


def _mark(fills: list[FillRow], mids: MidLine, at_ns: int) -> float:
    """Yen made by the fills up to `at_ns`, coin valued at that instant's mid."""
    cash = coin = fees = 0.0
    for f in fills:
        if f.ts_ns > at_ns:
            break
        cash -= f.sign * f.price * f.amount
        coin += f.sign * f.amount
        fees += f.fee
    mid = mids.at(at_ns)
    return cash + coin * (mid or 0.0) - fees


def breakdown(fills: list[FillRow], mids: MidLine, end_ns: int) -> Breakdown:
    fills = sorted(fills, key=lambda f: f.ts_ns)
    out = Breakdown(fills=len(fills))
    for f in fills:
        out.buys += f.sign > 0
        out.volume += f.price * f.amount
        out.fees += f.fee
        mid = mids.at(f.ts_ns)
        if mid is None:
            continue
        out.edge += f.sign * (mid - f.price) * f.amount
        for horizon, name in ((10, "move_10s"), (60, "move_60s")):
            later = mids.at(min(f.ts_ns + horizon * NS, end_ns))
            if later is not None:
                setattr(out, name, getattr(out, name) + f.sign * (later - mid) * f.amount)
    out.pnl = _mark(fills, mids, end_ns)
    return out


def slices(fills: list[FillRow], mids: MidLine, start_ns: int, end_ns: int,
           step_s: int = 1800) -> list[tuple[int, float]]:
    """(slice start, yen made in the slice) for each `step_s` of the window."""
    fills = sorted(fills, key=lambda f: f.ts_ns)
    out, before = [], 0.0
    t = start_ns
    while t < end_ns:
        nxt = min(t + step_s * NS, end_ns)
        now = _mark(fills, mids, nxt)
        out.append((t, now - before))
        before, t = now, nxt
    return out


def worst(fills: list[FillRow], mids: MidLine, end_ns: int, n: int = 5) -> list[tuple]:
    """The fills the price moved hardest against within a minute."""
    rows = []
    for f in fills:
        mid, later = mids.at(f.ts_ns), mids.at(min(f.ts_ns + 60 * NS, end_ns))
        if mid is not None and later is not None:
            rows.append((f.sign * (later - f.price) * f.amount, f))
    return sorted(rows, key=lambda r: r[0])[:n]


def report(live: list[FillRow], sim: list[FillRow], mids: MidLine,
           start_ns: int, end_ns: int) -> str:
    from datetime import UTC, datetime

    def hhmm(ns: int) -> str:
        return datetime.fromtimestamp(ns / NS, UTC).strftime("%H:%M:%S")

    a, b = breakdown(live, mids, end_ns), breakdown(sim, mids, end_ns)
    lines = [
        "項目\t本番\t検証",
        f"約定\t{a.fills}\t{b.fills}",
        f"うち買い\t{a.buys}\t{b.buys}",
        f"約定額(円)\t{a.volume:,.0f}\t{b.volume:,.0f}",
        f"板の中値との差(円)\t{a.edge:+,.0f}\t{b.edge:+,.0f}",
        f"10秒後の値動き(円)\t{a.move_10s:+,.0f}\t{b.move_10s:+,.0f}",
        f"60秒後の値動き(円)\t{a.move_60s:+,.0f}\t{b.move_60s:+,.0f}",
        f"在庫の値動き(円)\t{a.carry:+,.0f}\t{b.carry:+,.0f}",
        f"手数料(円,マイナスはリベート)\t{a.fees:+,.0f}\t{b.fees:+,.0f}",
        f"損益(円)\t{a.pnl:+,.0f}\t{b.pnl:+,.0f}",
        "",
        "30分ごとの損益(UTC)\t本番\t検証",
    ]
    for (t, x), (_, y) in zip(slices(live, mids, start_ns, end_ns),
                              slices(sim, mids, start_ns, end_ns), strict=True):
        lines.append(f"{hhmm(t)[:5]}\t{x:+,.0f}\t{y:+,.0f}")
    for name, fills in (("本番", live), ("検証", sim)):
        lines += ["", f"{name}: 1分以内に一番逆に動いた約定\t損(円)\t売買\t値段\t量"]
        for loss, f in worst(fills, mids, end_ns):
            side = "買い" if f.sign > 0 else "売り"
            maker = "" if f.maker else "(成行)"
            lines.append(f"{hhmm(f.ts_ns)}\t{loss:+,.0f}\t{side}{maker}\t{f.price:g}\t{f.amount:g}")
    return "\n".join(lines)
