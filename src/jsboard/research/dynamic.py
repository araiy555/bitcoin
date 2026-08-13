"""Second-pass screen: does the price move faster than the spread pays?

`triage` reads the static shape of a book — fee, tick, spread, trade count —
and USUSDT cleared every one of its gates before losing 31 bps. What it could
not see is speed. USUSDT's spread was 9.53 bps and the mid moved 12.48 bps
against a fill within 100 milliseconds, so more than a whole spread went past
before a 100ms requote loop could react. Quoting there is not badly placed,
it is stale on arrival.

That gives one ratio worth screening on:

    adverse selection over 100ms  ÷  spread

Above 1 the market takes more than the spread pays, and no placement, size or
inventory rule recovers it at that reaction time. WIFUSDT sat at 0.31 and
died of other causes; USUSDT sat at 1.31 and died of this one.

**No paper market maker runs here, deliberately.** Every public print is a
fill we would have had standing at the front of that queue, so the tape alone
measures maker-side adverse selection: an aggressive buy lifts an offer, so
the maker sold; an aggressive sell hits a bid, so the maker bought. Taking
the sign from the aggressor removes the fill model from the measurement
entirely — no queue position to assume, no gap-throughs to miss — and the
sample size is simply how many times the market traded.

The 0.5 threshold for a research candidate is a judgement, not a measurement.
A ratio just under 1 leaves nothing for fees, inventory and the optimism
already known to be in the fill model, so the bar sits at half. Anything that
clears it still has to be settled on net P&L, not on this ratio.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from ..mm.markout import MarkOutTracker

NS_PER_MS = 1_000_000

GOOD_RATIO = 0.5
"""Below this a symbol is worth recording. Judgement, not a measured edge."""

MARGINAL_RATIO = 1.0
"""At or above this the market takes more than the spread pays."""


@dataclass(slots=True)
class SymbolProbe:
    """Live spread and maker-side mark-out for one symbol."""

    symbol: str
    horizon_s: float = 0.1
    markout: MarkOutTracker = field(init=False)
    spread_samples: list[float] = field(default_factory=list)
    bid: float = 0.0
    ask: float = 0.0
    trades: int = 0
    book_updates: int = 0
    last_ns: int = 0

    def __post_init__(self) -> None:
        self.markout = MarkOutTracker(horizons_s=(self.horizon_s,))

    # ------------------------------------------------------------ ingestion

    def _clock(self, ts_ns: int) -> int:
        """Exchange time, forced non-decreasing.

        Book and trade updates arrive on two sockets, so a few can land out of
        order. The mark-out queues settle from the front and assume time only
        moves forward, so a late stamp is clamped rather than allowed to
        unsettle entries that already matured.
        """
        self.last_ns = max(self.last_ns, ts_ns)
        return self.last_ns

    @property
    def mid(self) -> float | None:
        if self.bid <= 0 or self.ask <= self.bid:
            return None
        return (self.bid + self.ask) / 2.0

    def on_book(self, bid: float, ask: float, ts_ns: int) -> None:
        self.bid, self.ask = bid, ask
        mid = self.mid
        if mid is None:
            return
        self.book_updates += 1
        self.spread_samples.append((ask - bid) / mid * 10_000.0)
        self.markout.poll(self._clock(ts_ns), mid)

    def on_trade(self, aggressor_sign: int, ts_ns: int) -> None:
        """Record the fill a maker would have had on the other side.

        `aggressor_sign` is +1 when the taker bought — which means the maker
        sold, so the maker's own side is the opposite.
        """
        mid = self.mid
        if mid is None or aggressor_sign == 0:
            return
        self.trades += 1
        now = self._clock(ts_ns)
        self.markout.on_fill(now, -aggressor_sign, mid, 1.0, mid_ticks=mid)
        self.markout.poll(now, mid)

    # --------------------------------------------------------------- reads

    @property
    def spread_bps(self) -> float:
        if not self.spread_samples:
            return math.nan
        return statistics.median(self.spread_samples)

    @property
    def markout_bps(self) -> float:
        """Signed: negative means the market moved against the maker."""
        return self.markout.windows[0].mean_bps

    @property
    def settled(self) -> int:
        return self.markout.windows[0].n

    @property
    def ratio(self) -> float:
        """Adverse selection as a fraction of the spread. Lower is better."""
        spread = self.spread_bps
        mo = self.markout_bps
        if math.isnan(spread) or math.isnan(mo) or spread <= 0:
            return math.nan
        return -mo / spread

    def verdict(self, *, min_trades: int = 500) -> str:
        if self.settled < min_trades:
            return "サンプル不足"
        ratio = self.ratio
        if math.isnan(ratio):
            return "サンプル不足"
        if ratio >= MARGINAL_RATIO:
            return "不可"
        if ratio >= GOOD_RATIO:
            return "見込み薄"
        return "研究候補"


def rank(probes: list[SymbolProbe], *, min_trades: int = 500) -> list[SymbolProbe]:
    """Everything measurable, best ratio first; unmeasurable last."""
    def key(p: SymbolProbe) -> tuple[int, float]:
        ratio = p.ratio
        unusable = p.settled < min_trades or math.isnan(ratio)
        return (1 if unusable else 0, ratio if not math.isnan(ratio) else math.inf)

    return sorted(probes, key=key)


def tally(probes: list[SymbolProbe], *, min_trades: int = 500) -> dict[str, int]:
    counts = {"研究候補": 0, "見込み薄": 0, "不可": 0, "サンプル不足": 0}
    for probe in probes:
        counts[probe.verdict(min_trades=min_trades)] += 1
    return counts
