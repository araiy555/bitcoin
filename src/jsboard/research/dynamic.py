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
fill a maker would have had standing in that queue, so the tape alone
measures maker-side adverse selection: an aggressive buy lifts an offer, so
the maker sold; an aggressive sell hits a bid, so the maker bought. Taking
the sign from the aggressor keeps a fill model out of the sign, and the
sample size is simply how often the market traded.

**Which prints count is not obvious, and it changes the answer by a factor of
five.** The first version of this screen averaged every print with equal
weight and scored USUSDT at 0.23; an hour of recorded depth, replayed through
the paper venue, scored the same symbol at 1.31. The gap is not the market
moving. It is two selection biases, both in the flattering direction:

  **size.** A thousand one-lot prints and one large sweep are not equally
  informative, and it is the sweeps that carry information. Averaging per
  print buries them; every mark-out here is weighted by traded size instead.

  **the queue.** A maker joining the touch does not trade on every print — it
  trades once a print is large enough to consume the size resting in front of
  it. Those prints are exactly the aggressive ones, so a maker's realised fills
  are drawn from the worst tail of the tape rather than from its average.

Both are reported. `ratio` averages every print and is the optimistic bound;
`sweep_ratio` counts only prints that cleared the visible touch and is the
closer analogue of what a maker actually gets filled on. The verdict uses the
second whenever there are enough of them, and the gap between the two is
itself worth reading: a symbol where they agree is a calm book, and one where
they diverge is a book whose information arrives in bursts.

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
    sweep_markout: MarkOutTracker = field(init=False)
    """The same measurement over prints that cleared the visible touch — the
    ones a maker resting in that queue would actually have traded on."""
    spread_samples: list[float] = field(default_factory=list)
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    trades: int = 0
    sweeps: int = 0
    book_updates: int = 0
    last_ns: int = 0

    def __post_init__(self) -> None:
        self.markout = MarkOutTracker(horizons_s=(self.horizon_s,))
        self.sweep_markout = MarkOutTracker(horizons_s=(self.horizon_s,))

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

    def on_book(
        self, bid: float, ask: float, ts_ns: int, bid_qty: float = 0.0, ask_qty: float = 0.0
    ) -> None:
        self.bid, self.ask = bid, ask
        self.bid_qty, self.ask_qty = bid_qty, ask_qty
        mid = self.mid
        if mid is None:
            return
        self.book_updates += 1
        self.spread_samples.append((ask - bid) / mid * 10_000.0)
        now = self._clock(ts_ns)
        self.markout.poll(now, mid)
        self.sweep_markout.poll(now, mid)

    def on_trade(self, aggressor_sign: int, ts_ns: int, qty: float = 1.0) -> None:
        """Record the fill a maker would have had on the other side.

        `aggressor_sign` is +1 when the taker bought — which means the maker
        sold, so the maker's own side is the opposite. `qty` is the traded
        size and becomes the weight, because a sweep and a dust print are not
        equally informative and averaging them per print hides the sweeps.
        """
        mid = self.mid
        if mid is None or aggressor_sign == 0 or qty <= 0:
            return
        self.trades += 1
        now = self._clock(ts_ns)
        self.markout.on_fill(now, -aggressor_sign, mid, qty, mid_ticks=mid)
        self.markout.poll(now, mid)

        # A maker in the queue trades only once the print clears what is
        # resting in front of it. An aggressive buy eats the offer, so the
        # size that matters is the one on that side.
        resting = self.ask_qty if aggressor_sign > 0 else self.bid_qty
        if resting > 0 and qty >= resting:
            self.sweeps += 1
            self.sweep_markout.on_fill(now, -aggressor_sign, mid, qty, mid_ticks=mid)
            self.sweep_markout.poll(now, mid)

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
    def sweep_markout_bps(self) -> float:
        return self.sweep_markout.windows[0].mean_bps

    @property
    def settled(self) -> int:
        return self.markout.windows[0].n

    @property
    def sweeps_settled(self) -> int:
        return self.sweep_markout.windows[0].n

    def _ratio(self, markout_bps: float) -> float:
        spread = self.spread_bps
        if math.isnan(spread) or math.isnan(markout_bps) or spread <= 0:
            return math.nan
        return -markout_bps / spread

    @property
    def ratio(self) -> float:
        """Every print, size-weighted. The optimistic bound."""
        return self._ratio(self.markout_bps)

    @property
    def sweep_ratio(self) -> float:
        """Only prints that cleared the touch — closer to a maker's own fills."""
        return self._ratio(self.sweep_markout_bps)

    def decisive_ratio(self, *, min_sweeps: int = 100) -> float:
        """Whichever ratio the sample supports, preferring the honest one."""
        if self.sweeps_settled >= min_sweeps:
            return self.sweep_ratio
        return self.ratio

    def verdict(self, *, min_trades: int = 500, min_sweeps: int = 100) -> str:
        if self.settled < min_trades:
            return "サンプル不足"
        ratio = self.decisive_ratio(min_sweeps=min_sweeps)
        if math.isnan(ratio):
            return "サンプル不足"
        if ratio >= MARGINAL_RATIO:
            return "不可"
        if ratio >= GOOD_RATIO:
            return "見込み薄"
        return "研究候補"


def rank(
    probes: list[SymbolProbe], *, min_trades: int = 500, min_sweeps: int = 100
) -> list[SymbolProbe]:
    """Everything measurable, best ratio first; unmeasurable last."""

    def key(p: SymbolProbe) -> tuple[int, float]:
        ratio = p.decisive_ratio(min_sweeps=min_sweeps)
        unusable = p.settled < min_trades or math.isnan(ratio)
        return (1 if unusable else 0, ratio if not math.isnan(ratio) else math.inf)

    return sorted(probes, key=key)


def tally(
    probes: list[SymbolProbe], *, min_trades: int = 500, min_sweeps: int = 100
) -> dict[str, int]:
    counts = {"研究候補": 0, "見込み薄": 0, "不可": 0, "サンプル不足": 0}
    for probe in probes:
        counts[probe.verdict(min_trades=min_trades, min_sweeps=min_sweeps)] += 1
    return counts
