"""Post-fill mark-out: did the price move against us right after we filled?

A maker's gross edge is the spread it captures. A maker's loss is adverse
selection — being filled precisely by the counterparty who turns out to be
right. Both land in the same realised P&L number, so that number alone cannot
say which one is driving the result, and the two call for opposite responses:
a fee problem is fixed by a fee tier, an adverse-selection problem is not
fixed by anything cheap.

Mark-out separates them. For each of our fills, look at the mid a fixed time
later and measure how far it moved *in our favour*:

    our buy  at 100, mid 10s later 100.05  ->  +5 bps  (we were paid to buy)
    our sell at 100, mid 10s later 100.05  ->  -5 bps  (we were picked off)

Read the horizons together. Negative at 1s means we are being run over by
faster flow. Negative at 60s while positive at 1s means we are quoting into a
trend rather than a mean-reverting flow. Positive throughout while P&L is
negative means the fee is the entire problem.

Mark-out is measured against the mid, so it deliberately ignores the spread
we earned — it is the adverse-selection term on its own, not a second P&L.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

NS_PER_S = 1_000_000_000


@dataclass(slots=True)
class _Pending:
    """One fill waiting for its horizon to elapse."""

    due_ns: int
    sign: int
    fill_ticks: float
    mid_ticks: float
    weight: float


@dataclass(slots=True)
class MarkOutWindow:
    """Size-weighted mark-out at a single horizon, against two references.

    Which reference matters depends on the question, and conflating them is
    the easy mistake:

      vs_mid   mid at the fill → mid at the horizon. Adverse selection on its
               own: how much the market moved against us after trading with
               us, with the spread we earned excluded.
      vs_fill  our fill price → mid at the horizon. What the fill is worth if
               unwound at mid, so it carries the half-spread inside it.

    On a symbol whose tick is several basis points wide the gap between them
    is larger than either number, and reading `vs_fill` as adverse selection
    turns a loss into an apparent gain.
    """

    horizon_s: float
    n: int = 0
    weight: float = 0.0
    weighted_vs_mid: float = 0.0
    weighted_vs_fill: float = 0.0
    pending: deque[_Pending] = field(default_factory=deque)

    @property
    def mean_bps(self) -> float:
        """Adverse selection: size-weighted, NaN until something has matured."""
        return self.weighted_vs_mid / self.weight if self.weight > 0 else math.nan

    @property
    def mean_vs_fill_bps(self) -> float:
        """The same fills measured from our own price, spread included."""
        return self.weighted_vs_fill / self.weight if self.weight > 0 else math.nan

    @property
    def unsettled(self) -> int:
        """Fills too recent to have reached this horizon yet."""
        return len(self.pending)


@dataclass(slots=True)
class MarkOutTracker:
    """Accumulates mark-out across several horizons at once.

    Prices go in as ticks and come out as basis points, so the caller never
    has to convert: the ratio is unit-free as long as fill price and mid are
    quoted the same way.
    """

    horizons_s: tuple[float, ...] = (1.0, 10.0, 60.0)
    windows: list[MarkOutWindow] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.windows:
            self.windows = [MarkOutWindow(h) for h in self.horizons_s]

    def on_fill(
        self,
        now_ns: int,
        sign: int,
        price_ticks: float,
        weight: float = 1.0,
        mid_ticks: float | None = None,
    ) -> None:
        """Register one of our fills. `sign` is +1 when we bought, -1 when we sold.

        `mid_ticks` is the mid at the moment of the fill and is the reference
        adverse selection is measured from. Without it there is no way to
        separate the spread we earned from the move that followed, so the fill
        is not recorded rather than being recorded against the wrong baseline.
        """
        if weight <= 0 or price_ticks <= 0 or sign == 0:
            return
        if mid_ticks is None or mid_ticks <= 0:
            return
        for window in self.windows:
            # now_ns is monotonic and the offset is constant per window, so
            # each deque stays sorted by due time and settles from the front.
            due = now_ns + int(window.horizon_s * NS_PER_S)
            window.pending.append(_Pending(due, sign, price_ticks, mid_ticks, weight))

    def poll(self, now_ns: int, mid_ticks: float | None) -> None:
        """Settle every fill whose horizon has elapsed, at the current mid."""
        if mid_ticks is None or mid_ticks <= 0:
            # No usable mid: leave them pending rather than scoring against a
            # price we do not have. They mature on a later poll.
            return
        for window in self.windows:
            queue = window.pending
            while queue and queue[0].due_ns <= now_ns:
                fill = queue.popleft()
                vs_mid = fill.sign * (mid_ticks - fill.mid_ticks) / fill.mid_ticks * 10_000.0
                vs_fill = fill.sign * (mid_ticks - fill.fill_ticks) / fill.fill_ticks * 10_000.0
                window.n += 1
                window.weight += fill.weight
                window.weighted_vs_mid += vs_mid * fill.weight
                window.weighted_vs_fill += vs_fill * fill.weight

    def summary(self) -> list[dict[str, float]]:
        return [
            {
                "horizon_s": w.horizon_s,
                "mean_bps": w.mean_bps,
                "vs_fill_bps": w.mean_vs_fill_bps,
                "n": float(w.n),
                "unsettled": float(w.unsettled),
            }
            for w in self.windows
        ]
