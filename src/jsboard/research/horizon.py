"""Is there room to be right in, before asking whether we can be right.

A direction prediction pays `move` when correct and costs `move` when wrong,
and pays the round-trip taker fee either way. So with accuracy `p` on a move
of typical size `m`:

    expected = (2p - 1) · m − cost

Setting that to zero gives the accuracy the strategy would need:

    p* = (1 + cost / m) / 2

That single number decides whether the idea is worth pursuing at a given
horizon, and it needs no model — only the size of the moves and the fee. When
`cost` exceeds `m`, `p*` exceeds 1 and **no predictor of any quality wins**,
because being right every single time still does not cover the fee. That is
not a hard problem, it is an impossible one, and it is cheap to check first.

The scan reached the same conclusion from the other side: symbols failed on
the fee rather than on the book. This asks whether the direction idea meets
the same wall at 1–10 seconds, using years of real trades instead of an
argument.

`mean_abs_move` is deliberately the mean, not the median. A strategy that
trades only when it expects a large move sees something closer to an upper
tail than to the middle, so the median would understate what a selective
strategy can reach — and the mean is already generous enough to be a fair
ceiling.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .archive import SecondBar


@dataclass(frozen=True, slots=True)
class HorizonStats:
    """What the market offers at one holding period, before any model."""

    horizon_s: int
    samples: int
    mean_abs_bps: float
    median_abs_bps: float
    p75_abs_bps: float
    p90_abs_bps: float
    p99_abs_bps: float
    cost_bps: float
    mean_top10_bps: float = 0.0
    mean_top1_bps: float = 0.0

    @property
    def required_accuracy(self) -> float:
        """Direction accuracy needed to break even on an average move."""
        return self._accuracy_for(self.mean_abs_bps)

    def _accuracy_for(self, move_bps: float) -> float:
        if move_bps <= 0:
            return math.inf
        return (1.0 + self.cost_bps / move_bps) / 2.0

    @property
    def required_accuracy_top10(self) -> float:
        """Same, if only the largest tenth of moves were ever traded."""
        return self._accuracy_for(self.mean_top10_bps)

    @property
    def required_accuracy_top1(self) -> float:
        return self._accuracy_for(self.mean_top1_bps)

    @property
    def selective_is_possible(self) -> bool:
        """Whether trading only the big moves escapes the fee at all.

        This is the last door left when the average move loses. It does not
        say the selection is achievable — knowing in advance which seconds
        are about to move is its own prediction problem, and a harder one
        than direction. It says only whether the arithmetic permits it.
        """
        return self.required_accuracy_top10 < 1.0

    @property
    def is_possible(self) -> bool:
        return self.required_accuracy < 1.0

    @property
    def perfect_foresight_bps(self) -> float:
        """Per-trade edge with a flawless predictor. The ceiling, not a target."""
        return self.mean_abs_bps - self.cost_bps

    @property
    def tradeable_fraction(self) -> float:
        """Share of moments whose move alone would clear the fee."""
        return self._fraction_over

    _fraction_over: float = 0.0


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def forward_returns(bars: list[SecondBar], horizon_s: int) -> list[float]:
    """Signed returns in bps over `horizon_s`, in seconds of real time.

    Bars exist only for seconds that traded, so the series is indexed by
    timestamp rather than by position — stepping N rows ahead would mean "N
    prints later", which on a quiet market is minutes rather than seconds.

    The price at `t + horizon` is the last print at or before that instant. A
    window with no prints in it therefore returns zero, which is the truth:
    nothing traded, so the price did not move.

    Windows extending past the last bar are dropped rather than filled. There
    is no future to compare against, and filling them with the current price
    would manufacture a run of zero-return samples at the end of every day and
    drag the average move down.
    """
    if horizon_s <= 0:
        raise ValueError("horizon must be positive")
    if not bars:
        return []

    secs = [b.sec for b in bars]
    prices = [b.last for b in bars]
    last_sec = secs[-1]

    out: list[float] = []
    for bar in bars:
        target = bar.sec + horizon_s
        if target > last_sec:
            continue
        j = bisect.bisect_right(secs, target) - 1
        out.append((prices[j] - bar.last) / bar.last * 10_000.0)
    return out


def analyse(bars: list[SecondBar], horizon_s: int, cost_bps: float) -> HorizonStats:
    moves = [abs(r) for r in forward_returns(bars, horizon_s)]
    moves.sort()
    n = len(moves)
    mean = sum(moves) / n if n else 0.0
    over = sum(1 for m in moves if m > cost_bps) / n if n else 0.0
    # Conditional means, not the percentile itself: a strategy that trades
    # the top decile earns the average of that decile, which sits well above
    # the number at its boundary.
    top10 = moves[int(n * 0.9) :]
    top1 = moves[int(n * 0.99) :]
    return HorizonStats(
        horizon_s=horizon_s,
        samples=n,
        mean_abs_bps=mean,
        median_abs_bps=_percentile(moves, 0.50),
        p75_abs_bps=_percentile(moves, 0.75),
        p90_abs_bps=_percentile(moves, 0.90),
        p99_abs_bps=_percentile(moves, 0.99),
        cost_bps=cost_bps,
        mean_top10_bps=sum(top10) / len(top10) if top10 else 0.0,
        mean_top1_bps=sum(top1) / len(top1) if top1 else 0.0,
        _fraction_over=over,
    )


def round_trip_cost_bps(taker_bps: float, slippage_ticks: float, tick_bps: float) -> float:
    """Entry and exit, both crossing the spread.

    Two fees and two crossings — the common mistake is counting one of each,
    which halves the wall the strategy has to clear.
    """
    return 2.0 * taker_bps + 2.0 * slippage_ticks * tick_bps
