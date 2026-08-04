"""Is there room to be right in, before asking whether we can be right.

A direction trade earns its move when right, loses its move when wrong, and
pays the round-trip fee either way. In general:

    expected = p · win − (1 − p) · loss − cost
    p*       = (loss + cost) / (win + loss)

`win` and `loss` are the average move on the trades taken, split by outcome —
they are *not* the same quantity, and a strategy with a stop or a target makes
them deliberately different.

**What this module reports is the symmetric special case**, `win == loss ==
mean|move|`, which collapses to `p* = (1 + cost/m) / 2`. That is the right
number for exactly one strategy: enter at every timepoint, hold a fixed
duration, exit regardless. It is the cheapest thing to check first and it
settles that one design, but it does not generalise. `required_accuracy_for`
takes the two magnitudes separately for anything that does.

The scope of a `p* > 1` verdict is therefore narrow and worth stating
precisely: **at every timepoint, taker in and taker out, fixed horizon, no
predictor of any quality wins.** It says nothing about a strategy that trades
only on some condition. Conditioning on a large print, a side of the book
vanishing, spot and perp diverging, a liquidation, or a jump in volatility
gives a conditional mean move that can be far above the unconditional one —
which is a different question, and this module does not answer it.

The scan reached the same wall from the other side: symbols failed on the fee
rather than on the book. This asks whether the fixed-horizon direction idea
meets it too, using years of real trades instead of an argument.

`mean_abs_move` is the mean rather than the median on purpose. A selective
strategy sees something closer to an upper tail than to the middle, so the
median would understate what selection can reach.
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
        """Break-even accuracy **assuming wins and losses are the same size**.

        True for entering at every timepoint and exiting on a timer, which is
        what this module measures. A strategy with a target or a stop breaks
        that assumption — use `required_accuracy_for` there.
        """
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


def required_accuracy_for(win_bps: float, loss_bps: float, cost_bps: float) -> float:
    """Break-even accuracy when a win and a loss are different sizes.

        p* = (loss + cost) / (win + loss)

    A target-and-stop strategy sets these deliberately: taking +20bps and
    cutting at −10bps needs 63.3% at a 9bps cost, where the symmetric reading
    of the same 15bps average move demands 80%. Reporting only the symmetric
    number would rule out designs that are not actually ruled out.

    Returns infinity when neither side can move, since no accuracy helps.
    """
    total = win_bps + loss_bps
    if total <= 0:
        return math.inf
    return (loss_bps + cost_bps) / total


def expected_bps(
    accuracy: float, win_bps: float, loss_bps: float, cost_bps: float
) -> float:
    """Expected result per round trip. Negative means the design loses."""
    return accuracy * win_bps - (1.0 - accuracy) * loss_bps - cost_bps


def round_trip_cost_bps(taker_bps: float, slippage_ticks: float, tick_bps: float) -> float:
    """Entry and exit, both crossing the spread.

    Two fees and two crossings — the common mistake is counting one of each,
    which halves the wall the strategy has to clear.
    """
    return 2.0 * taker_bps + 2.0 * slippage_ticks * tick_bps
