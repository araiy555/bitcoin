"""Mark-out split by the signal that was showing when the fill landed.

Every mark-out number this project has produced so far is an average over
every fill. `-0.83bps at 100ms` says the typical fill is adverse; it does not
say whether *all* fills are adverse or whether a minority of catastrophic
ones are dragging an otherwise profitable majority down. Those two worlds
call for opposite decisions — the first closes the strategy, the second says
quote only in the good band — and the aggregate cannot tell them apart.

So: bucket the fills by the toxicity score at the moment they filled, and
report mark-out per bucket.

Two things this has to get right.

**The gate must be off while measuring.** With the gate running, fills only
happen in the band the gate allows, so the very buckets the study exists to
examine are the ones with no data in them.

**Buys and sells are kept apart.** `MarkOutTracker` already signs its
mark-out so that "in our favour" is positive, which makes a buy and a sell
directly comparable — but a signal that predicts direction will help one side
and hurt the other, and averaging the two hides exactly that. Separate rows
are what make an asymmetric signal visible.

The score is read at fill time rather than at quote time. With a requote
interval of a millisecond the two are nearly the same moment, and fill time
is the one that is unambiguously observable in a replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..mm.markout import MarkOutTracker

DEFAULT_EDGES: tuple[float, ...] = (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0)
DEFAULT_HORIZONS: tuple[float, ...] = (0.1, 1.0, 10.0)


def bucket_of(score: float, edges: tuple[float, ...]) -> int:
    """Index of the band `score` falls in, clamped to the ends.

    Clamping rather than dropping: a score beyond the outermost edge is the
    most extreme case of the thing being measured, and discarding it would
    quietly remove the rows most likely to carry the answer.
    """
    if math.isnan(score):
        return -1
    for i in range(len(edges) - 1):
        if score < edges[i + 1]:
            return max(0, i)
    return len(edges) - 2


@dataclass(slots=True)
class EdgeStudy:
    """Mark-out per (score band, side), sharing one settling clock."""

    edges: tuple[float, ...] = DEFAULT_EDGES
    horizons_s: tuple[float, ...] = DEFAULT_HORIZONS
    _cells: dict[tuple[int, int], MarkOutTracker] = field(default_factory=dict)

    def _tracker(self, bucket: int, sign: int) -> MarkOutTracker:
        key = (bucket, sign)
        cell = self._cells.get(key)
        if cell is None:
            cell = MarkOutTracker(horizons_s=self.horizons_s)
            self._cells[key] = cell
        return cell

    def on_fill(
        self,
        now_ns: int,
        score: float,
        sign: int,
        price_ticks: float,
        weight: float,
        mid_ticks: float | None,
    ) -> None:
        bucket = bucket_of(score, self.edges)
        if bucket < 0 or sign == 0:
            return
        self._tracker(bucket, sign).on_fill(
            now_ns, sign, price_ticks, weight, mid_ticks=mid_ticks
        )

    def poll(self, now_ns: int, mid_ticks: float | None) -> None:
        for cell in self._cells.values():
            cell.poll(now_ns, mid_ticks)

    def rows(self) -> list[dict]:
        """One row per (band, side) that saw a fill, in band order."""
        out: list[dict] = []
        for bucket in range(len(self.edges) - 1):
            for sign in (1, -1):
                cell = self._cells.get((bucket, sign))
                if cell is None:
                    continue
                row: dict = {
                    "low": self.edges[bucket],
                    "high": self.edges[bucket + 1],
                    "side": "buy" if sign > 0 else "sell",
                    "n": 0.0,
                }
                for window in cell.summary():
                    row[f"mo_{window['horizon_s']:g}s"] = window["mean_bps"]
                    row["n"] = max(row["n"], window["n"])
                out.append(row)
        return out
