"""Pre-trade directional-toxicity gate.

The fair-value estimator moves quotes toward pressure but still leaves both
sides resting. That is not enough when the market selectively trades with the
side that is about to become stale. This gate uses only information already
visible at the quoting decision:

* recent signed aggressive trade flow,
* multi-level book imbalance, and
* the top-of-book microprice displacement.

A positive score means upward pressure, so selling is the toxic side. A
negative score means downward pressure, so buying is the toxic side. Moderate
pressure removes only that side; extreme pressure pulls both sides. No future
price or post-fill mark-out enters the decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.types import Side


@dataclass(slots=True)
class ToxicityConfig:
    threshold: float = 0.0
    """One-sided gate threshold in [0, 1]. Zero disables the gate."""

    pull_threshold: float = 0.90
    """Absolute score at which both sides are pulled."""

    depth_levels: int = 5
    flow_weight: float = 0.50
    book_weight: float = 0.30
    microprice_weight: float = 0.20


@dataclass(frozen=True, slots=True)
class ToxicityDecision:
    score: float
    allowed_sides: frozenset[Side]
    reason: str

    @property
    def pulled(self) -> bool:
        return not self.allowed_sides

    @property
    def one_sided(self) -> bool:
        return len(self.allowed_sides) == 1

    def permits(self, side: Side) -> bool:
        return side in self.allowed_sides


@dataclass(slots=True)
class ToxicityGate:
    config: ToxicityConfig = field(default_factory=ToxicityConfig)

    @staticmethod
    def _clip(value: float) -> float:
        return max(-1.0, min(1.0, value))

    def score(self, market: MarketView) -> float:
        cfg = self.config
        snap = market.snapshot(cfg.depth_levels)
        mid = snap.mid
        spread = snap.spread
        micro = snap.microprice

        micro_signal = 0.0
        if mid is not None and micro is not None and spread and spread > 0:
            micro_signal = self._clip((micro - mid) / (spread / 2.0))

        flow = self._clip(market.flow.value)
        book = self._clip(snap.imbalance(cfg.depth_levels))
        total_weight = abs(cfg.flow_weight) + abs(cfg.book_weight) + abs(cfg.microprice_weight)
        if total_weight <= 0:
            return 0.0

        return self._clip(
            (
                cfg.flow_weight * flow
                + cfg.book_weight * book
                + cfg.microprice_weight * micro_signal
            )
            / total_weight
        )

    def evaluate(self, market: MarketView) -> ToxicityDecision:
        cfg = self.config
        both = frozenset({Side.BUY, Side.SELL})
        if cfg.threshold <= 0:
            return ToxicityDecision(0.0, both, "disabled")

        score = self.score(market)
        pull = max(cfg.threshold, cfg.pull_threshold)
        if abs(score) >= pull:
            return ToxicityDecision(score, frozenset(), f"toxicity {score:+.2f}; pull all")
        if score >= cfg.threshold:
            return ToxicityDecision(
                score,
                frozenset({Side.BUY}),
                f"upward toxicity {score:+.2f}; no asks",
            )
        if score <= -cfg.threshold:
            return ToxicityDecision(
                score,
                frozenset({Side.SELL}),
                f"downward toxicity {score:+.2f}; no bids",
            )
        return ToxicityDecision(score, both, f"toxicity {score:+.2f}; neutral")
