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

An optional fourth input looks at another venue. A market that leads this one
(the future for an ETF, the other exchange for a coin) shows where this book
is about to go; when it sits more than `lead_threshold_bps` away, the side it
is moving toward is withdrawn. The persistent gap between the two venues is
tracked with a slow average and removed first, so a standing basis is not
read as a move.
"""

from __future__ import annotations

import math
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

    lead_threshold_bps: float = 0.0
    """Withdraw the side the lead venue is moving toward once it sits this far
    from our mid, net of the usual basis. Zero disables it."""

    lead_basis_halflife_s: float = 60.0
    """How slowly the standing gap between the venues is learned. Much shorter
    and a real move is absorbed into the basis before it can be acted on."""


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
    lead: object | None = None
    """Anything with ``estimate(market) -> price in our ticks | None``; in
    practice :class:`jsboard.sim.pair.CrossMarketFairValue` over the other
    venue's book."""
    lead_blocks: int = 0
    _basis_bps: float | None = None
    _basis_ns: int = 0

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

    def lead_bps(self, market: MarketView) -> float | None:
        """How far the lead venue sits from our mid, net of the usual gap.

        Positive means the lead is above us: our book is about to rise, and
        an ask left resting is the one that gets picked off.
        """
        if self.lead is None:
            return None
        mid = market.mid
        lead = self.lead.estimate(market)
        if not mid or lead is None:
            return None
        if lead <= 0:
            return None
        # A log ratio, not a difference over our mid: the lead may be priced
        # in another currency (dollars against yen), and a difference would
        # shrink every move on it by the exchange rate. As a ratio the rate is
        # a standing offset, which the basis below absorbs.
        raw = math.log(lead / mid) * 1e4
        now = int(market.clock())
        if self._basis_bps is None:
            self._basis_bps, self._basis_ns = raw, now
            return 0.0
        gap = raw - self._basis_bps
        dt_s = max(0.0, (now - self._basis_ns) / 1e9)
        halflife = self.config.lead_basis_halflife_s
        weight = 1.0 - 0.5 ** (dt_s / halflife) if halflife > 0 else 1.0
        self._basis_bps += weight * (raw - self._basis_bps)
        self._basis_ns = now
        return gap

    def evaluate(self, market: MarketView) -> ToxicityDecision:
        decision = self._evaluate_own_book(market)
        cfg = self.config
        if cfg.lead_threshold_bps <= 0:
            return decision
        gap = self.lead_bps(market)
        if gap is None:
            return decision
        if gap >= cfg.lead_threshold_bps:
            blocked, reason = Side.SELL, f"lead {gap:+.1f}bps above; no asks"
        elif gap <= -cfg.lead_threshold_bps:
            blocked, reason = Side.BUY, f"lead {gap:+.1f}bps below; no bids"
        else:
            return decision
        if blocked not in decision.allowed_sides:
            return decision
        self.lead_blocks += 1
        allowed = decision.allowed_sides - {blocked}
        return ToxicityDecision(decision.score, allowed, reason)

    def _evaluate_own_book(self, market: MarketView) -> ToxicityDecision:
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
