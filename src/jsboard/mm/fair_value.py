"""Fair-value estimation.

The mid is the wrong anchor for a market maker: it ignores which side of the
book is thick, and a maker who quotes symmetrically around it gets adversely
selected. We start from the microprice instead and nudge it with two slower
signals — depth imbalance behind the touch, and recent trade flow.

All prices here are floats in *ticks*. Downstream rounding to a real tick
happens in the quoter, once, so intermediate estimates keep their precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.market import MarketView


@dataclass(slots=True)
class FairValueConfig:
    depth_levels: int = 5
    """How far behind the touch the imbalance signal looks."""

    imbalance_weight: float = 0.35
    """Fraction of the half-spread to lean by at full depth imbalance."""

    flow_weight: float = 0.25
    """Fraction of the half-spread to lean by at full trade-flow imbalance."""

    smoothing_halflife: float = 3.0
    """EWMA halflife in updates. 0 disables smoothing."""

    max_adjust_ticks: float = 0.0
    """Hard cap on the total nudge away from the microprice. 0 = spread-based."""


@dataclass(slots=True)
class FairValueEstimator:
    config: FairValueConfig = None  # type: ignore[assignment]
    _ewma: float | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = FairValueConfig()

    @property
    def _alpha(self) -> float:
        hl = self.config.smoothing_halflife
        if hl <= 0:
            return 1.0
        return 1.0 - math.exp(-math.log(2.0) / hl)

    def reset(self) -> None:
        self._ewma = None

    def estimate(self, market: MarketView) -> float | None:
        """Fair value in ticks, or None if the book is not two-sided."""
        snap = market.snapshot(self.config.depth_levels)
        anchor = snap.microprice
        if anchor is None:
            return None

        spread = snap.spread or 1
        half = spread / 2.0
        cfg = self.config

        imb = snap.imbalance(cfg.depth_levels)
        flow = market.flow.value

        adjust = half * (cfg.imbalance_weight * imb + cfg.flow_weight * flow)
        cap = cfg.max_adjust_ticks if cfg.max_adjust_ticks > 0 else half
        adjust = max(-cap, min(cap, adjust))

        raw = anchor + adjust

        if self._ewma is None:
            self._ewma = raw
        else:
            a = self._alpha
            self._ewma = (1 - a) * self._ewma + a * raw

        # Never let the estimate drift outside the touch — a fair value beyond
        # the best bid/ask says the market is free money, which it is not.
        if snap.best_bid is not None and snap.best_ask is not None:
            self._ewma = max(float(snap.best_bid), min(float(snap.best_ask), self._ewma))

        return self._ewma
