"""Pre-trade risk gate.

Every quoting cycle passes through `RiskManager.evaluate` before any order is
sent. The gate can allow quoting, restrict it to one side, or halt outright.
Halting is *sticky*: a breached drawdown limit does not un-breach itself when
the next tick looks better, so the kill switch stays latched until someone
resets it deliberately.

The staleness checks matter more than they look. Quoting against a book that
stopped updating ten seconds ago is how a maker ends up as the only bid in a
falling market.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.types import Side
from .inventory import PnLTracker, Position


class RiskAction(enum.Enum):
    QUOTE = "QUOTE"
    """Normal two-sided quoting."""

    ONE_SIDED = "ONE_SIDED"
    """Quote only the side that reduces inventory."""

    PULL = "PULL"
    """Cancel everything, but stay running and re-evaluate next tick."""

    HALT = "HALT"
    """Latched stop. Cancel everything and do not resume."""


@dataclass(frozen=True, slots=True)
class RiskDecision:
    action: RiskAction
    reason: str = ""
    allowed_sides: frozenset[Side] = field(default=frozenset({Side.BUY, Side.SELL}))

    @property
    def can_quote(self) -> bool:
        return self.action in (RiskAction.QUOTE, RiskAction.ONE_SIDED)

    def permits(self, side: Side) -> bool:
        return self.can_quote and side in self.allowed_sides


@dataclass(slots=True)
class RiskLimits:
    max_position_lots: int = 1000
    max_notional: float = 250_000.0
    max_drawdown: float = 2_000.0
    max_book_age_ms: float = 2_000.0
    max_spread_ticks: int = 100
    max_sigma_bps: float = 25.0
    """Per-update realised vol that counts as a dislocation. In bps so the
    limit means the same thing regardless of tick size or price level."""
    min_book_levels: int = 2


@dataclass(slots=True)
class RiskManager:
    limits: RiskLimits = field(default_factory=RiskLimits)
    pnl: PnLTracker = field(default_factory=PnLTracker)
    halted: bool = False
    halt_reason: str = ""
    last_decision: RiskDecision | None = None

    def reset(self) -> None:
        """Clear a latched halt. Deliberate operator action, never automatic."""
        self.halted = False
        self.halt_reason = ""

    def evaluate(self, market: MarketView, position: Position) -> RiskDecision:
        decision = self._evaluate(market, position)
        self.last_decision = decision
        return decision

    def _evaluate(self, market: MarketView, position: Position) -> RiskDecision:
        lim = self.limits

        if self.halted:
            return RiskDecision(RiskAction.HALT, self.halt_reason, frozenset())

        # --- equity / drawdown (latching) ----------------------------------
        mark = market.mid
        equity = position.total_pnl(mark)
        self.pnl.update(equity)
        if lim.max_drawdown > 0 and self.pnl.drawdown > lim.max_drawdown:
            return self._halt(
                f"drawdown {self.pnl.drawdown:,.2f} exceeded limit {lim.max_drawdown:,.2f}"
            )

        # --- market-data health --------------------------------------------
        if not market.is_live:
            return RiskDecision(RiskAction.PULL, f"feed is {market.status}", frozenset())

        age = market.age_ms
        if age > lim.max_book_age_ms:
            return RiskDecision(RiskAction.PULL, f"book is stale by {age:,.0f}ms", frozenset())

        snap = market.snapshot(lim.min_book_levels)
        if len(snap.bids) < lim.min_book_levels or len(snap.asks) < lim.min_book_levels:
            return RiskDecision(RiskAction.PULL, "book too thin to quote", frozenset())

        spread = snap.spread
        if spread is None or spread > lim.max_spread_ticks:
            return RiskDecision(RiskAction.PULL, f"spread {spread} ticks is dislocated", frozenset())

        sigma_bps = market.vol.bps
        if lim.max_sigma_bps > 0 and sigma_bps > lim.max_sigma_bps:
            return RiskDecision(
                RiskAction.PULL, f"volatility {sigma_bps:,.1f}bps too high", frozenset()
            )

        # --- position limits -------------------------------------------------
        if mark is not None and lim.max_notional > 0 and position.notional(mark) > lim.max_notional:
            reducing = Side.SELL if position.lots > 0 else Side.BUY
            return RiskDecision(
                RiskAction.ONE_SIDED,
                f"notional {position.notional(mark):,.0f} over limit; reducing only",
                frozenset({reducing}),
            )

        if lim.max_position_lots > 0 and abs(position.lots) >= lim.max_position_lots:
            reducing = Side.SELL if position.lots > 0 else Side.BUY
            return RiskDecision(
                RiskAction.ONE_SIDED,
                f"position {position.lots} lots at limit; reducing only",
                frozenset({reducing}),
            )

        return RiskDecision(RiskAction.QUOTE, "ok")

    def _halt(self, reason: str) -> RiskDecision:
        self.halted = True
        self.halt_reason = reason
        return RiskDecision(RiskAction.HALT, reason, frozenset())
