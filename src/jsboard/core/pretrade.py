"""Shared pre-trade primitives for every cross-market strategy.

The strategies used to carry private copies of the same three decisions:
whether two books were comparable, how far a market order could really fill,
and whether the visible edge survived costs.  Keeping those copies separate
made it possible for ``xarb`` and ``pair`` to disagree about the same books.

This module is deliberately strategy-free.  It knows nothing about z-scores,
basis, or triangular paths; it only answers the mechanical questions that all
of them must answer in exactly the same way.
"""

from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass

from .types import BookSnapshot, Instrument, Side


class RejectCode(enum.StrEnum):
    METADATA_INVALID = "METADATA_INVALID"
    FEED_DEGRADED = "FEED_DEGRADED"
    BOOK_GAP = "BOOK_GAP"
    STALE = "STALE"
    SKEW = "SKEW"
    NO_DEPTH = "NO_DEPTH"
    MIN_NOTIONAL = "MIN_NOTIONAL"
    BORROW_UNAVAILABLE = "BORROW_UNAVAILABLE"
    FUNDING_UNAVAILABLE = "FUNDING_UNAVAILABLE"
    POSITION_LIMIT = "POSITION_LIMIT"
    GROSS_TOO_SMALL = "GROSS_TOO_SMALL"
    NET_NEGATIVE = "NET_NEGATIVE"
    STRESS_NEGATIVE = "STRESS_NEGATIVE"
    COOLDOWN = "COOLDOWN"
    DUPLICATE_SIGNAL = "DUPLICATE_SIGNAL"


@dataclass(frozen=True, slots=True)
class MarketDataPoint:
    """The minimum state needed to decide whether a book is usable."""

    source: str
    updated_ns: int
    book_valid: bool = True
    feed_state: str = "live"
    metadata_valid: bool = True
    source_mode: str = "ws"

    def age_ms(self, now_ns: int) -> float:
        if self.updated_ns <= 0:
            return float("inf")
        return max(0.0, (now_ns - self.updated_ns) / 1e6)


@dataclass(frozen=True, slots=True)
class QualityDecision:
    allowed: bool
    reject_codes: tuple[RejectCode, ...]
    ages_ms: dict[str, float]
    skew_ms: float


def assess_market_data(
    points: list[MarketDataPoint] | tuple[MarketDataPoint, ...],
    *,
    now_ns: int,
    max_age_ms: float,
    max_skew_ms: float,
) -> QualityDecision:
    """Apply the common metadata/feed/book/age/skew gate.

    All failures are returned, not only the first one.  A run with no usable
    data must remain distinguishable from a healthy run with no opportunity.
    """
    if max_age_ms < 0 or max_skew_ms < 0:
        raise ValueError("age and skew limits must be non-negative")

    codes: list[RejectCode] = []
    ages = {point.source: point.age_ms(now_ns) for point in points}
    timestamps = [point.updated_ns for point in points if point.updated_ns > 0]
    skew_ms = (
        (max(timestamps) - min(timestamps)) / 1e6 if len(timestamps) >= 2 else 0.0
    )

    if not points or any(not point.metadata_valid for point in points):
        codes.append(RejectCode.METADATA_INVALID)
    if any(point.feed_state.lower() in {"disconnected", "resyncing", "degraded"} for point in points):
        codes.append(RejectCode.FEED_DEGRADED)
    if any(not point.book_valid for point in points):
        codes.append(RejectCode.BOOK_GAP)
    if any(not math.isfinite(age) or age > max_age_ms for age in ages.values()):
        codes.append(RejectCode.STALE)
    if skew_ms > max_skew_ms:
        codes.append(RejectCode.SKEW)

    unique = tuple(dict.fromkeys(codes))
    return QualityDecision(not unique, unique, ages, skew_ms)


@dataclass(frozen=True, slots=True)
class BookWalk:
    side: Side
    requested_lots: int
    filled_lots: int
    avg_price_ticks: float = 0.0
    best_price_ticks: int = 0
    notional_quote: float = 0.0
    slippage_bps: float = 0.0

    @property
    def remaining_lots(self) -> int:
        return max(0, self.requested_lots - self.filled_lots)

    @property
    def complete(self) -> bool:
        return self.requested_lots > 0 and self.remaining_lots == 0

    def filled_base(self, instrument: Instrument) -> float:
        return instrument.qty_f(self.filled_lots)

    def unfilled_base(self, instrument: Instrument) -> float:
        return instrument.qty_f(self.remaining_lots)

    def price(self, instrument: Instrument) -> float:
        return self.avg_price_ticks * float(instrument.tick_size)


def walk_book(
    snapshot: BookSnapshot,
    instrument: Instrument,
    side: Side,
    qty_base: float,
    *,
    max_levels: int | None = None,
) -> BookWalk:
    """Walk visible depth for a base-asset quantity without inventing fills."""
    if qty_base < 0:
        raise ValueError("quantity must be non-negative")
    if max_levels is not None and max_levels <= 0:
        raise ValueError("max_levels must be positive")

    requested = instrument.to_lots(qty_base)
    if requested <= 0:
        return BookWalk(side, requested_lots=0, filled_lots=0)
    levels = snapshot.asks if side is Side.BUY else snapshot.bids
    if max_levels is not None:
        levels = levels[:max_levels]
    best = levels[0].price if levels else 0
    remaining = requested
    filled = 0
    tick_lots = 0
    for level in levels:
        if remaining <= 0:
            break
        take = min(max(0, level.qty), remaining)
        if take <= 0:
            continue
        tick_lots += level.price * take
        filled += take
        remaining -= take

    if filled <= 0:
        return BookWalk(side, requested, 0, best_price_ticks=best)
    average = tick_lots / filled
    if best > 0:
        adverse_ticks = average - best if side is Side.BUY else best - average
        slippage_bps = max(0.0, adverse_ticks / best * 10_000.0)
    else:
        slippage_bps = 0.0
    notional = (
        tick_lots
        * float(instrument.tick_size)
        * float(instrument.lot_size)
    )
    return BookWalk(
        side=side,
        requested_lots=requested,
        filled_lots=filled,
        avg_price_ticks=average,
        best_price_ticks=best,
        notional_quote=notional,
        slippage_bps=slippage_bps,
    )


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Every pre-trade deduction, expressed in basis points.

    ``funding_bps`` is a signed cost: positive means paid, negative means the
    strategy expects to receive it.  Every other component is non-negative.
    """

    entry_fees_bps: float = 0.0
    expected_exit_fees_bps: float = 0.0
    entry_depth_slippage_bps: float = 0.0
    expected_exit_slippage_bps: float = 0.0
    funding_bps: float = 0.0
    borrow_bps: float = 0.0
    hedge_latency_buffer_bps: float = 0.0
    fill_model_buffer_bps: float = 0.0
    safety_margin_bps: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name != "funding_bps" and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def total_bps(self) -> float:
        return sum(asdict(self).values())

    def net_bps(self, gross_edge_bps: float) -> float:
        return gross_edge_bps - self.total_bps

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CostDecision:
    gross_edge_bps: float
    expected_net_bps: float
    stress_net_bps: float
    accepted: bool
    reject_codes: tuple[RejectCode, ...]
    costs: CostBreakdown
    stress_costs: CostBreakdown


def assess_costs(
    gross_edge_bps: float,
    costs: CostBreakdown,
    *,
    min_expected_net_bps: float = 0.0,
    stress_costs: CostBreakdown | None = None,
    min_gross_bps: float = 0.0,
) -> CostDecision:
    """Make the shared Gross -> expected Net -> stress Net decision."""
    if min_expected_net_bps < 0:
        raise ValueError("minimum expected net must be non-negative")
    stressed = stress_costs or costs
    expected = costs.net_bps(gross_edge_bps)
    stress = stressed.net_bps(gross_edge_bps)
    codes: list[RejectCode] = []
    if gross_edge_bps < min_gross_bps:
        codes.append(RejectCode.GROSS_TOO_SMALL)
    if expected < min_expected_net_bps:
        codes.append(RejectCode.NET_NEGATIVE)
    if stress < min_expected_net_bps:
        codes.append(RejectCode.STRESS_NEGATIVE)
    unique = tuple(dict.fromkeys(codes))
    return CostDecision(
        gross_edge_bps,
        expected,
        stress,
        not unique,
        unique,
        costs,
        stressed,
    )
