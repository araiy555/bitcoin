"""Immediate hedging of maker fills in a second instrument.

The market maker earns its spread on one book and then carries the position
until something closes it. That carry is where the measured loss lives. A
hedge answers it directly: the instant a maker fill lands, take the opposite
side in a correlated instrument and pay to cross, turning an open directional
position into a spread position.

The hedge is not free, and the point of simulating it is to charge honestly
for it:

  * **crossing the spread** — the hedge is a taker, so it lifts the offer or
    hits the bid rather than resting. That is booked as negative spread
    capture, the mirror of what the maker leg earns.
  * **walking the book** — a hedge larger than the touch eats successive
    levels, so the average price is worse than the best. Sizes that exceed
    the visible book are only partially hedged rather than being filled at an
    invented price.
  * **the taker fee**, which is several times the maker fee.
  * **residual risk** — the two legs are marked against their own mids, so
    any basis move between them lands in the result instead of being assumed
    away.

Accounting reuses the maker leg's identity rather than inventing a second
one. The hedge gets its own PnLAttribution, in which "spread capture" is the
(negative) cost of crossing and "inventory" is the hedge position's own P&L.
The all-in result is the two attributions added together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.types import Instrument
from ..mm.attribution import PnLAttribution
from ..mm.inventory import FeeSchedule


@dataclass(slots=True)
class HedgeConfig:
    ratio: float = 1.0
    """Fraction of the maker fill to offset. 0 disables hedging entirely."""

    taker_bps: float = 5.0
    """What crossing costs on the hedge venue."""

    max_levels: int = 20
    """How deep into the book one hedge may reach before giving up."""


@dataclass(slots=True)
class HedgeResult:
    filled_base: float = 0.0
    avg_price_ticks: float = 0.0
    unfilled_base: float = 0.0


@dataclass(slots=True)
class Hedger:
    """Crosses the hedge book to offset maker fills as they happen."""

    instrument: Instrument
    market: MarketView
    config: HedgeConfig = field(default_factory=HedgeConfig)
    attribution: PnLAttribution = field(init=False)
    fees: FeeSchedule = field(init=False)

    hedges: int = 0
    skipped_no_book: int = 0
    unfilled_base: float = 0.0

    def __post_init__(self) -> None:
        self.attribution = PnLAttribution(self.instrument)
        self.fees = FeeSchedule(maker_bps=0.0, taker_bps=self.config.taker_bps)

    # ------------------------------------------------------------- plumbing

    @property
    def _clock_ns(self) -> int | None:
        return self.market.last_update_ns or None

    def on_market(self) -> None:
        """Mark the hedge position after the hedge book moves."""
        self.attribution.on_mid(self.market.mid, self._clock_ns)

    # -------------------------------------------------------------- hedging

    def walk(self, sign: int, qty_base: float) -> HedgeResult:
        """Average price for crossing `qty_base` in `sign` direction.

        `sign` is +1 to buy the hedge instrument, -1 to sell. Only visible
        depth is consumed; whatever the book cannot supply comes back as
        unfilled rather than being filled at a price nobody showed.
        """
        snapshot = self.market.book.snapshot(self.config.max_levels)
        levels = snapshot.asks if sign > 0 else snapshot.bids
        remaining = self.instrument.to_lots(qty_base)
        if remaining <= 0:
            return HedgeResult()

        filled = 0
        cost = 0.0
        for level in levels:
            if remaining <= 0:
                break
            take = min(level.qty, remaining)
            if take <= 0:
                continue
            cost += level.price * take
            filled += take
            remaining -= take

        if filled <= 0:
            return HedgeResult(unfilled_base=qty_base)
        return HedgeResult(
            filled_base=self.instrument.qty_f(filled),
            avg_price_ticks=cost / filled,
            unfilled_base=self.instrument.qty_f(remaining),
        )

    def on_maker_fill(self, maker_sign: int, qty_base: float) -> HedgeResult:
        """Offset a maker fill. `maker_sign` is +1 if the maker leg bought."""
        if self.config.ratio <= 0 or qty_base <= 0:
            return HedgeResult()

        mid = self.market.mid
        if mid is None:
            # No hedge book yet. Recorded rather than silently treated as a
            # free hedge, since an unhedged fill is exactly the exposure the
            # whole test is about.
            self.skipped_no_book += 1
            return HedgeResult(unfilled_base=qty_base)

        sign = -maker_sign
        result = self.walk(sign, qty_base * self.config.ratio)
        if result.filled_base <= 0:
            self.skipped_no_book += 1
            self.unfilled_base += result.unfilled_base
            return result

        notional = result.filled_base * result.avg_price_ticks * float(self.instrument.tick_size)
        fee = self.fees.cost(notional, is_maker=False)
        self.attribution.on_fill(
            price_ticks=result.avg_price_ticks,
            qty_lots=self.instrument.to_lots(result.filled_base),
            sign=sign,
            fee=fee,
            mid_ticks=mid,
            now_ns=self._clock_ns,
        )
        self.hedges += 1
        self.unfilled_base += result.unfilled_base
        return result

    def summary(self) -> dict:
        out = self.attribution.summary()
        out.update(
            {
                "hedges": float(self.hedges),
                "skipped": float(self.skipped_no_book),
                "unfilled_base": self.unfilled_base,
            }
        )
        return out
