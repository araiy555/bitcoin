"""Where a market maker's P&L actually came from.

A single P&L number cannot distinguish a strategy with no edge from one whose
edge is real but smaller than its costs, and the two call for opposite
responses. This splits the total into terms that sum back to it exactly:

    Total = Spread Capture + Inventory P&L − Fees

The identity is not an approximation. Mark equity as cash plus position times
mid, and every change falls into exactly one term:

  * a fill moves cash by −side·price·qty and position by +side·qty, so at an
    unchanged mid the equity change is side·(mid − price)·qty. That is the
    **spread capture**: what we earned for standing away from the mid. It is
    booked once, at the instant of the fill, and never revised.

  * between fills, equity moves by position times the change in mid. That is
    the **inventory P&L**: the cost or benefit of carrying what we bought.

Adverse selection then sits *inside* the inventory term rather than beside
it. Mark-out over the first τ seconds after each fill is the part of the
inventory move that follows our own fills; whatever is left is exposure we
took on and held. So:

    Inventory P&L = mark-out(τ) + residual carry

which is why a neutral mark-out does not by itself prove the loss is
inventory rather than adverse selection — it proves only that the loss is not
in the first τ seconds. Both readings need the terms measured, not inferred
from a gap between two numbers.

Hedging, when it lands, adds two more subtractions (hedge P&L and hedge cost)
without disturbing the identity above.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import Instrument


@dataclass(slots=True)
class PnLAttribution:
    """Accumulates the terms of Total = Spread + Inventory − Fees."""

    instrument: Instrument
    spread_capture: float = 0.0
    inventory_pnl: float = 0.0
    fees: float = 0.0
    position_lots: int = 0
    last_mid_ticks: float | None = None

    # How long the position was carried, which is the exposure the inventory
    # term is the price of. A run that is flat 99% of the time and one that
    # never goes flat can report the same inventory P&L for very different
    # reasons, and only the second is a case for hedging.
    elapsed_ns: int = 0
    exposed_ns: int = 0
    abs_position_ns: float = 0.0
    last_clock_ns: int | None = None

    fills_priced: int = 0
    fills_unpriced: int = 0
    """Fills booked with no mid available. Their spread capture is unknowable,
    so it is not guessed at — but the count has to be visible, since a run
    with many of them has an incomplete decomposition rather than a small
    residual."""

    @property
    def _tick(self) -> float:
        return float(self.instrument.tick_size)

    def on_clock(self, now_ns: int | None) -> None:
        """Advance the exposure clock, crediting the position we held over it."""
        if now_ns is None:
            return
        if self.last_clock_ns is not None:
            elapsed = now_ns - self.last_clock_ns
            if elapsed > 0:
                self.elapsed_ns += elapsed
                if self.position_lots:
                    self.exposed_ns += elapsed
                    self.abs_position_ns += abs(self.instrument.qty_f(self.position_lots)) * elapsed
        self.last_clock_ns = now_ns

    def on_mid(self, mid_ticks: float | None, now_ns: int | None = None) -> None:
        """Mark the carried position to a new mid."""
        self.on_clock(now_ns)
        if mid_ticks is None or mid_ticks <= 0:
            return
        if self.last_mid_ticks is not None and self.position_lots:
            move = (mid_ticks - self.last_mid_ticks) * self._tick
            self.inventory_pnl += move * self.instrument.qty_f(self.position_lots)
        self.last_mid_ticks = mid_ticks

    def on_fill(
        self,
        *,
        price_ticks: float,
        qty_lots: int,
        sign: int,
        fee: float,
        mid_ticks: float | None,
        now_ns: int | None = None,
    ) -> None:
        """Book one of our fills. `sign` is +1 when we bought, -1 when we sold."""
        if qty_lots <= 0:
            return
        # Carry the existing position up to this moment before the fill
        # changes it, or the new size would be credited with an older move.
        self.on_mid(mid_ticks, now_ns)

        if mid_ticks is None or mid_ticks <= 0:
            self.fills_unpriced += 1
        else:
            edge = sign * (mid_ticks - price_ticks) * self._tick
            self.spread_capture += edge * self.instrument.qty_f(qty_lots)
            self.fills_priced += 1

        self.position_lots += sign * qty_lots
        self.fees += fee

    @property
    def total(self) -> float:
        return self.spread_capture + self.inventory_pnl - self.fees

    @property
    def net_before_fees(self) -> float:
        """What the strategy earned before the venue took its cut."""
        return self.spread_capture + self.inventory_pnl

    @property
    def exposed_share(self) -> float:
        """Fraction of the session spent holding a position at all."""
        return self.exposed_ns / self.elapsed_ns if self.elapsed_ns else 0.0

    @property
    def mean_abs_position(self) -> float:
        """Time-weighted average absolute position, in base units."""
        return self.abs_position_ns / self.elapsed_ns if self.elapsed_ns else 0.0

    def per_round_trip_bps(self, matched_qty: float) -> dict[str, float]:
        """The same terms as basis points of the notional actually turned over."""
        mid = self.last_mid_ticks
        if matched_qty <= 0 or not mid:
            return {}
        notional = matched_qty * float(mid) * self._tick
        if notional <= 0:
            return {}
        scale = 10_000.0 / notional
        return {
            "spread_capture": self.spread_capture * scale,
            "inventory": self.inventory_pnl * scale,
            "net_before_fees": self.net_before_fees * scale,
            "fees": -self.fees * scale,
            "total": self.total * scale,
        }

    def summary(self) -> dict[str, float]:
        return {
            "spread_capture": self.spread_capture,
            "inventory": self.inventory_pnl,
            "fees": self.fees,
            "net_before_fees": self.net_before_fees,
            "total": self.total,
            "unpriced_fills": float(self.fills_unpriced),
            "exposed_share": self.exposed_share,
            "mean_abs_position": self.mean_abs_position,
        }
