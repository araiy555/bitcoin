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

from collections import deque
from dataclasses import dataclass, field

from ..core.types import Instrument

NS_PER_S = 1_000_000_000

AGE_BOUNDS_S = (0.1, 1.0, 10.0)
"""Upper edges of the holding-age buckets; the last bucket is everything older."""

AGE_LABELS = ("0-100ms", "100ms-1s", "1-10s", "10s+")


@dataclass(slots=True)
class _Lot:
    """One open parcel of inventory, and when we acquired it."""

    acquired_ns: int
    lots: int  # signed, same sign as the position it belongs to


def _bucket_index(age_s: float) -> int:
    for i, bound in enumerate(AGE_BOUNDS_S):
        if age_s < bound:
            return i
    return len(AGE_BOUNDS_S)


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

    # Inventory P&L split by how long the parcel had been held when the move
    # happened. Each open lot carries its acquisition time, and the lots sum
    # to the position, so the buckets sum to the inventory term exactly —
    # unlike subtracting a fill-aggregated mark-out from a time-aggregated
    # P&L, which are not the same base and do not cancel.
    open_lots: deque[_Lot] = field(default_factory=deque)
    inventory_by_age: list[float] = field(default_factory=lambda: [0.0] * (len(AGE_BOUNDS_S) + 1))
    inventory_unaged: float = 0.0
    """Inventory moves booked with no clock available, so they belong to no
    bucket. Kept separate rather than dumped into one, so the buckets plus
    this always reconstruct the total."""

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
        # Ages are taken at the start of the interval — how long each parcel
        # had been held when the move began — so capture it before the clock
        # advances. The bucket edges are far wider than one update interval,
        # so the choice of endpoint moves very little across a boundary.
        interval_start = self.last_clock_ns
        self.on_clock(now_ns)
        if mid_ticks is None or mid_ticks <= 0:
            return
        if self.last_mid_ticks is not None and self.position_lots:
            move = (mid_ticks - self.last_mid_ticks) * self._tick
            self.inventory_pnl += move * self.instrument.qty_f(self.position_lots)
            self._split_by_age(move, interval_start if interval_start is not None else now_ns)
        self.last_mid_ticks = mid_ticks

    def _split_by_age(self, move: float, at_ns: int | None) -> None:
        """Hand this move to the buckets its parcels were sitting in."""
        if at_ns is None:
            self.inventory_unaged += move * self.instrument.qty_f(self.position_lots)
            return
        for lot in self.open_lots:
            age_s = max(0.0, (at_ns - lot.acquired_ns) / NS_PER_S)
            self.inventory_by_age[_bucket_index(age_s)] += move * self.instrument.qty_f(lot.lots)

    def _book_lots(self, sign: int, qty_lots: int, now_ns: int | None) -> None:
        """Keep the FIFO parcels in step with the position.

        Reducing consumes the oldest parcel first, which is what makes an age
        bucket mean anything: inventory that turns over quickly never reaches
        the older buckets.
        """
        acquired = now_ns if now_ns is not None else 0
        adding = self.position_lots == 0 or (self.position_lots > 0) == (sign > 0)
        if adding:
            self.open_lots.append(_Lot(acquired, sign * qty_lots))
            return

        remaining = qty_lots
        while remaining > 0 and self.open_lots:
            head = self.open_lots[0]
            take = min(abs(head.lots), remaining)
            head.lots -= (1 if head.lots > 0 else -1) * take
            remaining -= take
            if head.lots == 0:
                self.open_lots.popleft()
        if remaining > 0:
            # Flipped through zero: the residual is a new parcel, acquired now.
            self.open_lots.append(_Lot(acquired, sign * remaining))

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

        self._book_lots(sign, qty_lots, now_ns)
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

    def max_maker_bps(self, matched_qty: float) -> float:
        """The highest maker fee this run could have paid and still broken even.

        Stated as one side, since that is how a fee schedule is quoted: the
        edge before fees has to cover the fee on both legs of a round trip.
        Reading it as "which VIP tier would this need" is more useful than
        asking whether one particular fee happens to work.
        """
        bps = self.per_round_trip_bps(matched_qty)
        if not bps:
            return float("nan")
        return bps["net_before_fees"] / 2.0

    def age_buckets(self, matched_qty: float = 0.0) -> list[tuple[str, float, float]]:
        """(label, quote currency, bps) per holding-age bucket."""
        bps_scale = 0.0
        mid = self.last_mid_ticks
        if matched_qty > 0 and mid:
            notional = matched_qty * float(mid) * self._tick
            if notional > 0:
                bps_scale = 10_000.0 / notional
        rows = [
            (label, value, value * bps_scale)
            for label, value in zip(AGE_LABELS, self.inventory_by_age, strict=True)
        ]
        if self.inventory_unaged:
            rows.append(("時計なし", self.inventory_unaged, self.inventory_unaged * bps_scale))
        return rows

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
            "inventory_by_age": dict(zip(AGE_LABELS, self.inventory_by_age, strict=True)),
            "inventory_unaged": self.inventory_unaged,
        }
