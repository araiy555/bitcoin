"""Position, average cost, and P&L accounting.

Average-cost method: adding to a position moves the average, reducing one
realises P&L against it, and flipping through zero closes the old position
before opening the new one at the fill price. Prices stay in ticks and sizes
in lots internally; everything the outside world reads is in quote currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import Fill, Instrument, Side


@dataclass(slots=True)
class FeeSchedule:
    """Venue fees in basis points of notional. Negative maker fee = rebate."""

    maker_bps: float = 1.0
    taker_bps: float = 4.0

    def cost(self, notional: float, is_maker: bool) -> float:
        return notional * (self.maker_bps if is_maker else self.taker_bps) / 10_000.0


@dataclass(slots=True)
class Position:
    """Signed inventory and realised/unrealised P&L for one instrument."""

    instrument: Instrument
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    lots: int = 0
    avg_price_ticks: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    volume_lots: int = 0
    buy_lots: int = 0
    sell_lots: int = 0
    fill_count: int = 0
    maker_fills: int = 0
    taker_fills: int = 0

    # ---------------------------------------------------------------- state

    @property
    def is_flat(self) -> bool:
        return self.lots == 0

    @property
    def qty(self) -> float:
        """Signed position in base units."""
        return self.instrument.qty_f(self.lots)

    @property
    def avg_price(self) -> float:
        return self.avg_price_ticks * float(self.instrument.tick_size)

    def notional(self, mark_ticks: float) -> float:
        return abs(self.qty) * mark_ticks * float(self.instrument.tick_size)

    # --------------------------------------------------------------- update

    def on_fill(self, price_ticks: int, qty_lots: int, side: Side, is_maker: bool) -> float:
        """Book a fill. Returns the realised P&L it produced (may be 0)."""
        if qty_lots <= 0:
            return 0.0

        notional = self.instrument.notional(price_ticks, qty_lots)
        fee = self.fees.cost(notional, is_maker)
        self.fees_paid += fee
        self.volume_lots += qty_lots
        if side is Side.BUY:
            self.buy_lots += qty_lots
        else:
            self.sell_lots += qty_lots
        self.fill_count += 1
        if is_maker:
            self.maker_fills += 1
        else:
            self.taker_fills += 1

        signed = side.sign * qty_lots
        realized = 0.0

        if self.lots == 0 or (self.lots > 0) == (signed > 0):
            # Opening or adding: blend into the average.
            total = abs(self.lots) + qty_lots
            self.avg_price_ticks = (
                self.avg_price_ticks * abs(self.lots) + price_ticks * qty_lots
            ) / total
            self.lots += signed
        else:
            # Reducing, closing, or flipping.
            closing = min(abs(self.lots), qty_lots)
            direction = 1 if self.lots > 0 else -1
            realized = (
                (price_ticks - self.avg_price_ticks)
                * direction
                * float(self.instrument.tick_size)
                * self.instrument.qty_f(closing)
            )
            self.realized_pnl += realized
            self.lots += signed
            if self.lots == 0:
                self.avg_price_ticks = 0.0
            elif (self.lots > 0) != (direction > 0):
                # Flipped through zero: the residual opened at this fill price.
                self.avg_price_ticks = float(price_ticks)

        self.realized_pnl -= fee
        return realized - fee

    def apply(self, fill: Fill, owner: str) -> float:
        """Book whichever side(s) of `fill` belong to `owner`."""
        total = 0.0
        if fill.maker_owner == owner:
            total += self.on_fill(fill.price, fill.qty, fill.aggressor.opposite, is_maker=True)
        if fill.taker_owner == owner:
            total += self.on_fill(fill.price, fill.qty, fill.aggressor, is_maker=False)
        return total

    # ----------------------------------------------------------------- P&L

    def unrealized_pnl(self, mark_ticks: float | None) -> float:
        if mark_ticks is None or self.lots == 0:
            return 0.0
        return (
            (mark_ticks - self.avg_price_ticks)
            * float(self.instrument.tick_size)
            * self.instrument.qty_f(self.lots)
        )

    def total_pnl(self, mark_ticks: float | None) -> float:
        return self.realized_pnl + self.unrealized_pnl(mark_ticks)

    @property
    def gross_pnl(self) -> float:
        """Realised P&L before fees — the spread we actually captured.

        `realized_pnl` has every fee already subtracted from it, so adding
        them back recovers the gross figure exactly. Kept as a property
        rather than a second accumulator so the two can never drift apart.
        """
        return self.realized_pnl + self.fees_paid

    def summary(self, mark_ticks: float | None) -> dict[str, float]:
        return {
            "position": self.qty,
            "avg_price": self.avg_price,
            "gross": self.gross_pnl,
            "bought": self.instrument.qty_f(self.buy_lots),
            "sold": self.instrument.qty_f(self.sell_lots),
            "realized": self.realized_pnl,
            "unrealized": self.unrealized_pnl(mark_ticks),
            "total": self.total_pnl(mark_ticks),
            "fees": self.fees_paid,
            "fills": float(self.fill_count),
            "volume": self.instrument.qty_f(self.volume_lots),
        }


@dataclass(slots=True)
class PnLTracker:
    """Equity high-water mark and drawdown, for the risk layer to act on."""

    peak: float = 0.0
    trough: float = 0.0
    last: float = 0.0

    def update(self, equity: float) -> None:
        self.last = equity
        self.peak = max(self.peak, equity)
        self.trough = min(self.trough, equity)

    @property
    def drawdown(self) -> float:
        """How far below the high-water mark we are, as a positive number."""
        return max(0.0, self.peak - self.last)
