"""Quote generation: where to stand, how wide, and how big.

The two ideas are Avellaneda–Stoikov's:

  reservation price   r = s − (inventory skew)
      Holding inventory makes you want to sell. Shifting the *whole* quote
      pair against the position means the market mean-reverts your inventory
      for you, instead of you crossing the spread to flatten.

  half-spread δ = (inventory risk term) + (liquidity term)
      Wider when volatile, wider when the book is thin and slow to refill.

The parameterisation is not theirs, and deliberately so. In the paper γ is
dimensional — γ·σ²·T only yields a price because γ carries units of
1/(price·quantity). Ported naively into tick space that makes every parameter
instrument-specific: the same γ that quotes a sane spread on a $0.01 tick
produces a five-figure one somewhere else, and σ² in ticks overflows any
sensible clamp long before it means anything.

So the same structure is expressed in units that survive a change of
instrument:

  skew_ticks  = (inventory_skew_ticks + gamma·σ_ticks) · q_norm
  half_spread = base + vol_multiplier·σ_ticks + liquidity_premium/κ

`q_norm` is inventory as a fraction of the position limit, so it is bounded in
[-1, 1]; σ is in ticks, so both terms are ticks; and γ, `vol_multiplier` and κ
are dimensionless and mean the same thing on every symbol. Skew stays linear
in inventory (as A–S has it) and scales with volatility (as σ² does, but
without the units problem).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..core.types import Side


@dataclass(frozen=True, slots=True)
class Quote:
    side: Side
    price: int  # ticks
    qty: int  # lots
    level: int = 0

    def key(self) -> tuple[int, int]:
        """Identity for diffing against live orders."""
        return (int(self.side), self.price)


@dataclass(frozen=True, slots=True)
class QuoteSet:
    bids: tuple[Quote, ...] = ()
    asks: tuple[Quote, ...] = ()
    fair_value: float | None = None
    reservation: float | None = None
    half_spread: float | None = None
    reason: str = ""

    def all(self) -> tuple[Quote, ...]:
        return self.bids + self.asks

    @property
    def is_empty(self) -> bool:
        return not self.bids and not self.asks


@dataclass(slots=True)
class QuoterConfig:
    # --- pricing (all dimensionless unless the name says ticks) ------------
    gamma: float = 0.6
    """Risk aversion: how many σ of skew to apply at a full position."""

    kappa: float = 1.4
    """Order-arrival intensity. Lower = thinner book = wider quotes."""

    inventory_skew_ticks: float = 1.0
    """Floor on inventory skew, in ticks, at a full position."""

    vol_multiplier: float = 0.05
    """Half-spread widening per tick of realised volatility.

    Small on purpose. Per-update volatility is not all adverse selection —
    most of it mean-reverts and is exactly what a maker earns the spread on.
    Charging the full σ prices us out of a liquid book entirely: on BTCUSDT,
    σ over 100ms runs to tens of ticks against a two-tick spread, so a
    multiplier near 1 quotes far outside the touch and never trades."""

    liquidity_premium_ticks: float = 1.0
    """Numerator of the thin-book term; the half-spread gains this over κ."""

    vol_floor_ticks: float = 0.5
    """Volatility floor, so a quiet book does not collapse the spread to zero."""

    # --- practical guardrails ---------------------------------------------
    min_half_spread_ticks: int = 1
    max_half_spread_ticks: int = 200
    max_skew_ticks: int = 400

    min_edge_bps: float = 0.0
    """Floor on the half-spread, in bps of mid — normally the maker fee.

    Without this the quoter will happily quote a half-spread of a few ticks
    while paying a fee an order of magnitude larger, and lose money on every
    round trip no matter how well it forecasts. A maker paying `f` bps per
    side must earn at least `f` bps per side to break even, so the floor is
    the fee itself."""

    allow_price_improvement: bool = True
    """Let a quote sit *inside* the spread when the model says it should.

    This is where queue position comes from: joining the touch puts us behind
    everyone already there, while improving by one tick puts us first. Turning
    it off restricts us to joining, which is safer against adverse selection
    and fills far less often."""

    # --- laddering ---------------------------------------------------------
    levels: int = 3
    level_step_ticks: int = 2
    level_size_decay: float = 0.7
    """Size multiplier applied per level out from the touch."""

    # --- sizing ------------------------------------------------------------
    base_size_lots: int = 100
    min_size_lots: int = 1
    max_position_lots: int = 1000
    inventory_taper: float = 1.0
    """How sharply size falls off as inventory approaches the limit."""


@dataclass(slots=True)
class Quoter:
    config: QuoterConfig = field(default_factory=QuoterConfig)

    def reservation_price(self, fair: float, inventory_lots: int, sigma_ticks: float) -> float:
        """Fair value shifted against the inventory we are carrying.

        Long inventory pushes the pair down so the bid backs off and the ask
        gets hit; short does the reverse.
        """
        cfg = self.config
        q = inventory_lots / max(1, cfg.max_position_lots)
        per_unit = cfg.inventory_skew_ticks + cfg.gamma * sigma_ticks
        skew = q * per_unit
        skew = max(-cfg.max_skew_ticks, min(cfg.max_skew_ticks, skew))
        return fair - skew

    def half_spread(self, sigma_ticks: float, mid_ticks: float | None = None) -> float:
        """Volatility premium plus a thin-book premium, floored and capped.

        The floor is whichever binds harder: the configured tick minimum, or
        the fee we must clear to make the round trip worth doing.
        """
        cfg = self.config
        vol_term = cfg.vol_multiplier * sigma_ticks
        liquidity_term = cfg.liquidity_premium_ticks / max(1e-9, cfg.kappa)
        raw = vol_term + liquidity_term

        floor = float(cfg.min_half_spread_ticks)
        if cfg.min_edge_bps > 0 and mid_ticks:
            floor = max(floor, cfg.min_edge_bps / 10_000.0 * mid_ticks)

        return max(floor, min(cfg.max_half_spread_ticks, raw))

    def _capacity(self, inventory_lots: int) -> tuple[int, int]:
        """Lots we may still buy and still sell before hitting the limit."""
        cap = self.config.max_position_lots
        return max(0, cap - inventory_lots), max(0, cap + inventory_lots)

    def _size_for(self, level: int, capacity: int, inventory_lots: int, side: Side) -> int:
        cfg = self.config
        size = cfg.base_size_lots * (cfg.level_size_decay**level)

        # Taper the side that would add to an existing position.
        if cfg.max_position_lots > 0 and cfg.inventory_taper > 0:
            q = inventory_lots / cfg.max_position_lots
            adding = (side is Side.BUY and q > 0) or (side is Side.SELL and q < 0)
            if adding:
                size *= max(0.0, 1.0 - cfg.inventory_taper * abs(q))

        return max(0, min(int(size), capacity))

    def quote(
        self,
        *,
        fair_value: float | None,
        sigma_ticks: float,
        inventory_lots: int,
        best_bid: int | None,
        best_ask: int | None,
    ) -> QuoteSet:
        cfg = self.config
        if fair_value is None or best_bid is None or best_ask is None:
            return QuoteSet(reason="book is not two-sided")

        sigma = max(cfg.vol_floor_ticks, sigma_ticks)
        reservation = self.reservation_price(fair_value, inventory_lots, sigma)
        half = self.half_spread(sigma, mid_ticks=(best_bid + best_ask) / 2.0)

        buy_capacity, sell_capacity = self._capacity(inventory_lots)
        bids: list[Quote] = []
        asks: list[Quote] = []

        for level in range(cfg.levels):
            offset = half + level * cfg.level_step_ticks

            bid_px = int(math.floor(reservation - offset))
            ask_px = int(math.ceil(reservation + offset))

            # Hard constraint: a maker never crosses. A bid must stay below the
            # best offer and an ask above the best bid, or it is a taker order.
            bid_px = min(bid_px, best_ask - 1)
            ask_px = max(ask_px, best_bid + 1)

            if not cfg.allow_price_improvement:
                # Join the queue at best, never step in front of it.
                bid_px = min(bid_px, best_bid)
                ask_px = max(ask_px, best_ask)

            bid_qty = self._size_for(level, buy_capacity, inventory_lots, Side.BUY)
            if bid_qty >= cfg.min_size_lots:
                bids.append(Quote(Side.BUY, bid_px, bid_qty, level))
                buy_capacity -= bid_qty

            ask_qty = self._size_for(level, sell_capacity, inventory_lots, Side.SELL)
            if ask_qty >= cfg.min_size_lots:
                asks.append(Quote(Side.SELL, ask_px, ask_qty, level))
                sell_capacity -= ask_qty

        return QuoteSet(
            bids=_dedupe(bids),
            asks=_dedupe(asks),
            fair_value=fair_value,
            reservation=reservation,
            half_spread=half,
            reason="ok",
        )


def _dedupe(quotes: list[Quote]) -> tuple[Quote, ...]:
    """Collapse ladder levels that rounded onto the same tick."""
    if not quotes:
        return ()
    merged: dict[int, Quote] = {}
    for q in quotes:
        existing = merged.get(q.price)
        if existing is None:
            merged[q.price] = q
        else:
            merged[q.price] = Quote(q.side, q.price, existing.qty + q.qty, existing.level)
    ordered = sorted(merged.values(), key=lambda q: q.price, reverse=quotes[0].side is Side.BUY)
    return tuple(ordered)
