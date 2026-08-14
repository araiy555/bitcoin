"""Cross-market pricing for a maker leg with an immediate taker hedge.

The ordinary market maker prices one book from that same book.  This module
does something materially different: the hedge book is the fair-value
anchor, and every proposed maker quote has to survive the actual executable
price of the opposite hedge before it is allowed onto the venue.

That is the smallest useful relative-value strategy.  It does not predict a
coin's direction and it does not pretend that the hedge trades at mid.  A
maker buy is worth placing only when the hedge bids can absorb the same base
quantity after both fees; a maker sell is judged against the hedge asks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.types import Instrument, Side
from ..mm.quoter import Quote, QuoteSet
from .hedge import Hedger


@dataclass(slots=True)
class CrossMarketFairValue:
    """Expose the hedge microprice in the maker instrument's tick units."""

    maker_instrument: Instrument
    hedge_instrument: Instrument
    hedge_market: MarketView

    def estimate(self, _maker_market: MarketView) -> float | None:
        fair = self.hedge_market.microprice
        if fair is None:
            return None
        quote_price = fair * float(self.hedge_instrument.tick_size)
        return quote_price / float(self.maker_instrument.tick_size)

    def reset(self) -> None:
        """Match the estimator interface used by :class:`MarketMaker`."""


@dataclass(frozen=True, slots=True)
class PairEdge:
    side: Side
    maker_price: float
    hedge_price: float
    qty_base: float
    gross_bps: float
    fee_bps: float
    net_bps: float
    executable: bool = True
    reason: str = "ok"


@dataclass(slots=True)
class PairGateStats:
    evaluations: int = 0
    quotes_tested: int = 0
    quotes_passed: int = 0
    rejected_no_book: int = 0
    rejected_stale: int = 0
    rejected_depth: int = 0
    rejected_edge: int = 0
    best_net_bps: float = float("-inf")

    @property
    def pass_share(self) -> float:
        return self.quotes_passed / self.quotes_tested if self.quotes_tested else 0.0


@dataclass(slots=True)
class PairQuoteGate:
    """Keep only quotes whose immediately executable hedge is profitable."""

    maker_instrument: Instrument
    hedge_instrument: Instrument
    maker_market: MarketView
    hedge_market: MarketView
    hedger: Hedger
    maker_bps: float = 0.0
    taker_bps: float = 4.0
    min_net_bps: float = 0.0
    max_hedge_age_ms: float = 250.0
    clock: Callable[[], int] | None = None
    stats: PairGateStats = field(default_factory=PairGateStats)

    def _now_ns(self) -> int:
        if self.clock is not None:
            return self.clock()
        return max(self.maker_market.last_update_ns, self.hedge_market.last_update_ns)

    def _book_is_fresh(self) -> bool:
        updated = self.hedge_market.last_update_ns
        if not updated:
            return False
        age_ms = max(0.0, (self._now_ns() - updated) / 1e6)
        return age_ms <= self.max_hedge_age_ms

    def edge(self, quote: Quote) -> PairEdge:
        """Price one proposed maker quote against visible hedge depth."""
        qty_base = self.maker_instrument.qty_f(quote.qty)
        maker_price = self.maker_instrument.price_f(quote.price)

        if self.hedge_market.mid is None:
            return PairEdge(
                quote.side, maker_price, 0.0, qty_base, 0.0, 0.0, float("-inf"),
                executable=False, reason="hedge book missing",
            )
        if not self._book_is_fresh():
            return PairEdge(
                quote.side, maker_price, 0.0, qty_base, 0.0, 0.0, float("-inf"),
                executable=False, reason="hedge book stale",
            )

        walked = self.hedger.walk(-quote.side.sign, qty_base)
        # Partial hedges leave directional inventory.  They are not a smaller
        # opportunity; they are a different and riskier trade, so reject them.
        tolerance = max(1e-12, float(self.hedge_instrument.lot_size) / 2.0)
        if walked.filled_base <= 0 or walked.unfilled_base > tolerance:
            return PairEdge(
                quote.side, maker_price, 0.0, qty_base, 0.0, 0.0, float("-inf"),
                executable=False, reason="insufficient hedge depth",
            )

        hedge_price = walked.avg_price_ticks * float(self.hedge_instrument.tick_size)
        maker_notional = qty_base * maker_price
        hedge_notional = walked.filled_base * hedge_price
        if maker_notional <= 0:
            return PairEdge(
                quote.side, maker_price, hedge_price, qty_base, 0.0, 0.0,
                float("-inf"), executable=False, reason="zero maker notional",
            )

        # Buy maker / sell hedge: hedge - maker.  Sell maker / buy hedge:
        # maker - hedge.  `side.sign` gives both with one expression.
        gross_quote = quote.side.sign * (hedge_price - maker_price) * qty_base
        fees = (
            maker_notional * self.maker_bps / 10_000.0
            + hedge_notional * self.taker_bps / 10_000.0
        )
        scale = 10_000.0 / maker_notional
        return PairEdge(
            side=quote.side,
            maker_price=maker_price,
            hedge_price=hedge_price,
            qty_base=qty_base,
            gross_bps=gross_quote * scale,
            fee_bps=fees * scale,
            net_bps=(gross_quote - fees) * scale,
        )

    def filter(self, quotes: QuoteSet) -> QuoteSet:
        self.stats.evaluations += 1
        kept_bids: list[Quote] = []
        kept_asks: list[Quote] = []

        for quote in quotes.all():
            self.stats.quotes_tested += 1
            edge = self.edge(quote)
            if edge.executable:
                self.stats.best_net_bps = max(self.stats.best_net_bps, edge.net_bps)
            elif edge.reason == "hedge book missing":
                self.stats.rejected_no_book += 1
                continue
            elif edge.reason == "hedge book stale":
                self.stats.rejected_stale += 1
                continue
            else:
                self.stats.rejected_depth += 1
                continue

            if edge.net_bps < self.min_net_bps:
                self.stats.rejected_edge += 1
                continue
            self.stats.quotes_passed += 1
            (kept_bids if quote.side is Side.BUY else kept_asks).append(quote)

        reason = quotes.reason
        if quotes.all() and not kept_bids and not kept_asks:
            reason = f"pair edge below {self.min_net_bps:g} bps"
        return QuoteSet(
            bids=tuple(kept_bids),
            asks=tuple(kept_asks),
            fair_value=quotes.fair_value,
            reservation=quotes.reservation,
            half_spread=quotes.half_spread,
            reason=reason,
        )
