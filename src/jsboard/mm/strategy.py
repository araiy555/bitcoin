"""The market maker: wires feed → book → signals → risk → quotes → venue.

The requote path is a *diff*, not a replace. Recomputing quotes every tick and
blindly cancel-replacing would throw away queue position on every level that
did not actually move — and queue position is most of a maker's edge. So each
cycle compares the desired ladder against what is already resting and touches
only the difference, tolerating small size drift rather than re-queuing for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.market import MARKET_OWNER, MarketView
from ..core.types import Fill, Instrument, Side
from ..feed.base import DepthDelta, DepthSnapshot, FeedEvent, TradeTick
from ..sim.paper import PAPER_OWNER, PaperVenue
from .fair_value import FairValueEstimator
from .inventory import Position
from .quoter import Quote, Quoter, QuoteSet
from .risk import RiskAction, RiskDecision, RiskManager


@dataclass(slots=True)
class StrategyConfig:
    requote_interval_ms: float = 250.0
    """Minimum wall-clock gap between requote cycles."""

    size_tolerance: float = 0.25
    """Requote a level only if desired size drifts this far from resting size."""

    max_orders_per_cycle: int = 12
    """Backstop against a pathological cycle flooding the venue."""


@dataclass(slots=True)
class StrategyStats:
    cycles: int = 0
    orders_placed: int = 0
    orders_cancelled: int = 0
    orders_kept: int = 0
    fills: int = 0
    last_decision: str = ""
    last_requote_ns: int = 0


@dataclass(slots=True)
class MarketMaker:
    instrument: Instrument
    market: MarketView
    venue: PaperVenue
    position: Position
    quoter: Quoter = field(default_factory=Quoter)
    fair_value: FairValueEstimator = field(default_factory=FairValueEstimator)
    risk: RiskManager = field(default_factory=RiskManager)
    config: StrategyConfig = field(default_factory=StrategyConfig)
    stats: StrategyStats = field(default_factory=StrategyStats)
    clock: object = time.time_ns
    last_quotes: QuoteSet = field(default_factory=QuoteSet)
    last_decision: RiskDecision | None = None
    recent_fills: list[Fill] = field(default_factory=list)

    # ------------------------------------------------------------ ingestion

    def on_event(self, event: FeedEvent) -> list[Fill]:
        """Feed one market event through the whole pipeline."""
        self.market.apply(event)

        fills: list[Fill] = []
        if isinstance(event, TradeTick):
            fills = self.venue.on_trade(event)
            for fill in fills:
                self.position.apply(fill, PAPER_OWNER)
            if fills:
                self.stats.fills += len(fills)
                self.recent_fills.extend(fills)
                del self.recent_fills[:-100]
        elif isinstance(event, (DepthDelta, DepthSnapshot)):
            for price, qty in event.bids:
                self.venue.on_depth(Side.BUY, price, qty)
            for price, qty in event.asks:
                self.venue.on_depth(Side.SELL, price, qty)

        return fills

    # -------------------------------------------------------------- quoting

    @property
    def sigma_ticks(self) -> float:
        """Realised vol expressed in ticks, which is what the quoter wants."""
        mid = self.market.mid
        if mid is None:
            return 0.0
        return self.market.vol.sigma * mid

    def should_requote(self) -> bool:
        elapsed_ms = (self.clock() - self.stats.last_requote_ns) / 1e6
        return elapsed_ms >= self.config.requote_interval_ms

    def requote(self, force: bool = False) -> QuoteSet:
        if not force and not self.should_requote():
            return self.last_quotes

        self.stats.cycles += 1
        self.stats.last_requote_ns = self.clock()

        decision = self.risk.evaluate(self.market, self.position)
        self.last_decision = decision
        self.stats.last_decision = f"{decision.action.value}: {decision.reason}"

        if not decision.can_quote:
            cancelled = self.venue.cancel_all()
            self.stats.orders_cancelled += cancelled
            self.last_quotes = QuoteSet(reason=decision.reason)
            return self.last_quotes

        desired = self.quoter.quote(
            fair_value=self.fair_value.estimate(self.market),
            sigma_ticks=self.sigma_ticks,
            inventory_lots=self.position.lots,
            best_bid=self.market.book.best_bid(),
            best_ask=self.market.book.best_ask(),
        )

        if decision.action is RiskAction.ONE_SIDED:
            desired = QuoteSet(
                bids=desired.bids if decision.permits(Side.BUY) else (),
                asks=desired.asks if decision.permits(Side.SELL) else (),
                fair_value=desired.fair_value,
                reservation=desired.reservation,
                half_spread=desired.half_spread,
                reason=decision.reason,
            )

        self._reconcile(desired)
        self.last_quotes = desired
        return desired

    def _reconcile(self, desired: QuoteSet) -> None:
        """Cancel what no longer belongs, keep what still does, place the rest."""
        live = {(int(o.side), o.price): o for o in self.venue.open_orders()}
        wanted = {q.key(): q for q in desired.all()}
        placed = 0

        for key, order in list(live.items()):
            quote = wanted.get(key)
            if quote is None:
                self.venue.cancel(order.order_id)
                self.stats.orders_cancelled += 1
                continue

            # Keep the order — and its queue position — unless size drifted far.
            drift = abs(order.remaining - quote.qty) / max(1, quote.qty)
            if drift <= self.config.size_tolerance:
                self.stats.orders_kept += 1
                wanted.pop(key)
            else:
                self.venue.cancel(order.order_id)
                self.stats.orders_cancelled += 1

        for quote in wanted.values():
            if placed >= self.config.max_orders_per_cycle:
                break
            depth = self._visible_depth(quote)
            opposite = (
                self.market.book.best_ask() if quote.side is Side.BUY else self.market.book.best_bid()
            )
            if self.venue.place(quote, visible_depth=depth, best_opposite=opposite) is not None:
                self.stats.orders_placed += 1
                placed += 1

    def _visible_depth(self, quote: Quote) -> int:
        """Public size already resting at this price — our queue to get through."""
        levels = self.market.book.bids if quote.side is Side.BUY else self.market.book.asks
        level = levels.get(quote.price)
        if level is None:
            return 0
        return sum(o.remaining for o in level.queue if o.owner == MARKET_OWNER and o.is_live)

    # ---------------------------------------------------------------- reads

    def flatten(self) -> int:
        """Pull all quotes. Does not trade out of the position."""
        n = self.venue.cancel_all()
        self.stats.orders_cancelled += n
        return n

    def summary(self) -> dict:
        mark = self.market.mid
        out = self.position.summary(mark)
        out.update(
            {
                "mid_ticks": mark,
                "fair_value": self.last_quotes.fair_value,
                "reservation": self.last_quotes.reservation,
                "half_spread": self.last_quotes.half_spread,
                "sigma_ticks": self.sigma_ticks,
                "resting": len(self.venue.open_orders()),
                "cycles": self.stats.cycles,
                "placed": self.stats.orders_placed,
                "cancelled": self.stats.orders_cancelled,
                "kept": self.stats.orders_kept,
                "decision": self.stats.last_decision,
                "halted": self.risk.halted,
            }
        )
        return out
