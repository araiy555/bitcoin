"""The market maker: wires feed → book → signals → risk → quotes → venue.

The requote path is a *diff*, not a replace. Recomputing quotes every tick and
blindly cancel-replacing would throw away queue position on every level that
did not actually move — and queue position is most of a maker's edge. So each
cycle compares the desired ladder against what is already resting and touches
only the difference, tolerating small size drift rather than re-queuing for it.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.market import MARKET_OWNER, MarketView
from ..core.types import Fill, Instrument, Side
from ..feed.base import DepthDelta, DepthSnapshot, FeedEvent, TradeTick
from ..sim.paper import PAPER_OWNER, PaperVenue
from .attribution import PnLAttribution
from .fair_value import FairValueEstimator
from .inventory import Position
from .markout import MarkOutTracker
from .quoter import Quote, Quoter, QuoteSet
from .risk import RiskAction, RiskDecision, RiskManager
from .toxicity import ToxicityDecision, ToxicityGate


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

    # Where our quotes actually landed relative to the touch, in ticks:
    # negative improves on the best price, 0 joins it, positive rests behind
    # it. A maker that never reaches the touch cannot fill however long it
    # runs, and that is invisible in a placed/cancelled count.
    placement_ticks: Counter[int] = field(default_factory=Counter)
    queue_ahead_ratio_sum: float = 0.0
    queue_ahead_ratio_n: int = 0

    toxicity_evaluations: int = 0
    toxicity_one_sided: int = 0
    toxicity_pulls: int = 0
    toxicity_bid_blocks: int = 0
    toxicity_ask_blocks: int = 0
    last_toxicity_score: float = 0.0

    @property
    def mean_queue_ahead_ratio(self) -> float:
        """Size resting in front of a touch-joining quote, as a multiple of ours."""
        if self.queue_ahead_ratio_n == 0:
            return math.nan
        return self.queue_ahead_ratio_sum / self.queue_ahead_ratio_n


@dataclass(slots=True)
class MarketMaker:
    instrument: Instrument
    market: MarketView
    venue: PaperVenue
    position: Position
    quoter: Quoter = field(default_factory=Quoter)
    fair_value: FairValueEstimator = field(default_factory=FairValueEstimator)
    risk: RiskManager = field(default_factory=RiskManager)
    toxicity: ToxicityGate = field(default_factory=ToxicityGate)
    config: StrategyConfig = field(default_factory=StrategyConfig)
    stats: StrategyStats = field(default_factory=StrategyStats)
    markout: MarkOutTracker = field(default_factory=MarkOutTracker)
    attribution: PnLAttribution = field(init=False)
    clock: object = time.time_ns
    last_quotes: QuoteSet = field(default_factory=QuoteSet)
    last_decision: RiskDecision | None = None
    last_toxicity: ToxicityDecision | None = None
    recent_fills: list[Fill] = field(default_factory=list)
    quote_filter: Callable[[QuoteSet], QuoteSet] | None = None
    """Optional final gate applied before orders reach the venue.

    Cross-market strategies use this to reject a proposed quote when its
    immediately executable hedge does not clear all costs.  The ordinary
    single-book strategy leaves it unset and is unchanged.
    """

    def __post_init__(self) -> None:
        # The attribution needs the instrument to convert ticks and lots into
        # quote currency, so it cannot be a plain default_factory.
        self.attribution = PnLAttribution(self.instrument)

    @property
    def _data_time_ns(self) -> int | None:
        """Time as the data sees it, or nothing before the first stamped event.

        The virtual clock falls back to the wall clock until a timestamped
        event arrives, so an interval spanning that switch is measured between
        two different clocks — a gap that differs on every run and makes a
        replay unrepeatable. Exposure is time at risk *in the market*, so the
        market's own clock is also the right one to measure it on.
        """
        return self.market.last_update_ns or None

    # ------------------------------------------------------------ ingestion

    def on_event(self, event: FeedEvent) -> list[Fill]:
        """Feed one market event through the whole pipeline."""
        self.market.apply(event)

        fills: list[Fill] = []
        if isinstance(event, TradeTick):
            # The mid *before* our own fills are booked is the reference both
            # the attribution and the mark-out measure against: it is the
            # price the market showed at the instant we traded.
            mid = self.market.mid
            fills = self.venue.on_trade(event)
            now = self.clock()
            for fill in fills:
                # Our side of the trade, which is the opposite of the taker's
                # when we were the maker. Both sides are possible on one fill
                # only if we somehow traded with ourselves; book each anyway.
                sides = []
                if fill.maker_owner == PAPER_OWNER:
                    sides.append(fill.aggressor.opposite.sign)
                if fill.taker_owner == PAPER_OWNER:
                    sides.append(fill.aggressor.sign)
                if not sides:
                    continue
                before = self.position.fees_paid
                self.position.apply(fill, PAPER_OWNER)
                fee = self.position.fees_paid - before
                for sign in sides:
                    self.markout.on_fill(now, sign, fill.price, fill.qty, mid_ticks=mid)
                    self.attribution.on_fill(
                        price_ticks=fill.price,
                        qty_lots=fill.qty,
                        sign=sign,
                        fee=fee / len(sides),
                        mid_ticks=mid,
                        now_ns=self._data_time_ns,
                    )
            if fills:
                self.stats.fills += len(fills)
                self.recent_fills.extend(fills)
                del self.recent_fills[:-100]
        elif isinstance(event, (DepthDelta, DepthSnapshot)):
            for price, qty in event.bids:
                self.venue.on_depth(Side.BUY, price, qty)
            for price, qty in event.asks:
                self.venue.on_depth(Side.SELL, price, qty)

        mid_now = self.market.mid
        self.markout.poll(self.clock(), mid_now)
        self.attribution.on_mid(mid_now, self._data_time_ns)
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
        if elapsed_ms < 0:
            # The clock went backwards. A replay does this at the first event,
            # when the source of "now" switches from the wall clock to the
            # recording's own timestamps; an NTP step does it in production.
            # Re-baseline instead of waiting for a deadline that has already
            # passed, which would freeze quoting for the rest of the run.
            return True
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

        toxicity = self.toxicity.evaluate(self.market)
        self.last_toxicity = toxicity
        self._record_toxicity(toxicity)
        if toxicity.pulled:
            cancelled = self.venue.cancel_all()
            self.stats.orders_cancelled += cancelled
            self.stats.last_decision = f"PULL: {toxicity.reason}"
            self.last_quotes = QuoteSet(reason=toxicity.reason)
            return self.last_quotes

        desired = self.quoter.quote(
            fair_value=self.fair_value.estimate(self.market),
            sigma_ticks=self.sigma_ticks,
            inventory_lots=self.position.lots,
            best_bid=self.market.book.best_bid(),
            best_ask=self.market.book.best_ask(),
        )

        allowed = decision.allowed_sides & toxicity.allowed_sides
        if allowed != frozenset({Side.BUY, Side.SELL}):
            reasons = []
            if decision.action is RiskAction.ONE_SIDED:
                reasons.append(decision.reason)
            if toxicity.one_sided:
                reasons.append(toxicity.reason)
            desired = QuoteSet(
                bids=desired.bids if Side.BUY in allowed else (),
                asks=desired.asks if Side.SELL in allowed else (),
                fair_value=desired.fair_value,
                reservation=desired.reservation,
                half_spread=desired.half_spread,
                reason="; ".join(reasons),
            )
            self.stats.last_decision = f"ONE_SIDED: {desired.reason}"

        if self.quote_filter is not None:
            desired = self.quote_filter(desired)

        self._reconcile(desired)
        self.last_quotes = desired
        return desired

    def _record_toxicity(self, decision: ToxicityDecision) -> None:
        """Count how often the filter removes quote exposure."""
        self.stats.toxicity_evaluations += 1
        self.stats.last_toxicity_score = decision.score
        if decision.one_sided:
            self.stats.toxicity_one_sided += 1
        if decision.pulled:
            self.stats.toxicity_pulls += 1
        if not decision.permits(Side.BUY):
            self.stats.toxicity_bid_blocks += 1
        if not decision.permits(Side.SELL):
            self.stats.toxicity_ask_blocks += 1

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
                self._record_placement(quote, depth)

    def _record_placement(self, quote: Quote, depth: int) -> None:
        """Note how far behind the touch this quote landed, and its queue."""
        best = (
            self.market.book.best_bid() if quote.side is Side.BUY else self.market.book.best_ask()
        )
        if best is None:
            return
        distance = best - quote.price if quote.side is Side.BUY else quote.price - best
        self.stats.placement_ticks[distance] += 1
        if distance <= 0 and quote.qty > 0:
            # Only the touch tells us anything about reachability; a quote
            # three ticks back has a queue we were never going to clear.
            self.stats.queue_ahead_ratio_sum += depth / quote.qty
            self.stats.queue_ahead_ratio_n += 1

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
        tox_n = self.stats.toxicity_evaluations
        toxicity = {
            "threshold": self.toxicity.config.threshold,
            "last_score": self.stats.last_toxicity_score,
            "evaluations": tox_n,
            "one_sided": self.stats.toxicity_one_sided,
            "pulls": self.stats.toxicity_pulls,
            "one_sided_share": self.stats.toxicity_one_sided / tox_n if tox_n else 0.0,
            "pull_share": self.stats.toxicity_pulls / tox_n if tox_n else 0.0,
            "bid_block_share": self.stats.toxicity_bid_blocks / tox_n if tox_n else 0.0,
            "ask_block_share": self.stats.toxicity_ask_blocks / tox_n if tox_n else 0.0,
        }
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
                "markout": self.markout.summary(),
                "attribution": self.attribution.summary(),
                "placement": dict(self.stats.placement_ticks),
                "queue_ahead_ratio": self.stats.mean_queue_ahead_ratio,
                "prints_seen": self.venue.prints_seen,
                "prints_at_our_price": self.venue.prints_at_our_price,
                "queue_absorbed": self.instrument.qty_f(self.venue.queue_absorbed_lots),
                "toxicity": toxicity,
            }
        )
        return out
