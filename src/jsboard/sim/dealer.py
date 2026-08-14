"""Binance spot/perpetual route selection for a cross-market dealer.

The pair simulator prices one user-selected maker/hedge direction.  A dealer
cannot assume that direction in advance: at every observable market update it
must compare both venues and both sides, include executable hedge depth and
costs, and route a quote only to the best surviving opportunity.

This module is deliberately pre-trade only.  A selected route is a quote
candidate, not a fill and not realised profit.  Queue position and round-trip
PnL remain the job of the existing pair replay after the route is selected.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.types import Instrument, Side
from ..feed.base import FeedEvent, MarkPrice
from ..mm.quoter import Quote
from .hedge import HedgeConfig, Hedger
from .pair import PairQuoteGate


@dataclass(frozen=True, slots=True)
class DealerRoute:
    maker_source: str
    hedge_source: str
    maker_side: Side

    @property
    def label(self) -> str:
        action = "買い" if self.maker_side is Side.BUY else "売り"
        hedge_action = "売り" if self.maker_side is Side.BUY else "買い"
        return f"{self.maker_source}{action} → {self.hedge_source}{hedge_action}"


@dataclass(frozen=True, slots=True)
class DealerOpportunity:
    route: DealerRoute
    maker_price: float = 0.0
    hedge_price: float = 0.0
    qty_base: float = 0.0
    gross_bps: float = 0.0
    fee_bps: float = 0.0
    funding_bps: float = 0.0
    net_bps: float = float("-inf")
    executable: bool = False
    candidate: bool = False
    reason: str = "book missing"


@dataclass(slots=True)
class DealerRouteStats:
    observations: int = 0
    executable: int = 0
    candidates: int = 0
    selected: int = 0
    best_net_bps: float = float("-inf")
    executable_edges: list[float] = field(default_factory=list)

    @property
    def executable_share(self) -> float:
        return self.executable / self.observations if self.observations else 0.0

    @property
    def candidate_share(self) -> float:
        return self.candidates / self.observations if self.observations else 0.0

    @property
    def median_net_bps(self) -> float:
        return statistics.median(self.executable_edges) if self.executable_edges else float("nan")


@dataclass(slots=True)
class BinanceDealer:
    """Compare every spot/perpetual maker route on one observable clock."""

    instruments: dict[str, Instrument]
    maker_bps: dict[str, float]
    taker_bps: dict[str, float]
    size_base: float
    min_net_bps: float = 1.0
    max_age_ms: float = 250.0
    funding_horizon_h: float = 8.0
    depth: int = 20
    now_ns: int = 0
    markets: dict[str, MarketView] = field(init=False)
    gates: dict[tuple[str, str], PairQuoteGate] = field(init=False)
    stats: dict[DealerRoute, DealerRouteStats] = field(init=False)
    perp_mark: MarkPrice | None = None
    evaluations: int = 0

    def __post_init__(self) -> None:
        if set(self.instruments) != {"spot", "perp"}:
            raise ValueError("dealerにはspotとperpの両方が必要です")
        if self.size_base <= 0:
            raise ValueError("size_baseは0より大きくしてください")
        if self.min_net_bps < 0 or self.max_age_ms < 0 or self.funding_horizon_h < 0:
            raise ValueError("edge、age、funding horizonは0以上で指定してください")

        self.markets = {
            source: MarketView(instrument=instrument, depth=self.depth, clock=self._clock)
            for source, instrument in self.instruments.items()
        }
        self.gates = {}
        for maker_source, hedge_source in (("spot", "perp"), ("perp", "spot")):
            hedge = Hedger(
                self.instruments[hedge_source],
                self.markets[hedge_source],
                HedgeConfig(
                    ratio=1.0,
                    taker_bps=self.taker_bps[hedge_source],
                    max_levels=self.depth,
                ),
            )
            self.gates[(maker_source, hedge_source)] = PairQuoteGate(
                maker_instrument=self.instruments[maker_source],
                hedge_instrument=self.instruments[hedge_source],
                maker_market=self.markets[maker_source],
                hedge_market=self.markets[hedge_source],
                hedger=hedge,
                maker_bps=self.maker_bps[maker_source],
                taker_bps=self.taker_bps[hedge_source],
                min_net_bps=self.min_net_bps,
                max_hedge_age_ms=self.max_age_ms,
                clock=self._clock,
            )
        routes = (
            DealerRoute("spot", "perp", Side.BUY),
            DealerRoute("spot", "perp", Side.SELL),
            DealerRoute("perp", "spot", Side.BUY),
            DealerRoute("perp", "spot", Side.SELL),
        )
        self.stats = {route: DealerRouteStats() for route in routes}

    def _clock(self) -> int:
        return self.now_ns

    def apply(self, source: str, event: FeedEvent, received_ns: int = 0) -> None:
        if source not in self.markets:
            return
        event_ns = getattr(event, "ts_ns", 0)
        self.now_ns = max(self.now_ns, received_ns, event_ns)
        if source == "perp" and isinstance(event, MarkPrice):
            self.perp_mark = event
        self.markets[source].apply(event)

    def _funding_bps(
        self, route: DealerRoute, maker_price: float, hedge_price: float, qty_base: float
    ) -> float:
        mark = self.perp_mark
        if mark is None or mark.next_funding_ns <= 0 or self.funding_horizon_h <= 0:
            return 0.0
        horizon_ns = int(self.funding_horizon_h * 3_600 * 1e9)
        if not (self.now_ns <= mark.next_funding_ns <= self.now_ns + horizon_ns):
            return 0.0

        if route.maker_source == "perp":
            perp_sign = route.maker_side.sign
            perp_price = maker_price
        else:
            perp_sign = -route.maker_side.sign
            perp_price = hedge_price
        maker_notional = maker_price * qty_base
        perp_notional = perp_price * qty_base
        if maker_notional <= 0:
            return 0.0
        # Positive funding is paid by a perp long and received by a short.
        return (
            -perp_sign
            * mark.funding_rate
            * perp_notional
            / maker_notional
            * 10_000.0
        )

    def opportunity(self, route: DealerRoute) -> DealerOpportunity:
        maker_market = self.markets[route.maker_source]
        if maker_market.age_ms > self.max_age_ms:
            return DealerOpportunity(route=route, reason="maker book stale")
        maker_price_ticks = (
            maker_market.book.best_bid()
            if route.maker_side is Side.BUY
            else maker_market.book.best_ask()
        )
        if maker_price_ticks is None:
            return DealerOpportunity(route=route, reason="maker book missing")
        maker_instrument = self.instruments[route.maker_source]
        qty_lots = maker_instrument.to_lots(self.size_base)
        if qty_lots <= 0:
            return DealerOpportunity(route=route, reason="maker lot below minimum")

        edge = self.gates[(route.maker_source, route.hedge_source)].edge(
            Quote(route.maker_side, maker_price_ticks, qty_lots)
        )
        if not edge.executable:
            return DealerOpportunity(
                route=route,
                maker_price=edge.maker_price,
                hedge_price=edge.hedge_price,
                qty_base=edge.qty_base,
                gross_bps=edge.gross_bps,
                fee_bps=edge.fee_bps,
                reason=edge.reason,
            )
        funding_bps = self._funding_bps(
            route, edge.maker_price, edge.hedge_price, edge.qty_base
        )
        net_bps = edge.net_bps + funding_bps
        return DealerOpportunity(
            route=route,
            maker_price=edge.maker_price,
            hedge_price=edge.hedge_price,
            qty_base=edge.qty_base,
            gross_bps=edge.gross_bps,
            fee_bps=edge.fee_bps,
            funding_bps=funding_bps,
            net_bps=net_bps,
            executable=True,
            candidate=net_bps >= self.min_net_bps,
            reason="ok" if net_bps >= self.min_net_bps else "edge below threshold",
        )

    def evaluate(self) -> list[DealerOpportunity]:
        self.evaluations += 1
        opportunities = [self.opportunity(route) for route in self.stats]
        for opportunity in opportunities:
            stats = self.stats[opportunity.route]
            stats.observations += 1
            if opportunity.executable:
                stats.executable += 1
                stats.executable_edges.append(opportunity.net_bps)
                stats.best_net_bps = max(stats.best_net_bps, opportunity.net_bps)
            if opportunity.candidate:
                stats.candidates += 1
        candidates = [row for row in opportunities if row.candidate]
        if candidates:
            best = max(candidates, key=lambda row: row.net_bps)
            self.stats[best.route].selected += 1
        return opportunities

