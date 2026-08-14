"""Directional toxicity is causal, signed, and enforced before quoting."""

from decimal import Decimal

import pytest

from jsboard.core.market import MarketView
from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, FeedStatus, TradeTick
from jsboard.mm.inventory import FeeSchedule, Position
from jsboard.mm.quoter import Quoter, QuoterConfig
from jsboard.mm.risk import RiskLimits, RiskManager
from jsboard.mm.strategy import MarketMaker
from jsboard.mm.toxicity import ToxicityConfig, ToxicityGate
from jsboard.sim.paper import PaperVenue

INST = Instrument("TEST", Decimal("1"), Decimal("1"), "X", "USD")


def market_with(bid_qty: int, ask_qty: int, aggressor: Side | None = None) -> MarketView:
    market = MarketView(INST)
    market.apply(FeedStatus("live"))
    market.apply(
        DepthSnapshot(
            bids=((100, bid_qty), (99, bid_qty)),
            asks=((102, ask_qty), (103, ask_qty)),
            last_update_id=1,
            ts_ns=market.clock(),
        )
    )
    if aggressor is not None:
        market.apply(
            TradeTick(
                price=102 if aggressor is Side.BUY else 100,
                qty=100,
                aggressor=aggressor,
                trade_id=1,
                ts_ns=market.clock(),
            )
        )
    return market


def gate(threshold=0.4, pull=0.95) -> ToxicityGate:
    return ToxicityGate(ToxicityConfig(threshold=threshold, pull_threshold=pull))


class TestScore:
    def test_buy_pressure_blocks_the_ask(self):
        decision = gate().evaluate(market_with(900, 100, Side.BUY))

        assert decision.score > 0
        assert decision.permits(Side.BUY)
        assert not decision.permits(Side.SELL)
        assert "no asks" in decision.reason

    def test_sell_pressure_blocks_the_bid(self):
        decision = gate().evaluate(market_with(100, 900, Side.SELL))

        assert decision.score < 0
        assert decision.permits(Side.SELL)
        assert not decision.permits(Side.BUY)
        assert "no bids" in decision.reason

    def test_balanced_book_is_two_sided(self):
        decision = gate().evaluate(market_with(500, 500))

        assert decision.score == pytest.approx(0.0)
        assert decision.permits(Side.BUY)
        assert decision.permits(Side.SELL)

    def test_extreme_pressure_pulls_everything(self):
        decision = gate().evaluate(market_with(10_000, 1, Side.BUY))

        assert decision.score >= 0.95
        assert decision.pulled

    def test_zero_threshold_preserves_the_old_strategy(self):
        decision = gate(threshold=0.0).evaluate(market_with(10_000, 1, Side.BUY))

        assert decision.score == 0.0
        assert decision.permits(Side.BUY)
        assert decision.permits(Side.SELL)


def maker(market: MarketView, toxicity: ToxicityGate) -> MarketMaker:
    return MarketMaker(
        instrument=INST,
        market=market,
        venue=PaperVenue(INST),
        position=Position(INST, FeeSchedule()),
        quoter=Quoter(
            QuoterConfig(
                levels=1,
                base_size_lots=10,
                max_position_lots=100,
                min_half_spread_ticks=1,
            )
        ),
        risk=RiskManager(
            RiskLimits(
                max_position_lots=100,
                max_notional=1_000_000,
                max_drawdown=1_000_000,
                max_book_age_ms=10_000,
            )
        ),
        toxicity=toxicity,
    )


class TestStrategyIntegration:
    def test_upward_pressure_places_no_sell_orders(self):
        mm = maker(market_with(900, 100, Side.BUY), gate())

        quotes = mm.requote(force=True)

        assert quotes.bids
        assert quotes.asks == ()
        assert mm.venue.open_orders()
        assert all(order.side is Side.BUY for order in mm.venue.open_orders())
        assert mm.summary()["toxicity"]["one_sided"] == 1

    def test_extreme_pressure_cancels_existing_orders(self):
        market = market_with(500, 500)
        mm = maker(market, gate())
        mm.requote(force=True)
        assert mm.venue.open_orders()

        market.apply(
            DepthSnapshot(
                bids=((100, 10_000), (99, 10_000)),
                asks=((102, 1), (103, 1)),
                last_update_id=2,
                ts_ns=market.clock(),
            )
        )
        market.apply(TradeTick(102, 100, Side.BUY, trade_id=2, ts_ns=market.clock()))
        quotes = mm.requote(force=True)

        assert quotes.is_empty
        assert mm.venue.open_orders() == []
        assert mm.summary()["toxicity"]["pulls"] == 1
