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


class FixedLead:
    """A lead venue whose price, in our ticks, the test sets directly."""

    def __init__(self, price):
        self.price = price

    def estimate(self, _market):
        return self.price


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now


def lead_gate(lead, threshold_bps=5.0, halflife_s=60.0, own=0.0) -> ToxicityGate:
    return ToxicityGate(
        ToxicityConfig(
            threshold=own, lead_threshold_bps=threshold_bps, lead_basis_halflife_s=halflife_s
        ),
        lead=lead,
    )


class TestLeadVenue:
    """The stock rule: when the future moves 5bps away, pull the side it moves toward."""

    def setup_market(self):
        market = market_with(500, 500)  # mid 101
        clock = Clock()
        market.clock = clock
        return market, clock

    def test_a_standing_gap_is_learned_not_acted_on(self):
        market, _ = self.setup_market()
        # 10bps apart from the start: that is the basis, not a move.
        g = lead_gate(FixedLead(101 * 1.001))
        decision = g.evaluate(market)
        assert decision.permits(Side.BUY) and decision.permits(Side.SELL)

    def test_lead_jumping_up_pulls_the_ask_only(self):
        market, clock = self.setup_market()
        lead = FixedLead(101.0)
        g = lead_gate(lead)
        g.evaluate(market)  # learns a zero basis
        clock.now += 250_000_000
        lead.price = 101 * 1.0006  # +6bps
        decision = g.evaluate(market)
        assert decision.permits(Side.BUY)
        assert not decision.permits(Side.SELL)
        assert "no asks" in decision.reason
        assert g.lead_blocks == 1

    def test_lead_dropping_pulls_the_bid_only(self):
        market, clock = self.setup_market()
        lead = FixedLead(101.0)
        g = lead_gate(lead)
        g.evaluate(market)
        clock.now += 250_000_000
        lead.price = 101 * 0.9994
        decision = g.evaluate(market)
        assert decision.permits(Side.SELL)
        assert not decision.permits(Side.BUY)

    def test_a_move_under_the_threshold_leaves_both_sides(self):
        market, clock = self.setup_market()
        lead = FixedLead(101.0)
        g = lead_gate(lead)
        g.evaluate(market)
        clock.now += 250_000_000
        lead.price = 101 * 1.0004  # +4bps < 5
        decision = g.evaluate(market)
        assert decision.permits(Side.BUY) and decision.permits(Side.SELL)

    def test_the_basis_catches_up_over_the_halflife(self):
        market, clock = self.setup_market()
        lead = FixedLead(101.0)
        g = lead_gate(lead, halflife_s=1.0)
        g.evaluate(market)
        lead.price = 101 * 1.001
        for _ in range(20):
            clock.now += 1_000_000_000
            decision = g.evaluate(market)
        # A gap that has persisted for twenty halflives is the new normal.
        assert decision.permits(Side.BUY) and decision.permits(Side.SELL)

    def test_no_lead_book_means_no_opinion(self):
        market, _ = self.setup_market()
        decision = lead_gate(FixedLead(None)).evaluate(market)
        assert decision.permits(Side.BUY) and decision.permits(Side.SELL)

    def test_disabled_without_a_threshold(self):
        market, clock = self.setup_market()
        lead = FixedLead(101.0)
        g = lead_gate(lead, threshold_bps=0.0)
        g.evaluate(market)
        lead.price = 101 * 1.01
        assert g.evaluate(market).permits(Side.SELL)

    def test_combines_with_the_own_book_gate(self):
        # Own book says "no asks"; lead says "no bids" → nothing left to quote.
        market = market_with(900, 100, Side.BUY)
        clock = Clock()
        market.clock = clock
        lead = FixedLead(101.0)
        g = lead_gate(lead, own=0.4)
        g.evaluate(market)
        clock.now += 250_000_000
        lead.price = market.mid * 0.999
        assert g.evaluate(market).pulled

    def test_the_maker_places_no_ask_when_the_lead_is_up(self):
        market, clock = self.setup_market()
        lead = FixedLead(101.0)
        mm = maker(market, lead_gate(lead))
        mm.requote(force=True)
        clock.now += 250_000_000
        lead.price = 101 * 1.001
        quotes = mm.requote(force=True)
        assert quotes.bids and quotes.asks == ()
        assert mm.summary()["toxicity"]["lead_share"] > 0
