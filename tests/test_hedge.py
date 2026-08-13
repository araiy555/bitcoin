"""Immediate hedging: crossing cost, book walking, and what stays unhedged.

The hedge exists to remove the inventory term. The point of simulating it is
to charge for it honestly, so these pin the charges rather than the offset.
"""

from decimal import Decimal

import pytest

from jsboard.core.market import MARKET_OWNER, MarketView
from jsboard.core.types import Instrument
from jsboard.sim.hedge import HedgeConfig, Hedger

INST = Instrument("HEDGE", tick_size=Decimal("1"), lot_size=Decimal("1"), base="H", quote="U")


def hedger(**kw):
    market = MarketView(instrument=INST, depth=20)
    return Hedger(instrument=INST, market=market, config=HedgeConfig(**kw))


def seed(h, bids, asks):
    h.market.book.replace_l2(list(bids), list(asks), owner=MARKET_OWNER)


class TestWalkingTheBook:
    def test_a_small_hedge_takes_the_touch_only(self):
        h = hedger()
        seed(h, [(99, 50)], [(101, 50)])
        result = h.walk(+1, 10)
        assert result.filled_base == pytest.approx(10)
        assert result.avg_price_ticks == pytest.approx(101)
        assert result.unfilled_base == pytest.approx(0)

    def test_a_large_hedge_pays_a_worse_average(self):
        h = hedger()
        seed(h, [(99, 50)], [(101, 10), (102, 10), (103, 10)])
        result = h.walk(+1, 30)
        assert result.filled_base == pytest.approx(30)
        assert result.avg_price_ticks == pytest.approx((101 + 102 + 103) / 3)

    def test_selling_walks_down_the_bids(self):
        h = hedger()
        seed(h, [(99, 10), (98, 10)], [(101, 50)])
        result = h.walk(-1, 20)
        assert result.avg_price_ticks == pytest.approx(98.5)

    def test_depth_that_is_not_there_comes_back_unfilled(self):
        h = hedger()
        seed(h, [(99, 50)], [(101, 5)])
        result = h.walk(+1, 20)
        assert result.filled_base == pytest.approx(5)
        assert result.unfilled_base == pytest.approx(15)

    def test_an_empty_side_fills_nothing(self):
        h = hedger()
        seed(h, [(99, 50)], [])
        result = h.walk(+1, 10)
        assert result.filled_base == pytest.approx(0)
        assert result.unfilled_base == pytest.approx(10)

    def test_it_stops_at_the_configured_depth(self):
        h = hedger(max_levels=2)
        seed(h, [(99, 50)], [(101, 5), (102, 5), (103, 5)])
        result = h.walk(+1, 15)
        assert result.filled_base == pytest.approx(10)
        assert result.unfilled_base == pytest.approx(5)


class TestHedgingAFill:
    def test_the_hedge_takes_the_opposite_side(self):
        h = hedger()
        seed(h, [(99, 50)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)  # we bought, so hedge sells
        assert h.attribution.position_lots == -10

    def test_crossing_is_booked_as_a_cost_not_a_gain(self):
        h = hedger(taker_bps=0.0)
        seed(h, [(99, 50)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        # Sold at 99 against a mid of 100: a full tick of spread paid away.
        assert h.attribution.spread_capture == pytest.approx(-10.0)

    def test_the_taker_fee_is_charged(self):
        h = hedger(taker_bps=10.0)
        seed(h, [(99, 50)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        assert h.attribution.fees == pytest.approx(10 * 99 * 10 / 10_000.0)

    def test_a_ratio_scales_the_hedge_down(self):
        h = hedger(ratio=0.5)
        seed(h, [(99, 50)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        assert h.attribution.position_lots == -5

    def test_a_zero_ratio_hedges_nothing(self):
        h = hedger(ratio=0.0)
        seed(h, [(99, 50)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        assert h.hedges == 0
        assert h.attribution.position_lots == 0

    def test_no_book_is_recorded_rather_than_hedged_for_free(self):
        h = hedger()
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        assert h.hedges == 0
        assert h.skipped_no_book == 1
        assert h.attribution.total == pytest.approx(0.0)

    def test_a_partial_hedge_leaves_the_remainder_exposed(self):
        h = hedger()
        seed(h, [(99, 4)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        assert h.attribution.position_lots == -4
        assert h.unfilled_base == pytest.approx(6)


class TestOffset:
    def test_a_perfectly_correlated_hedge_cancels_the_carry(self):
        """The hedge's whole purpose, isolated from its costs."""
        h = hedger(taker_bps=0.0)
        seed(h, [(99, 50)], [(101, 50)])
        h.on_maker_fill(maker_sign=+1, qty_base=10)  # short 10 on the hedge
        h.market.book.replace_l2([(109, 50)], [(111, 50)], owner=MARKET_OWNER)
        h.on_market()
        # Mid moved 100 -> 110 while short 10: the hedge loses exactly what a
        # long 10 on the maker leg would have gained.
        assert h.attribution.inventory_pnl == pytest.approx(-100.0)

    def test_summary_reports_what_went_unhedged(self):
        h = hedger()
        h.on_maker_fill(maker_sign=+1, qty_base=10)
        assert h.summary()["skipped"] == 1.0
