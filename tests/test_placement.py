"""Placement diagnostics: how far behind the touch our quotes actually land.

An hour of quoting that produces no fills is uninformative on its own. These
cover the counters that make it informative — distance from the touch, and
how much size stood in front of a quote that did join it.
"""

import math
from decimal import Decimal

import pytest

from jsboard.core.market import MARKET_OWNER, MarketView
from jsboard.core.types import Instrument, Side
from jsboard.mm.inventory import Position
from jsboard.mm.quoter import Quote, Quoter, QuoterConfig
from jsboard.mm.strategy import MarketMaker
from jsboard.sim.paper import PaperVenue

INST = Instrument("TEST", tick_size=Decimal("1"), lot_size=Decimal("1"), base="T", quote="U")


@pytest.fixture
def mm():
    market = MarketView(instrument=INST, depth=8)
    return MarketMaker(
        instrument=INST,
        market=market,
        venue=PaperVenue(instrument=INST),
        position=Position(instrument=INST),
        quoter=Quoter(QuoterConfig(base_size_lots=5, max_position_lots=100)),
    )


def seed_book(mm, best_bid=100, best_ask=101, qty=50):
    """Put a two-sided market on the book, owned by the market not by us."""
    mm.market.book.replace_l2([(best_bid, qty)], [(best_ask, qty)], owner=MARKET_OWNER)


class TestDistanceFromTouch:
    def test_joining_the_best_bid_is_zero(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 100, 5), depth=50)
        assert mm.stats.placement_ticks == {0: 1}

    def test_improving_on_the_best_bid_is_negative(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 101, 5), depth=0)
        assert mm.stats.placement_ticks == {-1: 1}

    def test_resting_behind_the_best_bid_is_positive(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 97, 5), depth=0)
        assert mm.stats.placement_ticks == {3: 1}

    def test_the_ask_side_measures_in_the_same_direction(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.SELL, 104, 5), depth=0)
        mm._record_placement(Quote(Side.SELL, 101, 5), depth=50)
        mm._record_placement(Quote(Side.SELL, 100, 5), depth=0)
        assert mm.stats.placement_ticks == {3: 1, 0: 1, -1: 1}

    def test_an_empty_side_records_nothing(self, mm):
        mm._record_placement(Quote(Side.BUY, 100, 5), depth=0)
        assert mm.stats.placement_ticks == {}


class TestQueueAheadRatio:
    def test_measured_only_at_or_inside_the_touch(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 97, 5), depth=999)  # behind: ignored
        assert mm.stats.queue_ahead_ratio_n == 0
        assert math.isnan(mm.stats.mean_queue_ahead_ratio)

    def test_reported_as_a_multiple_of_our_size(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 100, 5), depth=50)
        assert mm.stats.mean_queue_ahead_ratio == pytest.approx(10.0)

    def test_averaged_across_placements(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 100, 5), depth=50)  # 10x
        mm._record_placement(Quote(Side.SELL, 101, 5), depth=10)  # 2x
        assert mm.stats.mean_queue_ahead_ratio == pytest.approx(6.0)

    def test_an_empty_level_is_a_ratio_of_zero(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 101, 5), depth=0)
        assert mm.stats.mean_queue_ahead_ratio == pytest.approx(0.0)


class TestSummary:
    def test_summary_carries_the_diagnostics(self, mm):
        seed_book(mm)
        mm._record_placement(Quote(Side.BUY, 100, 5), depth=50)
        s = mm.summary()
        assert s["placement"] == {0: 1}
        assert s["queue_ahead_ratio"] == pytest.approx(10.0)
        assert s["prints_seen"] == 0
        assert s["prints_at_our_price"] == 0
