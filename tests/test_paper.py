"""Paper venue: queue position, latency, partial fills, and price hits."""

from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import TradeTick
from jsboard.mm.quoter import Quote
from jsboard.sim.paper import PaperConfig, PaperVenue

INST = Instrument("TEST", tick_size=Decimal("1"), lot_size=Decimal("1"))


class FakeClock:
    """Manually advanced clock, so latency is exact instead of flaky."""

    def __init__(self, now_ns: int = 1_000_000_000) -> None:
        self.now = now_ns

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += int(ms * 1e6)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def venue(clock):
    return PaperVenue(instrument=INST, config=PaperConfig(latency_ms=10.0), clock=clock)


def bid(price, qty):
    return Quote(Side.BUY, price, qty)


def ask(price, qty):
    return Quote(Side.SELL, price, qty)


def sell_into(price, qty, ts=0):
    """An aggressive sell — hits resting bids."""
    return TradeTick(price=price, qty=qty, aggressor=Side.SELL, ts_ns=ts)


def buy_into(price, qty, ts=0):
    """An aggressive buy — lifts resting asks."""
    return TradeTick(price=price, qty=qty, aggressor=Side.BUY, ts_ns=ts)


class TestLatency:
    def test_order_does_not_fill_before_it_is_live(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=101)

        assert venue.on_trade(sell_into(100, 10)) == []

    def test_order_fills_once_latency_elapses(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=101)
        clock.advance_ms(11)

        fills = venue.on_trade(sell_into(100, 10))

        assert len(fills) == 1
        assert fills[0].qty == 10


class TestQueuePosition:
    def test_queue_ahead_absorbs_the_print(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=50, best_opposite=101)
        clock.advance_ms(11)

        # 30 lots is not enough to clear the 50 ahead of us.
        assert venue.on_trade(sell_into(100, 30)) == []
        assert venue.orders[1].queue_ahead == 20

    def test_fill_begins_once_the_queue_clears(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=50, best_opposite=101)
        clock.advance_ms(11)
        venue.on_trade(sell_into(100, 50))

        fills = venue.on_trade(sell_into(100, 4))

        assert len(fills) == 1
        assert fills[0].qty == 4

    def test_one_print_can_clear_the_queue_and_fill(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=50, best_opposite=101)
        clock.advance_ms(11)

        fills = venue.on_trade(sell_into(100, 55))

        assert fills[0].qty == 5

    def test_price_improvement_starts_at_the_front(self, venue, clock):
        venue.place(bid(101, 10), visible_depth=0, best_opposite=105)
        clock.advance_ms(11)

        fills = venue.on_trade(sell_into(101, 10))

        assert len(fills) == 1

    def test_cancels_ahead_shorten_the_queue(self, clock):
        venue = PaperVenue(
            instrument=INST,
            config=PaperConfig(latency_ms=0.0, cancel_ahead_ratio=0.5),
            clock=clock,
        )
        venue.on_depth(Side.BUY, 100, 100)
        venue.place(bid(100, 10), visible_depth=100, best_opposite=101)

        # 40 lots vanished without a print; half are assumed to be ahead of us.
        venue.on_depth(Side.BUY, 100, 60)

        assert venue.orders[1].queue_ahead == 80

    def test_conservative_ratio_ignores_cancels(self, clock):
        venue = PaperVenue(
            instrument=INST,
            config=PaperConfig(latency_ms=0.0, cancel_ahead_ratio=0.0),
            clock=clock,
        )
        venue.on_depth(Side.BUY, 100, 100)
        venue.place(bid(100, 10), visible_depth=100, best_opposite=101)
        venue.on_depth(Side.BUY, 100, 20)

        assert venue.orders[1].queue_ahead == 100


class TestPriceHits:
    def test_sell_through_our_bid_fills_us(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=105)
        clock.advance_ms(11)

        assert len(venue.on_trade(sell_into(98, 10))) == 1

    def test_sell_above_our_bid_does_not(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=105)
        clock.advance_ms(11)

        assert venue.on_trade(sell_into(102, 10)) == []

    def test_buy_through_our_ask_fills_us(self, venue, clock):
        venue.place(ask(110, 10), visible_depth=0, best_opposite=105)
        clock.advance_ms(11)

        assert len(venue.on_trade(buy_into(112, 10))) == 1

    def test_aggressor_side_selects_the_right_book(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=110)
        clock.advance_ms(11)

        # An aggressive *buy* cannot fill our bid.
        assert venue.on_trade(buy_into(100, 10)) == []


class TestFills:
    def test_partial_fill_leaves_the_remainder_resting(self, venue, clock):
        order = venue.place(bid(100, 10), visible_depth=0, best_opposite=101)
        clock.advance_ms(11)
        venue.on_trade(sell_into(100, 4))

        assert order.remaining == 6
        assert order.order_id in venue.orders

    def test_full_fill_removes_the_order(self, venue, clock):
        order = venue.place(bid(100, 10), visible_depth=0, best_opposite=101)
        clock.advance_ms(11)
        venue.on_trade(sell_into(100, 10))

        assert order.order_id not in venue.orders

    def test_a_print_sweeps_best_price_first(self, venue, clock):
        venue.place(bid(101, 5), visible_depth=0, best_opposite=110)
        venue.place(bid(100, 5), visible_depth=0, best_opposite=110)
        clock.advance_ms(11)

        fills = venue.on_trade(sell_into(99, 7))

        assert [f.price for f in fills] == [101, 100]
        assert [f.qty for f in fills] == [5, 2]

    def test_fill_is_recorded_as_maker(self, venue, clock):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=101)
        clock.advance_ms(11)
        fill = venue.on_trade(sell_into(100, 10))[0]

        assert fill.maker_owner == "mm"
        assert fill.aggressor is Side.SELL


class TestRejects:
    def test_crossing_quote_is_refused(self, venue):
        assert venue.place(bid(105, 10), visible_depth=0, best_opposite=105) is None
        assert venue.rejected == 1

    def test_crossing_ask_is_refused(self, venue):
        assert venue.place(ask(95, 10), visible_depth=0, best_opposite=95) is None

    def test_zero_size_is_refused(self, venue):
        assert venue.place(bid(100, 0), visible_depth=0, best_opposite=101) is None

    def test_no_opposite_price_is_allowed(self, venue):
        assert venue.place(bid(100, 10), visible_depth=0, best_opposite=None) is not None


class TestLifecycle:
    def test_cancel_removes_the_order(self, venue):
        order = venue.place(bid(100, 10), visible_depth=0, best_opposite=101)

        assert venue.cancel(order.order_id) is True
        assert venue.cancel(order.order_id) is False
        assert venue.open_orders() == []

    def test_cancel_all_reports_the_count(self, venue):
        venue.place(bid(100, 10), visible_depth=0, best_opposite=110)
        venue.place(ask(105, 10), visible_depth=0, best_opposite=100)

        assert venue.cancel_all() == 2
        assert venue.resting_lots == 0


class TestReachCounters:
    """Zero fills has two causes; the counters have to tell them apart."""

    def test_a_print_away_from_our_price_counts_but_does_not_reach(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(20)
        venue.on_trade(TradeTick(price=99, qty=10, aggressor=Side.BUY, trade_id=1))
        assert venue.prints_seen == 1
        assert venue.prints_at_our_price == 0
        assert venue.filled_lots == 0

    def test_a_print_at_our_price_reaches_even_when_the_queue_eats_it(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=50, best_opposite=101)
        clock.advance_ms(20)
        fills = venue.on_trade(TradeTick(price=100, qty=10, aggressor=Side.SELL, trade_id=1))
        assert fills == []
        assert venue.prints_at_our_price == 1
        assert venue.queue_absorbed_lots == 10
        assert venue.filled_lots == 0

    def test_absorbed_and_filled_are_counted_separately(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=3, best_opposite=101)
        clock.advance_ms(20)
        venue.on_trade(TradeTick(price=100, qty=6, aggressor=Side.SELL, trade_id=1))
        assert venue.queue_absorbed_lots == 3
        assert venue.filled_lots == 3

    def test_counters_accumulate_across_prints(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(20)
        for i in range(3):
            venue.on_trade(TradeTick(price=98, qty=1, aggressor=Side.BUY, trade_id=i))
        assert venue.prints_seen == 3
        assert venue.prints_at_our_price == 0


class TestGapThrough:
    """Fills the tape never explains — the model's old blind spot.

    A resting bid the public ask has fallen to or below is a crossed market.
    It cannot survive: whoever posted that ask crossed our price to reach it,
    and we were the better-priced resting order. Counting only prints skipped
    these, and it skipped them one-sidedly — the market that jumps through a
    quote is the market that has just gone against it.
    """

    def test_a_book_that_jumps_past_our_bid_fills_it(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=50, best_opposite=101)
        clock.advance_ms(20)

        fills = venue.on_book(best_bid=98, best_ask=99)

        assert [(f.price, f.qty) for f in fills] == [(100, 5)]
        assert fills[0].aggressor is Side.SELL
        assert venue.gap_fills == 1
        assert venue.gap_filled_lots == 5

    def test_a_book_that_jumps_past_our_ask_fills_it(self, venue, clock):
        venue.place(ask(100, 5), visible_depth=50, best_opposite=99)
        clock.advance_ms(20)

        fills = venue.on_book(best_bid=101, best_ask=102)

        assert [(f.price, f.qty) for f in fills] == [(100, 5)]
        assert fills[0].aggressor is Side.BUY

    def test_the_queue_does_not_protect_a_level_that_was_swept(self, venue, clock):
        """A print says size traded; a moved touch says the level is gone."""
        venue.place(bid(100, 5), visible_depth=1_000_000, best_opposite=101)
        clock.advance_ms(20)

        assert venue.on_book(best_bid=90, best_ask=91) != []

    def test_a_touch_that_merely_reaches_us_does_not_fill(self, venue, clock):
        """Our bid at the public bid is ordinary; only a cross is not."""
        venue.place(bid(100, 5), visible_depth=10, best_opposite=101)
        clock.advance_ms(20)

        assert venue.on_book(best_bid=100, best_ask=101) == []
        assert venue.gap_fills == 0

    def test_an_order_still_in_flight_is_not_filled(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(1)  # latency is 10ms

        assert venue.on_book(best_bid=98, best_ask=99) == []

    def test_a_one_sided_book_fills_nothing_it_cannot_see(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(20)

        assert venue.on_book(best_bid=None, best_ask=None) == []

    def test_it_can_be_switched_off_to_reproduce_the_old_numbers(self, clock):
        off = PaperVenue(
            instrument=INST,
            config=PaperConfig(latency_ms=10.0, gap_through_fills=False),
            clock=clock,
        )
        off.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(20)

        assert off.on_book(best_bid=90, best_ask=91) == []
        assert off.open_orders() != []

    def test_a_gap_fill_leaves_no_order_behind_to_fill_twice(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(20)

        venue.on_book(best_bid=98, best_ask=99)

        assert venue.open_orders() == []
        assert venue.on_book(best_bid=98, best_ask=99) == []
        assert venue.on_trade(sell_into(100, 5)) == []

    def test_the_fill_carries_the_event_time_not_the_wall_clock(self, venue, clock):
        venue.place(bid(100, 5), visible_depth=0, best_opposite=101)
        clock.advance_ms(20)

        fills = venue.on_book(best_bid=98, best_ask=99, ts_ns=777)

        assert fills[0].ts_ns == 777
