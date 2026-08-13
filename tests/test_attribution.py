"""P&L attribution: Total = Spread Capture + Inventory − Fees.

The identity is the point. If the three terms do not add back to the P&L the
position accounting reports, the split is a story rather than a measurement,
so the reconciliation cases matter more than the individual ones.
"""

from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.mm.attribution import PnLAttribution
from jsboard.mm.inventory import FeeSchedule, Position

# One tick = 1 quote unit, one lot = 1 base unit: the arithmetic stays visible.
INST = Instrument("TEST", tick_size=Decimal("1"), lot_size=Decimal("1"), base="T", quote="U")


@pytest.fixture
def attr():
    return PnLAttribution(INST)


class TestSpreadCapture:
    def test_buying_below_the_mid_earns_the_distance(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        assert attr.spread_capture == pytest.approx(10.0)

    def test_selling_above_the_mid_earns_it_too(self, attr):
        attr.on_fill(price_ticks=101, qty_lots=10, sign=-1, fee=0.0, mid_ticks=100)
        assert attr.spread_capture == pytest.approx(10.0)

    def test_paying_up_through_the_mid_is_negative(self, attr):
        attr.on_fill(price_ticks=102, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        assert attr.spread_capture == pytest.approx(-20.0)

    def test_it_is_booked_once_and_never_revised(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        attr.on_mid(140)
        assert attr.spread_capture == pytest.approx(10.0)


class TestInventory:
    def test_a_flat_book_carries_nothing(self, attr):
        attr.on_mid(100)
        attr.on_mid(150)
        assert attr.inventory_pnl == pytest.approx(0.0)

    def test_a_long_gains_when_the_mid_rises(self, attr):
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        attr.on_mid(103)
        assert attr.inventory_pnl == pytest.approx(30.0)

    def test_a_short_loses_when_the_mid_rises(self, attr):
        attr.on_fill(price_ticks=100, qty_lots=10, sign=-1, fee=0.0, mid_ticks=100)
        attr.on_mid(103)
        assert attr.inventory_pnl == pytest.approx(-30.0)

    def test_a_fill_does_not_earn_carry_from_before_it_existed(self, attr):
        attr.on_mid(100)
        attr.on_mid(110)  # nothing held through this move
        attr.on_fill(price_ticks=110, qty_lots=10, sign=+1, fee=0.0, mid_ticks=110)
        assert attr.inventory_pnl == pytest.approx(0.0)

    def test_carry_is_settled_before_the_position_changes(self, attr):
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        # Mid moves to 110, then we add. Only the first 10 lots earn the move.
        attr.on_fill(price_ticks=110, qty_lots=90, sign=+1, fee=0.0, mid_ticks=110)
        assert attr.inventory_pnl == pytest.approx(100.0)


class TestFeesAndTotal:
    def test_fees_subtract(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=4.0, mid_ticks=100)
        assert attr.total == pytest.approx(6.0)

    def test_unpriced_fills_are_counted_not_guessed(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=0.0, mid_ticks=None)
        assert attr.spread_capture == pytest.approx(0.0)
        assert attr.summary()["unpriced_fills"] == 1.0

    def test_a_zero_size_fill_books_nothing(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=0, sign=+1, fee=9.0, mid_ticks=100)
        assert attr.total == pytest.approx(0.0)


class TestReconciliation:
    """The terms must add back to what the position accounting reports."""

    def book(self, fills, final_mid, maker_bps=2.0):
        fees = FeeSchedule(maker_bps=maker_bps, taker_bps=4.0)
        position = Position(instrument=INST, fees=fees)
        attr = PnLAttribution(INST)
        for price, qty, side, mid in fills:
            attr.on_mid(mid)
            before = position.fees_paid
            position.on_fill(price, qty, side, is_maker=True)
            attr.on_fill(
                price_ticks=price,
                qty_lots=qty,
                sign=side.sign,
                fee=position.fees_paid - before,
                mid_ticks=mid,
            )
        attr.on_mid(final_mid)
        return position, attr

    def test_a_round_trip_reconciles(self):
        position, attr = self.book(
            [(99, 10, Side.BUY, 100), (101, 10, Side.SELL, 100)], final_mid=100
        )
        assert attr.total == pytest.approx(position.total_pnl(100))

    def test_an_open_position_reconciles_against_the_mark(self):
        position, attr = self.book([(99, 10, Side.BUY, 100)], final_mid=120)
        assert attr.total == pytest.approx(position.total_pnl(120))

    def test_a_losing_run_reconciles(self):
        position, attr = self.book(
            [(99, 10, Side.BUY, 100), (90, 10, Side.SELL, 91)], final_mid=91
        )
        assert attr.total == pytest.approx(position.total_pnl(91))

    def test_a_flip_through_zero_reconciles(self):
        position, attr = self.book(
            [(99, 10, Side.BUY, 100), (105, 30, Side.SELL, 104)], final_mid=104
        )
        assert attr.total == pytest.approx(position.total_pnl(104))

    def test_the_split_is_not_trivial(self):
        # A reconciliation that passes because both terms are zero proves
        # nothing about the decomposition.
        _, attr = self.book([(99, 10, Side.BUY, 100)], final_mid=120)
        assert attr.spread_capture > 0
        assert attr.inventory_pnl > 0
        assert attr.fees > 0


class TestPerRoundTripBps:
    def test_terms_scale_to_the_notional_turned_over(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        attr.on_fill(price_ticks=101, qty_lots=10, sign=-1, fee=0.0, mid_ticks=100)
        # 10 units matched at a mid of 100 = 1,000 of notional; 20 earned.
        bps = attr.per_round_trip_bps(10.0)
        assert bps["spread_capture"] == pytest.approx(200.0)

    def test_nothing_matched_reports_nothing(self, attr):
        assert attr.per_round_trip_bps(0.0) == {}


class TestExposureClock:
    """How long the position was carried, which the inventory term is the price of."""

    S = 1_000_000_000

    def test_a_flat_session_is_never_exposed(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_mid(101, now_ns=10 * self.S)
        assert attr.exposed_share == pytest.approx(0.0)
        assert attr.mean_abs_position == pytest.approx(0.0)

    def test_exposure_starts_at_the_fill_not_before(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_mid(100, now_ns=5 * self.S)  # flat for 5s
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100,
                     now_ns=5 * self.S)
        attr.on_mid(100, now_ns=10 * self.S)  # long for 5s
        assert attr.exposed_share == pytest.approx(0.5)

    def test_mean_position_is_time_weighted_not_fill_weighted(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(100, now_ns=9 * self.S)  # 10 units for 9s
        attr.on_fill(price_ticks=100, qty_lots=10, sign=-1, fee=0.0, mid_ticks=100,
                     now_ns=9 * self.S)
        attr.on_mid(100, now_ns=10 * self.S)  # flat for 1s
        assert attr.mean_abs_position == pytest.approx(9.0)

    def test_a_short_counts_as_exposure_too(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=4, sign=-1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(100, now_ns=10 * self.S)
        assert attr.exposed_share == pytest.approx(1.0)
        assert attr.mean_abs_position == pytest.approx(4.0)

    def test_time_is_ignored_when_no_clock_is_supplied(self, attr):
        attr.on_mid(100)
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        attr.on_mid(110)
        assert attr.elapsed_ns == 0
        assert attr.exposed_share == pytest.approx(0.0)
        # ... and the P&L terms are unaffected by the missing clock.
        assert attr.inventory_pnl == pytest.approx(100.0)

    def test_a_clock_that_goes_backwards_does_not_subtract_time(self, attr):
        attr.on_mid(100, now_ns=10 * self.S)
        attr.on_mid(100, now_ns=1 * self.S)
        assert attr.elapsed_ns == 0


class TestAgeBuckets:
    """Inventory P&L split by how long the parcel had been held.

    Subtracting a fill-aggregated mark-out from a time-aggregated inventory
    P&L does not give "the loss after 100ms" — the two have different bases
    and overlapping positions do not cancel. This splits the accounting
    itself, so the buckets sum back to the term they came from.
    """

    S = 1_000_000_000

    def total_buckets(self, attr):
        return sum(attr.inventory_by_age) + attr.inventory_unaged

    def test_buckets_sum_to_the_inventory_term(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        for i in range(1, 40):
            attr.on_mid(100 + i, now_ns=i * self.S // 2)
        assert self.total_buckets(attr) == pytest.approx(attr.inventory_pnl)

    def test_a_move_inside_100ms_lands_in_the_first_bucket(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(101, now_ns=self.S // 20)  # 50ms later
        assert attr.inventory_by_age[0] == pytest.approx(10.0)
        assert sum(attr.inventory_by_age[1:]) == pytest.approx(0.0)

    def test_an_old_parcel_lands_in_the_last_bucket(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(100, now_ns=30 * self.S)
        attr.on_mid(101, now_ns=31 * self.S)
        assert attr.inventory_by_age[-1] == pytest.approx(10.0)
        assert sum(attr.inventory_by_age[:-1]) == pytest.approx(0.0)

    def test_two_parcels_of_different_ages_split_one_move(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(100, now_ns=30 * self.S)
        # A second parcel, brand new, while the first is 30s old.
        attr.on_fill(price_ticks=100, qty_lots=5, sign=+1, fee=0.0, mid_ticks=100,
                     now_ns=30 * self.S)
        attr.on_mid(101, now_ns=30 * self.S + self.S // 20)
        assert attr.inventory_by_age[0] == pytest.approx(5.0)   # the new parcel
        assert attr.inventory_by_age[-1] == pytest.approx(10.0)  # the old one
        assert self.total_buckets(attr) == pytest.approx(attr.inventory_pnl)

    def test_reducing_consumes_the_oldest_parcel_first(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(100, now_ns=30 * self.S)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100,
                     now_ns=30 * self.S)
        # Sell 10: FIFO retires the 30s-old parcel, leaving only the fresh one.
        attr.on_fill(price_ticks=100, qty_lots=10, sign=-1, fee=0.0, mid_ticks=100,
                     now_ns=30 * self.S)
        attr.on_mid(101, now_ns=30 * self.S + self.S // 20)
        assert attr.inventory_by_age[0] == pytest.approx(10.0)
        assert attr.inventory_by_age[-1] == pytest.approx(0.0)

    def test_lots_stay_in_step_with_the_position(self, attr):
        attr.on_mid(100, now_ns=0)
        moves = [(+1, 10), (+1, 5), (-1, 12), (-1, 20), (+1, 3)]
        for i, (sign, qty) in enumerate(moves):
            attr.on_fill(price_ticks=100, qty_lots=qty, sign=sign, fee=0.0, mid_ticks=100,
                         now_ns=i * self.S)
            assert sum(x.lots for x in attr.open_lots) == attr.position_lots

    def test_a_flip_through_zero_reconciles_and_restarts_the_age(self, attr):
        attr.on_mid(100, now_ns=0)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100, now_ns=0)
        attr.on_mid(100, now_ns=30 * self.S)
        attr.on_fill(price_ticks=100, qty_lots=30, sign=-1, fee=0.0, mid_ticks=100,
                     now_ns=30 * self.S)
        assert attr.position_lots == -20
        attr.on_mid(101, now_ns=30 * self.S + self.S // 20)
        assert attr.inventory_by_age[0] == pytest.approx(-20.0)
        assert self.total_buckets(attr) == pytest.approx(attr.inventory_pnl)

    def test_without_a_clock_the_move_is_held_apart_not_bucketed(self, attr):
        attr.on_mid(100)
        attr.on_fill(price_ticks=100, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        attr.on_mid(110)
        assert sum(attr.inventory_by_age) == pytest.approx(0.0)
        assert attr.inventory_unaged == pytest.approx(100.0)
        assert self.total_buckets(attr) == pytest.approx(attr.inventory_pnl)


class TestBreakevenFee:
    def test_the_edge_has_to_cover_both_legs(self, attr):
        attr.on_fill(price_ticks=99, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        attr.on_fill(price_ticks=101, qty_lots=10, sign=-1, fee=0.0, mid_ticks=100)
        # 200 bps of edge over the round trip tolerates 100 bps per side.
        assert attr.max_maker_bps(10.0) == pytest.approx(100.0)

    def test_a_losing_run_tolerates_a_negative_fee_only(self, attr):
        attr.on_fill(price_ticks=101, qty_lots=10, sign=+1, fee=0.0, mid_ticks=100)
        assert attr.max_maker_bps(10.0) < 0

    def test_nothing_matched_has_no_answer(self, attr):
        import math

        assert math.isnan(attr.max_maker_bps(0.0))
