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
