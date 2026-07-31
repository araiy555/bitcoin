"""Position accounting: average cost, realised P&L, flips, and fees."""

from decimal import Decimal

import pytest

from jsboard.core.types import Fill, Instrument, Side
from jsboard.mm.inventory import FeeSchedule, PnLTracker, Position

# tick $1, lot 1.0 — keeps the arithmetic in these tests readable.
INST = Instrument("TEST", tick_size=Decimal("1"), lot_size=Decimal("1"), base="X", quote="USD")


@pytest.fixture
def pos():
    return Position(instrument=INST, fees=FeeSchedule(maker_bps=0.0, taker_bps=0.0))


def test_opening_a_long_sets_average(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)

    assert pos.lots == 10
    assert pos.avg_price == 100
    assert pos.realized_pnl == 0.0


def test_adding_blends_the_average(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)
    pos.on_fill(110, 10, Side.BUY, is_maker=True)

    assert pos.lots == 20
    assert pos.avg_price == pytest.approx(105.0)


def test_closing_realises_the_difference(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)
    realized = pos.on_fill(110, 10, Side.SELL, is_maker=True)

    assert pos.lots == 0
    assert realized == pytest.approx(100.0)  # 10 lots * $10
    assert pos.realized_pnl == pytest.approx(100.0)


def test_partial_close_keeps_the_average(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)
    pos.on_fill(110, 4, Side.SELL, is_maker=True)

    assert pos.lots == 6
    assert pos.avg_price == pytest.approx(100.0)
    assert pos.realized_pnl == pytest.approx(40.0)


def test_short_side_realises_with_the_right_sign(pos):
    pos.on_fill(100, 10, Side.SELL, is_maker=True)
    assert pos.lots == -10

    # Buying back cheaper is a profit for a short.
    realized = pos.on_fill(90, 10, Side.BUY, is_maker=True)
    assert realized == pytest.approx(100.0)


def test_flip_through_zero_reopens_at_the_fill_price(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)
    pos.on_fill(110, 25, Side.SELL, is_maker=True)

    assert pos.lots == -15
    assert pos.avg_price == pytest.approx(110.0)
    # Only the 10 lots that closed the long are realised.
    assert pos.realized_pnl == pytest.approx(100.0)


def test_unrealized_tracks_the_mark(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)

    assert pos.unrealized_pnl(105) == pytest.approx(50.0)
    assert pos.unrealized_pnl(95) == pytest.approx(-50.0)
    assert pos.unrealized_pnl(None) == 0.0


def test_total_pnl_combines_both_legs(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)
    pos.on_fill(110, 5, Side.SELL, is_maker=True)

    assert pos.total_pnl(120) == pytest.approx(50.0 + 100.0)


def test_flat_position_has_no_unrealized(pos):
    pos.on_fill(100, 10, Side.BUY, is_maker=True)
    pos.on_fill(150, 10, Side.SELL, is_maker=True)

    assert pos.is_flat
    assert pos.unrealized_pnl(999) == 0.0


class TestFees:
    def test_maker_fee_reduces_realised_pnl(self):
        pos = Position(instrument=INST, fees=FeeSchedule(maker_bps=10.0, taker_bps=20.0))
        pos.on_fill(100, 10, Side.BUY, is_maker=True)

        # 10bps of $1,000 notional.
        assert pos.fees_paid == pytest.approx(1.0)
        assert pos.realized_pnl == pytest.approx(-1.0)

    def test_taker_costs_more_than_maker(self):
        pos = Position(instrument=INST, fees=FeeSchedule(maker_bps=10.0, taker_bps=20.0))
        pos.on_fill(100, 10, Side.BUY, is_maker=False)

        assert pos.fees_paid == pytest.approx(2.0)

    def test_rebate_is_a_credit(self):
        pos = Position(instrument=INST, fees=FeeSchedule(maker_bps=-2.0))
        pos.on_fill(100, 10, Side.BUY, is_maker=True)

        assert pos.fees_paid == pytest.approx(-0.2)
        assert pos.realized_pnl == pytest.approx(0.2)

    def test_maker_and_taker_counts_are_tracked(self, pos):
        pos.on_fill(100, 1, Side.BUY, is_maker=True)
        pos.on_fill(100, 1, Side.BUY, is_maker=False)

        assert (pos.maker_fills, pos.taker_fills) == (1, 1)


class TestApplyFill:
    def test_our_maker_side_is_booked_opposite_the_aggressor(self, pos):
        fill = Fill(
            price=100,
            qty=5,
            maker_id=1,
            taker_id=2,
            maker_owner="mm",
            taker_owner="market",
            aggressor=Side.BUY,
            ts_ns=0,
        )
        pos.apply(fill, "mm")

        # The market bought, so we (the maker) sold.
        assert pos.lots == -5

    def test_unrelated_fill_is_ignored(self, pos):
        fill = Fill(
            price=100,
            qty=5,
            maker_id=1,
            taker_id=2,
            maker_owner="someone",
            taker_owner="else",
            aggressor=Side.BUY,
            ts_ns=0,
        )
        pos.apply(fill, "mm")

        assert pos.lots == 0

    def test_zero_quantity_is_a_no_op(self, pos):
        assert pos.on_fill(100, 0, Side.BUY, is_maker=True) == 0.0
        assert pos.fill_count == 0


class TestDrawdown:
    def test_drawdown_measures_from_the_high_water_mark(self):
        t = PnLTracker()
        for equity in (0, 100, 250, 180):
            t.update(equity)

        assert t.peak == 250
        assert t.drawdown == pytest.approx(70)

    def test_new_high_resets_the_drawdown(self):
        t = PnLTracker()
        for equity in (0, 100, 50, 150):
            t.update(equity)

        assert t.drawdown == 0.0
