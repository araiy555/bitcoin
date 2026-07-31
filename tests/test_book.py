"""Matching engine tests: priority, TIF semantics, STP, and L2 loading."""

from decimal import Decimal

import pytest

from jsboard.core.book import OrderBook, STPPolicy
from jsboard.core.types import Instrument, Order, OrderStatus, Side, TimeInForce

BTC = Instrument("BTCUSDT", tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"))


def limit(side, price, qty, owner="anon", tif=TimeInForce.GTC):
    return Order(side=side, price=price, qty=qty, owner=owner, tif=tif)


@pytest.fixture
def book():
    return OrderBook(stp=STPPolicy.NONE)


def test_resting_orders_set_the_touch(book):
    book.submit(limit(Side.BUY, 100, 5))
    book.submit(limit(Side.BUY, 99, 7))
    book.submit(limit(Side.SELL, 102, 3))

    assert book.best_bid() == 100
    assert book.best_ask() == 102
    assert book.spread == 2
    assert book.mid == 101.0


def test_price_priority_beats_arrival_order(book):
    book.submit(limit(Side.SELL, 105, 5))
    book.submit(limit(Side.SELL, 103, 5))  # arrives later, but better priced

    res = book.submit(limit(Side.BUY, 110, 5))

    assert len(res.fills) == 1
    assert res.fills[0].price == 103


def test_time_priority_within_a_level(book):
    first = book.submit(limit(Side.SELL, 100, 4)).order
    second = book.submit(limit(Side.SELL, 100, 4)).order

    res = book.submit(limit(Side.BUY, 100, 6))

    assert [(f.maker_id, f.qty) for f in res.fills] == [(first.order_id, 4), (second.order_id, 2)]
    assert second.remaining == 2


def test_fills_execute_at_the_maker_price(book):
    book.submit(limit(Side.SELL, 100, 10))
    res = book.submit(limit(Side.BUY, 120, 10))

    assert res.fills[0].price == 100
    assert res.avg_price == 100
    assert res.order.status is OrderStatus.FILLED


def test_partial_fill_rests_the_remainder(book):
    book.submit(limit(Side.SELL, 100, 3))
    res = book.submit(limit(Side.BUY, 100, 10))

    assert res.filled_qty == 3
    assert res.resting is True
    assert book.best_bid() == 100
    assert book.depth_at(Side.BUY, 100) == 7


def test_walking_multiple_levels(book):
    book.submit(limit(Side.SELL, 100, 2))
    book.submit(limit(Side.SELL, 101, 2))
    book.submit(limit(Side.SELL, 102, 2))

    res = book.submit(limit(Side.BUY, 102, 5))

    assert [f.price for f in res.fills] == [100, 101, 102]
    assert res.filled_qty == 5
    assert book.depth_at(Side.SELL, 102) == 1


class TestTimeInForce:
    def test_ioc_cancels_the_unfilled_remainder(self, book):
        book.submit(limit(Side.SELL, 100, 3))
        res = book.submit(limit(Side.BUY, 100, 10, tif=TimeInForce.IOC))

        assert res.filled_qty == 3
        assert res.resting is False
        assert book.best_bid() is None

    def test_fok_rejects_when_it_cannot_fill_in_full(self, book):
        book.submit(limit(Side.SELL, 100, 3))
        res = book.submit(limit(Side.BUY, 100, 10, tif=TimeInForce.FOK))

        assert res.fills == []
        assert res.order.status is OrderStatus.REJECTED
        assert book.depth_at(Side.SELL, 100) == 3  # untouched

    def test_fok_fills_when_liquidity_suffices(self, book):
        book.submit(limit(Side.SELL, 100, 6))
        book.submit(limit(Side.SELL, 101, 6))
        res = book.submit(limit(Side.BUY, 101, 10, tif=TimeInForce.FOK))

        assert res.filled_qty == 10

    def test_post_only_is_rejected_when_it_would_cross(self, book):
        book.submit(limit(Side.SELL, 100, 5))
        res = book.submit(limit(Side.BUY, 100, 5, tif=TimeInForce.POST_ONLY))

        assert res.order.status is OrderStatus.REJECTED
        assert "post-only" in res.reject_reason

    def test_post_only_rests_when_it_does_not_cross(self, book):
        book.submit(limit(Side.SELL, 100, 5))
        res = book.submit(limit(Side.BUY, 99, 5, tif=TimeInForce.POST_ONLY))

        assert res.resting is True
        assert book.best_bid() == 99

    def test_market_order_sweeps_the_book(self, book):
        book.submit(limit(Side.SELL, 100, 2))
        book.submit(limit(Side.SELL, 105, 2))

        res = book.submit(Order(side=Side.BUY, price=None, qty=4, tif=TimeInForce.IOC))

        assert res.filled_qty == 4
        assert [f.price for f in res.fills] == [100, 105]

    def test_market_order_rejects_gtc(self):
        with pytest.raises(ValueError, match="market order cannot be GTC"):
            Order(side=Side.BUY, price=None, qty=1, tif=TimeInForce.GTC)


class TestCancelAndAmend:
    def test_cancel_removes_depth_and_empties_the_level(self, book):
        o = book.submit(limit(Side.BUY, 100, 5)).order
        assert book.cancel(o.order_id) is not None

        assert book.best_bid() is None
        assert book.depth_at(Side.BUY, 100) == 0
        assert book.cancel(o.order_id) is None  # idempotent

    def test_cancelled_order_in_mid_queue_is_skipped(self, book):
        a = book.submit(limit(Side.SELL, 100, 5)).order
        b = book.submit(limit(Side.SELL, 100, 5)).order
        c = book.submit(limit(Side.SELL, 100, 5)).order
        book.cancel(b.order_id)

        res = book.submit(limit(Side.BUY, 100, 10))

        assert [f.maker_id for f in res.fills] == [a.order_id, c.order_id]

    def test_size_reduction_keeps_queue_position(self, book):
        first = book.submit(limit(Side.SELL, 100, 10)).order
        second = book.submit(limit(Side.SELL, 100, 10)).order
        book.amend(first.order_id, qty=4)

        res = book.submit(limit(Side.BUY, 100, 5))

        assert [(f.maker_id, f.qty) for f in res.fills] == [
            (first.order_id, 4),
            (second.order_id, 1),
        ]

    def test_price_change_loses_queue_position(self, book):
        first = book.submit(limit(Side.SELL, 100, 5)).order
        second = book.submit(limit(Side.SELL, 100, 5)).order
        amended = book.amend(first.order_id, price=100, qty=8).order

        res = book.submit(limit(Side.BUY, 100, 6))

        # `second` is now at the front; the replacement went to the back.
        assert res.fills[0].maker_id == second.order_id
        assert res.fills[1].maker_id == amended.order_id

    def test_cancel_all_scopes_to_owner(self, book):
        book.submit(limit(Side.BUY, 100, 5, owner="mm"))
        book.submit(limit(Side.BUY, 99, 5, owner="mm"))
        book.submit(limit(Side.BUY, 98, 5, owner="other"))

        cancelled = book.cancel_all(owner="mm")

        assert len(cancelled) == 2
        assert book.best_bid() == 98


class TestSelfTradePrevention:
    def test_cancel_maker_drops_the_resting_order(self):
        book = OrderBook(stp=STPPolicy.CANCEL_MAKER)
        book.submit(limit(Side.SELL, 100, 5, owner="mm"))
        book.submit(limit(Side.SELL, 100, 5, owner="other"))

        res = book.submit(limit(Side.BUY, 100, 5, owner="mm"))

        assert len(res.fills) == 1
        assert res.fills[0].maker_owner == "other"

    def test_cancel_taker_stops_the_aggressor(self):
        book = OrderBook(stp=STPPolicy.CANCEL_TAKER)
        book.submit(limit(Side.SELL, 100, 5, owner="mm"))

        res = book.submit(limit(Side.BUY, 100, 5, owner="mm"))

        assert res.fills == []
        assert res.order.status is OrderStatus.CANCELLED


class TestL2Loading:
    def test_replace_l2_builds_the_book(self, book):
        book.replace_l2(bids=[(100, 5), (99, 3)], asks=[(101, 4), (102, 6)])

        assert book.best_bid() == 100
        assert book.best_ask() == 101
        assert book.depth_at(Side.SELL, 102) == 6

    def test_replace_l2_leaves_our_own_orders_alone(self, book):
        mine = book.submit(limit(Side.BUY, 100, 7, owner="mm")).order
        book.replace_l2(bids=[(100, 5)], asks=[(101, 4)])

        assert book.depth_at(Side.BUY, 100) == 12  # 5 market + 7 ours
        assert book.get(mine.order_id) is not None

    def test_replace_l2_clears_stale_market_levels(self, book):
        book.replace_l2(bids=[(100, 5), (99, 5)], asks=[])
        book.replace_l2(bids=[(100, 5)], asks=[])

        assert book.depth_at(Side.BUY, 99) == 0

    def test_apply_l2_delta_sets_and_clears(self, book):
        book.apply_l2_delta(Side.BUY, 100, 5)
        assert book.depth_at(Side.BUY, 100) == 5

        book.apply_l2_delta(Side.BUY, 100, 8)
        assert book.depth_at(Side.BUY, 100) == 8

        book.apply_l2_delta(Side.BUY, 100, 0)
        assert book.depth_at(Side.BUY, 100) == 0
        assert book.best_bid() is None

    def test_market_order_priority_is_preserved_across_deltas(self, book):
        ours = book.submit(limit(Side.BUY, 100, 3, owner="mm")).order
        book.apply_l2_delta(Side.BUY, 100, 5)
        book.apply_l2_delta(Side.BUY, 100, 9)

        assert book.get(ours.order_id).is_live
        assert book.depth_at(Side.BUY, 100) == 12


class TestSnapshot:
    def test_snapshot_orders_and_truncates(self, book):
        for p in (95, 96, 97, 98, 99):
            book.submit(limit(Side.BUY, p, 1))
        for p in (101, 102, 103):
            book.submit(limit(Side.SELL, p, 2))

        snap = book.snapshot(depth=3)

        assert [lvl.price for lvl in snap.bids] == [99, 98, 97]
        assert [lvl.price for lvl in snap.asks] == [101, 102, 103]

    def test_microprice_leans_toward_the_thin_side(self, book):
        book.submit(limit(Side.BUY, 100, 90))
        book.submit(limit(Side.SELL, 102, 10))
        snap = book.snapshot()

        # Heavy bid, light ask -> fair value pushed up toward the ask.
        assert snap.mid == 101.0
        assert snap.microprice > 101.0
        assert snap.imbalance() == pytest.approx(0.8)

    def test_imbalance_is_zero_on_a_balanced_book(self, book):
        book.submit(limit(Side.BUY, 100, 5))
        book.submit(limit(Side.SELL, 102, 5))

        assert book.snapshot().imbalance() == 0.0


class TestInstrument:
    def test_tick_and_lot_conversion_roundtrips(self):
        assert BTC.to_ticks("64250.37") == 6425037
        assert BTC.to_price(6425037) == Decimal("64250.37")
        assert BTC.to_lots("0.00123") == 123
        assert BTC.to_qty(123) == Decimal("0.00123")

    def test_quantity_rounds_down_never_up(self):
        # 0.000019 is 1.9 lots; inventing the extra lot would overstate size.
        assert BTC.to_lots("0.000019") == 1

    def test_price_rounds_to_nearest_tick(self):
        assert BTC.to_ticks("100.005") == 10001
        assert BTC.to_ticks("100.004") == 10000
