"""bitbank's socket.io stream, as the venue we would quote on.

The failure worth guarding is a diff applied after a newer whole book: it
would put back levels the venue had already cleared.
"""

from decimal import Decimal

from jsboard.core.types import Side
from jsboard.feed.base import DepthDelta, DepthSnapshot, TradeTick
from jsboard.feed.bitbank import BitbankFeed, instrument_from_pair

PAIR = {"name": "ada_jpy", "base_asset": "ada", "quote_asset": "jpy",
        "price_digits": 3, "amount_digits": 4}


def feed():
    return BitbankFeed(instrument_from_pair(PAIR))


def whole(seq, bids, asks):
    return {"room_name": "depth_whole_ada_jpy",
            "message": {"data": {"bids": bids, "asks": asks, "sequenceId": str(seq),
                                 "timestamp": 1_790_000_000_000}}}


def diff(seq, b=(), a=()):
    return {"room_name": "depth_diff_ada_jpy",
            "message": {"data": {"b": list(b), "a": list(a), "s": str(seq), "t": 1_790_000_000_500}}}


class TestSpec:
    def test_digits_become_steps(self):
        inst = instrument_from_pair(PAIR)
        assert inst.tick_size == Decimal("0.001")
        assert inst.lot_size == Decimal("0.0001")
        assert (inst.base, inst.quote) == ("ADA", "JPY")


class TestBook:
    def test_whole_book_is_a_snapshot(self):
        [event] = feed().parse_message(whole(10, [["37.120", "500"]], [["37.140", "250.5"]]))
        assert isinstance(event, DepthSnapshot)
        assert event.bids == ((37120, 5_000_000),)
        assert event.asks == ((37140, 2_505_000),)

    def test_diffs_before_the_first_whole_book_are_dropped(self):
        assert feed().parse_message(diff(5, b=[["37.1", "1"]])) == []

    def test_a_stale_diff_is_dropped_a_fresh_one_applied(self):
        f = feed()
        f.parse_message(whole(10, [["37.120", "500"]], [["37.140", "250"]]))
        assert f.parse_message(diff(9, b=[["37.130", "1"]])) == []
        [event] = f.parse_message(diff(11, b=[["37.130", "1"]], a=[["37.140", "0"]]))
        assert isinstance(event, DepthDelta)
        assert event.bids == ((37130, 10_000),)
        assert event.asks == ((37140, 0),)  # zero removes the level


class TestTrades:
    def test_side_is_the_taker(self):
        events = feed().parse_message(
            {"room_name": "transactions_ada_jpy",
             "message": {"data": {"transactions": [
                 {"transaction_id": 1, "side": "buy", "price": "37.140", "amount": "10", "executed_at": 1},
                 {"transaction_id": 2, "side": "sell", "price": "37.120", "amount": "3", "executed_at": 2},
             ]}}}
        )
        assert [e.aggressor for e in events] == [Side.BUY, Side.SELL]
        assert all(isinstance(e, TradeTick) for e in events)


class TestFrames:
    def test_engine_io_frames(self):
        assert BitbankFeed.decode_frame('0{"sid":"x"}')[0] == "open"
        assert BitbankFeed.decode_frame("40")[0] == "connected"
        assert BitbankFeed.decode_frame("2")[0] == "ping"
        kind, body = BitbankFeed.decode_frame('42["message",{"room_name":"r"}]')
        assert kind == "message" and body == {"room_name": "r"}
