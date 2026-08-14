"""Bybit public order-book normalisation."""

from decimal import Decimal

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthDelta, DepthSnapshot, MarkPrice
from jsboard.feed.bybit import BybitFeed


def test_bybit_parses_snapshot_and_absolute_delta():
    instrument = Instrument("BTCUSDT", Decimal("0.1"), Decimal("0.001"), "BTC", "USDT")
    feed = BybitFeed(instrument, category="linear", depth=50)
    snapshot = feed.parse_message(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": "snapshot",
            "ts": 1_700_000_000_000,
            "data": {
                "u": 42,
                "b": [["100.1", "0.005"]],
                "a": [["100.2", "0.007"]],
            },
        }
    )
    assert isinstance(snapshot, DepthSnapshot)
    assert snapshot.bids == ((1001, 5),)
    assert snapshot.asks == ((1002, 7),)
    assert snapshot.last_update_id == 42

    delta = feed.parse_message(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "ts": 1_700_000_000_010,
            "data": {"u": 43, "b": [["100.1", "0"]], "a": [["100.3", "0.009"]]},
        }
    )
    assert isinstance(delta, DepthDelta)
    assert delta.bids == ((1001, 0),)
    assert delta.asks == ((1003, 9),)
    assert delta.first_id == delta.final_id == 43


def test_bybit_update_id_one_resets_book_even_if_labelled_delta():
    instrument = Instrument("BTCUSDT", Decimal("0.1"), Decimal("0.001"))
    feed = BybitFeed(instrument)
    event = feed.parse_message(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "ts": 1,
            "data": {"u": 1, "b": [["100", "1"]], "a": [["101", "1"]]},
        }
    )
    assert isinstance(event, DepthSnapshot)


def test_bybit_ticker_keeps_unchanged_funding_fields_across_deltas():
    instrument = Instrument("BTCUSDT", Decimal("0.1"), Decimal("0.001"))
    feed = BybitFeed(instrument)
    first = feed.parse_message(
        {
            "topic": "tickers.BTCUSDT",
            "type": "snapshot",
            "ts": 1000,
            "data": {
                "markPrice": "100.1",
                "indexPrice": "100.0",
                "fundingRate": "0.0001",
                "nextFundingTime": "2000",
            },
        }
    )
    assert isinstance(first, MarkPrice)
    second = feed.parse_message(
        {
            "topic": "tickers.BTCUSDT",
            "type": "delta",
            "ts": 1100,
            "data": {"markPrice": "100.2"},
        }
    )
    assert isinstance(second, MarkPrice)
    assert second.mark == 1002
    assert second.funding_rate == 0.0001
    assert second.next_funding_ns == 2_000_000_000
