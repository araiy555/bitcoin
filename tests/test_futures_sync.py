"""USDⓈ-M futures feed: the depth handshake, which is *not* the spot one.

The dangerous property of the two protocols is that they are close enough
that the wrong one mostly works. These tests pin the differences directly:
futures brackets `lastUpdateId` itself rather than the one after it, and
checks continuity against the exchange-supplied `pu` rather than an inferred
`prev_u + 1`. A stream that satisfies one rule and violates the other is
constructed explicitly, because that is the case a shared implementation
would silently get wrong.
"""

import asyncio
import json
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import (
    DepthDelta,
    DepthSnapshot,
    FeedStatus,
    Liquidation,
    MarkPrice,
    OpenInterest,
    TradeTick,
)
from jsboard.feed.binance_futures import BinanceFuturesFeed

PERP = Instrument("BTCUSDT", tick_size=Decimal("0.1"), lot_size=Decimal("0.001"))


class FakeSocket:
    def __init__(self, frames):
        self.frames = list(frames)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for frame in self.frames:
            await asyncio.sleep(0)
            yield json.dumps(frame)


def depth(U, u, pu, bids=(), asks=()):
    return {
        "stream": "btcusdt@depth@100ms",
        "data": {
            "e": "depthUpdate",
            "E": 1_700_000_000_000,
            "T": 1_700_000_000_000,
            "s": "BTCUSDT",
            "U": U,
            "u": u,
            "pu": pu,
            "b": [[str(p), str(q)] for p, q in bids],
            "a": [[str(p), str(q)] for p, q in asks],
        },
    }


def agg_trade(price, qty, buyer_is_maker):
    return {
        "stream": "btcusdt@aggTrade",
        "data": {
            "e": "aggTrade", "E": 1_700_000_000_000, "s": "BTCUSDT", "a": 7,
            "p": str(price), "q": str(qty), "T": 1_700_000_000_000, "m": buyer_is_maker,
        },
    }


def mark_price(mark, index, rate, next_funding=1_700_000_600_000):
    return {
        "stream": "btcusdt@markPrice@1s",
        "data": {
            "e": "markPriceUpdate", "E": 1_700_000_000_000, "s": "BTCUSDT",
            "p": str(mark), "i": str(index), "P": str(mark), "r": str(rate), "T": next_funding,
        },
    }


def force_order(price, qty, side, avg=None):
    return {
        "stream": "btcusdt@forceOrder",
        "data": {
            "e": "forceOrder", "E": 1_700_000_000_000,
            "o": {
                "s": "BTCUSDT", "S": side, "o": "LIMIT", "f": "IOC", "q": str(qty),
                "p": str(price), "ap": str(avg) if avg is not None else "0",
                "X": "FILLED", "T": 1_700_000_000_000,
            },
        },
    }


def make_feed(snapshot_ids, monkeypatch):
    feed = BinanceFuturesFeed(PERP)
    ids = list(snapshot_ids)

    async def fake_snapshot():
        last = ids.pop(0) if len(ids) > 1 else ids[0]
        return DepthSnapshot(bids=((640_000, 1),), asks=((640_001, 1),), last_update_id=last)

    async def no_polling():
        await asyncio.Event().wait()

    monkeypatch.setattr(feed, "fetch_snapshot", fake_snapshot)
    monkeypatch.setattr(feed, "_poll_open_interest", no_polling)
    return feed


async def collect(feed, frames):
    return [event async for event in feed._sync(FakeSocket(frames))]


class TestConstruction:
    def test_rejects_a_spot_only_depth_interval(self):
        with pytest.raises(ValueError, match="futures depth diffs"):
            BinanceFuturesFeed(PERP, depth_ms=1000)

    def test_accepts_the_futures_intervals(self):
        for ms in (100, 250, 500):
            assert BinanceFuturesFeed(PERP, depth_ms=ms).depth_ms == ms

    def test_rejects_a_spot_only_snapshot_limit(self):
        # 5000 is valid on spot and not on futures.
        with pytest.raises(ValueError, match="snapshot_limit"):
            BinanceFuturesFeed(PERP, snapshot_limit=5000)

    def test_subscribes_to_all_four_streams(self):
        url = BinanceFuturesFeed(PERP)._stream_url

        assert url.startswith("wss://fstream.binance.com/stream")
        for stream in ("@depth@100ms", "@aggTrade", "@markPrice@1s", "@forceOrder"):
            assert stream in url


class TestParsers:
    def test_depth_event_carries_pu(self):
        feed = BinanceFuturesFeed(PERP)
        delta, pu = feed._parse_depth_event(depth(157, 160, 149)["data"])

        assert (delta.first_id, delta.final_id, pu) == (157, 160, 149)

    def test_depth_uses_transaction_time_not_event_time(self):
        """`T` is when the book changed; `E` is when the message was built."""
        feed = BinanceFuturesFeed(PERP)
        raw = depth(1, 2, 0)["data"]
        raw["E"] = 1_700_000_009_000
        raw["T"] = 1_700_000_000_000

        delta, _ = feed._parse_depth_event(raw)

        assert delta.ts_ns == 1_700_000_000_000 * 1_000_000

    def test_buyer_is_maker_means_the_seller_crossed(self):
        feed = BinanceFuturesFeed(PERP)

        assert feed._parse_trade(agg_trade("64000.0", "1.5", True)["data"]).aggressor is Side.SELL
        assert feed._parse_trade(agg_trade("64000.0", "1.5", False)["data"]).aggressor is Side.BUY

    def test_mark_price_carries_index_and_funding(self):
        feed = BinanceFuturesFeed(PERP)
        mp = feed._parse_mark_price(mark_price("64010.0", "64000.0", "0.00012")["data"])

        assert mp.mark == 640_100
        assert mp.index == 640_000
        assert mp.funding_rate == pytest.approx(0.00012)
        assert mp.next_funding_ns == 1_700_000_600_000 * 1_000_000

    def test_negative_funding_is_preserved(self):
        """Shorts paying longs is a real state, not a parse error."""
        feed = BinanceFuturesFeed(PERP)
        mp = feed._parse_mark_price(mark_price("64000.0", "64010.0", "-0.00025")["data"])

        assert mp.funding_rate < 0

    def test_liquidation_prefers_the_average_fill_price(self):
        feed = BinanceFuturesFeed(PERP)
        liq = feed._parse_liquidation(force_order("64000.0", "2.5", "SELL", avg="63990.0")["data"])

        assert liq.price == 639_900
        assert liq.qty == 2_500
        assert liq.side is Side.SELL

    def test_liquidation_falls_back_to_the_order_price(self):
        feed = BinanceFuturesFeed(PERP)
        liq = feed._parse_liquidation(force_order("64000.0", "1.0", "BUY")["data"])

        assert liq.price == 640_000
        assert liq.side is Side.BUY


@pytest.mark.asyncio
class TestFuturesSync:
    async def test_first_event_brackets_last_update_id_itself(self):
        """Futures straddles lastUpdateId; spot straddles lastUpdateId + 1."""
        feed = make_feed([100], pytest.MonkeyPatch())
        events = await collect(feed, [depth(95, 100, 90), depth(101, 105, 100)])

        deltas = [e for e in events if isinstance(e, DepthDelta)]
        assert (deltas[0].first_id, deltas[0].final_id) == (95, 100)

    async def test_events_entirely_before_the_snapshot_are_dropped(self):
        feed = make_feed([100], pytest.MonkeyPatch())
        events = await collect(
            feed, [depth(80, 90, 75), depth(95, 100, 90), depth(101, 105, 100)]
        )

        deltas = [(d.first_id, d.final_id) for d in events if isinstance(d, DepthDelta)]
        assert (80, 90) not in deltas

    async def test_continuity_is_checked_against_pu(self):
        feed = make_feed([100], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [depth(95, 100, 90), depth(101, 105, 100), depth(106, 110, 105)],
        )

        assert not any(isinstance(e, FeedStatus) and e.state == "resyncing" for e in events)
        assert len([e for e in events if isinstance(e, DepthDelta)]) == 3

    async def test_a_pu_mismatch_triggers_resync(self):
        feed = make_feed([100, 300], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [
                depth(95, 100, 90),
                depth(101, 105, 100),
                depth(120, 125, 118),  # pu=118 but the last u was 105
                depth(295, 300, 290),
            ],
        )

        statuses = [e for e in events if isinstance(e, FeedStatus)]
        assert any(s.state == "resyncing" and "pu=118" in s.detail for s in statuses)

    async def test_a_stream_that_only_the_spot_rule_would_accept_is_rejected(self):
        """The discriminating case a shared implementation would get wrong.

        Ids stay contiguous by the spot test (`U == prev_u + 1`) while `pu`
        says a message was missed. Spot logic sees a healthy stream; futures
        logic must resynchronise.
        """
        feed = make_feed([100, 400], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [
                depth(95, 100, 90),
                depth(101, 105, 100),
                depth(106, 110, 999),  # contiguous U, but pu is wrong
                depth(395, 400, 390),
            ],
        )

        assert any(
            isinstance(e, FeedStatus) and e.state == "resyncing" for e in events
        ), "futures must reject a stream that only satisfies the spot rule"

    async def test_a_stream_that_only_the_futures_rule_accepts_is_kept(self):
        """The mirror case: `U` jumps, but `pu` chains correctly.

        Update-id ranges need not be adjacent on futures — only `pu` matters.
        Applying the spot rule here would resync a perfectly good stream.
        """
        feed = make_feed([100], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [
                depth(95, 100, 90),
                depth(140, 150, 100),  # U jumps well past prev_u + 1
                depth(180, 190, 150),
            ],
        )

        assert not any(isinstance(e, FeedStatus) and e.state == "resyncing" for e in events)
        assert len([e for e in events if isinstance(e, DepthDelta)]) == 3

    async def test_a_snapshot_older_than_the_buffer_forces_a_fresh_one(self):
        feed = make_feed([100, 200], pytest.MonkeyPatch())
        events = await collect(feed, [depth(150, 160, 149), depth(195, 200, 190)])

        statuses = [e.detail for e in events if isinstance(e, FeedStatus)]
        assert any("did not join" in s for s in statuses)

    async def test_it_recovers_and_keeps_streaming_after_a_gap(self):
        feed = make_feed([100, 300], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [
                depth(95, 100, 90),
                depth(200, 205, 199),  # gap
                depth(295, 300, 290),
                depth(301, 310, 300),
            ],
        )

        deltas = [(d.first_id, d.final_id) for d in events if isinstance(d, DepthDelta)]
        assert (95, 100) in deltas
        assert (301, 310) in deltas
        assert len([e for e in events if isinstance(e, DepthSnapshot)]) == 2


@pytest.mark.asyncio
class TestNonBookStreams:
    async def test_derivative_events_pass_through_during_sync(self):
        """Mark price and liquidations must not wait on the book handshake."""
        feed = make_feed([100], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [
                mark_price("64010.0", "64000.0", "0.0001"),
                force_order("63900.0", "5.0", "SELL", avg="63890.0"),
                agg_trade("64000.0", "0.5", True),
                depth(95, 100, 90),
            ],
        )

        assert any(isinstance(e, MarkPrice) for e in events)
        assert any(isinstance(e, Liquidation) for e in events)
        assert any(isinstance(e, TradeTick) for e in events)

    async def test_unknown_message_types_are_ignored(self):
        feed = make_feed([100], pytest.MonkeyPatch())
        events = await collect(
            feed,
            [{"stream": "x", "data": {"e": "kline", "k": {}}}, depth(95, 100, 90)],
        )

        assert len([e for e in events if isinstance(e, DepthDelta)]) == 1

    async def test_queued_open_interest_is_emitted(self, monkeypatch):
        feed = make_feed([100], monkeypatch)
        feed._oi_queue.put_nowait(OpenInterest(lots=123_456))

        events = await collect(feed, [depth(95, 100, 90), depth(101, 105, 100)])

        oi = [e for e in events if isinstance(e, OpenInterest)]
        assert oi and oi[0].lots == 123_456
