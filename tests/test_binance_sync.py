"""Binance snapshot/diff handshake, driven by a fake socket.

These exercise the part that cannot be checked by eye: whether the stitched
book is actually continuous. A sync bug does not raise — it silently produces
a book that is a few levels wrong forever, so the gap and join conditions get
tested directly.
"""

import asyncio
import json
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthDelta, DepthSnapshot, FeedStatus, TradeTick
from jsboard.feed.binance import BinanceFeed

INST = Instrument("BTCUSDT", tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"))


class FakeSocket:
    """Async-iterates canned frames, yielding to the loop between each."""

    def __init__(self, frames):
        self.frames = list(frames)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for frame in self.frames:
            # Let pending tasks (the snapshot fetch) make progress.
            await asyncio.sleep(0)
            yield json.dumps(frame)


def depth_frame(U, u, bids=(), asks=()):
    return {
        "stream": "btcusdt@depth@100ms",
        "data": {
            "e": "depthUpdate",
            "E": 1_700_000_000_000,
            "s": "BTCUSDT",
            "U": U,
            "u": u,
            "b": [[str(p), str(q)] for p, q in bids],
            "a": [[str(p), str(q)] for p, q in asks],
        },
    }


def trade_frame(price, qty, buyer_is_maker):
    return {
        "stream": "btcusdt@aggTrade",
        "data": {
            "e": "aggTrade",
            "E": 1_700_000_000_000,
            "s": "BTCUSDT",
            "a": 42,
            "p": str(price),
            "q": str(qty),
            "T": 1_700_000_000_000,
            "m": buyer_is_maker,
        },
    }


def make_feed(snapshot_ids, monkeypatch):
    """A feed whose REST snapshot returns each id in `snapshot_ids` in turn."""
    feed = BinanceFeed(INST)
    ids = list(snapshot_ids)

    async def fake_snapshot():
        last = ids.pop(0) if len(ids) > 1 else ids[0]
        return DepthSnapshot(
            bids=((6_400_000, 100),), asks=((6_400_100, 100),), last_update_id=last
        )

    monkeypatch.setattr(feed, "fetch_snapshot", fake_snapshot)
    return feed


async def collect(feed, frames):
    return [event async for event in feed._sync(FakeSocket(frames))]


class TestParsers:
    def test_buyer_is_maker_means_the_seller_crossed(self):
        feed = BinanceFeed(INST)
        trade = feed._parse_trade(trade_frame("64000.00", "0.5", True)["data"])

        assert trade.aggressor is Side.SELL
        assert trade.price == 6_400_000
        assert trade.qty == 50_000

    def test_buyer_is_taker_means_the_buyer_crossed(self):
        feed = BinanceFeed(INST)
        trade = feed._parse_trade(trade_frame("64000.00", "0.5", False)["data"])

        assert trade.aggressor is Side.BUY

    def test_depth_event_converts_to_ticks_and_lots(self):
        feed = BinanceFeed(INST)
        delta = feed._parse_depth_event(
            depth_frame(10, 12, bids=[("63999.99", "1.5")], asks=[("64000.01", "2.0")])["data"]
        )

        assert delta.bids == ((6_399_999, 150_000),)
        assert delta.asks == ((6_400_001, 200_000),)
        assert (delta.first_id, delta.final_id) == (10, 12)

    def test_rejects_unsupported_depth_interval(self):
        with pytest.raises(ValueError, match="100ms or 1000ms"):
            BinanceFeed(INST, depth_ms=500)

    def test_rejects_invalid_snapshot_limit(self):
        with pytest.raises(ValueError, match="snapshot_limit"):
            BinanceFeed(INST, snapshot_limit=777)


@pytest.mark.asyncio
class TestSync:
    async def test_events_older_than_the_snapshot_are_dropped(self, monkeypatch):
        feed = make_feed([100], monkeypatch)
        events = await collect(
            feed,
            [
                depth_frame(90, 95),  # entirely before the snapshot
                depth_frame(96, 100),  # ends exactly at it
                depth_frame(101, 105, bids=[("63999.99", "1.0")]),  # straddles
                depth_frame(106, 110),
            ],
        )

        deltas = [e for e in events if isinstance(e, DepthDelta)]
        assert [(d.first_id, d.final_id) for d in deltas] == [(101, 105), (106, 110)]

    async def test_snapshot_is_emitted_before_any_delta(self, monkeypatch):
        feed = make_feed([100], monkeypatch)
        events = await collect(feed, [depth_frame(101, 105), depth_frame(106, 110)])

        kinds = [type(e) for e in events]
        assert kinds.index(DepthSnapshot) < kinds.index(DepthDelta)

    async def test_first_applied_event_must_straddle_the_snapshot(self, monkeypatch):
        # Snapshot at 100, but the stream starts at 105 — a hole. The feed must
        # not pretend it is synced; it should ask for a fresh image.
        feed = make_feed([100, 200], monkeypatch)
        events = await collect(feed, [depth_frame(105, 110), depth_frame(201, 205)])

        statuses = [e.detail for e in events if isinstance(e, FeedStatus)]
        assert any("did not join" in s for s in statuses)

    async def test_a_gap_in_the_stream_triggers_resync(self, monkeypatch):
        feed = make_feed([100, 300], monkeypatch)
        events = await collect(
            feed,
            [
                depth_frame(101, 105),
                depth_frame(106, 110),
                depth_frame(120, 125),  # 111..119 missing
                depth_frame(301, 305),
            ],
        )

        statuses = [e for e in events if isinstance(e, FeedStatus)]
        assert any(s.state == "resyncing" and "gap at update id 111" in s.detail for s in statuses)

    async def test_contiguous_stream_never_resyncs(self, monkeypatch):
        feed = make_feed([100], monkeypatch)
        events = await collect(
            feed,
            [depth_frame(101, 105), depth_frame(106, 110), depth_frame(111, 115)],
        )

        assert not any(isinstance(e, FeedStatus) and e.state == "resyncing" for e in events)
        assert len([e for e in events if isinstance(e, DepthDelta)]) == 3

    async def test_it_recovers_and_keeps_streaming_after_a_gap(self, monkeypatch):
        feed = make_feed([100, 300], monkeypatch)
        events = await collect(
            feed,
            [
                depth_frame(101, 105),
                depth_frame(200, 205),  # gap -> resync, snapshot now returns 300
                depth_frame(301, 310),
                depth_frame(311, 315),
            ],
        )

        deltas = [(d.first_id, d.final_id) for d in events if isinstance(d, DepthDelta)]
        assert (101, 105) in deltas
        assert (311, 315) in deltas
        # Exactly two snapshots: the original and the recovery.
        assert len([e for e in events if isinstance(e, DepthSnapshot)]) == 2

    async def test_trades_pass_through_during_and_after_sync(self, monkeypatch):
        feed = make_feed([100], monkeypatch)
        events = await collect(
            feed,
            [
                trade_frame("64000.00", "0.1", True),
                depth_frame(101, 105),
                trade_frame("64000.50", "0.2", False),
            ],
        )

        trades = [e for e in events if isinstance(e, TradeTick)]
        assert len(trades) == 2
        assert [t.aggressor for t in trades] == [Side.SELL, Side.BUY]

    async def test_unknown_message_types_are_ignored(self, monkeypatch):
        feed = make_feed([100], monkeypatch)
        events = await collect(
            feed,
            [
                {"stream": "x", "data": {"e": "kline", "k": {}}},
                depth_frame(101, 105),
            ],
        )

        assert len([e for e in events if isinstance(e, DepthDelta)]) == 1
