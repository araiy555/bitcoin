"""MarketView: feed ingestion, derived signals, and the synthetic/replay feeds."""

import math
from decimal import Decimal

import pytest

from jsboard.core.market import FlowImbalance, MarketView, RollingVol
from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthDelta, DepthSnapshot, FeedStatus, TradeTick
from jsboard.feed.replay import JsonlRecorder, ReplayFeed, SyntheticFeed

INST = Instrument("TEST", tick_size=Decimal("1"), lot_size=Decimal("1"), base="X", quote="USD")
BTC = Instrument("BTCUSDT", tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"))


@pytest.fixture
def market():
    m = MarketView(instrument=INST)
    m.apply(FeedStatus("live"))
    m.apply(
        DepthSnapshot(
            bids=((100, 30), (99, 40)), asks=((102, 10), (103, 20)), last_update_id=1
        )
    )
    return m


class TestIngestion:
    def test_snapshot_builds_the_book(self, market):
        assert market.book.best_bid() == 100
        assert market.book.best_ask() == 102
        assert market.is_live

    def test_delta_updates_a_level(self, market):
        market.apply(DepthDelta(bids=((100, 55),), asks=(), first_id=2, final_id=2))

        assert market.book.depth_at(Side.BUY, 100) == 55

    def test_delta_with_zero_removes_a_level(self, market):
        market.apply(DepthDelta(bids=((100, 0),), asks=(), first_id=2, final_id=2))

        assert market.book.best_bid() == 99

    def test_snapshot_replaces_rather_than_merges(self, market):
        market.apply(
            DepthSnapshot(bids=((50, 5),), asks=((60, 5),), last_update_id=9)
        )

        assert market.book.best_bid() == 50
        assert market.book.depth_at(Side.BUY, 99) == 0

    def test_trades_land_on_the_tape(self, market):
        market.apply(TradeTick(price=101, qty=5, aggressor=Side.BUY))

        assert len(market.tape) == 1
        assert market.recent_trades()[0].price == 101

    def test_resync_status_is_counted(self, market):
        market.apply(FeedStatus("resyncing", "gap"))

        assert market.resync_count == 1
        assert not market.is_live


class TestSignals:
    def test_microprice_leans_toward_the_thin_side(self, market):
        # Bid 30 vs ask 10 -> fair value sits above the 101 mid.
        assert market.mid == 101.0
        assert market.microprice > 101.0

    def test_imbalance_reflects_depth(self, market):
        assert market.imbalance(2) == pytest.approx((70 - 30) / 100)

    def test_vwap_weights_by_size(self, market):
        market.apply(TradeTick(price=100, qty=1, aggressor=Side.BUY))
        market.apply(TradeTick(price=110, qty=9, aggressor=Side.BUY))

        assert market.vwap() == pytest.approx(109.0)

    def test_vwap_is_none_without_prints(self, market):
        assert market.vwap() is None

    def test_mid_price_is_in_human_units(self):
        m = MarketView(instrument=BTC)
        m.apply(DepthSnapshot(bids=((6_400_000, 1),), asks=((6_400_002, 1),), last_update_id=1))

        assert m.mid_price == pytest.approx(64000.01)


class TestRollingVol:
    def test_flat_prices_give_zero_vol(self):
        v = RollingVol(halflife=5)
        for _ in range(20):
            v.update(100.0)

        assert v.sigma == pytest.approx(0.0)

    def test_larger_moves_give_larger_vol(self):
        calm, wild = RollingVol(halflife=5), RollingVol(halflife=5)
        for i in range(50):
            calm.update(100.0 + (i % 2) * 0.1)
            wild.update(100.0 + (i % 2) * 5.0)

        assert wild.sigma > calm.sigma

    def test_bps_is_sigma_scaled(self):
        v = RollingVol()
        v.update(100.0)
        v.update(101.0)

        assert v.bps == pytest.approx(v.sigma * 10_000)

    def test_non_positive_prices_are_ignored(self):
        v = RollingVol()
        v.update(100.0)
        v.update(0.0)
        v.update(-5.0)

        assert math.isfinite(v.sigma)


class TestFlowImbalance:
    def test_all_buys_reads_positive_one(self):
        f = FlowImbalance()
        for _ in range(10):
            f.update(Side.BUY, 1.0)

        assert f.value == pytest.approx(1.0)

    def test_balanced_flow_reads_zero(self):
        f = FlowImbalance(halflife=1e9)  # effectively no decay
        for _ in range(10):
            f.update(Side.BUY, 1.0)
            f.update(Side.SELL, 1.0)

        assert f.value == pytest.approx(0.0, abs=1e-6)

    def test_empty_flow_is_zero(self):
        assert FlowImbalance().value == 0.0

    def test_recent_flow_outweighs_old(self):
        f = FlowImbalance(halflife=2)
        for _ in range(20):
            f.update(Side.BUY, 1.0)
        for _ in range(5):
            f.update(Side.SELL, 1.0)

        assert f.value < 0


class TestSyntheticFeed:
    async def test_it_emits_a_snapshot_then_deltas(self):
        feed = SyntheticFeed(BTC, seed=1, tick_interval=0.0, max_events=20)
        events = [e async for e in feed.stream()]

        kinds = [type(e) for e in events]
        assert DepthSnapshot in kinds
        assert DepthDelta in kinds

    async def test_it_produces_prints(self):
        feed = SyntheticFeed(BTC, seed=1, tick_interval=0.0, max_events=200)
        events = [e async for e in feed.stream()]

        assert any(isinstance(e, TradeTick) for e in events)

    async def test_the_same_seed_gives_the_same_market(self):
        async def prices(seed):
            feed = SyntheticFeed(BTC, seed=seed, tick_interval=0.0, max_events=50)
            return [e.price for e in [x async for x in feed.stream()] if isinstance(e, TradeTick)]

        assert await prices(3) == await prices(3)
        assert await prices(3) != await prices(4)

    async def test_virtual_time_advances_without_sleeping(self):
        feed = SyntheticFeed(BTC, seed=1, tick_interval=0.0, max_events=10)
        stamps = [
            e.ts_ns
            for e in [x async for x in feed.stream()]
            if isinstance(e, (DepthDelta, DepthSnapshot))
        ]

        assert stamps == sorted(stamps)
        assert stamps[-1] > stamps[0]

    async def test_the_book_stays_two_sided(self):
        feed = SyntheticFeed(BTC, seed=5, tick_interval=0.0, max_events=100)
        market = MarketView(instrument=BTC)
        async for event in feed.stream():
            market.apply(event)
            if market.book.best_bid() is not None and market.book.best_ask() is not None:
                assert market.book.best_bid() < market.book.best_ask()


class TestReplay:
    async def test_roundtrip_through_jsonl(self, tmp_path):
        path = tmp_path / "capture.jsonl"
        feed = SyntheticFeed(BTC, seed=2, tick_interval=0.0, max_events=40)

        original = []
        with JsonlRecorder(path) as rec:
            async for event in feed.stream():
                rec.write(event)
                original.append(event)

        replayed = [e async for e in ReplayFeed(BTC, path, speed=0).stream()]

        # Replay brackets the capture with its own connecting/disconnected status.
        assert [type(e) for e in replayed[1:-1]] == [type(e) for e in original]
        trades_in = [e for e in original if isinstance(e, TradeTick)]
        trades_out = [e for e in replayed if isinstance(e, TradeTick)]
        assert [(t.price, t.qty, t.aggressor) for t in trades_in] == [
            (t.price, t.qty, t.aggressor) for t in trades_out
        ]

    async def test_replay_drives_a_market_view(self, tmp_path):
        path = tmp_path / "capture.jsonl"
        feed = SyntheticFeed(BTC, seed=2, tick_interval=0.0, max_events=60)
        with JsonlRecorder(path) as rec:
            async for event in feed.stream():
                rec.write(event)

        market = MarketView(instrument=BTC)
        async for event in ReplayFeed(BTC, path, speed=0).stream():
            market.apply(event)

        assert market.events_seen > 0
        assert market.book.best_bid() is not None
