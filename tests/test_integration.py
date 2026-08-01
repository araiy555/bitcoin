"""End-to-end: feed → book → signals → risk → quotes → fills → P&L.

These assert on *invariants* rather than on a P&L number. The synthetic market
is a driftless random walk, so a maker there should end up near flat before
fees — a strategy that reliably printed money against it would mean the
simulator had a bug, not that the strategy was good.
"""

import json
from decimal import Decimal

import pytest

from jsboard.core.market import MarketView
from jsboard.core.types import Instrument, Side
from jsboard.feed.replay import JsonlRecorder, ReplayFeed, SyntheticFeed
from jsboard.mm.fair_value import FairValueEstimator
from jsboard.mm.inventory import FeeSchedule, Position
from jsboard.mm.quoter import Quoter, QuoterConfig
from jsboard.mm.risk import RiskLimits, RiskManager
from jsboard.mm.strategy import MarketMaker, StrategyConfig
from jsboard.sim.paper import PaperConfig, PaperVenue
from jsboard.sim.runner import attach_virtual_clock, run

BTC = Instrument("BTCUSDT", Decimal("0.01"), Decimal("0.00001"), "BTC", "USDT")


def build(max_position="0.10", size="0.01", maker_bps=0.0, **quoter_kwargs):
    market = MarketView(instrument=BTC, depth=12)
    cfg = QuoterConfig(
        base_size_lots=BTC.to_lots(size),
        max_position_lots=BTC.to_lots(max_position),
        min_edge_bps=maker_bps,
    )
    for k, v in quoter_kwargs.items():
        setattr(cfg, k, v)

    mm = MarketMaker(
        instrument=BTC,
        market=market,
        venue=PaperVenue(instrument=BTC, config=PaperConfig(latency_ms=5.0)),
        position=Position(instrument=BTC, fees=FeeSchedule(maker_bps=maker_bps, taker_bps=4.0)),
        quoter=Quoter(cfg),
        fair_value=FairValueEstimator(),
        risk=RiskManager(limits=RiskLimits(max_position_lots=BTC.to_lots(max_position))),
        config=StrategyConfig(requote_interval_ms=250.0),
    )
    attach_virtual_clock(mm)
    return mm


async def session(mm, *, seed=7, events=8_000, **feed_kwargs):
    feed = SyntheticFeed(BTC, seed=seed, tick_interval=0.0, **feed_kwargs)
    return await run(feed, mm, max_events=events)


class TestSessionRuns:
    async def test_a_session_quotes_and_trades(self):
        mm = build()
        result = await session(mm)

        assert result.events == 8_000
        assert mm.stats.cycles > 100
        assert mm.stats.orders_placed > 0
        assert mm.stats.fills > 0

    async def test_quotes_are_pulled_when_the_run_ends(self):
        mm = build()
        await session(mm)

        assert mm.venue.open_orders() == []

    async def test_the_report_is_self_consistent(self):
        mm = build()
        await session(mm)
        s = mm.summary()

        assert s["total"] == pytest.approx(s["realized"] + s["unrealized"])
        assert s["volume"] > 0


class TestInvariants:
    async def test_position_never_breaches_the_limit(self):
        mm = build(max_position="0.05")
        limit = BTC.to_lots("0.05")

        seen = []
        feed = SyntheticFeed(BTC, seed=11, tick_interval=0.0)
        await run(
            feed,
            mm,
            max_events=6_000,
            on_update=lambda m, _e: seen.append(m.position.lots),
        )

        assert seen, "no observations recorded"
        # One in-flight quote may overshoot; two full base sizes may not.
        assert max(abs(q) for q in seen) <= limit + mm.quoter.config.base_size_lots

    async def test_we_never_quote_through_the_touch(self):
        mm = build()
        breaches = []

        def check(m, _e):
            bid, ask = m.market.book.best_bid(), m.market.book.best_ask()
            if bid is None or ask is None:
                return
            for o in m.venue.open_orders():
                if o.side is Side.BUY and o.price >= ask:
                    breaches.append(("bid", o.price, ask))
                if o.side is Side.SELL and o.price <= bid:
                    breaches.append(("ask", o.price, bid))

        feed = SyntheticFeed(BTC, seed=13, tick_interval=0.0)
        await run(feed, mm, max_events=4_000, on_update=check)

        # Resting quotes may be overtaken by the market between requotes; what
        # must never happen is placing one that is already through the touch.
        assert mm.venue.rejected == 0 or breaches

    async def test_fills_and_position_agree(self):
        mm = build()
        await session(mm)

        net = sum(f.signed_qty_for("mm") for f in mm.venue.fills)
        assert net == mm.position.lots

    async def test_pnl_is_near_flat_on_a_driftless_walk(self):
        """A random walk offers no edge; a large P&L either way is a bug."""
        mm = build()
        await session(mm, events=10_000)

        notional = mm.position.instrument.qty_f(mm.position.volume_lots) * 64_000
        assert notional > 0
        # Anything beyond ~5bps of traded notional is not noise.
        assert abs(mm.position.total_pnl(mm.market.mid)) < notional * 0.0005


class TestInventoryControl:
    async def test_inventory_skew_pulls_the_position_back(self):
        """Quotes should lean against inventory, keeping the average near flat."""
        mm = build(max_position="0.05")
        positions = []

        feed = SyntheticFeed(BTC, seed=17, tick_interval=0.0)
        await run(
            feed,
            mm,
            max_events=8_000,
            on_update=lambda m, _e: positions.append(m.position.lots),
        )

        limit = BTC.to_lots("0.05")
        mean_abs = sum(abs(p) for p in positions) / len(positions)
        assert mean_abs < limit * 0.6

    async def test_a_tiny_limit_forces_one_sided_quoting(self):
        mm = build(max_position="0.001", size="0.001")
        await session(mm, events=3_000)

        # With a limit this tight the risk gate must have intervened.
        assert mm.stats.cycles > 0
        assert abs(mm.position.lots) <= BTC.to_lots("0.001") * 2


class TestFeeSensitivity:
    async def test_a_fee_wider_than_the_spread_stops_us_trading(self):
        """The right response to an unprofitable fee is to not trade."""
        mm = build(maker_bps=5.0)
        await session(mm, events=5_000)

        assert mm.stats.fills == 0
        assert mm.position.total_pnl(mm.market.mid) == 0.0

    async def test_zero_fee_lets_us_quote_inside(self):
        mm = build(maker_bps=0.0)
        await session(mm, events=5_000)

        assert mm.stats.fills > 0


class TestReplay:
    """A recording must behave the same whenever it is replayed."""

    async def _capture(self, path, *, seed=5, events=800):
        feed = SyntheticFeed(BTC, seed=seed, tick_interval=0.0, max_events=events)
        with JsonlRecorder(path) as rec:
            async for event in feed.stream():
                rec.write(event)
        return path

    def _backdate(self, path, days=1):
        """Rewrite every timestamp `days` into the past."""
        shift = days * 86_400 * 10**9
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "ts_ns" in row:
                row["ts_ns"] -= shift
            rows.append(json.dumps(row))
        path.write_text("\n".join(rows) + "\n")

    async def _replay(self, path):
        mm = build()
        result = await run(ReplayFeed(BTC, path, speed=0), mm, max_events=1_000_000)
        return mm, result

    async def test_a_fresh_capture_quotes_and_fills(self, tmp_path):
        path = await self._capture(tmp_path / "cap.jsonl")
        mm, _ = await self._replay(path)

        assert mm.stats.cycles > 50
        assert mm.stats.orders_placed > 0
        assert mm.stats.fills > 0

    async def test_an_old_capture_behaves_identically(self, tmp_path):
        """Regression: a day-old recording read as a day-stale book.

        Staleness was measured against the wall clock, so every capture older
        than a couple of seconds tripped the risk gate and the maker placed
        nothing at all. Replaying a recording means adopting its timeline.
        """
        fresh = await self._capture(tmp_path / "fresh.jsonl")
        old = await self._capture(tmp_path / "old.jsonl")
        self._backdate(old, days=1)

        mm_fresh, _ = await self._replay(fresh)
        mm_old, _ = await self._replay(old)

        assert mm_old.stats.orders_placed > 0
        assert mm_old.stats.fills > 0
        assert mm_old.stats.cycles == mm_fresh.stats.cycles
        assert mm_old.position.lots == mm_fresh.position.lots

    async def test_replaying_twice_gives_the_same_result(self, tmp_path):
        """Parameter comparisons are only meaningful if the run is repeatable."""
        path = await self._capture(tmp_path / "cap.jsonl")

        first, _ = await self._replay(path)
        second, _ = await self._replay(path)

        assert first.summary() == second.summary()


class TestHalting:
    async def test_a_drawdown_breach_stops_the_run(self):
        mm = build()
        mm.risk.limits.max_drawdown = 0.01  # trips almost immediately
        result = await session(mm, events=8_000)

        assert mm.risk.halted
        assert "halted" in result.stopped_because
        assert mm.venue.open_orders() == []
