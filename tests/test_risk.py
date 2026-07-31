"""Risk gate: staleness, dislocation, position limits, and the latched halt."""

from decimal import Decimal

import pytest

from jsboard.core.market import MarketView
from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, FeedStatus
from jsboard.mm.inventory import FeeSchedule, Position
from jsboard.mm.risk import RiskAction, RiskLimits, RiskManager

INST = Instrument("TEST", tick_size=Decimal("1"), lot_size=Decimal("1"), base="X", quote="USD")


class FakeClock:
    def __init__(self, now_ns=1_000_000_000):
        self.now = now_ns

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += int(ms * 1e6)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def market(clock):
    m = MarketView(instrument=INST, clock=clock)
    m.apply(FeedStatus("live"))
    m.apply(
        DepthSnapshot(
            bids=((100, 50), (99, 50), (98, 50)),
            asks=((102, 50), (103, 50), (104, 50)),
            last_update_id=1,
            ts_ns=clock(),
        )
    )
    return m


@pytest.fixture
def position():
    return Position(instrument=INST, fees=FeeSchedule(maker_bps=0.0))


@pytest.fixture
def risk():
    return RiskManager(
        limits=RiskLimits(
            max_position_lots=100,
            max_notional=1_000_000.0,
            max_drawdown=500.0,
            max_book_age_ms=1_000.0,
            max_spread_ticks=10,
            max_sigma_bps=100.0,
        )
    )


def test_healthy_market_allows_quoting(risk, market, position):
    decision = risk.evaluate(market, position)

    assert decision.action is RiskAction.QUOTE
    assert decision.permits(Side.BUY) and decision.permits(Side.SELL)


class TestMarketDataHealth:
    def test_stale_book_pulls_quotes(self, risk, market, position, clock):
        clock.advance_ms(1_500)

        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.PULL
        assert "stale" in decision.reason
        assert not decision.can_quote

    def test_disconnected_feed_pulls_quotes(self, risk, market, position):
        market.apply(FeedStatus("disconnected", "socket closed"))

        assert risk.evaluate(market, position).action is RiskAction.PULL

    def test_wide_spread_pulls_quotes(self, risk, market, position, clock):
        market.apply(
            DepthSnapshot(
                bids=((100, 50), (99, 50)),
                asks=((200, 50), (201, 50)),
                last_update_id=2,
                ts_ns=clock(),
            )
        )

        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.PULL
        assert "dislocated" in decision.reason

    def test_thin_book_pulls_quotes(self, risk, market, position, clock):
        market.apply(
            DepthSnapshot(bids=((100, 50),), asks=(), last_update_id=3, ts_ns=clock())
        )

        assert risk.evaluate(market, position).action is RiskAction.PULL

    def test_volatility_spike_pulls_quotes(self, risk, market, position, clock):
        risk.limits.max_sigma_bps = 0.001
        for price in (100, 140, 90, 160):
            market.apply(
                DepthSnapshot(
                    bids=((price, 50), (price - 1, 50)),
                    asks=((price + 2, 50), (price + 3, 50)),
                    last_update_id=price,
                    ts_ns=clock(),
                )
            )
            market.vol.update(float(price))

        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.PULL
        assert "volatility" in decision.reason


class TestPositionLimits:
    def test_long_at_the_limit_quotes_the_sell_side_only(self, risk, market, position):
        position.on_fill(100, 100, Side.BUY, is_maker=True)

        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.ONE_SIDED
        assert decision.permits(Side.SELL)
        assert not decision.permits(Side.BUY)

    def test_short_at_the_limit_quotes_the_buy_side_only(self, risk, market, position):
        position.on_fill(100, 100, Side.SELL, is_maker=True)

        decision = risk.evaluate(market, position)

        assert decision.permits(Side.BUY)
        assert not decision.permits(Side.SELL)

    def test_notional_limit_also_restricts_to_one_side(self, risk, market, position):
        risk.limits.max_notional = 500.0
        position.on_fill(100, 10, Side.BUY, is_maker=True)

        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.ONE_SIDED
        assert "notional" in decision.reason

    def test_within_limits_stays_two_sided(self, risk, market, position):
        position.on_fill(100, 50, Side.BUY, is_maker=True)

        assert risk.evaluate(market, position).action is RiskAction.QUOTE


class TestKillSwitch:
    def test_drawdown_breach_halts(self, risk, market, position):
        position.on_fill(1000, 10, Side.BUY, is_maker=True)  # mark is 101 -> big loss

        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.HALT
        assert risk.halted

    def test_halt_is_latched_even_once_healthy(self, risk, market, position):
        position.on_fill(1000, 10, Side.BUY, is_maker=True)
        risk.evaluate(market, position)

        # Flatten the position entirely; the halt must survive.
        position.lots = 0
        position.realized_pnl = 0.0
        decision = risk.evaluate(market, position)

        assert decision.action is RiskAction.HALT

    def test_reset_clears_the_halt(self, risk, market, position):
        position.on_fill(1000, 10, Side.BUY, is_maker=True)
        risk.evaluate(market, position)
        risk.reset()

        position.lots = 0
        position.realized_pnl = 0.0
        position.avg_price_ticks = 0.0

        assert risk.evaluate(market, position).action is RiskAction.QUOTE

    def test_drawdown_measures_from_the_peak_not_from_zero(self, risk, market, position):
        # Earn 400, then give back 450: equity is -50 but drawdown is 850.
        position.realized_pnl = 400.0
        risk.evaluate(market, position)
        position.realized_pnl = -450.0

        assert risk.evaluate(market, position).action is RiskAction.HALT
