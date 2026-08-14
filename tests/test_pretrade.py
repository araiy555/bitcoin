"""Shared quality, execution and all-cost gates from CMR-001."""

from decimal import Decimal

import pytest

from jsboard.core.pretrade import (
    CostBreakdown,
    MarketDataPoint,
    RejectCode,
    assess_costs,
    assess_market_data,
    walk_book,
)
from jsboard.core.types import BookSnapshot, Instrument, Level, Side

INST = Instrument("PAIR", Decimal("0.5"), Decimal("1"), "P", "U")


def snapshot():
    return BookSnapshot(
        bids=(Level(200, 5), Level(198, 10)),
        asks=(Level(202, 4), Level(204, 10)),
        ts_ns=1_000_000_000,
    )


class TestBookWalk:
    def test_buy_walks_asks_and_reports_depth_slippage(self):
        result = walk_book(snapshot(), INST, Side.BUY, 10)
        assert result.complete
        assert result.avg_price_ticks == pytest.approx((202 * 4 + 204 * 6) / 10)
        assert result.price(INST) == pytest.approx(101.6)
        assert result.notional_quote == pytest.approx(1_016)
        assert result.slippage_bps > 0

    def test_sell_walks_bids(self):
        result = walk_book(snapshot(), INST, Side.SELL, 10)
        assert result.complete
        assert result.avg_price_ticks == pytest.approx((200 * 5 + 198 * 5) / 10)

    def test_missing_depth_is_not_filled_at_an_invented_price(self):
        result = walk_book(snapshot(), INST, Side.BUY, 20)
        assert not result.complete
        assert result.filled_lots == 14
        assert result.remaining_lots == 6

    def test_configured_depth_is_a_hard_limit(self):
        result = walk_book(snapshot(), INST, Side.SELL, 10, max_levels=1)
        assert not result.complete
        assert result.filled_lots == 5


class TestMarketDataQuality:
    def test_two_fresh_synchronised_books_pass(self):
        decision = assess_market_data(
            (
                MarketDataPoint("binance", 1_000_000_000),
                MarketDataPoint("bybit", 1_050_000_000),
            ),
            now_ns=1_100_000_000,
            max_age_ms=250,
            max_skew_ms=100,
        )
        assert decision.allowed
        assert decision.skew_ms == pytest.approx(50)

    def test_all_data_failures_are_preserved(self):
        decision = assess_market_data(
            (
                MarketDataPoint(
                    "binance", 0, book_valid=False, feed_state="disconnected",
                    metadata_valid=False,
                ),
                MarketDataPoint("bybit", 2_000_000_000),
            ),
            now_ns=2_000_000_000,
            max_age_ms=250,
            max_skew_ms=100,
        )
        assert not decision.allowed
        assert set(decision.reject_codes) == {
            RejectCode.METADATA_INVALID,
            RejectCode.FEED_DEGRADED,
            RejectCode.BOOK_GAP,
            RejectCode.STALE,
        }

    def test_receive_time_skew_is_rejected(self):
        decision = assess_market_data(
            (
                MarketDataPoint("binance", 1_000_000_000),
                MarketDataPoint("bybit", 1_500_000_000),
            ),
            now_ns=1_500_000_000,
            max_age_ms=1_000,
            max_skew_ms=250,
        )
        assert decision.reject_codes == (RejectCode.SKEW,)


class TestAllCostDecision:
    def test_every_cost_component_is_deducted(self):
        costs = CostBreakdown(
            entry_fees_bps=4,
            expected_exit_fees_bps=4,
            entry_depth_slippage_bps=1,
            expected_exit_slippage_bps=1,
            funding_bps=-0.5,
            borrow_bps=0.5,
            hedge_latency_buffer_bps=1,
            fill_model_buffer_bps=2,
            safety_margin_bps=1,
        )
        assert costs.total_bps == pytest.approx(14)
        decision = assess_costs(20, costs, min_expected_net_bps=1)
        assert decision.accepted
        assert decision.expected_net_bps == pytest.approx(6)

    def test_positive_gross_is_rejected_when_net_is_too_small(self):
        decision = assess_costs(
            10,
            CostBreakdown(entry_fees_bps=6, expected_exit_fees_bps=6),
            min_expected_net_bps=1,
        )
        assert not decision.accepted
        assert RejectCode.NET_NEGATIVE in decision.reject_codes

    def test_stress_case_must_pass_independently(self):
        base = CostBreakdown(entry_fees_bps=2)
        stress = CostBreakdown(entry_fees_bps=2, fill_model_buffer_bps=5)
        decision = assess_costs(6, base, stress_costs=stress, min_expected_net_bps=1)
        assert decision.expected_net_bps == pytest.approx(4)
        assert decision.stress_net_bps == pytest.approx(-1)
        assert decision.reject_codes == (RejectCode.STRESS_NEGATIVE,)

    def test_non_funding_costs_cannot_be_negative(self):
        with pytest.raises(ValueError, match="entry_fees_bps"):
            CostBreakdown(entry_fees_bps=-1)
