"""First-pass screen: fee, tick width, spread.

The screen exists to reject, so the cases that matter are the ones it must
kill — above all the shape that cost this project days: a spread that is one
tick wide, on a tick worth several basis points, against a fee that eats it.
"""

import math

import pytest

from jsboard.research.scan import parse_tick_sizes
from jsboard.research.triage import build, rank, tally


def row(bid, ask, tick, symbol="XUSDT", volume=1e9):
    return build(symbol, bid=bid, ask=ask, tick_size=tick, quote_volume=volume)


class TestTickWidth:
    def test_a_coarse_tick_is_worth_many_bps(self):
        # WIFUSDT as it actually was: 0.0001 on a 0.1356 price.
        r = row(0.13555, 0.13565, 0.0001)
        assert r.tick_bps == pytest.approx(7.37, abs=0.05)

    def test_a_fine_tick_is_worth_almost_nothing(self):
        # BTCUSDT: 0.01 on 64,000.
        r = row(63_999.99, 64_000.01, 0.01)
        assert r.tick_bps < 0.01

    def test_no_tick_is_an_infinite_width_rather_than_a_crash(self):
        assert row(100, 101, 0).tick_bps == math.inf


class TestSpreadInTicks:
    def test_a_one_tick_spread_reads_as_one_tick(self):
        r = row(0.13555, 0.13565, 0.0001)
        assert r.spread_ticks == pytest.approx(1.0, abs=0.01)

    def test_a_wide_spread_reads_as_several(self):
        r = row(99.97, 100.03, 0.01)
        assert r.spread_ticks == pytest.approx(6.0, abs=0.01)


class TestVerdict:
    def test_the_case_that_cost_this_project_days(self):
        # 7.37 bps of spread against a 20 bps round trip. Not close.
        r = row(0.13555, 0.13565, 0.0001)
        assert r.headroom_bps(10.0) < 0
        assert r.verdict(10.0) == "手数料負け"

    def test_the_same_symbol_still_fails_on_tick_at_a_free_fee(self):
        # Even with no fee at all, a one-tick spread leaves no room to move.
        r = row(0.13555, 0.13565, 0.0001)
        assert r.headroom_bps(0.0) > 0
        assert r.verdict(0.0) == "tickが粗い"

    def test_a_wide_spread_on_a_fine_tick_survives(self):
        r = row(99.7, 100.3, 0.01)
        assert r.verdict(10.0) == "候補"

    def test_a_crossed_or_empty_book_produces_no_row(self):
        assert row(0, 100, 0.01) is None
        assert row(101, 100, 0.01) is None

    def test_headroom_is_the_spread_less_both_legs(self):
        r = row(99.5, 100.5, 0.01)
        assert r.headroom_bps(10.0) == pytest.approx(r.spread_bps - 20.0)


class TestRankAndTally:
    def market(self):
        return [
            row(0.13555, 0.13565, 0.0001, "WIFUSDT"),      # 手数料負け
            row(63_999.99, 64_000.01, 0.01, "BTCUSDT"),    # 手数料負け
            row(99.7, 100.3, 0.01, "MIDUSDT"),             # 候補, 60 bps
            row(9.95, 10.05, 0.001, "WIDEUSDT"),           # 候補, 100 bps
            # Clears the fee (30 bps of spread) but the spread is 1.5 ticks:
            # the gate the fee alone would have let through.
            row(99.85, 100.15, 0.2, "COARSEUSDT"),
        ]

    def test_survivors_come_back_widest_first(self):
        got = rank(self.market(), 10.0)
        assert [r.symbol for r in got] == ["WIDEUSDT", "MIDUSDT"]

    def test_the_tally_says_where_each_symbol_died(self):
        counts = tally(self.market(), 10.0)
        assert counts["候補"] == 2
        assert counts["手数料負け"] == 2
        assert counts["tickが粗い"] == 1
        assert sum(counts.values()) == 5

    def test_a_harsher_tick_requirement_kills_more(self):
        assert tally(self.market(), 10.0, min_ticks=1000.0)["候補"] == 0


class TestParseTickSizes:
    def test_one_call_yields_every_symbol(self):
        payload = {
            "symbols": [
                {"symbol": "AUSDT", "filters": [{"filterType": "PRICE_FILTER",
                                                 "tickSize": "0.001"}]},
                {"symbol": "BUSDT", "filters": [
                    {"filterType": "LOT_SIZE", "stepSize": "1"},
                    {"filterType": "PRICE_FILTER", "tickSize": "0.5"},
                ]},
            ]
        }
        assert parse_tick_sizes(payload) == {"AUSDT": "0.001", "BUSDT": "0.5"}

    def test_a_symbol_with_no_price_filter_is_left_out(self):
        payload = {"symbols": [{"symbol": "CUSDT", "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "1"}
        ]}]}
        assert parse_tick_sizes(payload) == {}

    def test_an_empty_payload_is_not_an_error(self):
        assert parse_tick_sizes({}) == {}
