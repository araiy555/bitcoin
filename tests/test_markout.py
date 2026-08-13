"""Mark-out accounting: sign, horizons, weighting, and what stays pending.

The reference matters more than anything else here. Adverse selection is
measured from the mid at the fill; measuring from our own price instead folds
the spread we earned into it, which on a wide-tick symbol is enough to turn a
loss into an apparent gain.
"""

from __future__ import annotations

import math

import pytest

from jsboard.mm.markout import NS_PER_S, MarkOutTracker


def tracker(*horizons: float) -> MarkOutTracker:
    return MarkOutTracker(horizons_s=horizons or (1.0,))


def fill(t, now, sign, price, weight=1.0, mid=None):
    t.on_fill(now, sign, price, weight, mid_ticks=price if mid is None else mid)


class TestReference:
    """The two references, and the gap between them."""

    def test_adverse_selection_ignores_the_spread_we_earned(self):
        t = tracker(1.0)
        # Bought a tick below a mid of 100, and the mid did not move.
        fill(t, 0, +1, 99.0, mid=100.0)
        t.poll(NS_PER_S, 100.0)
        assert t.windows[0].mean_bps == pytest.approx(0.0)

    def test_measured_from_our_price_the_same_fill_looks_profitable(self):
        t = tracker(1.0)
        fill(t, 0, +1, 99.0, mid=100.0)
        t.poll(NS_PER_S, 100.0)
        assert t.windows[0].mean_vs_fill_bps == pytest.approx(101.01, rel=1e-3)

    def test_a_fill_with_no_mid_is_not_recorded_against_a_guess(self):
        t = tracker(1.0)
        t.on_fill(0, +1, 99.0, 1.0, mid_ticks=None)
        t.poll(NS_PER_S, 100.0)
        assert t.windows[0].n == 0
        assert t.windows[0].unsettled == 0


class TestSign:
    def test_buy_then_price_up_is_favourable(self):
        t = tracker(1.0)
        fill(t, 0, +1, 100.0)
        t.poll(NS_PER_S, 100.5)
        assert t.windows[0].mean_bps == pytest.approx(50.0)

    def test_sell_then_price_up_is_adverse(self):
        t = tracker(1.0)
        fill(t, 0, -1, 100.0)
        t.poll(NS_PER_S, 100.5)
        assert t.windows[0].mean_bps == pytest.approx(-50.0)


class TestHorizons:
    def test_nothing_settles_before_the_horizon(self):
        t = tracker(10.0)
        fill(t, 0, +1, 100.0)
        t.poll(9 * NS_PER_S, 200.0)
        assert math.isnan(t.windows[0].mean_bps)
        assert t.windows[0].unsettled == 1
        assert t.windows[0].n == 0

    def test_each_horizon_settles_at_its_own_time(self):
        t = tracker(1.0, 10.0)
        fill(t, 0, +1, 100.0)
        t.poll(NS_PER_S, 101.0)
        t.poll(10 * NS_PER_S, 99.0)
        assert t.windows[0].mean_bps == pytest.approx(100.0)
        assert t.windows[1].mean_bps == pytest.approx(-100.0)

    def test_pending_settles_in_order(self):
        t = tracker(1.0)
        fill(t, 0, +1, 100.0)
        fill(t, 5 * NS_PER_S, +1, 100.0)
        t.poll(NS_PER_S, 101.0)
        assert t.windows[0].n == 1
        assert t.windows[0].unsettled == 1
        t.poll(6 * NS_PER_S, 99.0)
        assert t.windows[0].n == 2
        assert t.windows[0].mean_bps == pytest.approx(0.0)

    def test_defaults_start_at_the_shortest_the_data_resolves(self):
        # 100ms is the depth stream's own update interval: adverse selection
        # faster than that is invisible here, and bounding it matters.
        assert [w.horizon_s for w in MarkOutTracker().windows] == [0.1, 1.0, 10.0, 60.0]


class TestWeighting:
    def test_weighted_by_size_not_by_count(self):
        t = tracker(1.0)
        fill(t, 0, +1, 100.0, weight=9.0)
        fill(t, 0, +1, 100.0, weight=1.0)
        t.poll(NS_PER_S, 101.0)
        assert t.windows[0].weight == pytest.approx(10.0)
        assert t.windows[0].n == 2
        assert t.windows[0].mean_bps == pytest.approx(100.0)

    def test_big_fill_dominates_a_small_one(self):
        t = tracker(1.0)
        fill(t, 0, +1, 100.0, weight=99.0)
        t.poll(NS_PER_S, 101.0)
        fill(t, 2 * NS_PER_S, -1, 100.0, weight=1.0)
        t.poll(3 * NS_PER_S, 101.0)
        assert t.windows[0].mean_bps == pytest.approx((99 * 100.0 - 100.0) / 100.0)


class TestGuards:
    def test_a_missing_mid_at_settlement_defers_rather_than_scoring(self):
        t = tracker(1.0)
        fill(t, 0, +1, 100.0)
        t.poll(NS_PER_S, None)
        assert t.windows[0].unsettled == 1
        t.poll(2 * NS_PER_S, 101.0)
        assert t.windows[0].n == 1

    def test_nonsense_fills_are_ignored(self):
        t = tracker(1.0)
        fill(t, 0, +1, 100.0, weight=0.0)
        fill(t, 0, +1, 0.0, weight=1.0)
        fill(t, 0, 0, 100.0, weight=1.0)
        t.poll(NS_PER_S, 101.0)
        assert t.windows[0].n == 0
        assert t.windows[0].unsettled == 0


class TestSummary:
    def test_summary_reports_every_horizon_and_both_references(self):
        t = tracker(1.0, 10.0, 60.0)
        fill(t, 0, +1, 99.0, mid=100.0)
        t.poll(NS_PER_S, 101.0)
        rows = t.summary()
        assert [r["horizon_s"] for r in rows] == [1.0, 10.0, 60.0]
        assert rows[0]["n"] == 1.0
        assert rows[0]["mean_bps"] == pytest.approx(100.0)
        assert rows[0]["vs_fill_bps"] > rows[0]["mean_bps"]
        assert rows[1]["unsettled"] == 1.0
        assert math.isnan(rows[1]["mean_bps"])
