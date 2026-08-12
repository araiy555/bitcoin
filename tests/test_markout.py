"""Mark-out accounting: sign, horizons, weighting, and what stays pending."""

from __future__ import annotations

import math

import pytest

from jsboard.mm.markout import NS_PER_S, MarkOutTracker


def tracker(*horizons: float) -> MarkOutTracker:
    return MarkOutTracker(horizons_s=horizons or (1.0,))


def test_buy_then_price_up_is_favourable():
    t = tracker(1.0)
    t.on_fill(0, +1, 100.0, 1.0)
    t.poll(NS_PER_S, 100.5)
    assert t.windows[0].mean_bps == pytest.approx(50.0)


def test_sell_then_price_up_is_adverse():
    t = tracker(1.0)
    t.on_fill(0, -1, 100.0, 1.0)
    t.poll(NS_PER_S, 100.5)
    assert t.windows[0].mean_bps == pytest.approx(-50.0)


def test_nothing_settles_before_the_horizon():
    t = tracker(10.0)
    t.on_fill(0, +1, 100.0, 1.0)
    t.poll(9 * NS_PER_S, 200.0)
    assert math.isnan(t.windows[0].mean_bps)
    assert t.windows[0].unsettled == 1
    assert t.windows[0].n == 0


def test_each_horizon_settles_at_its_own_time():
    t = tracker(1.0, 10.0)
    t.on_fill(0, +1, 100.0, 1.0)
    t.poll(NS_PER_S, 101.0)  # +100 bps at 1s
    t.poll(10 * NS_PER_S, 99.0)  # -100 bps at 10s
    assert t.windows[0].mean_bps == pytest.approx(100.0)
    assert t.windows[1].mean_bps == pytest.approx(-100.0)


def test_weighted_by_size_not_by_count():
    t = tracker(1.0)
    t.on_fill(0, +1, 100.0, weight=9.0)  # +100 bps on nine lots
    t.on_fill(0, +1, 100.0, weight=1.0)  # ... but this one settles elsewhere
    t.poll(NS_PER_S, 101.0)
    # Both settle at the same mid here, so the check is that weight carried:
    assert t.windows[0].weight == pytest.approx(10.0)
    assert t.windows[0].n == 2
    assert t.windows[0].mean_bps == pytest.approx(100.0)


def test_big_fill_dominates_a_small_one():
    t = tracker(1.0)
    t.on_fill(0, +1, 100.0, weight=99.0)
    t.poll(NS_PER_S, 101.0)  # +100 bps, 99 lots
    t.on_fill(2 * NS_PER_S, -1, 100.0, weight=1.0)
    t.poll(3 * NS_PER_S, 101.0)  # -100 bps, 1 lot
    assert t.windows[0].mean_bps == pytest.approx((99 * 100.0 - 100.0) / 100.0)


def test_a_missing_mid_defers_rather_than_scoring():
    t = tracker(1.0)
    t.on_fill(0, +1, 100.0, 1.0)
    t.poll(NS_PER_S, None)
    assert t.windows[0].unsettled == 1
    t.poll(2 * NS_PER_S, 101.0)
    assert t.windows[0].n == 1


def test_nonsense_fills_are_ignored():
    t = tracker(1.0)
    t.on_fill(0, +1, 100.0, weight=0.0)
    t.on_fill(0, +1, 0.0, weight=1.0)
    t.on_fill(0, 0, 100.0, weight=1.0)
    t.poll(NS_PER_S, 101.0)
    assert t.windows[0].n == 0
    assert t.windows[0].unsettled == 0


def test_pending_settles_in_order():
    t = tracker(1.0)
    t.on_fill(0, +1, 100.0, 1.0)
    t.on_fill(5 * NS_PER_S, +1, 100.0, 1.0)
    t.poll(NS_PER_S, 101.0)
    assert t.windows[0].n == 1
    assert t.windows[0].unsettled == 1
    t.poll(6 * NS_PER_S, 99.0)
    assert t.windows[0].n == 2
    assert t.windows[0].mean_bps == pytest.approx(0.0)


def test_summary_reports_every_horizon():
    t = tracker(1.0, 10.0, 60.0)
    t.on_fill(0, +1, 100.0, 1.0)
    t.poll(NS_PER_S, 101.0)
    rows = t.summary()
    assert [r["horizon_s"] for r in rows] == [1.0, 10.0, 60.0]
    assert rows[0]["n"] == 1.0
    assert rows[1]["unsettled"] == 1.0
    assert math.isnan(rows[1]["mean_bps"])


def test_default_horizons_are_one_ten_sixty():
    assert [w.horizon_s for w in MarkOutTracker().windows] == [1.0, 10.0, 60.0]
