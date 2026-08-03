"""Signal evaluation, and the ways it can flatter itself.

Every test here guards against reporting an edge that is not there. The
failure mode is not a crash — it is a number that looks good: a signal scored
on the data that chose it, a split that leaks the future, a benchmark of 50%
in a market that rose all week.
"""

import pytest

from jsboard.research.archive import SecondBar
from jsboard.research.predict import (
    Series,
    always_long,
    candidates,
    evaluate,
    signal_value,
    split,
    volatility_threshold,
)


def make_bars(steps):
    """(price, buy_qty, sell_qty) per second, starting at t=0."""
    return [
        SecondBar(sec=i, last=p, high=p, low=p, buy_qty=b, sell_qty=s, trades=1)
        for i, (p, b, s) in enumerate(steps)
    ]


def flat(n, price=100.0, buy=1.0, sell=1.0):
    return make_bars([(price, buy, sell)] * n)


class TestWindows:
    def test_flow_is_a_share_of_volume_not_a_raw_total(self):
        # Unnormalised, a busy window outranks a lopsided one, and the signal
        # would mostly measure activity.
        quiet = Series.build(make_bars([(100.0, 2.0, 0.0)] * 10))
        busy = Series.build(make_bars([(100.0, 60.0, 40.0)] * 10))

        assert quiet.flow(9, 10) == pytest.approx(1.0)
        assert busy.flow(9, 10) == pytest.approx(0.2)

    def test_flow_is_zero_when_nothing_traded(self):
        s = Series.build(make_bars([(100.0, 0.0, 0.0)] * 5))

        assert s.flow(4, 5) == 0.0

    def test_windows_are_seconds_not_rows(self):
        # Bars only exist for seconds that traded; a 60s window over sparse
        # bars must not reach back further in time than 60 seconds.
        sparse = [
            SecondBar(0, 100.0, 100.0, 100.0, 1.0, 0.0, 1),
            SecondBar(500, 200.0, 200.0, 200.0, 1.0, 0.0, 1),
            SecondBar(501, 201.0, 201.0, 201.0, 1.0, 0.0, 1),
        ]
        s = Series.build(sparse)

        # The 60s window ending at bar 2 starts at t=441, so it must not see
        # the bar at t=0 and report a 100% move.
        assert s.momentum(2, 60) == pytest.approx((201.0 - 200.0) / 200.0 * 10_000)

    def test_forward_returns_stop_at_the_end_of_the_series(self):
        s = Series.build(flat(5))

        assert s.forward(4, 10) is None

    def test_volatility_sums_absolute_moves(self):
        s = Series.build(make_bars([(100.0, 1, 1), (101.0, 1, 1), (100.0, 1, 1)]))

        # ~100bps up then ~99bps down; a signed sum would report roughly zero.
        assert s.volatility(2, 10) > 190.0


class TestCandidates:
    def test_both_directions_are_offered(self):
        # Whether flow means continuation or reversal is a question to
        # measure, not one to answer in advance.
        names = [c[0] for c in candidates()]
        signs = {c[3] for c in candidates()}

        assert signs == {1, -1}
        assert any("逆張り" in n for n in names)

    def test_reversing_the_sign_reverses_the_value(self):
        s = Series.build(make_bars([(100.0, 3.0, 1.0)] * 10))

        assert signal_value(s, 9, "flow", 10, 1) == -signal_value(s, 9, "flow", 10, -1)


class TestSplit:
    def test_the_split_is_chronological(self):
        bars = flat(10)

        train, test = split(bars, 0.6)

        assert [b.sec for b in train] == list(range(6))
        assert [b.sec for b in test] == list(range(6, 10))

    def test_the_halves_do_not_overlap(self):
        bars = flat(100)

        train, test = split(bars, 0.7)

        assert max(b.sec for b in train) < min(b.sec for b in test)
        assert len(train) + len(test) == len(bars)


class TestEvaluate:
    def test_a_signal_that_leads_the_price_scores_well(self):
        # Runs long enough that the trailing window sits inside one of them
        # for most bars; only near a turn can the signal be wrong. Short runs
        # would make this a test of the boundaries rather than the signal.
        steps = []
        price = 100.0
        for i in range(2_000):
            up = (i // 200) % 2 == 0
            price += 0.1 if up else -0.1
            steps.append((price, 5.0 if up else 0.0, 0.0 if up else 5.0))
        s = Series.build(make_bars(steps))

        got = evaluate(
            s, "x", "flow", 10, 1, horizon_s=5, cost_bps=0.0,
            vol_window=10, vol_threshold=None,
        )

        assert got.samples > 1_000
        assert got.accuracy > 0.85

    def test_flat_windows_count_as_neither_hit_nor_miss(self):
        # Scoring a zero move as a miss would drag accuracy toward zero on a
        # quiet market; as a hit, toward one. It is simply not a test.
        s = Series.build(flat(300, buy=5.0, sell=0.0))

        got = evaluate(
            s, "x", "flow", 60, 1, horizon_s=10, cost_bps=0.0,
            vol_window=60, vol_threshold=None,
        )

        assert got.samples == 0
        assert got.accuracy == 0.0

    def test_bars_without_enough_history_are_skipped(self):
        # A 900s window over the first minute of data is not a 900s window.
        s = Series.build(make_bars([(100.0 + i * 0.01, 2.0, 1.0) for i in range(300)]))

        got = evaluate(
            s, "x", "flow", 900, 1, horizon_s=10, cost_bps=0.0,
            vol_window=900, vol_threshold=None,
        )

        assert got.samples == 0

    def test_the_volatility_filter_reduces_the_sample(self):
        steps = [(100.0 + (i % 7) * 0.5, 2.0, 1.0) for i in range(600)]
        s = Series.build(make_bars(steps))
        common = dict(horizon_s=10, cost_bps=0.0, vol_window=60)

        wide = evaluate(s, "x", "flow", 60, 1, vol_threshold=None, **common)
        narrow = evaluate(s, "x", "flow", 60, 1, vol_threshold=1e9, **common)

        assert wide.samples > 0
        assert narrow.samples == 0

    def test_edge_is_negative_when_accuracy_only_matches_the_fee(self):
        s = Series.build(make_bars([(100.0 + (i % 2), 2.0, 1.0) for i in range(400)]))

        got = evaluate(
            s, "x", "flow", 60, 1, horizon_s=10, cost_bps=1_000.0,
            vol_window=60, vol_threshold=None,
        )

        assert got.edge_bps < 0


class TestBaseline:
    def test_a_rising_market_makes_always_long_look_skilful(self):
        # This is why 50% is the wrong benchmark: here a coin that always
        # says up is right almost every time, with no information at all.
        s = Series.build(make_bars([(100.0 + i * 0.1, 1.0, 1.0) for i in range(500)]))

        base = always_long(s, horizon_s=10, cost_bps=0.0)

        assert base.accuracy > 0.95

    def test_a_falling_market_makes_it_look_hopeless(self):
        s = Series.build(make_bars([(200.0 - i * 0.1, 1.0, 1.0) for i in range(500)]))

        base = always_long(s, horizon_s=10, cost_bps=0.0)

        assert base.accuracy < 0.05


class TestVolatilityThreshold:
    def test_a_higher_quantile_demands_more_movement(self):
        steps = [(100.0 + (i % 11) * 0.3, 1.0, 1.0) for i in range(600)]
        s = Series.build(make_bars(steps))

        low = volatility_threshold(s, 60, 0.5)
        high = volatility_threshold(s, 60, 0.95)

        assert high >= low

    def test_a_motionless_series_has_a_zero_threshold(self):
        assert volatility_threshold(Series.build(flat(300)), 60, 0.9) == 0.0
