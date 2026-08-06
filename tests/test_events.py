"""Event study mechanics, and the ways a backtest flatters itself.

Nothing here is about whether a rule wins. It is about whether the number
reported for a rule can be believed: no lookahead, no overlapping trades
counted as independent, no ambiguous barrier resolved the favourable way, no
result carried by a handful of outliers.
"""

import math

import pytest

from jsboard.research.archive import SecondBar
from jsboard.research.events import (
    combine,
    score,
    simulate,
    threshold,
)
from jsboard.research.features import (
    MinuteBar,
    attach_open_interest,
    build,
    to_minutes,
)

MIN = 60


def secs(prices, start=0, step=1, buy=1.0, sell=1.0):
    return [
        SecondBar(
            sec=start + i * step, last=p, high=p, low=p,
            buy_qty=buy, sell_qty=sell, trades=1,
        )
        for i, p in enumerate(prices)
    ]


def minute_bars(n, price=100.0, start=0):
    return [
        MinuteBar(
            minute=start + i * MIN,
            spot_last=price, spot_buy=1.0, spot_sell=1.0, spot_trades=1,
            perp_last=price, perp_buy=1.0, perp_sell=1.0, perp_trades=1,
            perp_high=price, perp_low=price,
        )
        for i in range(n)
    ]


# ------------------------------------------------------------------ minutes


class TestMinuteBars:
    def test_seconds_fold_into_the_minute_that_contains_them(self):
        spot = secs([100.0, 101.0, 102.0], start=0)
        perp = secs([100.0, 101.0, 102.0], start=0)

        got = to_minutes(spot, perp)

        assert len(got) == 1
        assert got[0].minute == 0
        assert got[0].spot_last == 102.0  # the last print, not the first

    def test_a_minute_missing_one_product_is_dropped(self):
        # A basis computed against a stale spot print is not a basis, and
        # every condition here compares the two products.
        spot = secs([100.0], start=0)
        perp = secs([100.0], start=0) + secs([101.0], start=120)

        got = to_minutes(spot, perp)

        assert [b.minute for b in got] == [0]

    def test_buy_ratio_is_the_aggressor_share(self):
        perp = [SecondBar(0, 100.0, 100.0, 100.0, buy_qty=3.0, sell_qty=1.0, trades=1)]

        bar = to_minutes(secs([100.0]), perp)[0]

        assert bar.perp_buy_ratio == pytest.approx(0.75)

    def test_buy_ratio_is_balanced_when_nothing_traded(self):
        assert MinuteBar(minute=0).perp_buy_ratio == 0.5

    def test_basis_is_the_perp_against_spot(self):
        bar = MinuteBar(minute=0, spot_last=100.0, perp_last=100.5)

        assert bar.basis_bps == pytest.approx(50.0)

    def test_the_minute_high_and_low_span_the_seconds(self):
        perp = secs([100.0]) + [SecondBar(30, 101.0, 105.0, 99.0, 1.0, 1.0, 1)]

        bar = to_minutes(secs([100.0, 100.0], step=30), perp)[0]

        assert bar.perp_high == 105.0
        assert bar.perp_low == 99.0


class TestOpenInterest:
    def test_a_minute_takes_the_last_mark_at_or_before_it(self):
        # Taking the nearest mark would let a minute see a value published
        # minutes later — lookahead that leaves every test green.
        bars = minute_bars(10)
        marks = [(0, 100.0), (5 * MIN, 200.0)]

        attach_open_interest(bars, marks)

        assert bars[4].open_interest == 100.0
        assert bars[5].open_interest == 200.0
        assert bars[9].open_interest == 200.0

    def test_minutes_before_the_first_mark_stay_empty(self):
        bars = minute_bars(3)
        attach_open_interest(bars, [(10 * MIN, 500.0)])

        assert all(b.open_interest == 0.0 for b in bars)

    def test_no_marks_is_not_an_error(self):
        bars = minute_bars(3)
        attach_open_interest(bars, [])

        assert all(b.open_interest == 0.0 for b in bars)


# ----------------------------------------------------------------- features


class TestFeatures:
    def test_z_scores_look_only_backwards(self):
        # A spike at the end must not raise the z-score of the minutes before
        # it. The baseline varies, because a perfectly constant history has no
        # spread to score against and is covered by its own test.
        bars = minute_bars(200)
        for i, b in enumerate(bars):
            b.perp_buy = 1.0 + (i % 5) * 0.1
        before = build(bars, z_window=50)[100].volume_z

        bars[-1].perp_buy = 1_000.0
        after = build(bars, z_window=50)

        assert after[100].volume_z == pytest.approx(before)
        assert after[-1].volume_z > 3.0

    def test_a_constant_series_scores_zero_rather_than_infinity(self):
        # Dividing by a near-zero deviation manufactures signals out of
        # quiet stretches.
        rows = build(minute_bars(200), z_window=50)

        assert all(r.volume_z == 0.0 for r in rows)

    def test_an_early_window_without_history_scores_zero(self):
        rows = build(minute_bars(200), z_window=100)

        assert rows[3].volume_z == 0.0

    def test_returns_are_bps_over_the_stated_number_of_minutes(self):
        bars = minute_bars(10)
        for i, b in enumerate(bars):
            b.perp_last = 100.0 * (1.01**i)
            b.spot_last = b.perp_last

        rows = build(bars)

        assert rows[5].perp_ret_1m == pytest.approx(100.0, rel=1e-3)

    def test_book_features_are_absent_rather_than_zero(self):
        # Zero would read as "balanced book"; NaN reads as "not measured",
        # and a condition naming one simply never fires.
        rows = build(minute_bars(5))

        assert math.isnan(rows[0].book_imbalance_5)
        assert math.isnan(rows[0].ask_cancel_imbalance)

    def test_a_condition_on_an_unmeasured_feature_never_fires(self):
        rows = build(minute_bars(5))
        cond = threshold("book_imbalance_5", ">", 0.65)

        assert not any(cond(r) for r in rows)


# --------------------------------------------------------------- simulation


class TestSimulate:
    def test_trades_do_not_overlap(self):
        # Sixty readings of one hour are not sixty observations. Without
        # this, sample sizes are inflated by the holding period.
        bars = minute_bars(200)
        rows = build(bars)
        path = secs([100.0] * (200 * MIN))

        trades = simulate(
            rows, path, lambda f: True,
            direction=1, horizon_min=10, cost_bps=0.0,
        )

        assert trades
        for a, b in zip(trades, trades[1:], strict=False):
            assert b.entry_minute >= a.exit_minute

    def test_a_target_pays_the_target_not_the_close(self):
        prices = [100.0] * 300 + [101.0] * 300
        path = secs(prices)
        bars = minute_bars(10)
        for b in bars:
            b.perp_last = b.spot_last = 100.0
        rows = build(bars)

        trades = simulate(
            rows[:1], path, lambda f: True,
            direction=1, horizon_min=9, cost_bps=0.0,
            target_bps=50.0, stop_bps=50.0,
        )

        assert trades[0].reason == "target"
        assert trades[0].net_bps == pytest.approx(50.0, rel=1e-6)

    def test_a_bar_touching_both_barriers_resolves_as_a_stop(self):
        # The true order is unknowable at this resolution; assuming the
        # favourable one is how a backtest invents an edge.
        path = [SecondBar(0, 100.0, 100.0, 100.0, 1, 1, 1)] + [
            SecondBar(i, 100.0, 105.0, 95.0, 1, 1, 1) for i in range(60, 600)
        ]
        bars = minute_bars(10)
        rows = build(bars)

        trades = simulate(
            rows[:1], path, lambda f: True,
            direction=1, horizon_min=9, cost_bps=0.0,
            target_bps=100.0, stop_bps=100.0,
        )

        assert trades[0].reason == "stop"
        assert trades[0].net_bps < 0

    def test_cost_is_subtracted_from_the_gross_move(self):
        path = secs([100.0] * 300 + [101.0] * 300)
        rows = build(minute_bars(10))

        trades = simulate(
            rows[:1], path, lambda f: True,
            direction=1, horizon_min=9, cost_bps=9.0,
        )

        assert trades[0].gross_bps - trades[0].net_bps == pytest.approx(9.0)

    def test_a_short_earns_when_the_price_falls(self):
        path = secs([100.0] * 300 + [99.0] * 300)
        rows = build(minute_bars(10))

        trades = simulate(
            rows[:1], path, lambda f: True,
            direction=-1, horizon_min=9, cost_bps=0.0,
        )

        assert trades[0].net_bps > 0

    def test_entries_whose_horizon_runs_past_the_data_are_dropped(self):
        # Scoring them would need a future that does not exist.
        path = secs([100.0] * 120)
        rows = build(minute_bars(60))

        trades = simulate(
            rows, path, lambda f: True,
            direction=1, horizon_min=30, cost_bps=0.0,
        )

        assert trades == []

    def test_an_invalid_direction_is_refused(self):
        with pytest.raises(ValueError):
            simulate([], [], lambda f: True, direction=0, horizon_min=5, cost_bps=0.0)


# ------------------------------------------------------------------ scoring


def fake_trades(nets, cost=9.0, start=1_700_000_000):
    from jsboard.research.events import Trade

    out = []
    for i, n in enumerate(nets):
        # Encode the desired net by choosing the exit price.
        entry = 100.0
        exit_ = entry * (1 + (n + cost) / 10_000.0)
        out.append(
            Trade(
                entry_minute=start + i * 3600,
                exit_minute=start + i * 3600 + 600,
                entry_price=entry,
                exit_price=exit_,
                direction=1,
                cost_bps=cost,
                reason="timeout",
            )
        )
    return out


class TestScore:
    def test_profit_factor_is_gross_win_over_gross_loss(self):
        v = score("x", fake_trades([20.0, 20.0, -10.0, -10.0]))

        assert v.profit_factor == pytest.approx(2.0, rel=1e-3)

    def test_removing_the_best_ten_can_flip_the_verdict(self):
        # A rule carried by ten trades is a rule about ten trades.
        nets = [500.0] * 10 + [-1.0] * 90
        v = score("x", fake_trades(nets))

        assert v.mean_net_bps > 0
        assert v.mean_without_top10_bps < 0
        assert not v.passes

    def test_a_higher_fee_is_charged_against_every_trade(self):
        v = score("x", fake_trades([4.0] * 50, cost=9.0))

        # +4bps net at 9bps cost becomes -0.5bps at 13.5bps.
        assert v.mean_at_15x_cost_bps == pytest.approx(-0.5, abs=1e-6)

    def test_drawdown_is_the_worst_peak_to_trough(self):
        v = score("x", fake_trades([10.0, -30.0, 5.0]))

        assert v.max_drawdown_bps == pytest.approx(30.0, rel=1e-3)

    def test_a_losing_month_fails_the_gate(self):
        import datetime as dt

        jan = int(dt.datetime(2026, 1, 5, tzinfo=dt.UTC).timestamp())
        feb = int(dt.datetime(2026, 2, 5, tzinfo=dt.UTC).timestamp())
        good = fake_trades([20.0] * 60, start=jan)
        bad = fake_trades([-20.0] * 60, start=feb)

        v = score("x", good + bad)

        assert v.months_total == 2
        assert v.months_positive == 1
        assert not v.passes

    def test_too_few_trades_fails_however_good_they_look(self):
        v = score("x", fake_trades([50.0] * 20))

        assert v.mean_net_bps > 5
        assert not v.passes
        assert any("100" in f for f in v.failures())

    def test_an_empty_result_is_reported_rather_than_crashing(self):
        v = score("x", [])

        assert v.trades == 0
        assert not v.passes

    def test_a_rule_clearing_every_gate_passes(self):
        import datetime as dt

        start = int(dt.datetime(2026, 1, 1, tzinfo=dt.UTC).timestamp())
        # 150 trades, mostly +20 with some -5: median positive, PF high,
        # survives the trim and the fee bump, one period, all positive.
        nets = [20.0] * 120 + [-5.0] * 30
        v = score("x", fake_trades(nets, start=start))

        assert v.passes, v.failures()


class TestConditions:
    def test_thresholds_compare_in_the_stated_direction(self):
        from jsboard.research.features import Features

        f = Features(minute=0, price=100.0, volume_z=3.0)

        assert threshold("volume_z", ">", 2.0)(f)
        assert not threshold("volume_z", "<", 2.0)(f)

    def test_combine_requires_every_part(self):
        from jsboard.research.features import Features

        f = Features(minute=0, price=100.0, volume_z=3.0, perp_buy_ratio=0.7)
        both = combine(
            threshold("volume_z", ">", 2.0),
            threshold("perp_buy_ratio", ">", 0.65),
        )
        neither = combine(
            threshold("volume_z", ">", 2.0),
            threshold("perp_buy_ratio", ">", 0.9),
        )

        assert both(f)
        assert not neither(f)
