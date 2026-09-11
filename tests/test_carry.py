"""Funding carry: the worst case, not the average.

The average funding rate is the number everyone quotes and the one that
decides nothing. What these tests pin down is the machinery that says how long
an entry stayed underwater and how bad the negative stretches were, because
that is what separates a boring positive trade from a slow loss.
"""

import pytest

from jsboard.research.carry import CarryStudy, FundingPoint, parse_funding

HOUR_MS = 3_600_000
DAY_MS = 86_400_000
START = 1_700_000_000_000


def series(rates_bps, *, every_hours=8, start=START):
    """A funding history from a list of rates in basis points."""
    return [
        FundingPoint(ts_ms=start + i * every_hours * HOUR_MS, rate=bps / 10_000.0)
        for i, bps in enumerate(rates_bps)
    ]


def study(rates_bps, **kw):
    return CarryStudy(series(rates_bps, **{k: v for k, v in kw.items() if k == "every_hours"}),
                      **{k: v for k, v in kw.items() if k != "every_hours"})


class TestShape:
    def test_the_settlement_interval_is_measured_not_assumed(self):
        """A 4-hour symbol annualised as 8-hourly is wrong by half."""
        assert study([1] * 10, every_hours=4).interval_hours == 4.0
        assert study([1] * 10, every_hours=8).interval_hours == 8.0

    def test_annualising_follows_the_measured_interval(self):
        eight = study([1] * 10).annual_pct
        four = study([1] * 10, every_hours=4).annual_pct
        assert four == pytest.approx(eight * 2)

    def test_one_basis_point_every_eight_hours_is_about_eleven_percent(self):
        # 3 settlements a day * 365 * 1bp = 1095bp = 10.95%
        assert study([1] * 100).annual_pct == pytest.approx(10.95, abs=0.01)

    def test_an_empty_history_reports_zeroes_rather_than_dividing_by_zero(self):
        empty = CarryStudy([])
        assert empty.annual_pct == 0.0
        assert empty.max_drawdown_bps == 0.0
        assert empty.payback_summary()["judged"] == 0


class TestNegativeStretches:
    def test_the_share_of_settlements_that_cost_money(self):
        assert study([1, 1, -1, 1]).negative_share == 0.25

    def test_drawdown_is_the_fall_from_a_high_not_the_lowest_point(self):
        # Up 10, down 6, up again: the curve never goes below zero, but an
        # entry at the peak still gave back 6.
        assert study([5, 5, -3, -3, 5]).max_drawdown_bps == pytest.approx(6.0)

    def test_a_curve_that_only_rises_has_no_drawdown(self):
        assert study([1, 2, 3]).max_drawdown_bps == 0.0

    def test_time_below_a_previous_high_is_reported(self):
        # Peak at the second point, recovered four settlements later.
        underwater = study([5, 5, -2, -2, -2, 10]).longest_underwater_days
        assert underwater == pytest.approx(4 * 8 / 24, abs=1e-6)

    def test_the_worst_window_finds_the_bad_run(self):
        # 30 days at 8h settlements is 90 points; make a short window instead.
        s = CarryStudy(series([1] * 10 + [-5] * 10 + [1] * 10), drawdown_window_days=3)
        assert s.worst_window_bps() == pytest.approx(-45.0)


class TestPayback:
    def test_a_steady_rate_pays_back_on_schedule(self):
        # 1bp every 8h is 3bp a day; 28bp needs 10 days minus a rounding step.
        s = CarryStudy(series([1] * 200), entry_cost_bps=28.0)
        summary = s.payback_summary()
        assert summary["median_days"] == pytest.approx(28 / 3, abs=0.4)
        assert summary["never"] == 0

    def test_an_entry_that_never_recovers_is_counted(self):
        s = CarryStudy(series([0.01] * 300), entry_cost_bps=28.0)
        assert s.payback_summary(max_hold_days=30)["never_share"] == 1.0

    def test_recent_entries_are_not_counted_as_failures(self):
        """Otherwise every series looks worse the closer it gets to today."""
        s = CarryStudy(series([1] * 200), entry_cost_bps=28.0)
        summary = s.payback_summary(max_hold_days=20)
        # The last 20 days of entries had no chance to recover, so they are
        # excluded rather than recorded as never paying back.
        assert summary["judged"] < len(s)
        assert summary["never"] == 0

    def test_a_higher_entry_cost_takes_longer(self):
        cheap = CarryStudy(series([1] * 200), entry_cost_bps=10.0).payback_summary()
        dear = CarryStudy(series([1] * 200), entry_cost_bps=40.0).payback_summary()
        assert dear["median_days"] > cheap["median_days"]

    def test_only_funding_after_the_entry_counts(self):
        """An entry cannot be paid by settlements that happened before it."""
        rates = [1.0, 100.0] + [1.0] * 100  # one huge settlement early on
        s = CarryStudy(series(rates), entry_cost_bps=28.0)
        first, second = s.paybacks()[0], s.paybacks()[1]
        assert first.days == pytest.approx(8 / 24, abs=1e-6)  # the big one is ahead
        assert second.days > 9  # the big one is behind it and does not count

    def test_a_negative_cost_is_refused(self):
        with pytest.raises(ValueError):
            CarryStudy([], entry_cost_bps=-1.0)


class TestParsing:
    def test_binance_rows_become_signed_rates(self):
        points = parse_funding(
            [
                {"fundingTime": "1700000000000", "fundingRate": "0.00010000"},
                {"fundingTime": "1700028800000", "fundingRate": "-0.00005000"},
            ]
        )
        assert [p.bps for p in points] == [1.0, -0.5]

    def test_the_timestamp_survives_as_a_date(self):
        point = parse_funding([{"fundingTime": "1700000000000", "fundingRate": "0"}])[0]
        assert point.when.year == 2023


class TestCredentials:
    def test_the_fetcher_takes_no_key_arguments(self):
        import inspect

        from jsboard.research.carry import fetch_funding

        params = set(inspect.signature(fetch_funding).parameters)
        assert not (params & {"api_key", "api_secret", "key", "secret"})


class TestCommand:
    def test_the_carry_command_reads_a_saved_history(self, tmp_path, capsys):
        import asyncio
        import json

        from jsboard.cli import build_parser, cmd_carry

        path = tmp_path / "funding.json"
        path.write_text(
            json.dumps(
                [
                    {"fundingTime": START + i * 8 * HOUR_MS, "fundingRate": 0.0001}
                    for i in range(200)
                ]
            )
        )
        args = build_parser().parse_args(
            ["carry", "--from-file", str(path), "--plain"]
        )
        assert asyncio.run(cmd_carry(args)) == 0
        out = capsys.readouterr().out
        assert "回収" in out
        assert "年率" in out

    def test_an_empty_history_is_refused_rather_than_summarised(self, tmp_path):
        import asyncio
        import json

        from jsboard.cli import ConfigError, build_parser, cmd_carry

        path = tmp_path / "empty.json"
        path.write_text(json.dumps([]))
        args = build_parser().parse_args(["carry", "--from-file", str(path)])
        with pytest.raises(ConfigError):
            asyncio.run(cmd_carry(args))

    def test_a_missing_file_names_itself(self, tmp_path):
        import asyncio

        from jsboard.cli import ConfigError, build_parser, cmd_carry

        args = build_parser().parse_args(
            ["carry", "--from-file", str(tmp_path / "nope.json")]
        )
        with pytest.raises(ConfigError, match="nope.json"):
            asyncio.run(cmd_carry(args))


class TestTiming:
    def test_the_signal_cannot_see_the_settlement_it_is_positioned_for(self):
        """A rule that reads its own payment turns any series into a winner."""
        from jsboard.research.carry import run_timed

        # Flat, then one huge settlement. A lookahead rule would be in for it.
        points = series([0.0] * 5 + [1000.0] + [0.0] * 5)
        result = run_timed(points, lookback=2, enter_bps=1.0, exit_bps=0.5,
                           entry_cost_bps=0.0)
        assert result.net_bps <= 0.0

    def test_it_enters_after_the_signal_and_collects_what_follows(self):
        from jsboard.research.carry import run_timed

        points = series([5.0] * 10)
        result = run_timed(points, lookback=2, enter_bps=1.0, exit_bps=0.5,
                           entry_cost_bps=0.0)
        assert result.trips == 1
        assert result.net_bps == pytest.approx(5.0 * 8)  # the 8 after the window

    def test_a_series_that_never_qualifies_never_pays_the_entry(self):
        from jsboard.research.carry import run_timed

        result = run_timed(series([0.1] * 50), lookback=3, enter_bps=5.0,
                           exit_bps=1.0, entry_cost_bps=28.0)
        assert result.trips == 0
        assert result.net_bps == 0.0
        assert result.time_in_market == 0.0

    def test_each_re_entry_pays_the_cost_again(self):
        from jsboard.research.carry import run_timed

        # Rich, thin, rich again: two entries, two costs.
        points = series([5.0] * 6 + [0.0] * 6 + [5.0] * 6)
        result = run_timed(points, lookback=2, enter_bps=2.0, exit_bps=1.0,
                           entry_cost_bps=10.0)
        assert result.trips == 2
        assert result.net_bps < run_timed(
            points, lookback=2, enter_bps=2.0, exit_bps=1.0, entry_cost_bps=0.0
        ).net_bps

    def test_deployed_return_exceeds_the_full_period_one_when_it_sits_out(self):
        from jsboard.research.carry import run_timed

        points = series([5.0] * 10 + [0.0] * 30)
        result = run_timed(points, lookback=2, enter_bps=2.0, exit_bps=1.0,
                           entry_cost_bps=0.0)
        assert 0 < result.time_in_market < 1
        assert result.annual_pct_deployed > result.annual_pct_full

    def test_an_exit_above_the_entry_is_refused(self):
        from jsboard.research.carry import run_timed

        with pytest.raises(ValueError):
            run_timed(series([1.0] * 10), lookback=2, enter_bps=1.0, exit_bps=2.0)

    def test_a_history_shorter_than_the_lookback_reports_nothing(self):
        from jsboard.research.carry import run_timed

        result = run_timed(series([1.0] * 3), lookback=10, enter_bps=0.0, exit_bps=0.0)
        assert result.trips == 0
        assert result.equity_bps == []


def test_the_timing_grid_runs_and_names_the_baseline(tmp_path, capsys):
    import asyncio
    import json

    from jsboard.cli import build_parser, cmd_carry

    path = tmp_path / "funding.json"
    path.write_text(
        json.dumps(
            [
                {"fundingTime": START + i * 8 * HOUR_MS, "fundingRate": 0.00005}
                for i in range(400)
            ]
        )
    )
    args = build_parser().parse_args(
        ["carry", "--from-file", str(path), "--timing", "--plain"]
    )
    assert asyncio.run(cmd_carry(args)) == 0
    out = capsys.readouterr().out
    assert "常時保有" in out
