"""Funding gaps between venues.

The number this produces is a difference between two venues' published rates,
so the ways it can lie are all about comparability:

  * intervals differ — Binance settles every eight hours, Hyperliquid every
    hour, and treating those as equal is wrong by a factor of eight
  * settlements do not line up — a gap can only be collected if both legs
    actually paid, so a rate with no partner in range is not a gap
  * the entry is paid once on four legs, while the gap arrives per settlement
"""

import pytest

from jsboard.research.carry import FundingPoint
from jsboard.research.xcarry import (
    INTERVAL_HOURS,
    Spread,
    SpreadStudy,
    VenueFunding,
    pair_spreads,
)

HOUR = 3_600_000


def venue(name, interval_h, rates, *, start=0, step_h=None):
    step = int((step_h or interval_h) * HOUR)
    return VenueFunding(
        name,
        interval_h,
        [FundingPoint(start + i * step, r) for i, r in enumerate(rates)],
    )


class TestInterval:
    def test_a_rate_is_scored_per_hour_not_per_settlement(self):
        """0.01% every hour is eight times 0.01% every eight hours."""
        hourly = venue("hyperliquid", 1.0, [0.0001])
        eight = venue("binance", 8.0, [0.0001])

        assert hourly.hourly_bps(hourly.points[0]) == pytest.approx(1.0)
        assert eight.hourly_bps(eight.points[0]) == pytest.approx(0.125)

    def test_the_published_cadences_are_recorded(self):
        assert INTERVAL_HOURS["binance"] == 8.0
        assert INTERVAL_HOURS["hyperliquid"] == 1.0


class TestPairing:
    def test_the_higher_payer_is_the_short_leg(self):
        """Short where funding is high: that side is the one being paid."""
        a = venue("a", 8.0, [0.0002])
        b = venue("b", 8.0, [0.0001])

        spread = pair_spreads([a, b])[0]

        assert spread.short_venue == "a"
        assert spread.long_venue == "b"
        assert spread.hourly_bps == pytest.approx((2.0 - 1.0) / 8.0)

    def test_a_negative_rate_on_the_long_leg_widens_the_gap(self):
        """Being paid on both legs is the case worth finding."""
        a = venue("a", 8.0, [0.0002])
        b = venue("b", 8.0, [-0.0001])

        spread = pair_spreads([a, b])[0]

        assert spread.hourly_bps == pytest.approx((2.0 + 1.0) / 8.0)

    def test_a_settlement_with_no_partner_in_range_is_dropped(self):
        """No partner means nobody paid; inventing one manufactures a gap."""
        a = venue("a", 8.0, [0.0002], start=0)
        b = venue("b", 8.0, [0.0001], start=50 * HOUR)

        assert pair_spreads([a, b], tolerance_ms=30 * 60_000) == []

    def test_settlements_within_tolerance_are_matched(self):
        a = venue("a", 8.0, [0.0002], start=0)
        b = venue("b", 8.0, [0.0001], start=10 * 60_000)  # 10 minutes later

        assert len(pair_spreads([a, b], tolerance_ms=30 * 60_000)) == 1

    def test_one_venue_alone_has_nothing_to_compare(self):
        assert pair_spreads([venue("a", 8.0, [0.0002])]) == []

    def test_the_gap_is_never_negative(self):
        """The pair is chosen by which side pays more, so it cannot be short."""
        a = venue("a", 8.0, [0.0001, -0.0003, 0.0002])
        b = venue("b", 8.0, [0.0002, 0.0001, -0.0004])

        assert all(s.hourly_bps >= 0 for s in pair_spreads([a, b]))

    def test_different_cadences_still_pair(self):
        """Hourly against eight-hourly: only the aligned hours can match."""
        fast = venue("hyperliquid", 1.0, [0.00001] * 24, start=0)
        slow = venue("binance", 8.0, [0.0001] * 3, start=0)

        spreads = pair_spreads([fast, slow], tolerance_ms=30 * 60_000)

        assert spreads, "aligned settlements should have paired"
        # Binance 0.01%/8h = 0.125bps/h; Hyperliquid 0.001%/1h = 0.1bps/h.
        assert spreads[0].short_venue == "binance"


class TestStudy:
    def _study(self, hourly_bps, n=100, cost=28.0):
        spreads = [Spread(i * HOUR, "a", "b", hourly_bps) for i in range(n)]
        return SpreadStudy(spreads, entry_cost_bps=cost)

    def test_the_annual_figure_uses_hours_not_settlements(self):
        study = self._study(0.1)
        assert study.annual_pct == pytest.approx(0.1 * 365 * 24 / 100)

    def test_payback_is_the_entry_over_the_hourly_rate(self):
        study = self._study(0.5, cost=28.0)
        assert study.payback_hours == pytest.approx(56.0)

    def test_a_gap_that_never_pays_never_pays_back(self):
        assert self._study(0.0).payback_hours == float("inf")

    def test_an_empty_history_reports_nothing_rather_than_zero_percent(self):
        study = SpreadStudy([])
        assert study.n == 0
        assert study.annual_pct == 0.0

    def test_the_pairs_are_counted_so_one_venue_cannot_hide_the_rest(self):
        spreads = [
            Spread(0, "a", "b", 0.1),
            Spread(HOUR, "a", "b", 0.1),
            Spread(2 * HOUR, "b", "a", 0.2),
        ]
        counts = SpreadStudy(spreads).pair_counts()

        assert counts[("a", "b")] == 2
        assert counts[("b", "a")] == 1


class TestCredentials:
    def test_the_fetchers_take_no_key_arguments(self):
        """Funding history is public everywhere; a key here would be a leak."""
        import inspect

        from jsboard.research.xcarry import FETCHERS

        for name, fn in FETCHERS.items():
            params = set(inspect.signature(fn).parameters)
            assert not params & {"api_key", "secret", "key", "token"}, name


class TestTheCommand:
    """Two commands shipped broken on lines no unit test executed."""

    def _run(self, monkeypatch, capsys, *extra, rates=None):
        import asyncio

        from jsboard.cli import build_parser, cmd_xcarry
        from jsboard.research import xcarry as mod

        rates = rates or {"binance": 0.0002, "bybit": 0.0001}

        def make(name, rate):
            async def fetch(symbol, *, days, now_ms=None):
                return venue(name, INTERVAL_HOURS.get(name, 8.0), [rate] * 30)

            return fetch

        monkeypatch.setattr(
            mod, "FETCHERS", {k: make(k, v) for k, v in rates.items()}
        )
        args = build_parser().parse_args(
            ["xcarry", "--symbol", "BTCUSDT", "--days", "10",
             "--venues", ",".join(rates), *extra]
        )
        code = asyncio.run(cmd_xcarry(args))
        return code, capsys.readouterr().out

    def test_it_runs_and_reports_the_annual_figure(self, monkeypatch, capsys):
        code, out = self._run(monkeypatch, capsys)
        assert code == 0
        assert "年率換算" in out

    def test_a_gap_below_the_baseline_is_called_out(self, monkeypatch, capsys):
        """A tiny gap must not read as a win just because it is positive."""
        _, out = self._run(
            monkeypatch, capsys, "--baseline-annual-pct", "5.21",
            rates={"binance": 0.000011, "bybit": 0.00001},
        )
        assert "単独保有に届きません" in out

    def test_a_gap_above_the_baseline_is_called_out(self, monkeypatch, capsys):
        _, out = self._run(
            monkeypatch, capsys, "--baseline-annual-pct", "5.21",
            rates={"binance": 0.001, "bybit": -0.001},
        )
        assert "差のほうが大きい" in out

    def test_one_venue_is_refused_before_any_fetch(self, monkeypatch, capsys):
        import asyncio

        from jsboard.cli import ConfigError, build_parser, cmd_xcarry

        args = build_parser().parse_args(
            ["xcarry", "--symbol", "BTCUSDT", "--venues", "binance"]
        )
        with pytest.raises(ConfigError, match="2つ以上"):
            asyncio.run(cmd_xcarry(args))

    def test_an_unknown_venue_names_the_ones_that_work(self, monkeypatch):
        import asyncio

        from jsboard.cli import ConfigError, build_parser, cmd_xcarry

        args = build_parser().parse_args(
            ["xcarry", "--symbol", "BTCUSDT", "--venues", "binance,nasdaq"]
        )
        with pytest.raises(ConfigError, match="nasdaq"):
            asyncio.run(cmd_xcarry(args))

    def test_one_venue_failing_does_not_kill_the_run(self, monkeypatch, capsys):
        """A venue that is down must not discard the ones that answered."""
        import asyncio

        from jsboard.cli import build_parser, cmd_xcarry
        from jsboard.research import xcarry as mod

        async def ok(symbol, *, days, now_ms=None):
            return venue("binance", 8.0, [0.0002] * 30)

        async def ok2(symbol, *, days, now_ms=None):
            return venue("bybit", 8.0, [0.0001] * 30)

        async def broken(symbol, *, days, now_ms=None):
            raise RuntimeError("venue is down")

        monkeypatch.setattr(
            mod, "FETCHERS", {"binance": ok, "bybit": ok2, "hyperliquid": broken}
        )
        args = build_parser().parse_args(
            ["xcarry", "--symbol", "BTCUSDT", "--venues", "binance,bybit,hyperliquid"]
        )
        assert asyncio.run(cmd_xcarry(args)) == 0
        assert "取得できません" in capsys.readouterr().out
