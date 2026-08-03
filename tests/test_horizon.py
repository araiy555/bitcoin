"""The fee gate, and the archive parsing it rests on.

The number that matters here is `required_accuracy`. It answers whether a
direction strategy is worth building at all, before any model exists, and it
can exceed 1.0 — meaning a perfect predictor still loses to the fee. That
outcome has to survive as a distinct result rather than being clamped into
something that looks merely difficult.
"""

import csv
import io
import zipfile
from pathlib import Path

import pytest

from jsboard.research.archive import (
    SecondBar,
    archive_url,
    days_ending,
    load_seconds,
)
from jsboard.research.horizon import (
    analyse,
    forward_returns,
    round_trip_cost_bps,
)


def bars(prices, start=1_000_000):
    """One bar per second, ascending, with the given last prices."""
    return [
        SecondBar(sec=start + i, last=p, high=p, low=p, buy_qty=1.0, sell_qty=0.0, trades=1)
        for i, p in enumerate(prices)
    ]


def write_archive(tmp_path: Path, rows, *, header=False, name="x.csv") -> Path:
    path = tmp_path / "day.zip"
    buf = io.StringIO()
    writer = csv.writer(buf)
    if header:
        writer.writerow(
            ["agg_trade_id", "price", "quantity", "first", "last", "transact_time", "is_buyer_maker"]
        )
    writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(name, buf.getvalue())
    return path


# ------------------------------------------------------------------- naming


class TestUrls:
    def test_spot_and_perp_live_under_different_roots(self):
        import datetime

        day = datetime.date(2026, 8, 1)

        spot = archive_url("spot", "aggTrades", "btcusdt", day)
        perp = archive_url("perp", "aggTrades", "btcusdt", day)

        assert "/spot/daily/" in spot
        assert "/futures/um/daily/" in perp
        assert spot.endswith("BTCUSDT-aggTrades-2026-08-01.zip")

    def test_days_ending_is_oldest_first_and_inclusive(self):
        import datetime

        got = days_ending(datetime.date(2026, 8, 3), 3)

        assert got == [
            datetime.date(2026, 8, 1),
            datetime.date(2026, 8, 2),
            datetime.date(2026, 8, 3),
        ]


# ------------------------------------------------------------------ parsing


class TestLoadSeconds:
    def test_trades_collapse_into_one_bar_per_second(self, tmp_path):
        path = write_archive(
            tmp_path,
            [
                [1, "100.0", "1.0", 1, 1, 1_700_000_000_000, "true"],
                [2, "101.0", "2.0", 2, 2, 1_700_000_000_500, "false"],
                [3, "102.0", "3.0", 3, 3, 1_700_000_001_000, "false"],
            ],
        )

        got = load_seconds(path)

        assert [b.sec for b in got] == [1_700_000_000, 1_700_000_001]
        assert got[0].last == 101.0
        assert got[0].trades == 2

    def test_buyer_is_maker_means_the_seller_crossed(self, tmp_path):
        path = write_archive(
            tmp_path,
            [
                [1, "100.0", "5.0", 1, 1, 1_700_000_000_000, "true"],
                [2, "100.0", "2.0", 2, 2, 1_700_000_000_100, "false"],
            ],
        )

        bar = load_seconds(path)[0]

        assert bar.sell_qty == 5.0
        assert bar.buy_qty == 2.0
        assert bar.signed_qty == -3.0

    def test_a_header_row_is_skipped(self, tmp_path):
        # Older archives ship headerless and newer ones do not; a header
        # parsed as data would land at the epoch and skew every horizon.
        path = write_archive(
            tmp_path, [[1, "100.0", "1.0", 1, 1, 1_700_000_000_000, "true"]], header=True
        )

        got = load_seconds(path)

        assert len(got) == 1
        assert got[0].last == 100.0

    def test_microsecond_timestamps_are_recognised(self, tmp_path):
        # Binance moved some datasets to microseconds. Read as milliseconds
        # these land in the year 33,000 and every horizon comes out empty.
        path = write_archive(
            tmp_path, [[1, "100.0", "1.0", 1, 1, 1_700_000_000_000_000, "true"]]
        )

        got = load_seconds(path)

        assert got[0].sec == 1_700_000_000

    def test_high_and_low_track_the_second(self, tmp_path):
        path = write_archive(
            tmp_path,
            [
                [1, "100.0", "1.0", 1, 1, 1_700_000_000_000, "true"],
                [2, "105.0", "1.0", 2, 2, 1_700_000_000_100, "true"],
                [3, "98.0", "1.0", 3, 3, 1_700_000_000_200, "true"],
            ],
        )

        bar = load_seconds(path)[0]

        assert bar.high == 105.0
        assert bar.low == 98.0
        assert bar.last == 98.0


# ------------------------------------------------------------------ returns


class TestForwardReturns:
    def test_a_flat_tape_returns_nothing(self):
        got = forward_returns(bars([100.0] * 10), 1)

        assert got and all(r == 0.0 for r in got)

    def test_a_one_percent_rise_is_a_hundred_bps(self):
        got = forward_returns(bars([100.0, 101.0]), 1)

        assert got[0] == pytest.approx(100.0)

    def test_the_horizon_is_seconds_not_rows(self):
        # Bars exist only for seconds that traded. Stepping by index would
        # make a 5s horizon mean "5 prints later", which on a quiet market
        # is minutes.
        sparse = [
            SecondBar(1_000, 100.0, 100.0, 100.0, 1.0, 0.0, 1),
            SecondBar(1_060, 200.0, 200.0, 200.0, 1.0, 0.0, 1),
        ]

        got = forward_returns(sparse, 1)

        # 1_001 never traded, so the price a second later is still 100 —
        # not the 200 printed a minute afterwards.
        assert got == [0.0]

    def test_windows_past_the_last_bar_are_dropped(self):
        # Filling them with the current price would append a run of zero
        # moves to every day and understate what the market offers.
        got = forward_returns(bars([100.0, 101.0, 102.0]), 2)

        assert len(got) == 1
        assert got[0] == pytest.approx(200.0)

    def test_a_negative_horizon_is_refused(self):
        with pytest.raises(ValueError):
            forward_returns(bars([100.0, 101.0]), 0)


# --------------------------------------------------------------- the verdict


class TestRequiredAccuracy:
    def test_a_free_market_needs_only_a_coin_flip_edge(self):
        st = analyse(bars([100.0, 101.0, 100.0, 101.0, 100.0]), 1, cost_bps=0.0)

        assert st.required_accuracy == pytest.approx(0.5)
        assert st.is_possible

    def test_a_fee_equal_to_the_move_demands_perfection(self):
        # Charge exactly what the market offers: being right every single
        # time then breaks even, and anything less loses.
        series = bars([100.0, 101.0, 100.0, 101.0])
        move = analyse(series, 1, cost_bps=0.0).mean_abs_bps

        st = analyse(series, 1, cost_bps=move)

        assert st.required_accuracy == pytest.approx(1.0)
        assert not st.is_possible
        assert st.perfect_foresight_bps == pytest.approx(0.0)

    def test_a_fee_above_the_move_is_impossible_not_merely_hard(self):
        # The result that matters: no predictor of any quality wins here.
        st = analyse(bars([100.0, 100.1, 100.0, 100.1]), 1, cost_bps=50.0)

        assert st.required_accuracy > 1.0
        assert not st.is_possible
        assert st.perfect_foresight_bps < 0

    def test_perfect_foresight_is_the_move_minus_the_cost(self):
        st = analyse(bars([100.0, 101.0, 100.0, 101.0]), 1, cost_bps=20.0)

        assert st.perfect_foresight_bps == pytest.approx(st.mean_abs_bps - 20.0)

    def test_a_motionless_market_is_impossible_rather_than_undefined(self):
        st = analyse(bars([100.0] * 20), 1, cost_bps=5.0)

        assert st.mean_abs_bps == 0.0
        assert not st.is_possible

    def test_the_tradeable_fraction_counts_moves_over_the_fee(self):
        # Three of four steps move 100bps, one moves ~0.
        st = analyse(bars([100.0, 101.0, 100.0, 101.0, 101.0]), 1, cost_bps=50.0)

        assert 0.0 < st.tradeable_fraction < 1.0


class TestRoundTripCost:
    def test_both_legs_are_charged(self):
        # The common error is counting one fee and one crossing, which halves
        # the wall the strategy has to clear.
        assert round_trip_cost_bps(4.5, 0.0, 0.0) == pytest.approx(9.0)

    def test_slippage_is_counted_twice_as_well(self):
        assert round_trip_cost_bps(0.0, 1.0, 0.16) == pytest.approx(0.32)

    def test_a_free_venue_with_no_slippage_costs_nothing(self):
        assert round_trip_cost_bps(0.0, 0.0, 0.16) == 0.0
