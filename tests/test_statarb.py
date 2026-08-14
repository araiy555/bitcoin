import csv
import io
import json
import zipfile
from datetime import date

import pytest

from jsboard.cli import build_parser
from jsboard.research.statarb import (
    StrategyConfig,
    TradeReturn,
    archive_parts,
    build_point_in_time_universe,
    kline_url,
    load_dataset,
    load_hourly_closes,
    load_hourly_market,
    parse_hours,
    simulate_config,
    summarise_performance,
    walk_forward,
)


def test_statarb_commands_are_real_cli_commands():
    parser = build_parser()
    download = parser.parse_args(["statarb", "download", "--days", "365", "--top", "30"])
    backtest = parser.parse_args(
        [
            "statarb",
            "backtest",
            "--lookbacks",
            "6h,12h,24h,72h",
            "--holds",
            "4h,8h,24h",
            "--fees",
            "4",
            "--strategy",
            "loser-btc",
            "--entry-zs",
            "1,1.5,2",
            "--funding",
            "--walk-forward",
            "--point-in-time-universe",
            "--universe-top",
            "20",
            "--volume-lookback",
            "14d",
            "--min-history-days",
            "45",
        ]
    )
    assert download.func.__name__ == "cmd_statarb_download"
    assert backtest.func.__name__ == "cmd_statarb_backtest"
    assert backtest.funding is True
    assert backtest.walk_forward is True
    assert backtest.strategy == "loser-btc"
    assert backtest.target_vol == 20.0
    assert backtest.point_in_time_universe is True
    assert backtest.universe_top == 20
    assert backtest.volume_lookback == "14d"
    assert backtest.min_history_days == 45


@pytest.mark.parametrize(
    ("value", "expected"),
    [("6h", 6), ("24H", 24), ("3d", 72), ("12", 12)],
)
def test_parse_hours(value, expected):
    assert parse_hours(value) == expected


def test_archive_parts_use_monthly_files_then_daily_open_month():
    parts = archive_parts(date(2024, 1, 15), date(2024, 3, 3))
    assert [(part.scope, part.stamp) for part in parts[:2]] == [
        ("monthly", "2024-01"),
        ("monthly", "2024-02"),
    ]
    assert [(part.scope, part.stamp) for part in parts[2:]] == [
        ("daily", "2024-03-01"),
        ("daily", "2024-03-02"),
        ("daily", "2024-03-03"),
    ]
    assert kline_url("btcusdt", "5m", parts[0]).endswith(
        "/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip"
    )


def test_load_hourly_closes_keeps_last_five_minute_close(tmp_path):
    path = tmp_path / "bars.zip"
    rows = [
        [1704067200000, 100, 101, 99, 100.5, 1, 2, 10],  # 00:00
        [1704070500000, 100, 102, 99, 101.5, 1, 2, 20],  # 00:55
        [1704070800000, 101, 103, 100, 102.5, 1, 2, 30],  # 01:00
    ]
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("BTCUSDT-5m.csv", buffer.getvalue())

    prices = load_hourly_closes([path], date(2024, 1, 1), date(2024, 1, 1))
    _, execution, quote_volume = load_hourly_market(
        [path], date(2024, 1, 1), date(2024, 1, 1)
    )
    first_hour = 1704067200 // 3600
    assert prices[first_hour] == 101.5
    assert prices[first_hour + 1] == 102.5
    assert execution[first_hour] == 101.0
    assert quote_volume[first_hour] == 30.0
    assert quote_volume[first_hour + 1] == 30.0


def test_point_in_time_universe_uses_only_past_volume_and_minimum_history():
    prices = {
        symbol: {hour: 100.0 for hour in range(8)}
        for symbol in ("BTCUSDT", "AUSDT", "BUSDT")
    }
    quote_volume = {
        "AUSDT": {hour: 10.0 for hour in range(8)},
        "BUSDT": {**{hour: 1.0 for hour in range(6)}, 6: 1_000.0, 7: 1_000.0},
    }
    universe = build_point_in_time_universe(
        prices,
        quote_volume,
        top=1,
        volume_window_h=3,
        min_history_h=3,
        min_volume_coverage=1.0,
    )
    assert universe[2] == frozenset()
    assert universe[3] == frozenset({"AUSDT"})
    assert universe[5] == frozenset({"AUSDT"})
    assert universe[6] == frozenset({"BUSDT"})


def test_residual_reversion_longs_loser_and_shorts_winner_after_costs():
    prices = {
        symbol: {hour: 100.0 for hour in range(26)}
        for symbol in ("BTCUSDT", "AUSDT", "BUSDT", "CUSDT", "DUSDT")
    }
    # At hour 24 A is the largest loser and D the largest winner.  Both
    # converge at hour 25, so a long-A / short-D portfolio profits.
    prices["AUSDT"][24] = 90.0
    prices["BUSDT"][24] = 95.0
    prices["CUSDT"][24] = 105.0
    prices["DUSDT"][24] = 110.0

    trades = simulate_config(
        prices,
        {},
        StrategyConfig(1, 1),
        fee_bps=4.0,
        slippage_bps=1.0,
        include_funding=False,
        beta_window_h=24,
    )
    assert len(trades) == 1
    assert trades[0].long_symbols == ("AUSDT",)
    assert trades[0].short_symbols == ("DUSDT",)
    assert trades[0].cost_bps == 10.0
    assert trades[0].net_bps > 0


def test_positive_funding_is_paid_by_long_and_received_by_short():
    prices = {
        symbol: {hour: 100.0 for hour in range(26)}
        for symbol in ("BTCUSDT", "AUSDT", "BUSDT", "CUSDT", "DUSDT")
    }
    prices["AUSDT"][24] = 90.0
    prices["BUSDT"][24] = 95.0
    prices["CUSDT"][24] = 105.0
    prices["DUSDT"][24] = 110.0
    # Prefix arrays correspond to one settlement after entry, at hour 26.
    funding = {
        "AUSDT": ([26], [0.0, 0.001]),
        "DUSDT": ([26], [0.0, 0.002]),
    }
    trades = simulate_config(
        prices,
        funding,
        StrategyConfig(1, 1),
        fee_bps=0.0,
        slippage_bps=0.0,
        include_funding=True,
        beta_window_h=24,
    )
    # Half-weight long pays 10bps, half-weight short receives 20bps.
    assert trades[0].funding_bps == pytest.approx(5.0)


def _loser_prices():
    prices = {
        symbol: {hour: 100.0 for hour in range(26)}
        for symbol in ("BTCUSDT", "AUSDT", "BUSDT", "CUSDT", "DUSDT")
    }
    prices["AUSDT"][24] = 90.0
    prices["BUSDT"][24] = 99.0
    prices["CUSDT"][24] = 100.0
    prices["DUSDT"][24] = 101.0
    return prices


def test_loser_btc_only_buys_extreme_loser_and_hedges_with_btc():
    trades = simulate_config(
        _loser_prices(),
        {},
        StrategyConfig(1, 1, strategy="loser_btc", entry_z=1.0),
        fee_bps=4.0,
        slippage_bps=1.0,
        include_funding=False,
        beta_window_h=24,
    )
    assert len(trades) == 1
    assert trades[0].long_symbols == ("AUSDT",)
    assert trades[0].short_symbols == ("BTCUSDT",)
    assert trades[0].net_bps > 0


def test_loser_btc_executes_at_next_bar_open_not_observed_signal_close():
    prices = _loser_prices()
    execution = {symbol: dict(series) for symbol, series in prices.items()}
    # The signal observes A at 90, but the first tradable price is 95.
    execution["AUSDT"][24] = 95.0
    trades = simulate_config(
        prices,
        {},
        StrategyConfig(1, 1, strategy="loser_btc", entry_z=1.0),
        execution_prices=execution,
        fee_bps=0.0,
        slippage_bps=0.0,
        include_funding=False,
        beta_window_h=24,
    )
    assert trades[0].long_price_bps == pytest.approx(0.5 * (100 / 95 - 1) * 10_000)


def test_target_vol_only_scales_risk_down_and_scales_cost_with_it():
    trades = simulate_config(
        _loser_prices(),
        {},
        StrategyConfig(1, 1, strategy="loser_btc", entry_z=1.0),
        fee_bps=4.0,
        slippage_bps=1.0,
        include_funding=False,
        target_vol_pct=20.0,
        beta_window_h=24,
    )
    assert 0.0 < trades[0].gross_scale < 1.0
    assert trades[0].cost_bps == pytest.approx(10.0 * trades[0].gross_scale)


def test_loser_btc_holds_cash_when_no_residual_crosses_entry_z():
    trades = simulate_config(
        _loser_prices(),
        {},
        StrategyConfig(1, 1, strategy="loser_btc", entry_z=3.0),
        fee_bps=4.0,
        slippage_bps=1.0,
        include_funding=False,
        beta_window_h=24,
    )
    assert trades == []


def test_simulation_never_trades_symbol_outside_point_in_time_universe():
    trades = simulate_config(
        _loser_prices(),
        {},
        StrategyConfig(1, 1, strategy="loser_btc", entry_z=1.0),
        fee_bps=0.0,
        slippage_bps=0.0,
        include_funding=False,
        beta_window_h=24,
        eligible_by_hour={24: frozenset({"BUSDT", "CUSDT", "DUSDT"})},
    )
    assert trades == []

    no_universe = simulate_config(
        _loser_prices(),
        {},
        StrategyConfig(1, 1, strategy="loser_btc", entry_z=1.0),
        fee_bps=0.0,
        slippage_bps=0.0,
        include_funding=False,
        beta_window_h=24,
        eligible_by_hour={},
    )
    assert no_universe == []


def _negative_return(hour):
    return TradeReturn(
        hour=hour,
        price_bps=9.0,
        long_price_bps=9.0,
        short_price_bps=0.0,
        funding_bps=0.0,
        cost_bps=10.0,
        net_bps=-1.0,
        gross_scale=1.0,
        long_symbols=("AUSDT",),
        short_symbols=("BTCUSDT",),
        symbol_bps={"AUSDT": -1.0},
    )


def test_walk_forward_chooses_cash_instead_of_least_bad_strategy():
    results = []
    for lookback in (6, 12):
        rows = [_negative_return(hour) for hour in range(0, 1001, 10)]
        results.append(
            summarise_performance(
                StrategyConfig(lookback, 24, strategy="loser_btc", entry_z=1.0),
                rows,
                observed_days=42,
            )
        )
    folds, summary = walk_forward(results, min_train_trades=1)
    assert all(fold.selected is None for fold in folds)
    assert all(fold.test_trades == 0 for fold in folds)
    assert summary.total_bps == 0.0


def test_compound_account_stops_after_bankruptcy():
    fatal = _negative_return(1)
    fatal.net_bps = -11_000.0
    fatal.price_bps = -10_990.0
    recovery = _negative_return(2)
    recovery.net_bps = 50_000.0
    result = summarise_performance(
        StrategyConfig(6, 24), [fatal, recovery], observed_days=30
    )
    assert result.bankrupt is True
    assert result.final_equity == 0.0
    assert result.total_bps == -10_000.0
    assert result.trades == 1


def test_load_dataset_explains_required_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="statarb download"):
        load_dataset(tmp_path)


def test_manifest_is_plain_json_contract(tmp_path):
    manifest = {
        "version": 1,
        "start": "2024-01-01",
        "end": "2024-01-02",
        "interval": "5m",
        "universe": [],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    loaded, prices, execution, quote_volume, funding = load_dataset(tmp_path)
    assert loaded == manifest
    assert prices == {}
    assert execution == {}
    assert quote_volume == {}
    assert funding == {}
