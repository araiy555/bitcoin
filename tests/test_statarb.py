import csv
import io
import json
import zipfile
from datetime import date

import pytest

from jsboard.cli import build_parser
from jsboard.research.statarb import (
    StrategyConfig,
    archive_parts,
    kline_url,
    load_dataset,
    load_hourly_closes,
    parse_hours,
    simulate_config,
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
            "--funding",
            "--walk-forward",
        ]
    )
    assert download.func.__name__ == "cmd_statarb_download"
    assert backtest.func.__name__ == "cmd_statarb_backtest"
    assert backtest.funding is True
    assert backtest.walk_forward is True


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
        [1704067200000, 100, 101, 99, 100.5],  # 00:00
        [1704070500000, 100, 102, 99, 101.5],  # 00:55
        [1704070800000, 101, 103, 100, 102.5],  # 01:00
    ]
    buffer = io.StringIO()
    csv.writer(buffer).writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("BTCUSDT-5m.csv", buffer.getvalue())

    prices = load_hourly_closes([path], date(2024, 1, 1), date(2024, 1, 1))
    first_hour = 1704067200 // 3600
    assert prices[first_hour] == 101.5
    assert prices[first_hour + 1] == 102.5


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
    # Prefix arrays correspond to one settlement at hour 25.
    funding = {
        "AUSDT": ([25], [0.0, 0.001]),
        "DUSDT": ([25], [0.0, 0.002]),
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
    loaded, prices, funding = load_dataset(tmp_path)
    assert loaded == manifest
    assert prices == {}
    assert funding == {}
