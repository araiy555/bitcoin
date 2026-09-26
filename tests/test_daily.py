"""The morning check: yesterday's book, fixed settings, and a warning when it stops paying."""

import gzip
import json
import math
from decimal import Decimal

import pytest

from jsboard.research.daily import DayResult, Target, losing_streak, size_for, slack_text


def test_targets_name_a_known_venue():
    assert Target.parse("bitbank:ada_jpy") == Target("bitbank", "ada_jpy")
    with pytest.raises(ValueError):
        Target.parse("okx:ada")


def test_an_order_is_sized_in_yen_on_the_lot_grid():
    assert size_for(10_000, 37.2, Decimal("0.0001")) == "268.8172"
    assert size_for(10_000, 241.5, Decimal("10")) == "40"
    assert size_for(10, 50_000, Decimal("0.01")) == "0.01"  # never below one lot


def history(*days):
    return [{"results": [{"target": "bitbank:ada_jpy", "short_bps": v}]} for v in days]


def test_a_streak_counts_consecutive_negative_days_ending_today():
    assert losing_streak("bitbank:ada_jpy", -0.5, history(-1.0, -0.2, 3.0)) == 3
    assert losing_streak("bitbank:ada_jpy", -0.5, history(2.0)) == 1
    assert losing_streak("bitbank:ada_jpy", 0.4, history(-1.0)) == 0


def test_two_bad_days_raise_a_warning():
    ada = Target("bitbank", "ada_jpy")
    text = slack_text("2026-09-26", [DayResult(ada, 900, -0.3, -1.0)], history(-0.1))
    assert "2日連続マイナス" in text
    calm = slack_text("2026-09-26", [DayResult(ada, 900, 2.1, 0.8)], history(-0.1))
    assert "連続" not in calm


def test_a_missing_recording_says_so():
    text = slack_text("d", [DayResult(Target("bitbank", "sui_jpy"), note="録画なし")], [])
    assert "sui_jpy: 録画なし" in text


@pytest.mark.asyncio
async def test_the_morning_run_replays_a_day_and_files_the_report(monkeypatch, capsys):
    import io

    import jsboard.sim.s3 as s3
    from jsboard.cli import build_parser
    from jsboard.core.types import Side
    from jsboard.feed.base import DepthSnapshot, TradeTick
    from jsboard.feed.replay import _encode
    from jsboard.sim.s3 import S3Target

    ns = 1_000_000_000
    base = 1_790_380_800 * ns  # 2026-09-26 00:00 UTC
    bb = {"symbol": "ada_jpy", "tick_size": "0.001", "lot_size": "0.0001",
          "base": "ADA", "quote": "JPY", "market": "spot"}
    bn = {"symbol": "ADAUSDT", "tick_size": "0.0001", "lot_size": "1",
          "base": "ADA", "quote": "USDT", "market": "perp"}
    rows = [{"k": "status", "state": "live", "detail": "", "ts_ns": base, "src": "bitbank"}]
    for i in range(400):
        ts = base + i * 250_000_000
        mid = 37_200 + (i // 50) % 3
        b = _encode(DepthSnapshot(tuple((mid - 5 - j, 9_000_000) for j in range(5)),
                                  tuple((mid + 5 + j, 9_000_000) for j in range(5)), i, ts))
        b.update(src="bitbank", rx_ns=ts)
        # Prints that sweep a few levels, so a quote 2bps from mid is reached.
        t = _encode(TradeTick(mid + 12 if i % 2 else mid - 12, 5_000_000,
                              Side.BUY if i % 2 else Side.SELL, i, ts))
        t.update(src="bitbank", rx_ns=ts)
        n = _encode(DepthSnapshot(tuple((2480 - j, 90_000) for j in range(5)),
                                  tuple((2481 + j, 90_000) for j in range(5)), i, ts))
        n.update(src="binance", rx_ns=ts)
        rows += [b, t, n]
    target = S3Target(bucket="b", prefix="raw/live")
    objects = {
        target.key_for("ADA_JPY", base, 1): gzip.compress(
            ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
        ),
        target.meta_key("ADA_JPY"): json.dumps({"sources": {"bitbank": bb, "binance": bn}}).encode(),
    }

    class Bucket:
        def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):  # noqa: N803
            keys = sorted(k for k in objects if k.startswith(Prefix))
            return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": io.BytesIO(objects[Key])}

        def head_object(self, Bucket, Key):  # noqa: N803
            if Key not in objects:
                raise KeyError(Key)

        def put_object(self, Bucket, Key, Body):  # noqa: N803
            objects[Key] = Body

    monkeypatch.setattr(s3, "default_client", Bucket)
    args = build_parser().parse_args(["daily", "--s3-bucket", "b", "--date", "2026-09-26"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "bitbank ada_jpy: 10秒内計" in out
    report = json.loads(objects["reports/daily/2026-09-26.json"])
    [row] = report["results"]
    assert row["target"] == "bitbank:ada_jpy"
    assert row["note"] == "" and row["fills"] > 0, row
    assert row["short_bps"] is not None
