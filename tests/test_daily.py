"""The morning check: yesterday's book, fixed settings, and a warning when it stops paying."""

import gzip
import json
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
        # A trial is replayed beside the fixed settings, not instead of them.
        "control/settings/bitbank_ada_jpy.json": json.dumps(
            {"settings": {"inventory_skew_bps": 20}}
        ).encode(),
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

        def delete_object(self, Bucket, Key):  # noqa: N803
            objects.pop(Key, None)

    monkeypatch.setattr(s3, "default_client", Bucket)
    args = build_parser().parse_args(["daily", "--s3-bucket", "b", "--date", "2026-09-26"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "bitbank ada_jpy: *損益" in out
    report = json.loads(objects["reports/daily/2026-09-26.json"])
    [row] = report["results"]
    assert row["target"] == "bitbank:ada_jpy"
    assert row["note"] == "" and row["fills"] > 0, row
    assert row["short_bps"] is not None
    assert row["trial"]["settings"] == "在庫の片寄せ 20bps"
    assert row["trial"]["fills"] > 0
    assert "試験中（在庫の片寄せ 20bps）: *損益" in out
    assert row["pnl"] is not None and f"損益 {row['pnl']:+,}円" in out


def daily_history(*edges):
    return [{"results": [{"target": "bitbank:ada_jpy", "short_bps": e}]} for e in edges]


class TestReadiness:
    def test_ten_good_days_and_paper_ahead_is_ready(self):
        from jsboard.research.daily import readiness

        r = readiness("bitbank:ada_jpy", 3.0, daily_history(2, 1, -1, 4, 2, 3, 1, -0.5, 2), 5_000)
        assert (r.days, r.positive, r.streak_seen) == (10, 8, False)
        assert r.ready and "満たしました" in r.text()

    def test_a_two_day_streak_anywhere_in_the_window_blocks_it(self):
        from jsboard.research.daily import readiness

        r = readiness("bitbank:ada_jpy", 3.0, daily_history(2, -1, -1, 4, 2, 3, 1, 2, 2), 5_000)
        assert r.streak_seen and not r.ready

    def test_too_few_days_says_how_many_more(self):
        from jsboard.research.daily import readiness

        r = readiness("bitbank:ada_jpy", 3.0, daily_history(2, 1), 900)
        assert not r.ready and "あと 7 日分" in r.text()

    def test_paper_behind_blocks_it(self):
        from jsboard.research.daily import readiness

        r = readiness("bitbank:ada_jpy", 3.0, daily_history(*[2] * 9), -10)
        assert not r.ready


class FlagBucket:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


def test_halt_and_resume_set_and_clear_the_flag(monkeypatch):
    import asyncio

    import jsboard.sim.s3 as s3
    from jsboard.cli import build_parser

    bucket = FlagBucket()
    monkeypatch.setattr(s3, "default_client", lambda: bucket)
    def run(argv):
        args = build_parser().parse_args(argv)
        return asyncio.run(args.func(args))

    assert run(["halt", "--target", "bitbank:ada_jpy", "--s3-bucket", "b"]) == 0
    assert "control/halt/bitbank_ada_jpy.json" in bucket.objects
    assert run(["resume", "--target", "bitbank:ada_jpy", "--s3-bucket", "b"]) == 0
    assert bucket.objects == {}


def test_a_trial_that_trails_two_days_is_due_for_removal():
    from jsboard.research.daily import trial_trailing_days

    def day(fixed, trial):
        return {"target": "bitbank:ada_jpy", "after_fees_bps": fixed,
                "trial": {"after_fees_bps": trial}}

    key = "bitbank:ada_jpy"
    yesterday = [{"results": [day(1.0, 0.5)]}]
    assert trial_trailing_days(key, day(1.8, 1.2), yesterday) == 2
    assert trial_trailing_days(key, day(1.0, 1.5), yesterday) == 0  # ahead today
    assert trial_trailing_days(key, day(1.8, 1.2), [{"results": [day(1.0, 2.0)]}]) == 1
    assert trial_trailing_days(key, {"target": key, "after_fees_bps": 1.0}, yesterday) == 0


def test_tune_starts_shows_and_resets_a_trial(monkeypatch, capsys):
    import asyncio
    import io

    import jsboard.sim.s3 as s3
    from jsboard.cli import build_parser

    bucket = FlagBucket()
    bucket.get_object = lambda Bucket, Key: {"Body": io.BytesIO(bucket.objects[Key])}  # noqa: N803
    monkeypatch.setattr(s3, "default_client", lambda: bucket)

    def run(*argv):
        args = build_parser().parse_args(
            ["tune", "--target", "bitbank:ada_jpy", "--s3-bucket", "b", *argv]
        )
        return asyncio.run(args.func(args))

    assert run("--set", "inventory_skew_bps=20") == 0
    stored = json.loads(bucket.objects["control/settings/bitbank_ada_jpy.json"])
    assert stored["settings"] == {"inventory_skew_bps": 20.0}
    assert run() == 0
    assert "在庫の片寄せ 20bps" in capsys.readouterr().out
    assert run("--reset") == 0
    assert bucket.objects == {}
    with pytest.raises(Exception, match="変えられる設定"):
        run("--set", "gamma=3")
