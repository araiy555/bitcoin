"""Record the day's best-screened books without being asked."""

import json
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot, Feed, FeedStatus
from jsboard.research.daily import Target
from jsboard.research.jpscan import Book


class Quiet(Feed):
    async def stream(self):
        yield FeedStatus("live", "fake")
        yield DepthSnapshot(((100, 5),), ((101, 5),), 1)


class Bucket:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)

    def get_object(self, Bucket, Key):  # noqa: N803
        import io

        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[Key] = Body

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as fh:
            self.objects[key] = fh.read()


@pytest.mark.asyncio
async def test_a_day_records_always_plus_top_candidates_and_files_the_list(
    monkeypatch, tmp_path
):
    import jsboard.cli as cli
    import jsboard.sim.s3 as s3

    bucket = Bucket()
    monkeypatch.setattr(s3, "default_client", lambda: bucket)

    def book(venue, symbol, spread, volume, taker=12.0):
        return Book(venue, symbol, -2.0, taker, [spread], volume_jpy=volume)

    async def screen(samples, interval):
        return [
            book("bitbank", "sui_jpy", 9.0, 2e8),
            book("gmo", "SOL", 6.0, 3e8, taker=9.0),
            book("gmo", "XRP_JPY", 7.0, 5e9, taker=0.0),  # free taking: never picked
            book("bitbank", "xlm_jpy", 8.0, 1.8e8),
        ]

    recorded = []

    async def parts(target, binance_depth_ms=100):
        recorded.append(target)
        inst = Instrument(target.symbol, Decimal("0.001"), Decimal("1"), "X", "JPY")
        spec = {"symbol": target.symbol, "tick_size": "0.001", "lot_size": "1",
                "base": "X", "quote": "JPY", "market": "spot"}
        return ({target.venue: Quiet(inst), "binance": Quiet(inst)},
                {target.venue: spec, "binance": spec})

    monkeypatch.setattr(cli, "_screen_books", screen)
    monkeypatch.setattr(cli, "_lead_capture_parts", parts)
    args = cli.build_parser().parse_args([
        "autorecord", "--s3-bucket", "b", "--always", "bitbank:ada_jpy",
        "--exclude", "bitbank:sui_jpy", "--top", "2",
        "--workdir", str(tmp_path), "--duration", "0.2", "--days", "1",
    ])
    assert await args.func(args) == 0

    assert recorded == [
        Target("bitbank", "ada_jpy"), Target("gmo", "SOL"), Target("bitbank", "xlm_jpy")
    ]
    [listed] = [k for k in bucket.objects if k.startswith("control/targets/")]
    assert json.loads(bucket.objects[listed])["targets"] == [
        "bitbank:ada_jpy", "bitbank:xlm_jpy", "gmo:SOL"
    ]
    assert "raw/live/symbol=SOL/meta.json" in bucket.objects
