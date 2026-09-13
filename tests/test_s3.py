"""Shipping a running recording to S3.

The failure that matters is silent data loss: a part that never uploads, or
one deleted locally because an upload was assumed to have worked. Those cases
carry the weight here; the key layout is checked because a wrong partition
makes a day of recording unqueryable rather than merely untidy.
"""

import gzip
import json

import pytest

from jsboard.sim.s3 import RotatingJsonlSink, S3Target

NS = 1_000_000_000


class FakeS3:
    """Records uploads, and can be told to fail."""

    def __init__(self, fail_on=()):
        self.objects: dict[str, bytes] = {}
        self.fail_on = set(fail_on)
        self.calls = 0

    def upload_file(self, filename, bucket, key):
        self.calls += 1
        if key in self.fail_on or "*" in self.fail_on:
            raise RuntimeError("bucket unreachable")
        with open(filename, "rb") as fh:
            self.objects[f"{bucket}/{key}"] = fh.read()


class Clock:
    def __init__(self, now=1_757_000_000 * NS):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += int(seconds * NS)


def sink(tmp_path, client, clock=None, **kw):
    return RotatingJsonlSink(
        path=tmp_path / "cap.jsonl",
        target=S3Target(bucket="b", prefix="raw"),
        symbol="btcusdt",
        client=client,
        clock=clock or Clock(),
        **kw,
    )


def decompress(client, key):
    return gzip.decompress(client.objects[f"b/{key}"]).decode()


class TestKeyLayout:
    def test_partitions_by_symbol_date_and_hour(self):
        target = S3Target(bucket="b", prefix="raw")
        # 2026-09-09 07:00:00 UTC
        key = target.key_for("btcusdt", 1_788_930_000 * NS, seq=1)
        assert key.startswith("raw/symbol=BTCUSDT/date=")
        assert "/hour=" in key
        assert key.endswith("1788930000000000000-00001.jsonl.gz")

    def test_parts_closed_in_the_same_tick_get_different_keys(self):
        # A size-triggered rotation can close two parts inside one clock tick.
        # Sharing a key there would overwrite the first, and both uploads
        # would still report success.
        target = S3Target(bucket="b")
        assert target.key_for("X", NS, 1) != target.key_for("X", NS, 2)

    def test_a_prefix_with_slashes_does_not_double_them(self):
        key = S3Target(bucket="b", prefix="/data/raw/").key_for("X", NS)
        assert key.startswith("data/raw/symbol=X/")
        assert "//" not in key


class TestRotation:
    def test_time_closes_a_part_and_uploads_it(self, tmp_path):
        client, clock = FakeS3(), Clock()
        with sink(tmp_path, client, clock, rotate_seconds=60) as fh:
            fh.write('{"a":1}\n')
            clock.advance(61)
            fh.write('{"a":2}\n')
        assert client.calls == 2  # the rotated part, then the final one

    def test_size_closes_a_part_too(self, tmp_path):
        client = FakeS3()
        with sink(tmp_path, client, rotate_seconds=0, rotate_bytes=20) as fh:
            for i in range(5):
                fh.write(json.dumps({"i": i}) + "\n")
        assert client.calls == 2
        # Both parts survive: a frozen clock must not collapse them onto one key.
        assert len(client.objects) == 2

    def test_the_uploaded_object_holds_the_lines_it_was_given(self, tmp_path):
        client, clock = FakeS3(), Clock()
        with sink(tmp_path, client, clock, rotate_seconds=60) as fh:
            fh.write('{"a":1}\n')
        key = client.objects and next(iter(client.objects))
        assert gzip.decompress(client.objects[key]).decode() == '{"a":1}\n'

    def test_nothing_written_uploads_nothing(self, tmp_path):
        client = FakeS3()
        with sink(tmp_path, client):
            pass
        assert client.calls == 0
        assert not list(tmp_path.glob("*.jsonl"))

    def test_parts_do_not_overwrite_each_other(self, tmp_path):
        client, clock = FakeS3(), Clock()
        with sink(tmp_path, client, clock, rotate_seconds=60) as fh:
            fh.write("first\n")
            clock.advance(61)
            fh.write("second\n")
        bodies = {gzip.decompress(v).decode() for v in client.objects.values()}
        assert bodies == {"first\n", "second\n"}


class TestLocalCopies:
    def test_a_successful_upload_removes_the_local_part(self, tmp_path):
        with sink(tmp_path, FakeS3()) as fh:
            fh.write("x\n")
        assert not list(tmp_path.glob("*.jsonl"))
        assert not list(tmp_path.glob("*.gz"))

    def test_keep_local_leaves_it_behind(self, tmp_path):
        with sink(tmp_path, FakeS3(), keep_local=True) as fh:
            fh.write("x\n")
        assert len(list(tmp_path.glob("*.jsonl"))) == 1

    def test_a_failed_upload_never_deletes_the_evidence(self, tmp_path):
        client = FakeS3(fail_on=["*"])
        with sink(tmp_path, client) as fh:
            fh.write("x\n")
        # The bucket has nothing, so the only copy is the one on disk.
        assert client.objects == {}
        assert len(list(tmp_path.glob("*.jsonl"))) == 1

    def test_a_failure_is_reported_rather_than_swallowed(self, tmp_path):
        s = sink(tmp_path, FakeS3(fail_on=["*"]))
        with s as fh:
            fh.write("x\n")
        assert s.summary()["failed"]
        assert not s.summary()["uploaded"]

    def test_a_failure_does_not_stop_the_recording(self, tmp_path):
        client, clock = FakeS3(fail_on=["*"]), Clock()
        s = sink(tmp_path, client, clock, rotate_seconds=60)
        with s as fh:
            fh.write("a\n")
            clock.advance(61)
            fh.write("b\n")  # still writing after the first part failed
        assert len(s.summary()["failed"]) == 2


class TestCredentials:
    def test_the_sink_takes_no_key_arguments(self):
        import inspect

        params = set(inspect.signature(RotatingJsonlSink).parameters)
        assert not (params & {"aws_access_key_id", "aws_secret_access_key", "token"})

    def test_an_unusable_credential_chain_becomes_one_readable_line(self, monkeypatch):
        # boto3 raises its own exception types here, and the CLI only knows how
        # to print a RuntimeError. Without the translation the user gets a
        # 30-line traceback whose one useful line is "run aws configure".
        import sys
        import types

        from jsboard.sim import s3

        class Boom(Exception):
            pass

        fake = types.ModuleType("boto3")
        fake.client = lambda *a, **kw: (_ for _ in ()).throw(Boom("no credentials"))
        monkeypatch.setitem(sys.modules, "boto3", fake)

        with pytest.raises(RuntimeError, match="aws configure"):
            s3.default_client()


class TestFileLikeContract:
    """MultiCapture writes to this as if it were an open file."""

    def test_write_returns_a_count_and_flush_is_safe(self, tmp_path):
        with sink(tmp_path, FakeS3()) as fh:
            assert fh.write("abc\n") == 4
            fh.flush()

    def test_close_is_idempotent(self, tmp_path):
        s = sink(tmp_path, FakeS3())
        s.write("x\n")
        s.close()
        s.close()

    @pytest.mark.asyncio
    async def test_a_capture_writes_through_it(self, tmp_path):
        from decimal import Decimal

        from jsboard.core.types import Instrument, Side
        from jsboard.feed.base import Feed, TradeTick
        from jsboard.sim.capture import MultiCapture

        class OneTrade(Feed):
            async def stream(self):
                yield TradeTick(price=1, qty=1, aggressor=Side.BUY)

        inst = Instrument("BTCUSDT", Decimal("0.01"), Decimal("0.00001"))
        client = FakeS3()
        s = sink(tmp_path, client)
        feed = OneTrade(inst)
        await MultiCapture({"perp": feed}, tmp_path / "cap.jsonl", sink=s).run(max_events=1)
        s.close()
        assert client.calls == 1
        body = next(iter(client.objects.values()))
        assert b'"k": "trade"' in gzip.decompress(body)


class TestRecordingsInTheBucket:
    """A bucket you can only write to does not save the disk it was for.

    The whole reason for having somewhere else to put a day of data is that
    the laptop cannot hold it. A downloader that writes locally and then
    uploads still needs the local disk for the entire file — so a recording
    has to be readable straight out of the bucket, not merely archivable to
    it.
    """

    def test_a_uri_splits_into_bucket_and_key(self):
        from jsboard.sim.s3 import split_uri

        assert split_uri("s3://jsboard-capture/history/x.jsonl.gz") == (
            "jsboard-capture",
            "history/x.jsonl.gz",
        )

    def test_a_uri_without_a_key_is_refused(self):
        import pytest

        from jsboard.sim.s3 import split_uri

        with pytest.raises(ValueError):
            split_uri("s3://jsboard-capture")

    def test_a_local_path_is_not_mistaken_for_a_uri(self):
        from jsboard.sim.s3 import is_s3_uri

        assert not is_s3_uri("x.jsonl.gz")
        assert not is_s3_uri("/tmp/s3/x.jsonl")
        assert is_s3_uri("s3://b/k")

    def test_the_meta_file_sits_beside_the_object_either_way(self):
        from pathlib import Path

        from jsboard.sim.s3 import meta_uri

        assert meta_uri("s3://b/h/x.jsonl.gz") == "s3://b/h/x.jsonl.gz.meta.json"
        assert meta_uri(Path("x.jsonl.gz")) == "x.jsonl.gz.meta.json"

    def test_a_local_gzip_recording_still_reads(self, tmp_path):
        import gzip

        from jsboard.sim.s3 import open_text

        path = tmp_path / "r.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write('{"a":1}\n')
        with open_text(path) as fh:
            assert fh.read() == '{"a":1}\n'

    def test_a_gzipped_object_is_streamed_and_decompressed(self, monkeypatch):
        import gzip
        import io

        import jsboard.sim.s3 as s3

        body = gzip.compress(b'{"a":1}\n')

        class FakeClient:
            def get_object(self, Bucket, Key):  # noqa: N803 - boto3's spelling
                assert (Bucket, Key) == ("b", "h/x.jsonl.gz")
                return {"Body": io.BytesIO(body)}

        monkeypatch.setattr(s3, "default_client", FakeClient)
        with s3.open_text("s3://b/h/x.jsonl.gz") as fh:
            assert fh.read() == '{"a":1}\n'

    def test_an_uncompressed_object_is_read_as_text(self, monkeypatch):
        import io

        import jsboard.sim.s3 as s3

        class FakeClient:
            def get_object(self, Bucket, Key):  # noqa: N803
                return {"Body": io.BytesIO(b'{"a":1}\n')}

        monkeypatch.setattr(s3, "default_client", FakeClient)
        with s3.open_text("s3://b/h/x.jsonl") as fh:
            assert fh.read() == '{"a":1}\n'

    def test_a_replay_reads_a_recording_out_of_the_bucket(self, tmp_path, monkeypatch):
        """The reader the whole study goes through, pointed at an object."""
        import gzip
        import io

        import jsboard.sim.s3 as s3
        from jsboard.feed.replay import iter_tagged

        local = tmp_path / "r.jsonl.gz"
        with gzip.open(local, "wt", encoding="utf-8") as fh:
            fh.write(
                '{"k":"status","state":"live","detail":"archive","ts_ns":1,'
                '"src":"binance","rx_ns":1}\n'
            )
        blob = local.read_bytes()

        class FakeClient:
            def get_object(self, Bucket, Key):  # noqa: N803
                return {"Body": io.BytesIO(blob)}

        monkeypatch.setattr(s3, "default_client", FakeClient)
        assert [src for src, _ in iter_tagged("s3://b/h/r.jsonl.gz")] == ["binance"]
