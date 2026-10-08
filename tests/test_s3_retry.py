"""A dropped connection while reading a recording part is retried."""

import gzip

import jsboard.sim.s3 as s3


def test_a_part_is_fetched_again_after_a_reset(monkeypatch):
    calls = {"n": 0}
    blob = gzip.compress(b'{"a":1}\n{"a":2}\n')

    class Body:
        def read(self):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionResetError(104, "Connection reset by peer")
            return blob

    class Client:
        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": Body()}

    monkeypatch.setattr(s3, "default_client", Client)
    waits = []
    fh = s3._open_part("s3://b/raw/x/part.jsonl.gz", sleep=waits.append)
    assert fh.read().splitlines() == ['{"a":1}', '{"a":2}']
    assert calls["n"] == 3 and waits == [1, 2]


def test_it_gives_up_after_the_last_try(monkeypatch):
    class Body:
        def read(self):
            raise ConnectionResetError(104, "reset")

    class Client:
        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": Body()}

    monkeypatch.setattr(s3, "default_client", Client)
    try:
        s3._open_part("s3://b/k.gz", tries=2, sleep=lambda s: None)
    except ConnectionResetError:
        return
    raise AssertionError("expected the error after the last try")
