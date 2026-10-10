"""Quoting around the fair price on a replayed book, estimated cautiously."""

from decimal import Decimal

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.research.fairmm import run
from jsboard.research.fairprice import feature_names

GMO = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
PERP = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")
NS = 1_000_000_000
T0 = 1_790_380_800 * NS
MODEL = {"weights": {"2": [0.0] * len(feature_names(()))}, "cross": [], "hinge_bps": 8.0}


def book(bid, ask, t, qty=100):
    return DepthSnapshot(((bid, qty),), ((ask, qty),), 1, t)


def tape(gmo_at, prints, seconds=8):
    """Books every 100 ms (GMO's from `gmo_at(t)`), prints at given times."""
    rows = []
    for i in range(seconds * 10):
        t = T0 + i * NS // 10 + 1
        bid, ask = gmo_at(i)
        rows.append(("gmo", t, book(bid, ask, t)))
        rows.append(("bybit", t, book(10_000, 10_001, t)))
    for at_s, side, price, qty in prints:
        t = T0 + int(at_s * NS) + 2
        rows.append(("gmo", t, TradeTick(price, qty, side, ts_ns=t)))
    rows.sort(key=lambda r: r[1])
    return rows


def go(rows, **kw):
    kw.setdefault("widths", (0.4999,))
    return run(iter(rows), GMO, {"bybit": PERP}, MODEL, label="XRP",
               test=(T0 + NS, T0 + 100 * NS), shrink=0.0, place_ms=200, **kw)


def test_the_queue_ahead_must_trade_away_before_we_fill():
    # Our bid joins 100 coins at 100.000; 60 then 50 sell at that price.
    rows = tape(lambda i: (100_000, 100_010),
                [(3.0, Side.SELL, 100_000, 60), (4.0, Side.SELL, 100_000, 50)])
    r = go(rows, cancel_ms=(200.0,))
    [bk] = r.books
    assert bk.fills == 1 and abs(bk.inv - 10.0) < 1e-9


def test_a_print_through_our_level_fills_us():
    rows = tape(lambda i: (100_000, 100_010), [(3.0, Side.SELL, 99_990, 5)])
    [bk] = go(rows, cancel_ms=(200.0,)).books
    assert bk.fills == 1 and abs(bk.inv - 5.0) < 1e-9


def test_a_slow_cancel_is_picked_off_and_it_shows():
    # At 3 s the book drops one tick-width; our old bid at 100.000 is being
    # cancelled. Half a second later a seller hits 99.990: only the slow
    # cancel is still there to be filled, above the market.
    rows = tape(lambda i: (100_000, 100_010) if i < 30 else (99_990, 100_000),
                [(3.5, Side.SELL, 99_990, 60)])
    r = go(rows, cancel_ms=(200.0, 1000.0))
    fast, slow = r.books
    assert fast.late == 0 and fast.fills == 0
    assert slow.late == 1 and slow.fills == 1
    assert slow.after[1.0][1] / slow.after[1.0][0] < 0   # bought above the mid that followed


def test_inventory_stops_the_side_that_would_grow_it():
    prints = [(2.0 + k * 0.5, Side.SELL, 99_990, 10) for k in range(10)]
    rows = tape(lambda i: (100_000, 100_010), prints)
    [bk] = go(rows, cancel_ms=(200.0,), max_inv=20.0).books
    assert bk.inv <= 20.0 + 1e-9 and bk.max_inv == 20.0


async def test_the_command_starts_where_the_model_stopped_learning(monkeypatch, capsys, tmp_path):
    import gzip
    import io
    import json
    from datetime import UTC, datetime

    import jsboard.cli as cli
    import jsboard.sim.s3 as s3
    from jsboard.feed.replay import _encode
    from jsboard.sim.s3 import S3Target

    rows = tape(lambda i: (100_000, 100_010), [(3.0, Side.SELL, 99_990, 5)])
    lines = []
    for src, rx, event in rows:
        r = _encode(event)
        r.update(src=src, rx_ns=rx)
        lines.append(json.dumps(r))
    target = S3Target(bucket="b", prefix="raw/lead")
    spec = {"base": "XRP", "quote": "JPY"}
    objects = {
        target.key_for("XRP_JPY", T0, 1): gzip.compress(("\n".join(lines) + "\n").encode()),
        target.meta_key("XRP_JPY"): json.dumps({"sources": {
            "gmo": {**spec, "symbol": "XRP_JPY", "tick_size": "0.001", "lot_size": "1"},
            "bybit": {**spec, "symbol": "XRPUSDT", "tick_size": "0.0001", "lot_size": "1"},
        }}).encode(),
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

    def iso(ns):
        return datetime.fromtimestamp(ns / NS, UTC).isoformat()

    model = tmp_path / "XRP_JPY.json"
    model.write_text(json.dumps({**MODEL, "symbol": "XRP_JPY",
                                 "train": [f"{iso(T0 - 3600 * NS)},{iso(T0 + NS)}"]}))
    monkeypatch.setattr(s3, "default_client", Bucket)
    args = cli.build_parser().parse_args([
        "fairmm", "s3://b/raw/lead/", "--model", str(model), "--widths", "0.4999",
        "--cancel-ms", "200", "--shrink", "0",
        "--test", f"{iso(T0 + NS)},{iso(T0 + 100 * NS)}"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out.splitlines()
    row = next(x for x in out if x.startswith("200ms")).split("\t")
    assert row[4] == "1"   # one estimated fill
