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
    kw.setdefault("requote_bps", (0.5,))
    kw.setdefault("test", (T0 + NS, T0 + 100 * NS))
    return run(iter(rows), GMO, {"bybit": PERP}, MODEL, label="XRP",
               shrink=0.0, place_ms=200, **kw)


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


def test_a_quote_stays_put_until_its_target_moves_far_enough():
    # GMO's book wobbles by 3 ticks (0.3 bps) every 100 ms: requoting on
    # every change chases it, a 0.5 bps threshold leaves the quotes alone.
    rows = tape(lambda i: (100_000, 100_010) if i % 2 else (100_003, 100_013), [])
    every, calm = go(rows, cancel_ms=(200.0,), requote_bps=(0.0, 0.5)).books
    assert every.placed > 20 and every.cancels > 20
    assert calm.placed == 2 and calm.cancels == 0


def test_a_fill_on_a_cancelled_order_cannot_take_inventory_past_the_cap():
    # Long 10 with a bid resting and the cap at 20. The book drops; the old
    # bid is being cancelled (1 s) and still counts, so no new bid goes out
    # until it is gone. A seller then sweeps through both prices: only the
    # old bid is there, the inventory stops at the cap.
    rows = tape(lambda i: (100_000, 100_010) if i < 50 else (99_990, 100_000),
                [(3.0, Side.SELL, 99_990, 10), (5.5, Side.SELL, 99_970, 60)])
    [bk] = go(rows, cancel_ms=(1000.0,), max_inv=20.0).books
    assert bk.held > 0
    assert bk.late == 1 and abs(bk.late_qty - 10.0) < 1e-9
    assert abs(bk.inv - 20.0) < 1e-9 and bk.max_inv == 20.0


def test_hours_count_the_recording_not_the_gap_in_it():
    first = tape(lambda i: (100_000, 100_010), [])
    later = [(src, rx + 600 * NS, ev) for src, rx, ev in tape(lambda i: (100_000, 100_010), [])]
    r = go(sorted(first + later, key=lambda r: r[1]), cancel_ms=(200.0,),
           test=(T0 + NS, T0 + 1000 * NS))
    assert r.span_hours * 3600 > 600
    assert 10 < r.hours * 3600 < 17
    assert 0 < r.quoting_hours * 3600 <= r.hours * 3600 + 0.2   # a grid step at each edge


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
        # The same tape as GMO's spot book, where a maker is paid 1 bps.
        target.key_for("XRP", T0, 1): gzip.compress(("\n".join(lines) + "\n").encode()),
        target.meta_key("XRP"): json.dumps({"sources": {
            "gmo": {**spec, "symbol": "XRP", "tick_size": "0.001", "lot_size": "1",
                    "maker_bps": -1.0},
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
    rows = [x.split("\t") for x in out if x.startswith(("0.5bps", "1bps"))]
    assert [r[0] for r in rows] == ["0.5bps", "1bps"]   # both thresholds, same tape
    assert all(r[7] == "1" for r in rows)   # one estimated fill each
    assert any(x.startswith("期間 ") and "録画のある時間" in x for x in out)

    # A window that starts inside the model's training is refused.
    args = cli.build_parser().parse_args([
        "fairmm", "s3://b/raw/lead/", "--model", str(model), "--test", f"{iso(T0)},"])
    assert await args.func(args) == 1

    # The leverage model quoting GMO's spot book: the recorded rebate is used.
    args = cli.build_parser().parse_args([
        "fairmm", "s3://b/raw/lead/", "--model", str(model), "--symbol", "XRP",
        "--widths", "0.4999", "--cancel-ms", "200", "--shrink", "0", "--requote-bps", "0.5",
        "--test", f"{iso(T0 + NS)},{iso(T0 + 100 * NS)}"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert any(x.startswith("XRP:") and "リベート +1bps" in x for x in out)
    [row] = [x.split("\t") for x in out if x.startswith("0.5bps")]
    assert row[7] == "1" and float(row[19].replace(",", "")) > 0   # paid on the fill
