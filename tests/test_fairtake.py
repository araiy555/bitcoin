"""Taking GMO's gap to the leader's fair value, and closing when it is back."""

from decimal import Decimal

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot
from jsboard.research.fairtake import Trade, rows_of, run

GMO = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
PERP = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")
NS = 1_000_000_000
T0 = 1_790_380_800 * NS


def book(bid, ask, t, qty=1_000):
    return DepthSnapshot(((bid, qty),), ((ask, qty),), 1, t)


def tape(steps, gmo_at, lead_at):
    rows = []
    for i in range(steps):
        t = T0 + i * NS // 10 + 1
        gb = gmo_at(i)
        rows.append(("gmo", t, book(gb, gb + 2, t)))
        lb = lead_at(i)
        rows.append(("binance", t, book(lb, lb + 1, t)))
    return rows


def go(rows, **kw):
    kw.setdefault("gaps", (8.0,))
    kw.setdefault("size", 50.0)
    return run(iter(rows), GMO, {"binance": PERP}, label="XRP",
               test=(T0 + 70 * NS, T0 + 1_000 * NS), **kw)


def test_gmo_late_to_a_jump_is_bought_and_sold_back_at_fair():
    # A minute and more of calm, then Binance jumps 20 bps; GMO follows 2 s later.
    jump = 800
    rows = tape(1_200, lambda i: 150_000 if i < jump + 20 else 150_300,
                lambda i: 10_000 if i < jump else 10_020)
    [gp] = go(rows).gaps
    assert len(gp.trades) == 1
    tr = gp.trades[0]
    assert tr.side == 1 and not tr.forced
    assert tr.bps > 10 and tr.gmo_pnl > 0
    assert 1.5 < tr.held_s < 3.0


def test_a_gap_that_never_closes_is_left_after_the_longest_hold():
    jump = 800
    rows = tape(2_000, lambda i: 150_000, lambda i: 10_000 if i < jump else 10_020)
    [gp] = go(rows, max_hold_s=30.0).gaps
    assert gp.trades and gp.trades[0].forced
    assert 29.0 < gp.trades[0].held_s < 31.0


def test_no_gap_no_trade():
    rows = tape(1_200, lambda i: 150_000, lambda i: 10_000)
    [gp] = go(rows).gaps
    assert gp.trades == []


def test_gmo_floors_each_closed_positions_profit_to_the_yen():
    up = Trade(1, 0, 100.0, exit=100.0526, size=50)    # +2.63 yen
    down = Trade(1, 0, 100.0, exit=99.994, size=50)    # -0.30 yen
    assert up.gmo_pnl == 2 and down.gmo_pnl == -1


def test_every_gap_is_reported_with_its_halves():
    jump = 800
    rows = tape(1_200, lambda i: 150_000 if i < jump + 20 else 150_300,
                lambda i: 10_000 if i < jump else 10_020)
    r = go(rows, gaps=(4.0, 8.0, 50.0))
    lines = rows_of(r)
    assert [x.split("\t")[0] for x in lines] == ["4bps", "8bps", "50bps"]
    assert lines[2].split("\t")[1] == "0"                  # 50 bps never came
    assert r.hours > 0 and r.mid_ns > T0


async def test_the_command_reads_a_recording(monkeypatch, capsys):
    import gzip
    import io
    import json
    from datetime import UTC, datetime

    import jsboard.cli as cli
    import jsboard.sim.s3 as s3
    from jsboard.feed.replay import _encode
    from jsboard.sim.s3 import S3Target

    jump = 800
    rows = tape(1_200, lambda i: 150_000 if i < jump + 20 else 150_300,
                lambda i: 10_000 if i < jump else 10_020)
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
            "binance": {**spec, "symbol": "XRPUSDT", "tick_size": "0.0001", "lot_size": "1"},
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

    monkeypatch.setattr(s3, "default_client", Bucket)
    args = cli.build_parser().parse_args([
        "fairtake", "s3://b/raw/lead/", "--gaps", "8", "--test", f"{iso(T0 + 70 * NS)},{iso(T0 + 1000 * NS)}"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out.splitlines()
    row = next(x for x in out if x.startswith("8bps")).split("\t")
    assert row[1] == "1" and row[5].startswith("+")      # one trade, positive after GMO's flooring
