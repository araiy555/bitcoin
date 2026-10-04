"""Follow a jump in the lead market on GMO's real book."""

from decimal import Decimal

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot
from jsboard.research.leadlag import NS, analyse

GMO = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
LEAD = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")
T0 = 1_790_380_800_000_000_000


def book(bid, ask, t):
    return DepthSnapshot(((bid, 1_000),), ((ask, 1_000),), 1, t)


def tape(gmo_follows_at=None, gmo_jumps_with_lead=False):
    rows = []
    for i in range(200):  # 100ms steps for 20s
        t = T0 + i * NS // 10
        jumped = i >= 100
        lead = (10_020, 10_022) if jumped else (10_000, 10_002)
        follow = (gmo_jumps_with_lead and jumped) or (
            gmo_follows_at is not None and i >= gmo_follows_at)
        gmo = (100_200, 100_220) if follow else (100_000, 100_020)
        # When both move at once GMO's update is seen first: nothing to follow.
        rows += [("gmo", t, book(*gmo, t)), ("lead", t, book(*lead, t))]
    return rows


def run(rows):
    return analyse(iter(rows), GMO, LEAD, label="xrp", thresholds=[10], holds=[5],
                   size_jpy=10_000)


def test_buying_before_gmo_follows_pays_the_move_less_the_spread():
    r = run(tape(gmo_follows_at=105))  # GMO catches up half a second later
    [cell] = r.cells.values()
    assert cell.n == 1 and cell.pnl > 0
    # bought at the old ask 100.02, sold at the new bid 100.20
    assert abs(cell.bps() - (100.20 - 100.02) / 100.02 * 1e4) < 0.01


def test_no_trade_when_gmo_moved_with_the_lead():
    r = run(tape(gmo_jumps_with_lead=True))
    assert not r.cells


def test_a_lead_that_gmo_never_follows_costs_the_spread():
    [cell] = run(tape()).cells.values()
    assert cell.pnl < 0 and cell.wins == 0


async def test_the_command_follows_on_bitbank_from_a_live_recording(monkeypatch, capsys):
    import gzip
    import io
    import json

    import jsboard.cli as cli
    import jsboard.sim.s3 as s3
    from jsboard.feed.replay import _encode
    from jsboard.sim.s3 import S3Target

    rows = []
    for src, rx, event in tape(gmo_follows_at=105):
        r = _encode(event)
        r.update(src="bitbank" if src == "gmo" else "binance", rx_ns=rx)
        rows.append(r)
    target = S3Target(bucket="b", prefix="raw/live")
    spec = {"base": "XRP", "quote": "JPY"}
    objects = {
        target.key_for("XRP_JPY", T0, 1): gzip.compress(
            ("\n".join(json.dumps(r) for r in rows) + "\n").encode()),
        target.meta_key("XRP_JPY"): json.dumps({"sources": {
            "bitbank": {**spec, "symbol": "xrp_jpy", "tick_size": "0.001", "lot_size": "1"},
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

    monkeypatch.setattr(s3, "default_client", Bucket)
    args = cli.build_parser().parse_args([
        "leadlag", "s3://b/raw/live/", "--follower", "bitbank", "--leads", "binance",
        "--thresholds", "10", "--holds", "5", "--days", "3"])
    assert await args.func(args) == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("XRP_JPY(binance)"))
    # +18bps move bought at the ask, less 12bps each way at bitbank: a loss
    assert line.split("\t")[3] == "1" and line.split("\t")[4].startswith("-")


def test_direction_counts_whether_the_slow_market_followed():
    from jsboard.research.leadlag import direction

    followed = direction(iter(tape(gmo_follows_at=105)), GMO, LEAD, label="x", thresholds=[10])
    stayed = direction(iter(tape()), GMO, LEAD, label="x", thresholds=[10])
    assert followed.signals[10] >= 1
    d5 = followed.cells[(10, 5)]
    assert d5.same == d5.n and d5.total_bps > 0
    assert stayed.cells[(10, 5)].flat == stayed.cells[(10, 5)].n
