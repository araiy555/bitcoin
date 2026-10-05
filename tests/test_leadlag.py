"""Follow a jump in the lead market on GMO's real book."""

import math
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
    # The jump is followed; GMO catching up a moment later reads as the lead
    # lagging GMO, a second signal after which nothing moves.
    assert d5.same >= 1 and d5.opposite == 0 and d5.total_bps > 0
    assert stayed.cells[(10, 5)].flat == stayed.cells[(10, 5)].n


def test_slack_summary_picks_the_best_hold_not_the_longest():
    from jsboard.research.leadlag import Cell, LeadLagResult, slack_summary

    r = LeadLagResult("XRP_JPY(binance)")
    r.cells[("2026-10-05", 3.0, 1.0)] = Cell(n=30, wins=20, notional=300_000, pnl=60,
                                             mid_n=30, mid_bps=45.0)
    r.cells[("2026-10-05", 3.0, 60.0)] = Cell(n=30, wins=10, notional=300_000, pnl=-90)
    text = slack_summary([r], [1.0, 60.0])
    assert "1秒持つ → +2.0bps" in text and "60秒 -3.0" in text
    assert "手数料・スプレッド前）: 1秒 +1.5" in text


def test_gmo_mid_path_is_kept_from_the_signal_before_costs():
    r = analyse(iter(tape(gmo_follows_at=105)), GMO, LEAD, label="xrp", thresholds=[10],
                holds=[0.1, 5], size_jpy=10_000)
    early = next(c for (_, _, h), c in r.cells.items() if h == 0.1)
    late = next(c for (_, _, h), c in r.cells.items() if h == 5)
    assert early.mid_n == 1 and abs(early.mid()) < 1e-9  # GMO has not moved yet
    assert abs(late.mid() - math.log(100.21 / 100.01) * 1e4) < 0.01


def test_an_order_below_the_minimum_trades_the_minimum():
    r = analyse(iter(tape(gmo_follows_at=105)), GMO, LEAD, label="xrp", thresholds=[10],
                holds=[5], size_jpy=10_000, min_order=500)
    [cell] = r.cells.values()
    assert cell.n == 1 and abs(cell.notional - 500 * 100.02) < 1e-6
