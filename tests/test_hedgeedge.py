"""A bitbank maker fill hedged against GMO's recorded book."""

from decimal import Decimal

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.research.hedgeedge import NS, analyse

BB = Instrument("xrp_jpy", Decimal("0.001"), Decimal("0.0001"), "XRP", "JPY")
GM = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")


def book(bids, asks, t):
    return DepthSnapshot(tuple(bids), tuple(asks), 1, t)


def tape(gmo_bids, gmo_asks, after=None):
    """bitbank at 99.9/100.1 all along; one seller hits the bid at t=10s.
    `after` replaces GMO's book 100ms after the print (before our hedge)."""
    rows = [("bitbank", 0, book([(99_900, 10**6)], [(100_100, 10**6)], 0)),
            ("gmo", 0, book(gmo_bids, gmo_asks, 0))]
    rows.append(("bitbank", 10 * NS, TradeTick(99_900, 100_000, Side.SELL, 1, 10 * NS)))
    if after:
        rows.append(("gmo", 10 * NS + 100_000_000, book(*after, 10 * NS + 100_000_000)))
    for t in range(11, 400):  # time passes; nothing moves
        rows.append(("bitbank", t * NS, book([(99_900, 10**6)], [(100_100, 10**6)], t * NS)))
    return rows


def run(rows, **kw):
    return analyse(iter(rows), BB, GM, label="xrp", rebate_bps=2.0, hedge_fee_bps=0.0,
                   size_jpy=1_000.0, **kw)


def test_the_hedge_pays_what_the_book_asks_not_the_mid():
    # 10 XRP bought on bitbank at 99.9; GMO bids 5 @ 99.95 then 50 @ 99.90.
    r = run(tape([(99_950, 5), (99_900, 50)], [(100_050, 50)]))
    assert r.hedged == 1
    leg = r.horizons[60]
    maker = 10 * (100.0 - 99.9) + 99.9 * 10 * 2e-4      # half spread + rebate
    hedge = -(100.05 - (5 * 99.95 + 5 * 99.90) / 10) * 10  # sold walking bids, bought back at ask
    assert abs(leg.pnl - (maker + hedge)) < 1e-9 and leg.pnl < 0
    assert r.open_cost_bps > 0


def test_the_hedge_meets_the_book_as_it_stood_after_the_delay():
    early = run(tape([(99_950, 50)], [(100_050, 50)]))
    moved = run(tape([(99_950, 50)], [(100_050, 50)],
                     after=([(99_800, 50)], [(99_900, 50)])))  # GMO fell before we got there
    assert moved.horizons[60].pnl < early.horizons[60].pnl


def test_a_book_too_thin_to_hedge_is_counted_not_priced():
    r = run(tape([(99_950, 3)], [(100_050, 50)]))
    assert r.hedged == 0 and r.too_thin == 1


async def test_the_command_reads_a_pair_recording(monkeypatch, capsys):
    import gzip
    import io
    import json

    import jsboard.cli as cli
    import jsboard.sim.s3 as s3
    from jsboard.feed.replay import _encode
    from jsboard.sim.s3 import S3Target

    t0 = 1_790_380_800_000_000_000
    rows = []
    for src, rx, event in tape([(99_950, 50)], [(100_050, 50)]):
        r = _encode(event)
        r.update(src=src, rx_ns=t0 + rx)
        if "ts_ns" in r:
            r["ts_ns"] = t0 + rx
        rows.append(r)
    target = S3Target(bucket="b", prefix="raw/pair")
    spec = {"tick_size": "0.001", "base": "XRP", "quote": "JPY"}
    objects = {
        target.key_for("XRP_JPY", t0, 1): gzip.compress(
            ("\n".join(json.dumps(r) for r in rows) + "\n").encode()),
        target.meta_key("XRP_JPY"): json.dumps({"sources": {
            "bitbank": {**spec, "symbol": "xrp_jpy", "lot_size": "0.0001", "maker_bps": -2.0},
            "gmo": {**spec, "symbol": "XRP_JPY", "lot_size": "1", "taker_bps": 0.0,
                    "min_order": "1"},
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

    posted = []

    async def post(text):
        posted.append(text)
        return True

    monkeypatch.setattr(s3, "default_client", Bucket)
    monkeypatch.setattr(cli, "_post_slack", post)
    args = cli.build_parser().parse_args(
        ["hedgeedge", "s3://b/raw/pair/", "--size-jpy", "1000", "--slack"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "xrp_jpy→GMO XRP_JPY" in out and "60秒bps" in out
    assert "判定保留" in posted[0]  # one hedge is far too few to judge
