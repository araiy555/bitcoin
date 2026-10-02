"""A bitbank model for replaying the live trader, and the replay itself."""

import gzip
import io
import json
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.live.bitbank import NOT_FOUND, BitbankError, OrderRefused
from jsboard.sim.bitbank_sim import MS, SimBitbank, VenueBehaviour

INST = Instrument("ada_jpy", Decimal("0.001"), Decimal("0.0001"), "ADA", "JPY")
T0 = 1_790_380_800_000_000_000


def venue(**kw):
    b = VenueBehaviour(slow_land_share=0.0, refuse_share=0.0, **kw)
    sim = SimBitbank(INST, b, jpy=Decimal("50000"), coin=Decimal("500"))
    sim.now_ns = T0
    sim.on_market(DepthSnapshot(((36_990, 1_000_000),), ((37_010, 1_000_000),), 1, T0))
    sim.touch(36_990, 37_010)
    return sim


@pytest.mark.asyncio
async def test_an_order_rests_only_after_it_lands():
    sim = venue(land_ms=200)
    oid = await sim.order("ada_jpy", "buy", Decimal("37.000"), Decimal("10"))
    with pytest.raises(BitbankError) as exc:
        await sim.cancel("ada_jpy", oid)
    assert exc.value.code == NOT_FOUND
    assert await sim.active_orders("ada_jpy") == []
    sim.now_ns += 250 * MS
    assert await sim.active_orders("ada_jpy") == [oid]
    await sim.cancel("ada_jpy", oid)


@pytest.mark.asyncio
async def test_post_only_and_balance_rules():
    sim = venue()
    with pytest.raises(OrderRefused):
        await sim.order("ada_jpy", "buy", Decimal("37.010"), Decimal("10"))  # would take
    with pytest.raises(BitbankError) as exc:
        await sim.order("ada_jpy", "sell", Decimal("37.020"), Decimal("600"))  # 500 held
    assert exc.value.code == 60001
    await sim.order("ada_jpy", "buy", Decimal("37.000"), Decimal("1000"))  # 37,000 yen
    with pytest.raises(BitbankError):
        await sim.order("ada_jpy", "buy", Decimal("37.000"), Decimal("500"))  # locked


@pytest.mark.asyncio
async def test_only_executions_fill_and_a_cancel_in_flight_can_still_fill():
    sim = venue(land_ms=0, cancel_ms=250)
    oid = await sim.order("ada_jpy", "buy", Decimal("37.000"), Decimal("10"))
    # The book moving through our price fills nothing on bitbank's rule.
    sim.touch(36_980, 36_995)
    assert await sim.trade_history("ada_jpy", 0) == []
    await sim.cancel("ada_jpy", oid)
    sim.now_ns += 100 * MS  # the cancel has not landed yet
    sim.on_market(TradeTick(36_999, 1_000_000, Side.SELL, 5, sim.now_ns))
    [fill] = await sim.trade_history("ada_jpy", 0)
    assert fill["order_id"] == oid and fill["amount"] == "10.0000"
    assert sim.coin == Decimal("510")


def recording():
    """Two hours of a quiet two-sided ADA book with sellers and buyers
    crossing it, and a Binance lead that does not move."""
    ns = 1_000_000_000
    bb = {"symbol": "ada_jpy", "tick_size": "0.001", "lot_size": "0.0001",
          "base": "ADA", "quote": "JPY", "market": "spot"}
    bn = {"symbol": "ADAUSDT", "tick_size": "0.0001", "lot_size": "1",
          "base": "ADA", "quote": "USDT", "market": "perp"}
    from jsboard.feed.replay import _encode

    rows = [{"k": "status", "state": "live", "detail": "", "ts_ns": T0, "src": "bitbank",
             "rx_ns": T0}]
    for i in range(2_000):
        ts = T0 + i * ns
        mid = 37_200 + (i // 120) % 3
        b = _encode(DepthSnapshot(tuple((mid - 5 - j, 9_000_000) for j in range(5)),
                                  tuple((mid + 5 + j, 9_000_000) for j in range(5)), i, ts))
        b.update(src="bitbank", rx_ns=ts)
        t = _encode(TradeTick(mid + 6 if i % 2 else mid - 6, 20_000_000,
                              Side.BUY if i % 2 else Side.SELL, i, ts + ns // 2))
        t.update(src="bitbank", rx_ns=ts + ns // 2)
        n = _encode(DepthSnapshot(((2480, 90_000),), ((2481, 90_000),), i, ts))
        n.update(src="binance", rx_ns=ts)
        rows += [b, t, n]
    from jsboard.sim.s3 import S3Target

    target = S3Target(bucket="b", prefix="raw/live")
    return {
        target.key_for("ADA_JPY", T0, 1): gzip.compress(
            ("\n".join(json.dumps(r) for r in rows) + "\n").encode()),
        target.meta_key("ADA_JPY"): json.dumps(
            {"sources": {"bitbank": bb, "binance": bn}}).encode(),
    }


@pytest.mark.asyncio
async def test_the_live_trader_replays_against_the_model(monkeypatch, capsys):
    import jsboard.sim.s3 as s3
    from jsboard.cli import build_parser

    objects = recording()

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
    args = build_parser().parse_args([
        "livesim", "s3://b/raw/live/symbol=ADA_JPY/date=2026-09-26/",
        "--slow-land-share", "0", "--refuse-share", "0",
        "--vary", "cancel_ms=250,3000",
    ])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "2 通り" in out.replace("\n", "") and "損益円" in out
    lines = [line for line in out.splitlines() if line.startswith("cancel_ms=")]
    assert len(lines) == 2
    fills = [int(line.split("\t")[1].replace(",", "")) for line in lines]
    assert all(f > 0 for f in fills), out


def test_a_cancelled_order_keeps_its_balance_for_a_moment():
    from jsboard.live import clock
    from jsboard.live.venue import RELEASE_GRACE_S, LiveVenue
    from jsboard.mm.quoter import Quote

    now = [100.0]
    clock.use(lambda: now[0])
    try:
        v = LiveVenue(INST, quote_balance=Decimal("4000"), base_balance=Decimal(0))
        order = v.place(Quote(Side.BUY, 37_000, 1_000_000), 0, best_opposite=37_010)
        order.exchange_id = 1
        v.cancel(order.order_id)
        v.cancelled(order)
        assert not v.can_afford(Side.BUY, 37_000, 1_000_000)  # 3,700 yen still held
        now[0] += RELEASE_GRACE_S + 0.1
        assert v.can_afford(Side.BUY, 37_000, 1_000_000)
    finally:
        clock.reset()


@pytest.mark.asyncio
async def test_the_replay_lines_up_beside_the_live_fills(monkeypatch, capsys):
    import jsboard.cli as cli
    import jsboard.live.bitbank as bb
    import jsboard.sim.s3 as s3

    objects = recording()

    class Bucket:
        def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):  # noqa: N803
            keys = sorted(k for k in objects if k.startswith(Prefix))
            return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": io.BytesIO(objects[Key])}

        def head_object(self, Bucket, Key):  # noqa: N803
            if Key not in objects:
                raise KeyError(Key)

    class Account:
        def __init__(self, *keys, session=None):
            pass

        async def trades_between(self, pair, since_ms, end_ms):
            return [{"trade_id": 1, "executed_at": since_ms + 5_000, "side": "buy",
                     "price": "37.19", "amount": "100", "maker_taker": "maker"}]

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(s3, "default_client", Bucket)
    monkeypatch.setattr(bb, "BitbankPrivate", Account)
    monkeypatch.setattr(bb, "load_keys", lambda path: ("k", "s"))
    monkeypatch.setattr(cli, "make_session", Session)
    start = "2026-09-26T00:00"
    args = cli.build_parser().parse_args([
        "livesim", "s3://b/raw/live/symbol=ADA_JPY/date=2026-09-26/",
        "--since", start, "--until", "2026-09-26T00:30",
        "--slow-land-share", "0", "--refuse-share", "0", "--compare-live",
    ])
    assert await args.func(args) == 0
    out = capsys.readouterr().out.replace("\n", "\n")
    assert "本番の約定 1 件" in out and "板の中値との差" in out and "30分ごとの損益" in out


@pytest.mark.asyncio
async def test_the_print_ceiling_reads_a_recording(monkeypatch, capsys):
    import jsboard.sim.s3 as s3
    from jsboard.cli import build_parser

    objects = recording()

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
    args = build_parser().parse_args(
        ["printedge", "s3://b/raw/live/symbol=ADA_JPY/date=2026-09-26/"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "すべての約定" in out and "60秒後bps" in out and "スプレッド 中央値" in out
