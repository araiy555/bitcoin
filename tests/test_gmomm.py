"""Quoting GMO with limit orders: the cap, the stop, and the dry run."""

import asyncio

from jsboard.live.gmo import GmoError
from jsboard.live.gmomm import DryApi, Maker

NS = 1_000_000_000
T0 = 1_790_380_800 * NS


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> int:
        return self.t

    async def sleep(self, s: float) -> None:
        self.t += int(s * NS)
        await asyncio.sleep(0)


class Book:
    def __init__(self, bid=100.000, ask=100.010) -> None:
        self.bid, self.ask = bid, ask

    def touch(self):
        return self.bid, self.ask

    def level(self, side, price):
        return 100.0 if abs(price - (self.bid if side > 0 else self.ask)) < 1e-9 else 0.0

    def walk(self, side, qty):
        return self.ask if side > 0 else self.bid


def setup(**kw):
    clock, book = Clock(), Book()
    dry = DryApi(0.001, book.touch, book.level, book.walk, place_ms=200,
                 cancel_ms=kw.pop("cancel_ms", 1000), clock=clock)
    said = []

    async def notify(text):
        said.append(text)

    kw.setdefault("skew_bps", 0.0)
    m = Maker(api=kw.pop("api", dry), symbol="XRP_JPY", tick=0.001, clock=clock, sleep=clock.sleep,
              notify=notify, **kw)
    return m, dry, clock, book, said


async def turn(n=10):
    for _ in range(n):
        await asyncio.sleep(0)


async def quote(m, book, fair):
    m.requote(fair, book.bid, book.ask)
    await turn()


def sides(dry, side):
    return [o for o in dry.by_id.values() if o["side"] == side]


async def test_quotes_stay_put_until_their_target_moves_half_a_bp():
    m, dry, clock, book, _ = setup()
    await quote(m, book, 100.005)
    assert dry.sent == 2
    [bid], [ask] = sides(dry, 1), sides(dry, -1)
    assert abs(bid["price"] - 99.984) < 1e-9 and abs(ask["price"] - 100.026) < 1e-9
    await quote(m, book, 100.008)        # 0.3 bps: left alone
    assert dry.sent == 2 and m.stats.cancels == 0
    await quote(m, book, 100.016)        # 1.1 bps: both move
    assert dry.sent == 4 and m.stats.cancels == 2


async def test_an_order_being_cancelled_still_counts_against_the_cap():
    m, dry, clock, book, _ = setup()
    m.position = m.long_qty = dry.long = 20.0
    await quote(m, book, 100.005)
    assert len(sides(dry, 1)) == 1                       # 20 + 10 fits the 30 cap
    clock.t += NS // 2
    await quote(m, book, 99.990)                         # bid must move; old one is cancelling
    assert len(sides(dry, 1)) == 1 and m.stats.held >= 1  # 20 + 10 + 10 would not fit
    clock.t += NS // 4
    dry.on_trade(-1, 99.900, 50)                         # a seller sweeps the old bid
    await m.poll()
    assert m.position == 30.0 and m.stats.late == 1 and m.stats.late_qty == 10.0
    clock.t += 2 * NS
    await m.poll()
    await quote(m, book, 99.990)
    assert len(sides(dry, 1)) == 1 and m.position <= m.max_inv


async def test_a_new_bid_goes_out_once_gmo_confirms_the_old_one_is_gone():
    m, dry, clock, book, _ = setup()
    m.position = m.long_qty = dry.long = 20.0
    await quote(m, book, 100.005)
    clock.t += NS // 2
    await quote(m, book, 99.990)
    assert len(sides(dry, 1)) == 1
    clock.t += 2 * NS                                    # the cancel lands, nothing filled
    await m.poll()
    await quote(m, book, 99.990)
    assert len(sides(dry, 1)) == 2 and m.stats.cancel_dead_ms


async def test_the_side_that_shrinks_the_position_closes_it():
    m, dry, clock, book, _ = setup()
    await quote(m, book, 100.005)
    clock.t += NS // 2
    dry.on_trade(-1, 99.900, 10)                         # our bid is filled: long 10
    await m.poll()
    assert m.position == 10.0 and m.long_qty == 10.0
    await quote(m, book, 99.990)                         # both sides move
    asks = sides(dry, -1)
    assert asks[-1]["settle"] == "CLOSE" and sides(dry, 1)[-1]["settle"] == "OPEN"


async def test_a_loss_cancels_everything_and_closes_the_position():
    m, dry, clock, book, said = setup(stop_jpy=5.0, max_loss_jpy=10.0)
    await quote(m, book, 100.005)
    clock.t += NS // 2
    dry.on_trade(-1, 99.900, 10)
    await m.poll()
    book.bid, book.ask = 99.0, 99.01                     # the market falls 1%
    m.check_loss(book.bid, book.ask)
    await turn()
    await m._flatten
    assert m.stopped.startswith("損失が") and m.clean is True
    assert not await dry.active_orders("XRP_JPY")
    assert dry.long == dry.short == 0 and m.position == 0
    assert any("確認しました" in s for s in said)


async def test_an_order_whose_fate_is_unknown_stops_everything():
    clock, book = Clock(), Book()

    class Flaky(DryApi):
        calls = 0

        async def limit_open(self, *a, **k):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise ConnectionError("reset")
            return await super().limit_open(*a, **k)

    dry = Flaky(0.001, book.touch, book.level, book.walk, cancel_ms=200, clock=clock)
    m = Maker(api=dry, symbol="XRP_JPY", tick=0.001, clock=clock, sleep=clock.sleep, skew_bps=0.0)
    await quote(m, book, 100.005)
    await turn()
    await m._flatten
    assert m.stopped.startswith("注文の結果が分かりません") and m.clean is True
    assert not await dry.active_orders("XRP_JPY")


async def test_a_refused_key_stops_at_once():
    m, dry, clock, book, _ = setup()

    async def refuse(*a, **k):
        raise GmoError(["ERR-5012"], "/v1/order")

    dry.limit_open = refuse
    await quote(m, book, 100.005)
    await m._flatten
    assert "ERR-5012" in m.stopped


async def test_a_position_that_does_not_match_the_fills_stops():
    m, dry, clock, book, _ = setup()
    dry.long = 10.0                                      # someone else's position
    for _ in range(3):
        await m.poll(reconcile=True)
    await turn()
    await m._flatten
    assert "合いません" in m.stopped and dry.long == 0


async def test_a_fill_that_is_not_ours_stops():
    m, dry, clock, book, _ = setup()
    dry._book("manual-1", 1, "OPEN", 10.0, 100.0)
    await m.poll()
    await turn()
    await m._flatten
    assert "自分のものではない" in m.stopped and dry.long == 0


async def test_whatever_the_market_does_the_position_stays_inside_the_cap():
    import random

    rng = random.Random(7)
    m, dry, clock, book, _ = setup(cancel_ms=1000, max_inv=30.0, skew_bps=0.5)
    mid = 100.005
    worst = 0.0
    for step in range(3000):
        clock.t += NS // 10
        mid += rng.choice((-1, 0, 0, 1)) * 0.004
        book.bid, book.ask = round(mid - 0.005, 3), round(mid + 0.005, 3)
        if rng.random() < 0.3:   # sweeps either way, often through our levels
            side = rng.choice((1, -1))
            dry.on_trade(side, book.ask + 0.03 if side > 0 else book.bid - 0.03, rng.choice((5, 10, 40)))
        if step % 10 == 0:
            await m.poll(reconcile=True)
        await quote(m, book, mid + rng.uniform(-0.01, 0.01))
        worst = max(worst, abs(dry.long - dry.short))
    assert m.stats.late > 0 and m.stats.fills > 50       # cancels were picked off along the way
    assert worst <= 30.0 + 1e-9 and m.stopped is None


def test_the_command_quotes_in_a_dry_run_and_stops_clean(monkeypatch, tmp_path):
    import json
    from decimal import Decimal

    import jsboard.cli as cli
    from jsboard.core.types import Instrument, Side
    from jsboard.feed.base import DepthSnapshot, TradeTick
    from jsboard.research.fairprice import feature_names

    gmo = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
    perp = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")

    class Script:
        def __init__(self, price, tick, sweep_at=None):
            self.price, self.tick, self.sweep_at = price, tick, sweep_at

        async def __aiter__(self):
            b = round(self.price / self.tick)
            for i in range(40):
                await asyncio.sleep(0.05)
                yield DepthSnapshot(((b, 10_000),), ((b + 2, 10_000),), 1)
                if i == self.sweep_at:   # a seller sweeps well below the book
                    yield TradeTick(b - 100, 50, Side.SELL, ts_ns=0)
            await asyncio.sleep(3600)

    async def live_book(target):
        return gmo, Script(150.0, 0.001, sweep_at=30)

    async def bybit_inst(sym, cat="linear"):
        return perp

    async def no_binance(sym):
        raise RuntimeError("off")

    monkeypatch.setattr(cli, "_live_book", live_book)
    monkeypatch.setattr(cli, "fetch_bybit_instrument", bybit_inst)
    monkeypatch.setattr(cli, "BybitFeed", lambda inst, **kw: Script(1.0, 0.0001))
    monkeypatch.setattr(cli, "fetch_futures_instrument", no_binance)
    model = tmp_path / "XRP_JPY.json"
    model.write_text(json.dumps({"symbol": "XRP_JPY", "cross": [], "step_ms": 100,
                                 "weights": {"2": [0.0] * len(feature_names(()))}}))
    log = tmp_path / "mm.jsonl"
    args = cli.build_parser().parse_args([
        "gmomm", "--model", str(model), "--hours", "0.0007", "--log", str(log)])
    assert asyncio.run(args.func(args)) == 0
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert all(r["live"] is False for r in recs)
    assert any(r["ev"] == "placed" and r["side"] == "BUY" for r in recs)
    assert any(r["ev"] == "fill" and r["side"] == "BUY" for r in recs)
    [last] = [r for r in recs if r["ev"] == "stopped"]
    assert last["clean"] is True and last["position"] == 0


def test_live_without_keys_refuses_before_connecting(monkeypatch, tmp_path):
    import jsboard.cli as cli

    for k in ("GMO_API_KEY", "GMO_API_SECRET"):
        monkeypatch.delenv(k, raising=False)

    async def boom(target):
        raise AssertionError("must not connect")

    monkeypatch.setattr(cli, "_live_book", boom)
    args = cli.build_parser().parse_args([
        "gmomm", "--live", "--model", "unused.json", "--env-file", str(tmp_path / "none.env")])
    assert asyncio.run(args.func(args)) == 1
