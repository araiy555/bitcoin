"""The live GMO follow: signal, orders, records and stops, on a fake venue."""

import asyncio

from jsboard.live.gmo import Fill, GmoError, sign
from jsboard.live.gmofollow import NS, Detector, Follower, ShadowApi

T0 = 1_790_380_800 * NS


def test_sign_matches_gmo_scheme():
    import hashlib
    import hmac

    want = hmac.new(b"s", b"1POST/v1/order{}", hashlib.sha256).hexdigest()
    assert sign("s", "1POST/v1/order{}") == want


def feed(det, rows):
    return [x for x in (det.update(src, t, m) for src, t, m in rows) if x]


def test_detector_fires_once_per_move_and_only_when_the_lead_moved():
    det = Detector(threshold_bps=8)
    rows = [("gmo", T0, 100.0), ("bybit", T0, 1.0)]
    # The lead rises 10bps; GMO has not moved.
    rows += [("gmo", T0 + NS + 1, 100.0), ("bybit", T0 + NS + 2, 1.001)]
    rows += [("bybit", T0 + NS + 3, 1.0011)]  # same move: no second signal
    sigs = feed(det, rows)
    assert len(sigs) == 1 and sigs[0][0] == "bybit" and sigs[0][1] == 1
    # GMO catches up (moves alone): the gap flips, but the lead did not move.
    det2 = Detector(threshold_bps=8)
    rows2 = [("gmo", T0, 100.0), ("bybit", T0, 1.0), ("gmo", T0 + NS + 1, 100.1),
             ("bybit", T0 + NS + 2, 1.0)]
    assert feed(det2, rows2) == []


def test_reset_forgets_history_across_a_reconnect():
    det = Detector(threshold_bps=8)
    det.update("gmo", T0, 100.0)
    det.update("bybit", T0, 1.0)
    det.reset("bybit")
    assert det.update("bybit", T0 + 5 * NS, 1.01) is None


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.t += int(max(s, 0.001) * NS)
        await asyncio.sleep(0)


class Venue:
    """Fills market orders at a scripted price; can refuse."""

    def __init__(self, prices, refuse=None):
        self.prices = list(prices)
        self.refuse = refuse
        self.orders = []
        self._fills = {}

    async def market_open(self, symbol, side, size):
        if self.refuse:
            raise GmoError([self.refuse], "/v1/order")
        return self._fill("open", side, size, pid=7)

    async def market_close(self, symbol, side, positions):
        assert positions == [(7, "10")]
        return self._fill("close", side, sum(float(s) for _, s in positions), pid=7)

    def _fill(self, kind, side, size, pid):
        oid = str(len(self.orders) + 1)
        self.orders.append((kind, side, size))
        self._fills[oid] = [Fill(self.prices.pop(0), float(size), pid, 0.0, 0.0, "t")]
        return oid

    async def fills(self, oid):
        return self._fills.get(oid, [])


def follower(venue, clock, **kw):
    records, said = [], []

    async def notify(text):
        said.append(text)

    f = Follower(api=venue, symbol="XRP_JPY", size="10", walk=lambda side, q: 100.0,
                 touch=lambda: (99.99, 100.01), clock=clock, sleep=clock.sleep,
                 log=records.append, notify=notify, **kw)
    return f, records, said


def test_a_round_trip_is_recorded_with_the_expected_prices():
    clock = Clock()
    venue = Venue([100.0, 100.05])
    f, records, _ = follower(venue, clock)
    rec = asyncio.run(f.trade("bybit", 1, 9.0, clock()))
    assert venue.orders == [("open", "BUY", "10"), ("close", "SELL", 10.0)]
    assert abs(rec["pnl_jpy"] - 0.5) < 1e-9 and records == [rec]
    assert rec["expected_entry"] == 100.0 and "entry_slip_bps" in rec
    assert f.open_positions == [] and f.tally.n == 1


def test_the_loss_limit_stops_the_run():
    clock = Clock()
    venue = Venue([100.0, 50.0])  # 10 XRP lose 500 yen
    f, _, said = follower(venue, clock, max_loss_jpy=500)
    asyncio.run(f.trade("bybit", 1, 9.0, clock()))
    assert f.stopped and "損失" in f.stopped and said
    assert asyncio.run(f.trade("bybit", 1, 9.0, clock())) is None  # no more trades


def test_a_margin_error_stops_at_once():
    clock = Clock()
    f, records, _ = follower(Venue([], refuse="ERR-201"), clock)
    asyncio.run(f.trade("bybit", 1, 9.0, clock()))
    assert f.stopped and "ERR-201" in records[0]["error"]


def test_shadow_api_fills_at_the_book_and_sends_nothing():
    clock = Clock()
    api = ShadowApi(lambda side, q: 101.0 if side > 0 else 99.0, sleep=clock.sleep)
    f = Follower(api=api, symbol="XRP_JPY", size="10", walk=api.walk, touch=lambda: (99.0, 101.0),
                 clock=clock, sleep=clock.sleep)
    rec = asyncio.run(f.trade("binance", 1, 9.0, clock()))
    assert rec["entry_price"] == 101.0 and rec["exit_price"] == 99.0 and rec["pnl_jpy"] == -20.0


def test_the_command_runs_in_shadow_and_logs_a_trade(monkeypatch, tmp_path):
    from decimal import Decimal

    import jsboard.cli as cli
    from jsboard.core.types import Instrument
    from jsboard.feed.base import DepthSnapshot

    gmo = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
    perp = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")

    class Script:
        def __init__(self, prices, tick, delay):
            self.prices, self.tick, self.delay = prices, tick, delay

        async def __aiter__(self):
            for p in self.prices:
                await asyncio.sleep(self.delay)
                b = round(p / self.tick)
                yield DepthSnapshot(((b, 10_000),), ((b + 2, 10_000),), 1)
            await asyncio.sleep(3600)

    async def live_book(target):
        return gmo, Script([150.0] * 40, 0.001, 0.05)

    async def bybit_inst(sym, cat="linear"):
        return perp

    async def no_binance(sym):
        raise RuntimeError("off")

    # The lead holds still for a second and a half, then jumps 20bps.
    monkeypatch.setattr(cli, "_live_book", live_book)
    monkeypatch.setattr(cli, "fetch_bybit_instrument", bybit_inst)
    monkeypatch.setattr(cli, "BybitFeed", lambda inst, **kw: Script([1.0] * 30 + [1.002] * 10,
                                                                     0.0001, 0.05))
    monkeypatch.setattr(cli, "fetch_futures_instrument", no_binance)
    log = tmp_path / "t.jsonl"
    args = cli.build_parser().parse_args([
        "gmofollow", "--hold-s", "0.3", "--hours", "0.0007", "--log", str(log)])
    assert asyncio.run(args.func(args)) == 0
    import json

    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert recs and recs[0]["live"] is False and recs[0]["side"] == "BUY"
    assert recs[0]["lead"] == "bybit" and "pnl_jpy" in recs[0]


def test_live_without_keys_refuses_before_connecting(monkeypatch, tmp_path):
    import jsboard.cli as cli

    for k in ("GMO_API_KEY", "GMO_API_SECRET"):
        monkeypatch.delenv(k, raising=False)

    async def boom(target):
        raise AssertionError("must not connect")

    monkeypatch.setattr(cli, "_live_book", boom)
    args = cli.build_parser().parse_args([
        "gmofollow", "--live", "--env-file", str(tmp_path / "none.env")])
    assert asyncio.run(args.func(args)) == 1


def test_the_command_trades_on_a_fair_price_model(monkeypatch, tmp_path):
    import json
    from decimal import Decimal

    import jsboard.cli as cli
    from jsboard.core.types import Instrument
    from jsboard.feed.base import DepthSnapshot
    from jsboard.research.fairprice import feature_names

    gmo = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
    perp = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")

    class Script:
        def __init__(self, prices, tick, delay):
            self.prices, self.tick, self.delay = prices, tick, delay

        async def __aiter__(self):
            for p in self.prices:
                await asyncio.sleep(self.delay)
                b = round(p / self.tick)
                yield DepthSnapshot(((b, 10_000),), ((b + 2, 10_000),), 1)
            await asyncio.sleep(3600)

    async def live_book(target):
        return gmo, Script([150.0] * 40, 0.001, 0.05)

    async def bybit_inst(sym, cat="linear"):
        return perp

    async def no_binance(sym):
        raise RuntimeError("off")

    monkeypatch.setattr(cli, "_live_book", live_book)
    monkeypatch.setattr(cli, "fetch_bybit_instrument", bybit_inst)
    monkeypatch.setattr(cli, "BybitFeed", lambda inst, **kw: Script([1.0] * 15 + [1.002] * 25,
                                                                     0.0001, 0.05))
    monkeypatch.setattr(cli, "fetch_futures_instrument", no_binance)
    names = feature_names(())
    w = [0.0] * len(names)
    w[names.index("bybit 1秒")] = 1.0  # predict GMO follows Bybit's last second one for one
    model = tmp_path / "XRP_JPY.json"
    model.write_text(json.dumps({"symbol": "XRP_JPY", "names": names, "cross": [], "step_ms": 100,
                                 "weights": {"0.3": w}, "margin_bps": {"0.3": 1.0}}))
    log = tmp_path / "t.jsonl"
    args = cli.build_parser().parse_args([
        "gmofollow", "--model", str(model), "--hold-s", "0.3", "--hours", "0.0007", "--log", str(log)])
    assert asyncio.run(args.func(args)) == 0
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert recs and recs[0]["lead"] == "フェア価格" and recs[0]["side"] == "BUY"


def test_signals_while_busy_are_counted_and_the_log_summarised():
    from jsboard.live.gmofollow import summary

    clock = Clock()
    venue = Venue([100.0, 100.05])
    f, records, _ = follower(venue, clock)
    f.busy = True
    assert f.saw_signal("bybit", 1, 9.0, clock()) is False
    f.busy = False
    assert f.saw_signal("bybit", 1, 9.0, clock()) is True
    asyncio.run(f.trade("bybit", 1, 9.0, clock(), {"predicted_bps": 7.0, "cost_bps": 4.0}))
    assert f.tally.signals == 2 and f.tally.skipped == 1
    text = "\n".join(summary(records))
    assert "見送り 1" in text and "約定率 100%" in text and "+7.00" in text


def test_shadow_fills_at_the_book_after_the_latency():
    clock = Clock()
    book = {"ask": 100.0}
    api = ShadowApi(lambda side, q: book["ask"], latency_s=0.2, sleep=clock.sleep)

    async def go():
        task = asyncio.ensure_future(api.market_open("XRP_JPY", "BUY", "10"))
        await asyncio.sleep(0)
        book["ask"] = 100.3  # the book moves while the order is on its way
        oid = await task
        return (await api.fills(oid))[0].price

    assert asyncio.run(go()) == 100.3


def test_the_summary_shows_hours_and_the_result_without_the_extremes():
    from jsboard.live.gmofollow import summary

    recs = []
    for i in range(40):
        recs.append({"signal_ns": T0 + i * 600 * NS, "pnl_jpy": 0.1, "pnl_bps": 1.0})
    recs[0]["pnl_bps"] = 40.0  # one lucky trade
    text = "\n".join(summary(recs))
    assert "時間帯ごと" in text and "プラスだった時間帯" in text and "勝率 100%" in text
    assert "一番良い 2回を除くと 1回あたり +1.00bps" in text
