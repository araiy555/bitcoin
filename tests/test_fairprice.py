"""A fair price fitted on one period and traded on the next."""

import random
from decimal import Decimal

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot
from jsboard.research.fairprice import feature_names, run, solve

GMO = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
PERP = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")
NS = 1_000_000_000
T0 = 1_790_380_800_000_000_000


def book(bid, ask, t, size=100_000):
    return DepthSnapshot(((bid, size),), ((ask, size),), 1, t)


def tape(minutes=40, lag_steps=10, seed=7):
    """The perp jumps now and then; GMO copies it `lag_steps` x 100ms later."""
    rnd = random.Random(seed)
    lead = [100.0]
    for _ in range(minutes * 600):
        jump = rnd.choice((-1, 1)) * 0.15 if rnd.random() < 0.01 else 0.0
        lead.append(lead[-1] * (1 + jump / 100) + rnd.gauss(0, 0.0005))
    rows = []
    for i, p in enumerate(lead):
        t = T0 + i * NS // 10 + 1
        g = lead[max(0, i - lag_steps)] * 1.5  # yen per dollar, roughly
        gb = round(g / 0.001) - 30
        rows.append(("gmo", t, book(gb, gb + 60, t)))  # about 4bps wide
        lb = round(p / 0.0001)
        rows.append(("bybit", t, book(lb, lb + 1, t)))
    return rows, T0 + minutes * 300 * NS // 10, T0 + minutes * 600 * NS // 10


def test_solve():
    assert [round(v, 9) for v in solve([[2, 1], [1, 3]], [3, 5])] == [0.8, 1.4]


def test_a_lagging_follower_is_learned_and_traded_on_unseen_data():
    rows, mid, end = tape()
    r = run(iter(rows), GMO, {"bybit": PERP}, label="XRP", train=(0, mid), test=(mid, end),
            holds=(2.0,), latency_ms=200)
    w = dict(zip(feature_names(()), r.weights[2.0], strict=True))
    assert w["bybit 1秒"] > 0.1
    assert r.score[2.0].r2() > 0.2
    m = r.model[2.0]
    assert m.n > 5 and m.pnl > 0 and m.wins / m.n > 0.6
    assert r.rule[2.0].n > 0  # the one-signal rule ran on the same period


def test_another_coin_feeds_the_model():
    rows, mid, end = tape()
    rows += [("x:BTC_JPY", t, ev) for src, t, ev in rows if src == "bybit"]
    rows.sort(key=lambda r: r[1])
    r = run(iter(rows), GMO, {"bybit": PERP, "x:BTC_JPY": PERP}, label="XRP",
            train=(0, mid), test=(mid, end), holds=(2.0,), cross=("BTC_JPY",))
    assert len(r.weights[2.0]) == len(feature_names(("BTC_JPY",)))
    assert r.model[2.0].pnl > 0


def test_no_lag_no_trades():
    rows, mid, end = tape(lag_steps=0)
    r = run(iter(rows), GMO, {"bybit": PERP}, label="XRP", train=(0, mid), test=(mid, end),
            holds=(2.0,))
    assert r.model[2.0].n == 0


async def test_the_command_reads_a_recording_and_compares(monkeypatch, capsys, tmp_path):
    import gzip
    import io
    import json
    from datetime import UTC, datetime

    import jsboard.cli as cli
    import jsboard.sim.s3 as s3
    from jsboard.feed.replay import _encode
    from jsboard.sim.s3 import S3Target

    rows, mid, end = tape()
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
            "gmo": {**spec, "symbol": "XRP_JPY", "tick_size": "0.001", "lot_size": "1",
                    "taker_bps": 0.0, "min_order": "1"},
            "bybit": {**spec, "symbol": "XRPUSDT", "tick_size": "0.0001", "lot_size": "1"},
        }}).encode(),
        # BTC has a spec but no recording on this day: it must be skipped, not fatal.
        target.meta_key("BTC_JPY"): json.dumps({"sources": {
            "binance": {"symbol": "BTCUSDT", "tick_size": "0.1", "lot_size": "0.001"},
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
        "fairprice", "s3://b/raw/lead/", "--only", "XRP_JPY", "--cross", "BTC_JPY",
        "--train", f"{iso(T0)},{iso(mid)}", "--test", f"{iso(mid)},{iso(end)}", "--holds", "2",
        "--save-model", str(tmp_path)])
    assert await args.func(args) == 0
    out = capsys.readouterr().out.splitlines()
    model = next(x for x in out if "フェア価格モデル" in x).split("\t")
    assert int(model[5]) > 5 and model[6].startswith("+")
    assert any("単純な後追い" in x for x in out)
    saved = json.loads((tmp_path / "XRP_JPY.json").read_text())
    assert len(saved["weights"]["2"]) == len(saved["names"]) and "2" in saved["margin_bps"]


def test_the_entry_margin_is_chosen_on_training_data_only():
    rows, mid, end = tape()
    r = run(iter(rows), GMO, {"bybit": PERP}, label="XRP", train=(T0, mid), test=(mid, end),
            holds=(2.0,), min_cal_trades=5)
    assert r.cal_n > 0 and r.train_n > 0
    assert r.chosen[2.0] in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    best = max((b.pnl, m) for (m, h), b in r.cal.items() if b.n >= 5)[1]
    assert r.chosen[2.0] == best
    # Changing only the test period cannot change the pick.
    rows2 = [(s, t, ev) for s, t, ev in rows if t < mid]
    r2 = run(iter(rows2), GMO, {"bybit": PERP}, label="XRP", train=(T0, mid), test=(mid, end),
             holds=(2.0,), min_cal_trades=5)
    assert r2.chosen == {} or r2.chosen[2.0] == r.chosen[2.0]


def test_a_fixed_margin_fits_on_all_of_training():
    rows, mid, end = tape()
    r = run(iter(rows), GMO, {"bybit": PERP}, label="XRP", train=(T0, mid), test=(mid, end),
            holds=(2.0,), margin_bps=2.0)
    assert r.cal == {} and r.cal_n == 0 and r.chosen == {2.0: 2.0}


def test_the_split_counts_recorded_time_not_clock_time():
    from jsboard.research.fairprice import active_split

    m = 60 * NS
    # Ten hours recorded, a two-day gap, then two more hours.
    starts = [T0 + i * 5 * m for i in range(120)] + [T0 + 58 * 60 * m + i * 5 * m for i in range(24)]
    cut, hours = active_split(starts, (T0, T0 + 61 * 60 * m), frac=0.7)
    assert abs(hours - 12.0) < 0.2
    # 70% of ~12 recorded hours falls inside the first block, not in the gap.
    assert T0 + 8 * 60 * m < cut < T0 + 9 * 60 * m


def test_a_stopped_recording_is_not_learned_from():
    rows, mid, end = tape()
    # Cut ten minutes out of the middle of the training data.
    gap = (T0 + 5 * 60 * NS, T0 + 15 * 60 * NS)
    cut = [r for r in rows if not gap[0] <= r[1] < gap[1]]
    full = run(iter(rows), GMO, {"bybit": PERP}, label="X", train=(T0, mid), test=(mid, end),
               holds=(2.0,), margin_bps=0.0)
    holed = run(iter(cut), GMO, {"bybit": PERP}, label="X", train=(T0, mid), test=(mid, end),
                holds=(2.0,), margin_bps=0.0)
    # The ten silent minutes add no samples.
    assert holed.train_n <= full.train_n - 5 * 600
