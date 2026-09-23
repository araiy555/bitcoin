"""A second venue in the recording, watched but never traded.

The stock rule that passed was "when the future moves 5bps away, pull the side
it is moving toward". Here the other exchange plays the future. What matters
is that its book reaches the gate in file order — never ahead of our own —
and that it never reaches our book, where it would build one that never
existed.
"""

import json

import pytest

from jsboard.cli import ConfigError, attach_lead, build_instrument, build_maker, build_parser
from jsboard.feed.base import DepthSnapshot
from jsboard.feed.replay import ReplayFeed, _encode
from jsboard.sim.runner import attach_virtual_clock

BASE_NS = 1_757_000_000_000_000_000
SPEC = {
    "symbol": "BTCUSDT",
    "tick_size": "0.1",
    "lot_size": "0.001",
    "base": "BTC",
    "quote": "USDT",
    "market": "perp",
}


def book(mid: float, ts_ns: int) -> DepthSnapshot:
    bid, ask = round((mid - 0.05) * 10), round((mid + 0.05) * 10)
    return DepthSnapshot(
        bids=tuple((bid - i, 5000) for i in range(5)),
        asks=tuple((ask + i, 5000) for i in range(5)),
        last_update_id=ts_ns,
        ts_ns=ts_ns,
    )


def recording(tmp_path, points):
    path = tmp_path / "two.jsonl"
    rows = [{"k": "status", "state": "live", "detail": "", "ts_ns": BASE_NS, "src": "binance"}]
    for seconds, source, mid in points:
        now = BASE_NS + int(seconds * 1e9)
        row = _encode(book(mid, now))
        row.update({"src": source, "rx_ns": now})
        rows.append(row)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    path.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"sources": {"binance": SPEC, "bybit": SPEC}})
    )
    return path


def sweep_args(path, *extra):
    return build_parser().parse_args(
        ["sweep", str(path), "--source", "binance", "--symbol", "BTCUSDT", *extra]
    )


@pytest.mark.asyncio
async def test_the_lead_goes_to_the_callback_not_the_book(tmp_path):
    path = recording(tmp_path, [(0, "binance", 100.0), (1, "bybit", 100.2), (2, "binance", 100.0)])
    seen = []
    feed = ReplayFeed(
        build_instrument("BTCUSDT", "0.1", "0.001"),
        path,
        speed=0.0,
        source="binance",
        lead_source="bybit",
        on_lead=seen.append,
    )
    ours = [e async for e in feed.stream() if isinstance(e, DepthSnapshot)]
    assert len(ours) == 2
    assert len(seen) == 1
    assert seen[0].asks[0][0] == round(100.25 * 10)


def test_an_unknown_lead_is_refused(tmp_path):
    path = recording(tmp_path, [(0, "binance", 100.0)])
    args = sweep_args(path, "--lead-source", "okx")
    mm = build_maker(build_instrument("BTCUSDT", "0.1", "0.001"), args)
    with pytest.raises(ConfigError, match="okx"):
        attach_lead(mm, path, args)


def test_the_lead_cannot_be_our_own_venue(tmp_path):
    path = recording(tmp_path, [(0, "binance", 100.0)])
    args = sweep_args(path, "--lead-source", "binance")
    mm = build_maker(build_instrument("BTCUSDT", "0.1", "0.001"), args)
    with pytest.raises(ConfigError):
        attach_lead(mm, path, args)


@pytest.mark.asyncio
async def test_a_lead_that_jumps_first_pulls_our_ask(tmp_path):
    # Both venues sit together, then bybit jumps +20bps while binance lags.
    points = []
    for s in range(0, 10):
        points += [(s, "binance", 100.0), (s + 0.1, "bybit", 100.0)]
    points += [(10.0, "bybit", 100.2), (10.3, "binance", 100.0)]
    path = recording(tmp_path, points)

    args = sweep_args(
        path, "--lead-source", "bybit", "--lead-threshold-bps", "5", "--requote-ms", "0"
    )
    inst = build_instrument("BTCUSDT", "0.1", "0.001")
    mm = build_maker(inst, args)
    attach_virtual_clock(mm)
    feed = ReplayFeed(
        inst,
        path,
        speed=0.0,
        source="binance",
        lead_source="bybit",
        on_lead=attach_lead(mm, path, args),
    )
    async for event in feed.stream():
        mm.on_event(event)
        mm.requote()
    assert mm.toxicity.lead_blocks >= 1
    assert "no asks" in mm.last_toxicity.reason
