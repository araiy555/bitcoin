"""Multi-source recording.

The point of `MultiCapture` is not that it writes a file — it is that the
file preserves *which* venue each event came from and *when we learned it*.
Lose either and the recording cannot answer a lead-lag question, so both are
pinned here, along with the property that one dead feed does not stop the
other.
"""

import asyncio
import json
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import (
    DepthDelta,
    DepthSnapshot,
    Feed,
    FeedStatus,
    Liquidation,
    MarkPrice,
    OpenInterest,
    TradeTick,
)
from jsboard.feed.replay import _decode
from jsboard.sim.capture import MultiCapture, write_meta

INST = Instrument("BTCUSDT", tick_size=Decimal("0.01"), lot_size=Decimal("0.00001"))
PERP = Instrument("BTCUSDT", tick_size=Decimal("0.1"), lot_size=Decimal("0.001"))


class ScriptedFeed(Feed):
    """Emits a fixed list of events, optionally pausing between them."""

    def __init__(self, instrument, events, *, gap_s: float = 0.0, hang: bool = False):
        super().__init__(instrument)
        self.events = list(events)
        self.gap_s = gap_s
        self.hang = hang
        self.closed = False

    async def stream(self):
        try:
            for event in self.events:
                if self.gap_s:
                    await asyncio.sleep(self.gap_s)
                yield event
            if self.hang:
                await asyncio.sleep(3600)
        finally:
            self.closed = True


def read_rows(path):
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------ tagging


async def test_every_row_records_its_source(tmp_path):
    out = tmp_path / "cap.jsonl"
    capture = MultiCapture(
        {
            "spot": ScriptedFeed(INST, [TradeTick(price=1, qty=2, aggressor=Side.BUY)]),
            "perp": ScriptedFeed(PERP, [OpenInterest(lots=99)]),
        },
        out,
    )
    await capture.run(max_events=2)

    rows = read_rows(out)
    assert {r["src"] for r in rows} == {"spot", "perp"}
    by_src = {r["src"]: r for r in rows}
    assert by_src["spot"]["k"] == "trade"
    assert by_src["perp"]["k"] == "oi"


async def test_receive_time_is_stored_beside_exchange_time(tmp_path):
    # An event stamped long ago must still record when we actually saw it:
    # a strategy could not have acted on the exchange timestamp.
    out = tmp_path / "cap.jsonl"
    stale = TradeTick(price=1, qty=1, aggressor=Side.SELL, ts_ns=1_000)
    capture = MultiCapture({"spot": ScriptedFeed(INST, [stale])}, out)
    await capture.run(max_events=1)

    row = read_rows(out)[0]
    assert row["ts_ns"] == 1_000
    assert row["rx_ns"] > 1_000_000_000_000_000  # a real wall-clock stamp


async def test_rows_stay_decodable_after_the_capture_tags_them(tmp_path):
    # `src`/`rx_ns` are extra keys on the same rows the replay codec reads;
    # if they leak into the constructor the recording is unreplayable.
    out = tmp_path / "cap.jsonl"
    events = [
        DepthSnapshot(bids=((10, 5),), asks=((11, 5),), last_update_id=1),
        DepthDelta(bids=((10, 0),), asks=(), first_id=2, final_id=2),
        MarkPrice(mark=100, index=99, funding_rate=0.0001, next_funding_ns=5),
        Liquidation(price=100, qty=3, side=Side.SELL),
        FeedStatus("live", "ok"),
    ]
    capture = MultiCapture({"perp": ScriptedFeed(PERP, events)}, out)
    await capture.run(max_events=len(events))

    for row in read_rows(out):
        row.pop("src")
        row.pop("rx_ns")
        assert _decode(row) in events


# -------------------------------------------------------------- interleaving


async def test_sources_interleave_rather_than_running_one_after_the_other(tmp_path):
    # Two feeds ticking at different rates must appear mixed. If the capture
    # drained one feed before starting the other, every "spot" row would
    # precede every "perp" row and the arrival ordering — the measurement —
    # would be an artefact of the recorder.
    out = tmp_path / "cap.jsonl"
    spot = ScriptedFeed(
        INST, [TradeTick(price=i, qty=1, aggressor=Side.BUY) for i in range(6)], gap_s=0.01
    )
    perp = ScriptedFeed(PERP, [OpenInterest(lots=i) for i in range(6)], gap_s=0.01)
    await MultiCapture({"spot": spot, "perp": perp}, out).run(max_events=12)

    order = [r["src"] for r in read_rows(out)]
    transitions = sum(1 for a, b in zip(order, order[1:], strict=False) if a != b)
    assert transitions >= 4, order


async def test_receive_times_are_non_decreasing(tmp_path):
    out = tmp_path / "cap.jsonl"
    spot = ScriptedFeed(INST, [TradeTick(price=i, qty=1, aggressor=Side.BUY) for i in range(5)])
    perp = ScriptedFeed(PERP, [OpenInterest(lots=i) for i in range(5)])
    await MultiCapture({"spot": spot, "perp": perp}, out).run(max_events=10)

    rx = [r["rx_ns"] for r in read_rows(out)]
    assert rx == sorted(rx)


async def test_a_silent_source_does_not_stop_a_live_one(tmp_path):
    out = tmp_path / "cap.jsonl"
    live = ScriptedFeed(INST, [TradeTick(price=i, qty=1, aggressor=Side.BUY) for i in range(4)])
    silent = ScriptedFeed(PERP, [], hang=True)
    result = await MultiCapture({"spot": live, "perp": silent}, out).run(max_events=4)

    assert result.stats["spot"].events == 4
    assert result.stats["perp"].events == 0


# -------------------------------------------------------------------- stats


async def test_stats_break_down_by_kind_per_source(tmp_path):
    out = tmp_path / "cap.jsonl"
    perp = ScriptedFeed(
        PERP,
        [
            TradeTick(price=1, qty=1, aggressor=Side.BUY),
            TradeTick(price=2, qty=1, aggressor=Side.SELL),
            OpenInterest(lots=7),
        ],
    )
    result = await MultiCapture({"perp": perp}, out).run(max_events=3)

    assert result.stats["perp"].by_kind == {"trade": 2, "oi": 1}
    assert result.total_events == 3


async def test_disconnects_are_counted_and_the_last_state_is_kept(tmp_path):
    out = tmp_path / "cap.jsonl"
    feed = ScriptedFeed(
        INST,
        [
            FeedStatus("live", ""),
            FeedStatus("disconnected", "socket closed"),
            FeedStatus("disconnected", "again"),
            FeedStatus("live", "back"),
        ],
    )
    result = await MultiCapture({"spot": feed}, out).run(max_events=4)

    assert result.stats["spot"].errors == 2
    assert result.stats["spot"].status == "live"


async def test_last_seen_timestamp_ignores_events_without_one(tmp_path):
    # A zero stamp would otherwise read as "this feed went silent in 1970".
    out = tmp_path / "cap.jsonl"
    feed = ScriptedFeed(
        INST,
        [
            TradeTick(price=1, qty=1, aggressor=Side.BUY, ts_ns=5_000),
            TradeTick(price=2, qty=1, aggressor=Side.BUY, ts_ns=0),
        ],
    )
    result = await MultiCapture({"spot": feed}, out).run(max_events=2)

    assert result.stats["spot"].last_ts_ns == 5_000


# ---------------------------------------------------------------- lifecycle


async def test_max_events_stops_the_recording(tmp_path):
    out = tmp_path / "cap.jsonl"
    feed = ScriptedFeed(INST, [OpenInterest(lots=i) for i in range(100)], hang=True)
    result = await MultiCapture({"perp": feed}, out).run(max_events=5)

    assert result.total_events == 5
    assert len(read_rows(out)) == 5
    assert result.stopped_because == "max events reached"


async def test_duration_stops_a_feed_that_never_ends(tmp_path):
    out = tmp_path / "cap.jsonl"
    feed = ScriptedFeed(INST, [], hang=True)
    result = await MultiCapture({"spot": feed}, out).run(duration_s=0.2)

    assert result.stopped_because == "duration reached"
    assert result.duration_s >= 0.2


async def test_producers_are_shut_down_when_the_run_ends(tmp_path):
    out = tmp_path / "cap.jsonl"
    feed = ScriptedFeed(INST, [OpenInterest(lots=1)], hang=True)
    await MultiCapture({"perp": feed}, out).run(max_events=1)

    assert feed.closed


async def test_appending_keeps_the_earlier_recording(tmp_path):
    out = tmp_path / "cap.jsonl"
    for _ in range(2):
        feed = ScriptedFeed(INST, [OpenInterest(lots=1)])
        await MultiCapture({"perp": feed}, out).run(max_events=1)

    assert len(read_rows(out)) == 2


async def test_the_output_directory_is_created(tmp_path):
    out = tmp_path / "nested" / "deeper" / "cap.jsonl"
    feed = ScriptedFeed(INST, [OpenInterest(lots=1)])
    await MultiCapture({"perp": feed}, out).run(max_events=1)

    assert out.exists()


async def test_a_capture_with_no_sources_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        MultiCapture({}, tmp_path / "cap.jsonl")


async def test_the_event_callback_sees_the_source_and_the_event(tmp_path):
    out = tmp_path / "cap.jsonl"
    seen = []
    tick = TradeTick(price=1, qty=1, aggressor=Side.BUY)
    capture = MultiCapture(
        {"spot": ScriptedFeed(INST, [tick])}, out, on_event=lambda n, e: seen.append((n, e))
    )
    await capture.run(max_events=1)

    assert seen == [("spot", tick)]


# --------------------------------------------------------------------- meta


def test_meta_records_the_tick_and_lot_sizes(tmp_path):
    # A replay that guesses these silently rescales every price in the file.
    out = tmp_path / "cap.jsonl"
    meta = write_meta(
        out,
        {
            "spot": {"symbol": "BTCUSDT", "tick_size": "0.01", "lot_size": "0.00001"},
            "perp": {"symbol": "BTCUSDT", "tick_size": "0.1", "lot_size": "0.001"},
        },
    )

    assert meta.name == "cap.jsonl.meta.json"
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["sources"]["spot"]["tick_size"] == "0.01"
    assert payload["sources"]["perp"]["tick_size"] == "0.1"
