"""Record several venues at once onto one timeline.

Lead-lag work needs spot and perp events interleaved in the order they
actually happened, so the two feeds run concurrently and write into a single
file rather than one file each. Two separate recordings could be merged
afterwards, but only by trusting the exchange timestamps to be comparable
across products — and the arrival order is itself part of what we want to
measure.

Each line carries a `src` tag, so a replay can rebuild one book per source
while preserving the interleaving between them:

    {"src": "spot", "k": "delta", ...}
    {"src": "perp", "k": "mark",  ...}

The writer records the *receive* time alongside the exchange time. They are
not the same thing, and which one is right depends on the question: exchange
time for what the market did, receive time for what a strategy could have
known. Storing only one throws away the ability to ask the other later.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..core.market import MarketView
from ..feed.base import DepthDelta, DepthSnapshot, Feed, FeedEvent, FeedStatus, MarkPrice
from ..feed.replay import RX_KEY, SOURCE_KEY, _encode


@dataclass(slots=True)
class SourceStats:
    """Per-source counts, so a silent stream is visible while recording."""

    events: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    last_ts_ns: int = 0
    status: str = "connecting"
    errors: int = 0

    def record(self, kind: str, ts_ns: int) -> None:
        self.events += 1
        self.by_kind[kind] = self.by_kind.get(kind, 0) + 1
        if ts_ns:
            self.last_ts_ns = ts_ns


@dataclass(slots=True)
class CaptureResult:
    path: Path
    started_ns: int
    ended_ns: int
    stats: dict[str, SourceStats]
    stopped_because: str = ""

    @property
    def duration_s(self) -> float:
        return (self.ended_ns - self.started_ns) / 1e9

    @property
    def total_events(self) -> int:
        return sum(s.events for s in self.stats.values())


class MultiCapture:
    """Run several feeds concurrently and write their events to one JSONL."""

    def __init__(
        self,
        sources: dict[str, Feed],
        path: str | Path,
        *,
        on_event: Callable[[str, FeedEvent], None] | None = None,
        flush_every: int = 500,
        sink: object | None = None,
        basis_sample_ms: float | None = None,
        basis_depth: int = 20,
    ) -> None:
        if not sources:
            raise ValueError("capture needs at least one source")
        self.sources = sources
        self.path = Path(path)
        self.on_event = on_event
        self.flush_every = flush_every
        self.sink = sink
        """Where the lines go. A plain append-mode file when absent; anything
        with write/flush/close otherwise, which is how the S3 uploader slots in
        without the recorder knowing about buckets or rotation."""
        if basis_sample_ms is not None and basis_sample_ms <= 0:
            raise ValueError("basis sample interval must be positive")
        if basis_depth <= 0:
            raise ValueError("basis depth must be positive")
        self.basis_sample_ns = (
            None if basis_sample_ms is None else int(basis_sample_ms * 1e6)
        )
        self.basis_depth = basis_depth
        self._compact_views = {
            name: MarketView(feed.instrument, depth=basis_depth)
            for name, feed in sources.items()
        }
        self._compact_last_ns = {name: 0 for name in sources}
        self._compact_update_id = {name: 0 for name in sources}
        self.stats: dict[str, SourceStats] = {name: SourceStats() for name in sources}

    def _compact_basis_event(
        self,
        name: str,
        event: FeedEvent,
        monotonic_ns: int,
    ) -> FeedEvent | None:
        """Keep only sampled executable books and perpetual funding state.

        A funding/basis hold lasts hours, so storing every 100ms delta and
        every public trade is wasteful.  The complete live book is still
        rebuilt here; once per interval it is emitted as a full depth
        snapshot, which remains independently replayable.
        """
        if self.basis_sample_ns is None:
            return event
        if isinstance(event, (DepthSnapshot, DepthDelta)):
            view = self._compact_views[name]
            view.apply(event)
            if isinstance(event, DepthSnapshot):
                self._compact_update_id[name] = event.last_update_id
            else:
                self._compact_update_id[name] = event.final_id
            last_ns = self._compact_last_ns[name]
            if last_ns and monotonic_ns - last_ns < self.basis_sample_ns:
                return None
            snapshot = view.snapshot(self.basis_depth)
            if not snapshot.bids or not snapshot.asks:
                return None
            self._compact_last_ns[name] = monotonic_ns
            return DepthSnapshot(
                bids=tuple((level.price, level.qty) for level in snapshot.bids),
                asks=tuple((level.price, level.qty) for level in snapshot.asks),
                last_update_id=self._compact_update_id[name],
                ts_ns=getattr(event, "ts_ns", 0),
            )
        if isinstance(event, (MarkPrice, FeedStatus)):
            return event
        return None

    @staticmethod
    def _source_context(name: str, feed: Feed) -> dict[str, str]:
        """Derive stable envelope labels without teaching feeds about storage."""
        class_name = type(feed).__name__.lower()
        if name in {"binance", "bybit"}:
            venue, market = name, "perp"
        elif name in {"spot", "perp"}:
            venue, market = "binance", name
        else:
            venue, market = name, ""
        if "synthetic" in class_name:
            source_mode = "synthetic"
        elif "replay" in class_name:
            source_mode = "replay"
        elif any(token in class_name for token in ("binance", "bybit")):
            source_mode = "ws"
        else:
            source_mode = "unknown"
        instrument = feed.instrument
        pair = f"{instrument.base}/{instrument.quote}" if instrument.base and instrument.quote else instrument.symbol
        instrument_id = f"{pair}:{market}" if market else pair
        return {
            "venue": venue,
            "market_type": market,
            "symbol_native": instrument.symbol,
            "instrument_id": instrument_id,
            "source_mode": source_mode,
        }

    async def run(
        self, *, duration_s: float | None = None, max_events: int | None = None
    ) -> CaptureResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        queue: asyncio.Queue[tuple[str, FeedEvent, int, int] | None] = asyncio.Queue(maxsize=100_000)
        started = time.time_ns()
        capture_id = uuid.uuid4().hex
        contexts = {name: self._source_context(name, feed) for name, feed in self.sources.items()}
        reason = "stopped"

        producers = [
            asyncio.create_task(self._pump(name, feed, queue), name=f"capture:{name}")
            for name, feed in self.sources.items()
        ]

        written = 0
        try:
            with self._open_sink() as fh:
                while True:
                    if duration_s is not None and (time.time_ns() - started) / 1e9 >= duration_s:
                        reason = "duration reached"
                        break
                    if max_events is not None and written >= max_events:
                        reason = "max events reached"
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=0.5)
                    except TimeoutError:
                        # No data for half a second is normal on a quiet perp;
                        # loop so the stop conditions still get evaluated.
                        continue
                    if item is None:
                        continue

                    name, event, received_ns, monotonic_ns = item
                    event = self._compact_basis_event(name, event, monotonic_ns)
                    if event is None:
                        continue
                    row = _encode(event)
                    row[SOURCE_KEY] = name
                    row[RX_KEY] = received_ns
                    context = contexts[name]
                    row.update(
                        {
                            "schema_version": "1.0",
                            "capture_id": capture_id,
                            "event_seq": written + 1,
                            **context,
                            "event_type": row["k"],
                            "ts_exchange_ns": row.get("ts_ns"),
                            "ts_receive_ns": monotonic_ns,
                            "ts_wall_ns": received_ns,
                            "connection_id": f"{capture_id}:{name}",
                        }
                    )
                    fh.write(json.dumps(row) + "\n")
                    written += 1

                    self.stats[name].record(row["k"], row.get("ts_ns", 0))
                    if isinstance(event, FeedStatus):
                        self.stats[name].status = event.state
                        if event.state == "disconnected":
                            self.stats[name].errors += 1
                    if self.on_event is not None:
                        self.on_event(name, event)

                    if written % self.flush_every == 0:
                        fh.flush()
        except (KeyboardInterrupt, asyncio.CancelledError):
            reason = "interrupted"
        finally:
            for task in producers:
                task.cancel()
            for task in producers:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

        return CaptureResult(
            path=self.path,
            started_ns=started,
            ended_ns=time.time_ns(),
            stats=self.stats,
            stopped_because=reason,
        )

    def _open_sink(self):
        return self.sink if self.sink is not None else self.path.open("a", encoding="utf-8")

    async def _pump(self, name: str, feed: Feed, queue: asyncio.Queue) -> None:
        """Drain one feed into the shared queue, tagging the arrival time."""
        stream = feed.stream()
        try:
            async for event in stream:
                # Wall time keeps legacy replay compatible with exchange epoch
                # stamps. Monotonic time is the canonical receive clock for
                # cross-market ordering inside one capture.
                await queue.put((name, event, time.time_ns(), time.monotonic_ns()))
        except asyncio.CancelledError:
            raise
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()


def write_meta(path: Path, sources: dict[str, dict], **metadata) -> Path:
    """Store each source's instrument spec beside the capture.

    Without it a replay has to guess tick and lot sizes, and a wrong guess
    silently rescales every price in the file.
    """
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(
        json.dumps({"sources": sources, **metadata}, indent=2), encoding="utf-8"
    )
    return meta_path
