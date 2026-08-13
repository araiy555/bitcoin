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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..feed.base import Feed, FeedEvent, FeedStatus
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
    ) -> None:
        if not sources:
            raise ValueError("capture needs at least one source")
        self.sources = sources
        self.path = Path(path)
        self.on_event = on_event
        self.flush_every = flush_every
        self.stats: dict[str, SourceStats] = {name: SourceStats() for name in sources}

    async def run(
        self, *, duration_s: float | None = None, max_events: int | None = None
    ) -> CaptureResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        queue: asyncio.Queue[tuple[str, FeedEvent, int] | None] = asyncio.Queue(maxsize=100_000)
        started = time.time_ns()
        reason = "stopped"

        producers = [
            asyncio.create_task(self._pump(name, feed, queue), name=f"capture:{name}")
            for name, feed in self.sources.items()
        ]

        written = 0
        try:
            with self.path.open("a", encoding="utf-8") as fh:
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

                    name, event, received_ns = item
                    row = _encode(event)
                    row[SOURCE_KEY] = name
                    row[RX_KEY] = received_ns
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

    async def _pump(self, name: str, feed: Feed, queue: asyncio.Queue) -> None:
        """Drain one feed into the shared queue, tagging the arrival time."""
        stream = feed.stream()
        try:
            async for event in stream:
                await queue.put((name, event, time.time_ns()))
        except asyncio.CancelledError:
            raise
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()


def write_meta(path: Path, sources: dict[str, dict]) -> Path:
    """Store each source's instrument spec beside the capture.

    Without it a replay has to guess tick and lot sizes, and a wrong guess
    silently rescales every price in the file.
    """
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps({"sources": sources}, indent=2), encoding="utf-8")
    return meta_path
