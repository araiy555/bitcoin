"""Event loop: pump a feed through the market maker.

Kept deliberately thin so the same loop drives live trading, a replay, and the
test suite. Anything that wants to observe (the UI, a recorder, an assertion)
hooks in through `on_update` rather than being wired into the loop itself.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..feed.base import Feed, FeedEvent
from ..mm.strategy import MarketMaker

Observer = Callable[[MarketMaker, FeedEvent], None]


class FeedClock:
    """"Now" as the data sees it, rather than as the wall clock sees it.

    A zero-delay backtest chews through an hour of market time in a second, so
    wall-clock requote intervals fire three times and latency never elapses.
    Reading the clock off the last event timestamp keeps requoting, order-entry
    latency and staleness all on the same simulated timeline.
    """

    __slots__ = ("market", "_fallback")

    def __init__(self, market) -> None:
        self.market = market
        self._fallback = time.time_ns

    def __call__(self) -> int:
        return self.market.last_update_ns or self._fallback()


def attach_virtual_clock(mm: MarketMaker) -> FeedClock:
    """Put the maker, its venue and its market view on simulated time."""
    clock = FeedClock(mm.market)
    mm.clock = clock
    mm.venue.clock = clock
    mm.market.clock = clock
    return clock


@dataclass(slots=True)
class RunResult:
    events: int = 0
    fills: int = 0
    elapsed_s: float = 0.0
    summary: dict = field(default_factory=dict)
    stopped_because: str = ""


async def run(
    feed: Feed,
    mm: MarketMaker,
    *,
    duration_s: float | None = None,
    max_events: int | None = None,
    on_update: Observer | None = None,
    quote: bool = True,
) -> RunResult:
    """Drive `mm` from `feed` until a stop condition trips."""
    started = time.monotonic()
    result = RunResult()
    reason = "feed exhausted"

    stream = feed.stream()
    try:
        async for event in stream:
            result.events += 1
            result.fills += len(mm.on_event(event))

            if quote:
                mm.requote()

            if on_update is not None:
                on_update(mm, event)

            if max_events is not None and result.events >= max_events:
                reason = "max events reached"
                break
            if duration_s is not None and time.monotonic() - started >= duration_s:
                reason = "duration reached"
                break
            if mm.risk.halted:
                reason = f"halted: {mm.risk.halt_reason}"
                break
    except asyncio.CancelledError:
        reason = "cancelled"
        raise
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()
        mm.flatten()
        result.elapsed_s = time.monotonic() - started
        result.summary = mm.summary()
        result.stopped_because = reason

    return result
