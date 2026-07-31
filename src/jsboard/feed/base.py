"""Normalised market-data events and the feed interface.

Every adapter converts venue-specific messages into these types, in ticks and
lots, so that nothing downstream of here knows which exchange it is talking to.
"""

from __future__ import annotations

import abc
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..core.types import Instrument, Side


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    """A full L2 image. Replaces whatever the consumer currently holds."""

    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    last_update_id: int
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True)
class DepthDelta:
    """Absolute level quantities, not increments. `qty == 0` deletes a level.

    `first_id`/`final_id` are the venue's update-id range for this message and
    are what lets the consumer detect a dropped message.
    """

    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    first_id: int
    final_id: int
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True)
class TradeTick:
    """A public print. `aggressor` is the side that crossed the spread."""

    price: int
    qty: int
    aggressor: Side
    trade_id: int = 0
    ts_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True)
class FeedStatus:
    """Connection lifecycle, surfaced so the UI can show a stale book."""

    state: str  # connecting | live | resyncing | disconnected
    detail: str = ""
    ts_ns: int = field(default_factory=time.time_ns)


FeedEvent = DepthSnapshot | DepthDelta | TradeTick | FeedStatus


class Feed(abc.ABC):
    """A source of market data for one instrument."""

    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument

    @abc.abstractmethod
    def stream(self) -> AsyncIterator[FeedEvent]:
        """Yield events until cancelled. Must reconnect on its own."""
        raise NotImplementedError

    def __aiter__(self) -> AsyncIterator[FeedEvent]:
        return self.stream()
