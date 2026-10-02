"""The live code's notion of now, which a replay can take over.

Live, these are the system clocks. The bitbank simulator runs the very same
venue and executor code on recorded data, so it points both at the
recording's time instead; nothing else in the live path reads a clock.
"""

from __future__ import annotations

import time
from collections.abc import Callable

_monotonic: Callable[[], float] = time.monotonic
_wall: Callable[[], float] = time.time


def monotonic() -> float:
    return _monotonic()


def wall() -> float:
    return _wall()


def use(now_s: Callable[[], float]) -> None:
    """Drive both clocks from one source of seconds (the replay's)."""
    global _monotonic, _wall
    _monotonic = _wall = now_s


def reset() -> None:
    global _monotonic, _wall
    _monotonic, _wall = time.monotonic, time.time
