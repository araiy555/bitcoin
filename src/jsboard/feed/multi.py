"""Best quote and tape for many symbols at once.

The dynamic screen needs two things per symbol and nothing else: where the
touch is, and which side each print took. Running the full depth feed for
sixteen symbols would mean sixteen REST snapshots and sixteen book states to
keep in sync, all to derive a best bid and offer the venue already publishes
directly. `bookTicker` gives it in one message per change.

The two connections are not a choice. Probing established that on
fstream.binance.com the plain path serves depth and bookTicker while the
`/market` path serves aggTrade, markPrice and forceOrder — each declines what
the other carries, silently, without an error. So the quotes come from one
socket and the trades from the other.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

import websockets

from ..net import ssl_context

QUOTE_WS = "wss://fstream.binance.com/stream"
TRADE_WS = "wss://fstream.binance.com/market/stream"

NS_PER_MS = 1_000_000


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    ts_ns: int


@dataclass(frozen=True, slots=True)
class Print:
    symbol: str
    aggressor_sign: int
    """+1 when the taker bought, -1 when it sold."""
    ts_ns: int


def _chunk(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _parse(msg: dict) -> Quote | Print | None:
    data = msg.get("data", msg)
    if not isinstance(data, dict):
        return None
    kind = data.get("e")
    if kind == "bookTicker":
        try:
            bid, ask = float(data["b"]), float(data["a"])
        except (KeyError, TypeError, ValueError):
            return None
        ts = int(data.get("T") or data.get("E") or 0) * NS_PER_MS
        return Quote(data["s"], bid, ask, ts)
    if kind == "aggTrade":
        # `m` is "the buyer was the maker", so a true value means the
        # aggressor sold. Reading it the other way flips every sign in the
        # measurement, which would look like a working strategy.
        sign = -1 if data.get("m") else 1
        ts = int(data.get("T") or data.get("E") or 0) * NS_PER_MS
        return Print(data["s"], sign, ts)
    return None


@contextlib.contextmanager
def _quiet_sockets():
    """Mute the library's own error logging for the length of a run.

    Every socket here is under a reconnect loop that already handles a drop,
    and cancelling one at the deadline makes the library log a traceback from
    a callback no caller can catch. Left alone, a normal five-minute run ends
    in a wall of tracebacks that mean nothing, and a genuinely dead connection
    looks exactly the same as a healthy one. Connection health is reported by
    the event and print counts instead, which is what actually decides whether
    a run is usable.
    """
    # Two loggers, because the traceback comes from two places: the library
    # when a read fails, and asyncio's default handler when the failure lands
    # in a callback that no `await` can catch.
    loggers = [logging.getLogger("websockets"), logging.getLogger("asyncio")]
    before = [x.level for x in loggers]
    for x in loggers:
        x.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for x, level in zip(loggers, before, strict=True):
            x.setLevel(level)


async def _pump(url: str, out: asyncio.Queue) -> None:
    """One socket, reconnecting, decoded onto the shared queue."""
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ssl=ssl_context()) as ws:
                async for raw in ws:
                    event = _parse(json.loads(raw))
                    if event is not None:
                        await out.put(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a dropped socket is expected
            await asyncio.sleep(1.0)


async def stream(
    symbols: list[str], *, duration_s: float, per_socket: int = 100
) -> AsyncIterator[Quote | Print]:
    """Quotes and prints for every symbol, interleaved, for `duration_s`."""
    lower = [s.lower() for s in symbols]
    queue: asyncio.Queue = asyncio.Queue(maxsize=200_000)

    urls = []
    for group in _chunk([f"{s}@bookTicker" for s in lower], per_socket):
        urls.append(f"{QUOTE_WS}?streams={'/'.join(group)}")
    for group in _chunk([f"{s}@aggTrade" for s in lower], per_socket):
        urls.append(f"{TRADE_WS}?streams={'/'.join(group)}")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + duration_s
    with _quiet_sockets():
        tasks = [asyncio.create_task(_pump(url, queue)) for url in urls]
        try:
            while True:
                left = deadline - loop.time()
                if left <= 0:
                    return
                try:
                    yield await asyncio.wait_for(queue.get(), timeout=min(left, 1.0))
                except TimeoutError:
                    continue
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            # Cancelling a task does not run the transport's own
            # connection-lost callback; that lands on the loop a tick later.
            # Leaving the mute before then puts the traceback back.
            await asyncio.sleep(0.1)
