"""Binance USDⓈ-M perpetual futures market data.

Spot and USDⓈ-M futures are separate products on separate hosts, and the
depth-diff protocols are *not* the same. Reusing the spot adapter here would
produce a book that looks healthy and drifts wrong, which is the failure this
whole file exists to avoid.

  spot      first event must straddle:  U <= lastUpdateId + 1 <= u
            continuity:                 U == prev_u + 1

  futures   first event must straddle:  U <= lastUpdateId <= u
            continuity:                 pu == prev_u   ← a field spot has no

Two differences hide in there. Futures brackets `lastUpdateId` itself rather
than the one after it, and it carries `pu` (previous final update id) so
continuity is checked against a value the exchange states outright instead of
one we infer. Applying the spot rule to a futures stream mostly works, which
is exactly what makes it dangerous.

Beyond the book, a perpetual publishes state spot has no analogue for — mark
and index price, the funding it is paying, open interest, and forced orders.
Those are what make the perp lead or lag the spot market, so they are carried
as first-class events rather than derived later.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

import aiohttp
import websockets

from ..core.types import Instrument, Side
from ..net import make_session, ssl_context
from .base import (
    DepthDelta,
    DepthSnapshot,
    Feed,
    FeedEvent,
    FeedStatus,
    Liquidation,
    MarkPrice,
    OpenInterest,
    TradeTick,
)

log = logging.getLogger(__name__)

REST_BASE = "https://fapi.binance.com"
WS_BASE = "wss://fstream.binance.com/stream"

VALID_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000)
VALID_DEPTH_MS = (100, 250, 500)


class BinanceFuturesFeed(Feed):
    """Depth, trades, mark price, funding, open interest and liquidations."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        depth_ms: int = 100,
        snapshot_limit: int = 1000,
        open_interest_interval: float = 15.0,
        max_reconnect_delay: float = 30.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(instrument)
        if depth_ms not in VALID_DEPTH_MS:
            raise ValueError(f"futures depth diffs come at {VALID_DEPTH_MS}ms")
        if snapshot_limit not in VALID_DEPTH_LIMITS:
            raise ValueError(f"snapshot_limit must be one of {VALID_DEPTH_LIMITS}")
        self.depth_ms = depth_ms
        self.snapshot_limit = snapshot_limit
        self.open_interest_interval = open_interest_interval
        self.max_reconnect_delay = max_reconnect_delay
        self._session = session
        self._owns_session = session is None
        # Open interest has no stream, so it is polled beside the socket and
        # queued, keeping the main loop a single ordered sequence of events.
        self._oi_queue: asyncio.Queue[OpenInterest] = asyncio.Queue()

    @property
    def _symbol(self) -> str:
        return self.instrument.symbol.upper()

    @property
    def _stream_url(self) -> str:
        s = self.instrument.symbol.lower()
        streams = "/".join(
            (
                f"{s}@depth@{self.depth_ms}ms",
                f"{s}@aggTrade",
                f"{s}@markPrice@1s",
                f"{s}@forceOrder",
            )
        )
        return f"{WS_BASE}?streams={streams}"

    # ------------------------------------------------------------ transport

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = make_session()
            self._owns_session = True
        return self._session

    async def fetch_snapshot(self) -> DepthSnapshot:
        session = await self._ensure_session()
        async with session.get(
            f"{REST_BASE}/fapi/v1/depth",
            params={"symbol": self._symbol, "limit": self.snapshot_limit},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return DepthSnapshot(
            bids=self._levels(payload["bids"]),
            asks=self._levels(payload["asks"]),
            last_update_id=int(payload["lastUpdateId"]),
        )

    async def fetch_open_interest(self) -> OpenInterest:
        session = await self._ensure_session()
        async with session.get(
            f"{REST_BASE}/fapi/v1/openInterest",
            params={"symbol": self._symbol},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        stamped = int(payload.get("time", 0)) * 1_000_000
        lots = self.instrument.to_lots(payload["openInterest"])
        return OpenInterest(lots=lots, ts_ns=stamped) if stamped else OpenInterest(lots=lots)

    # -------------------------------------------------------------- parsing

    def _levels(self, rows) -> tuple[tuple[int, int], ...]:
        inst = self.instrument
        return tuple((inst.to_ticks(p), inst.to_lots(q)) for p, q in rows)

    def _parse_depth_event(self, data: dict) -> tuple[DepthDelta, int]:
        """Returns the delta and its `pu` (previous final update id)."""
        delta = DepthDelta(
            bids=self._levels(data["b"]),
            asks=self._levels(data["a"]),
            first_id=int(data["U"]),
            final_id=int(data["u"]),
            # `T` is transaction time and `E` event time; the former is when
            # the book actually changed, which is what lead-lag work needs.
            ts_ns=int(data.get("T", data["E"])) * 1_000_000,
        )
        return delta, int(data["pu"])

    def _parse_trade(self, data: dict) -> TradeTick:
        inst = self.instrument
        aggressor = Side.SELL if data["m"] else Side.BUY
        return TradeTick(
            price=inst.to_ticks(data["p"]),
            qty=inst.to_lots(data["q"]),
            aggressor=aggressor,
            trade_id=int(data.get("a", 0)),
            ts_ns=int(data["T"]) * 1_000_000,
        )

    def _parse_mark_price(self, data: dict) -> MarkPrice:
        inst = self.instrument
        return MarkPrice(
            mark=inst.to_ticks(data["p"]),
            index=inst.to_ticks(data["i"]),
            funding_rate=float(data["r"]),
            next_funding_ns=int(data["T"]) * 1_000_000,
            ts_ns=int(data["E"]) * 1_000_000,
        )

    def _parse_liquidation(self, data: dict) -> Liquidation:
        inst = self.instrument
        order = data["o"]
        # `ap` is the average fill price, reported as the *string* "0" while
        # nothing has filled — truthy, so it has to be compared numerically
        # rather than leaned on directly.
        avg = order.get("ap")
        price = avg if avg is not None and float(avg) > 0 else order["p"]
        return Liquidation(
            price=inst.to_ticks(price),
            qty=inst.to_lots(order["q"]),
            side=Side.BUY if order["S"].upper() == "BUY" else Side.SELL,
            ts_ns=int(order["T"]) * 1_000_000,
        )

    # --------------------------------------------------------------- stream

    async def stream(self) -> AsyncIterator[FeedEvent]:
        attempt = 0
        try:
            while True:
                try:
                    yield FeedStatus("connecting", self._stream_url)
                    async for event in self._session_once():
                        attempt = 0
                        yield event
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - any failure means reconnect
                    attempt += 1
                    delay = min(self.max_reconnect_delay, 2.0 ** min(attempt, 5))
                    log.warning("futures feed dropped (%s); reconnecting in %.0fs", exc, delay)
                    yield FeedStatus("disconnected", f"{type(exc).__name__}: {exc}")
                    await asyncio.sleep(delay)
        finally:
            if self._owns_session and self._session and not self._session.closed:
                await self._session.close()

    async def _session_once(self) -> AsyncIterator[FeedEvent]:
        async with websockets.connect(
            self._stream_url,
            ping_interval=20,
            ping_timeout=20,
            max_queue=2**16,
            ssl=ssl_context(),
        ) as ws:
            async for event in self._sync(ws):
                yield event

    async def _sync(self, ws) -> AsyncIterator[FeedEvent]:
        """Stitch the snapshot onto the diff stream using the *futures* rule."""
        buffer: list[tuple[DepthDelta, int]] = []
        snapshot: DepthSnapshot | None = None
        snapshot_task = asyncio.create_task(self.fetch_snapshot())
        oi_task = asyncio.create_task(self._poll_open_interest())
        prev_u: int | None = None

        try:
            async for raw in ws:
                # Open-interest polls arrive out of band; drain whatever is ready.
                while not self._oi_queue.empty():
                    yield self._oi_queue.get_nowait()

                msg = json.loads(raw)
                data = msg.get("data", msg)
                kind = data.get("e")

                if kind == "aggTrade":
                    yield self._parse_trade(data)
                    continue
                if kind == "markPriceUpdate":
                    yield self._parse_mark_price(data)
                    continue
                if kind == "forceOrder":
                    yield self._parse_liquidation(data)
                    continue
                if kind != "depthUpdate":
                    continue

                delta, pu = self._parse_depth_event(data)

                # --- phase 1: buffering until the snapshot lands ------------
                if snapshot is None:
                    buffer.append((delta, pu))
                    if not snapshot_task.done():
                        continue
                    snapshot = snapshot_task.result()

                    last = snapshot.last_update_id
                    buffer = [(d, p) for d, p in buffer if d.final_id >= last]
                    # Futures brackets lastUpdateId itself, not the one after.
                    if not buffer or not (buffer[0][0].first_id <= last <= buffer[0][0].final_id):
                        yield FeedStatus("resyncing", "snapshot did not join the diff stream")
                        snapshot = None
                        buffer = []
                        snapshot_task = asyncio.create_task(self.fetch_snapshot())
                        continue

                    yield snapshot
                    for buffered, _ in buffer:
                        yield buffered
                        prev_u = buffered.final_id
                    buffer = []
                    yield FeedStatus("live", f"synced at {last}")
                    continue

                # --- phase 2: steady state, checked against `pu` ------------
                if prev_u is not None and pu != prev_u:
                    log.warning(
                        "futures depth gap on %s: expected pu=%d, got pu=%d", self._symbol, prev_u, pu
                    )
                    yield FeedStatus("resyncing", f"gap: pu={pu} after u={prev_u}")
                    snapshot = None
                    prev_u = None
                    buffer = [(delta, pu)]
                    snapshot_task = asyncio.create_task(self.fetch_snapshot())
                    continue

                prev_u = delta.final_id
                yield delta
        finally:
            for task in (snapshot_task, oi_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                else:
                    task.exception() if not task.cancelled() else None

    async def _poll_open_interest(self) -> None:
        while True:
            try:
                self._oi_queue.put_nowait(await self.fetch_open_interest())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed poll is not fatal
                log.debug("open interest poll failed: %s", exc)
            await asyncio.sleep(self.open_interest_interval)
