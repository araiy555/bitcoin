"""Binance spot market-data adapter.

The whole point of this file is the snapshot/diff handshake. Binance streams
depth *diffs* continuously but serves the base image over REST, so the two
have to be stitched together without dropping or double-applying an update.
Their documented procedure, which `_sync` follows exactly:

  1. open the diff stream and start buffering
  2. fetch the REST snapshot, note `lastUpdateId`
  3. discard buffered events whose `u <= lastUpdateId`
  4. the first event applied must straddle the snapshot: `U <= lastUpdateId+1 <= u`
  5. from then on every event must satisfy `U == prev_u + 1`; a gap means we
     missed a message and the book is unreliable, so resynchronise

Skipping step 4 or 5 gives a book that looks fine and is quietly wrong, which
is the failure mode worth engineering against.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import aiohttp
import websockets

from ..core.types import Instrument, Side
from ..net import make_session, ssl_context
from .base import DepthDelta, DepthSnapshot, Feed, FeedEvent, FeedStatus, TradeTick

log = logging.getLogger(__name__)

REST_BASE = "https://api.binance.com"
WS_BASE = "wss://stream.binance.com:9443/stream"

# Binance rejects limit values outside this set for the depth endpoint.
VALID_DEPTH_LIMITS = (5, 10, 20, 50, 100, 500, 1000, 5000)


class BinanceFeed(Feed):
    """Live L2 depth and aggregated trades for one spot symbol."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        depth_ms: int = 100,
        snapshot_limit: int = 1000,
        max_reconnect_delay: float = 30.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(instrument)
        if depth_ms not in (100, 1000):
            raise ValueError("Binance publishes spot depth diffs at 100ms or 1000ms only")
        if snapshot_limit not in VALID_DEPTH_LIMITS:
            raise ValueError(f"snapshot_limit must be one of {VALID_DEPTH_LIMITS}")
        self.depth_ms = depth_ms
        self.snapshot_limit = snapshot_limit
        self.max_reconnect_delay = max_reconnect_delay
        self._session = session
        self._owns_session = session is None

    @property
    def _symbol(self) -> str:
        return self.instrument.symbol.upper()

    @property
    def _stream_url(self) -> str:
        s = self.instrument.symbol.lower()
        return f"{WS_BASE}?streams={s}@depth@{self.depth_ms}ms/{s}@aggTrade"

    # ------------------------------------------------------------ transport

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = make_session()
            self._owns_session = True
        return self._session

    async def fetch_snapshot(self) -> DepthSnapshot:
        session = await self._ensure_session()
        url = f"{REST_BASE}/api/v3/depth"
        params = {"symbol": self._symbol, "limit": self.snapshot_limit}
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return self._parse_snapshot(payload)

    def _parse_snapshot(self, payload: dict) -> DepthSnapshot:
        return DepthSnapshot(
            bids=self._levels(payload["bids"]),
            asks=self._levels(payload["asks"]),
            last_update_id=int(payload["lastUpdateId"]),
        )

    def _levels(self, rows) -> tuple[tuple[int, int], ...]:
        inst = self.instrument
        return tuple((inst.to_ticks(p), inst.to_lots(q)) for p, q in rows)

    def _parse_depth_event(self, data: dict) -> DepthDelta:
        return DepthDelta(
            bids=self._levels(data["b"]),
            asks=self._levels(data["a"]),
            first_id=int(data["U"]),
            final_id=int(data["u"]),
            ts_ns=int(data["E"]) * 1_000_000,
        )

    def _parse_trade(self, data: dict) -> TradeTick:
        inst = self.instrument
        # "m" is *buyer is maker*, so a true value means the seller crossed.
        aggressor = Side.SELL if data["m"] else Side.BUY
        return TradeTick(
            price=inst.to_ticks(data["p"]),
            qty=inst.to_lots(data["q"]),
            aggressor=aggressor,
            trade_id=int(data.get("a", 0)),
            ts_ns=int(data["T"]) * 1_000_000,
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
                    log.warning("binance feed dropped (%s); reconnecting in %.0fs", exc, delay)
                    yield FeedStatus("disconnected", f"{type(exc).__name__}: {exc}")
                    await asyncio.sleep(delay)
        finally:
            if self._owns_session and self._session and not self._session.closed:
                await self._session.close()

    async def _session_once(self) -> AsyncIterator[FeedEvent]:
        """One websocket connection, from handshake to failure."""
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
        """Stitch the REST snapshot onto the live diff stream, then relay."""
        buffer: list[DepthDelta] = []
        snapshot: DepthSnapshot | None = None
        snapshot_task = asyncio.create_task(self.fetch_snapshot())
        prev_u: int | None = None

        try:
            async for raw in ws:
                msg = json.loads(raw)
                data = msg.get("data", msg)
                kind = data.get("e")

                if kind == "aggTrade":
                    yield self._parse_trade(data)
                    continue
                if kind != "depthUpdate":
                    continue

                delta = self._parse_depth_event(data)

                # --- phase 1: buffering until the snapshot lands ------------
                if snapshot is None:
                    buffer.append(delta)
                    if not snapshot_task.done():
                        continue
                    snapshot = snapshot_task.result()

                    # Drop everything the snapshot already accounts for.
                    buffer = [d for d in buffer if d.final_id > snapshot.last_update_id]
                    target = snapshot.last_update_id + 1
                    if not buffer or not (buffer[0].first_id <= target <= buffer[0].final_id):
                        # The snapshot is older than our earliest buffered diff:
                        # there is an unrecoverable hole. Take a fresh image.
                        yield FeedStatus("resyncing", "snapshot did not join the diff stream")
                        snapshot = None
                        buffer = []
                        snapshot_task = asyncio.create_task(self.fetch_snapshot())
                        continue

                    yield snapshot
                    for buffered in buffer:
                        yield buffered
                        prev_u = buffered.final_id
                    buffer = []
                    yield FeedStatus("live", f"synced at {snapshot.last_update_id}")
                    continue

                # --- phase 2: steady state, gap-checked ---------------------
                if prev_u is not None and delta.first_id != prev_u + 1:
                    log.warning(
                        "depth gap on %s: expected U=%d, got U=%d", self._symbol, prev_u + 1, delta.first_id
                    )
                    yield FeedStatus("resyncing", f"gap at update id {prev_u + 1}")
                    snapshot = None
                    prev_u = None
                    buffer = [delta]
                    snapshot_task = asyncio.create_task(self.fetch_snapshot())
                    continue

                prev_u = delta.final_id
                yield delta
        finally:
            if not snapshot_task.done():
                snapshot_task.cancel()
            else:
                # Consume any exception so it does not surface as "never retrieved".
                snapshot_task.exception() if not snapshot_task.cancelled() else None
