"""Bybit V5 public order-book adapter.

Bybit sends a complete snapshot immediately after subscription and absolute
quantity deltas afterwards.  A later snapshot (or update id 1) replaces the
book, which is also how the venue recovers clients after a service restart.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import websockets

from ..core.types import Instrument
from ..net import ssl_context
from .base import DepthDelta, DepthSnapshot, Feed, FeedEvent, FeedStatus, MarkPrice

log = logging.getLogger(__name__)

PUBLIC_WS = {
    "spot": "wss://stream.bybit.com/v5/public/spot",
    "linear": "wss://stream.bybit.com/v5/public/linear",
}
VALID_DEPTHS = (1, 50, 200, 1000)


class BybitFeed(Feed):
    """Live Bybit L2 depth for one spot or USDT perpetual instrument."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        category: str = "linear",
        depth: int = 50,
        max_reconnect_delay: float = 30.0,
    ) -> None:
        super().__init__(instrument)
        if category not in PUBLIC_WS:
            raise ValueError(f"Bybit category must be one of {tuple(PUBLIC_WS)}")
        if depth not in VALID_DEPTHS:
            raise ValueError(f"Bybit depth must be one of {VALID_DEPTHS}")
        self.category = category
        self.depth = depth
        self.max_reconnect_delay = max_reconnect_delay
        self._ticker: dict = {}

    @property
    def topic(self) -> str:
        return f"orderbook.{self.depth}.{self.instrument.symbol.upper()}"

    @property
    def stream_url(self) -> str:
        return PUBLIC_WS[self.category]

    def _levels(self, rows) -> tuple[tuple[int, int], ...]:
        inst = self.instrument
        return tuple((inst.to_ticks(price), inst.to_lots(qty)) for price, qty in rows)

    def parse_message(self, payload: dict) -> DepthSnapshot | DepthDelta | MarkPrice | None:
        """Normalise one documented orderbook or derivatives ticker message."""
        topic = payload.get("topic")
        if topic == f"tickers.{self.instrument.symbol.upper()}":
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                return None
            # Derivatives ticker deltas omit unchanged fields, so retain the
            # last complete values rather than turning a missing rate into 0.
            self._ticker.update(data)
            mark = self._ticker.get("markPrice")
            index = self._ticker.get("indexPrice")
            funding = self._ticker.get("fundingRate")
            next_funding = self._ticker.get("nextFundingTime")
            if not all(value not in (None, "") for value in (mark, index, funding, next_funding)):
                return None
            ts_ns = int(payload.get("ts") or 0) * 1_000_000
            return MarkPrice(
                mark=self.instrument.to_ticks(mark),
                index=self.instrument.to_ticks(index),
                funding_rate=float(funding),
                next_funding_ns=int(next_funding) * 1_000_000,
                ts_ns=ts_ns,
            )
        if topic != self.topic:
            return None
        data = payload.get("data") or {}
        update_id = int(data.get("u", 0))
        ts_ns = int(data.get("cts") or payload.get("cts") or payload.get("ts") or 0) * 1_000_000
        bids = self._levels(data.get("b", ()))
        asks = self._levels(data.get("a", ()))
        if payload.get("type") == "snapshot" or update_id == 1:
            return DepthSnapshot(bids, asks, update_id, ts_ns)
        if payload.get("type") == "delta":
            return DepthDelta(bids, asks, update_id, update_id, ts_ns)
        return None

    async def stream(self) -> AsyncIterator[FeedEvent]:
        attempt = 0
        while True:
            try:
                yield FeedStatus("connecting", self.stream_url)
                async with websockets.connect(
                    self.stream_url,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=2**16,
                    ssl=ssl_context(),
                ) as ws:
                    topics = [self.topic]
                    if self.category == "linear":
                        topics.append(f"tickers.{self.instrument.symbol.upper()}")
                    await ws.send(json.dumps({"op": "subscribe", "args": topics}))
                    live = False
                    async for raw in ws:
                        payload = json.loads(raw)
                        if payload.get("success") is False:
                            raise RuntimeError(payload.get("ret_msg") or "Bybit subscription failed")
                        event = self.parse_message(payload)
                        if event is None:
                            continue
                        if not live:
                            live = True
                            attempt = 0
                            yield FeedStatus("live", self.topic)
                        yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect any broken stream
                attempt += 1
                delay = min(self.max_reconnect_delay, 2.0 ** min(attempt, 5))
                log.warning("bybit feed dropped (%s); reconnecting in %.0fs", exc, delay)
                yield FeedStatus("disconnected", f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(delay)
