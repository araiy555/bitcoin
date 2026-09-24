"""GMO Coin public order book and trades.

GMO pushes the whole book on every change (`orderbooks`), so each message is a
:class:`DepthSnapshot`. Trades are subscribed with ``TAKER_ONLY`` so that each
print arrives once, carrying the side that crossed the spread.

The venue accepts one subscribe request per second on a connection; the second
subscription is sent after a pause, or it is silently dropped and the
recording has a book with no trades.

Leverage symbols (``XRP_JPY``) and spot symbols (``XRP``) are separate books
with separate fees; the symbol is passed through exactly as the venue names it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal

import websockets

from ..core.types import Instrument, Side
from ..net import ssl_context
from .base import DepthSnapshot, Feed, FeedEvent, FeedStatus, TradeTick

log = logging.getLogger(__name__)

REST_URL = "https://api.coin.z.com/public/v1"
WS_URL = "wss://api.coin.z.com/ws/public/v1"
SUBSCRIBE_GAP_S = 1.2


def ts_ns(text: str | None) -> int:
    """`2026-09-24T01:02:03.456Z` → nanoseconds since the epoch."""
    if not text:
        return 0
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1e9)


def instrument_from_rule(rule: dict) -> Instrument:
    """One row of `/public/v1/symbols`."""
    symbol = rule["symbol"]
    base, _, quote = symbol.partition("_")
    return Instrument(
        symbol=symbol,
        tick_size=Decimal(str(rule["tickSize"])).normalize(),
        lot_size=Decimal(str(rule["sizeStep"])).normalize(),
        base=base,
        quote=quote or "JPY",
    )


async def fetch_instrument(symbol: str) -> Instrument:
    import aiohttp

    from ..net import make_session

    async with make_session() as session, session.get(
        f"{REST_URL}/symbols", timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    for rule in payload.get("data") or []:
        if rule.get("symbol") == symbol:
            return instrument_from_rule(rule)
    raise RuntimeError(f"{symbol} は GMOコインにありません（レバレッジは XRP_JPY の形）")


class GmoFeed(Feed):
    """Live GMO Coin book snapshots and taker trades for one symbol."""

    def __init__(self, instrument: Instrument, *, max_reconnect_delay: float = 30.0) -> None:
        super().__init__(instrument)
        self.symbol = instrument.symbol
        self.max_reconnect_delay = max_reconnect_delay
        self._book_seq = 0

    def _levels(self, rows) -> tuple[tuple[int, int], ...]:
        inst = self.instrument
        out = []
        for row in rows or ():
            lots = inst.to_lots(row["size"])
            if lots > 0:
                out.append((inst.to_ticks(row["price"]), lots))
        return tuple(out)

    def parse_message(self, payload: dict) -> list[FeedEvent]:
        if payload.get("symbol") != self.symbol:
            return []
        channel = payload.get("channel")
        if channel == "orderbooks":
            self._book_seq += 1
            return [
                DepthSnapshot(
                    bids=self._levels(payload.get("bids")),
                    asks=self._levels(payload.get("asks")),
                    last_update_id=self._book_seq,
                    ts_ns=ts_ns(payload.get("timestamp")),
                )
            ]
        if channel == "trades":
            side = payload.get("side")
            qty = self.instrument.to_lots(payload.get("size") or 0)
            if side not in ("BUY", "SELL") or qty <= 0:
                return []
            return [
                TradeTick(
                    price=self.instrument.to_ticks(payload["price"]),
                    qty=qty,
                    aggressor=Side.BUY if side == "BUY" else Side.SELL,
                    ts_ns=ts_ns(payload.get("timestamp")),
                )
            ]
        return []

    async def stream(self) -> AsyncIterator[FeedEvent]:
        attempt = 0
        while True:
            try:
                yield FeedStatus("connecting", WS_URL)
                async with websockets.connect(
                    WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=2**16,
                    ssl=ssl_context(),
                ) as ws:
                    await ws.send(
                        json.dumps(
                            {"command": "subscribe", "channel": "orderbooks", "symbol": self.symbol}
                        )
                    )
                    await asyncio.sleep(SUBSCRIBE_GAP_S)
                    await ws.send(
                        json.dumps(
                            {
                                "command": "subscribe",
                                "channel": "trades",
                                "symbol": self.symbol,
                                "option": "TAKER_ONLY",
                            }
                        )
                    )
                    live = False
                    async for raw in ws:
                        payload = json.loads(raw)
                        if "error" in payload:
                            raise RuntimeError(str(payload["error"]))
                        for event in self.parse_message(payload):
                            if not live and isinstance(event, DepthSnapshot):
                                live = True
                                attempt = 0
                                yield FeedStatus("live", f"gmo {self.symbol}")
                            if live:
                                yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect any broken stream
                attempt += 1
                delay = min(self.max_reconnect_delay, 2.0 ** min(attempt, 5))
                log.warning("gmo feed dropped (%s); reconnecting in %.0fs", exc, delay)
                yield FeedStatus("disconnected", f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(delay)
