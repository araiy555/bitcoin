"""bitbank public order book and trades, over its socket.io stream.

bitbank publishes a full book (`depth_whole_<pair>`, top 200 levels) and
absolute-quantity diffs between them (`depth_diff_<pair>`), both stamped with a
sequence id. A diff older than the last whole book is dropped; applying it
would resurrect levels the whole book had already removed. Trades come on
`transactions_<pair>`, where `side` is the taker's.

The stream speaks socket.io (engine.io v4) over a plain WebSocket: the client
answers the server's "2" pings with "3", joins rooms with
`42["join-room", "<room>"]`, and receives `42["message", {...}]`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from decimal import Decimal

import websockets

from ..core.types import Instrument, Side
from ..net import ssl_context
from .base import DepthDelta, DepthSnapshot, Feed, FeedEvent, FeedStatus, TradeTick

log = logging.getLogger(__name__)

PAIRS_URL = "https://api.bitbank.cc/v1/spot/pairs"
WS_URL = "wss://stream.bitbank.cc/socket.io/?EIO=4&transport=websocket"


def instrument_from_pair(info: dict) -> Instrument:
    """One entry of `/v1/spot/pairs`."""
    return Instrument(
        symbol=info["name"],
        tick_size=Decimal(1).scaleb(-int(info["price_digits"])),
        lot_size=Decimal(1).scaleb(-int(info["amount_digits"])),
        base=str(info.get("base_asset", "")).upper(),
        quote=str(info.get("quote_asset", "")).upper(),
    )


async def fetch_instrument(pair: str) -> Instrument:
    import aiohttp

    from ..net import make_session

    async with make_session() as session, session.get(
        PAIRS_URL, timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    for info in (payload.get("data") or {}).get("pairs") or []:
        if info.get("name") == pair:
            return instrument_from_pair(info)
    raise RuntimeError(f"{pair} は bitbank にありません（例: ada_jpy）")


class BitbankFeed(Feed):
    """Live bitbank book (whole + diffs) and taker trades for one pair."""

    def __init__(self, instrument: Instrument, *, max_reconnect_delay: float = 30.0) -> None:
        super().__init__(instrument)
        self.pair = instrument.symbol
        self.max_reconnect_delay = max_reconnect_delay
        self._whole_seq = -1

    @property
    def rooms(self) -> tuple[str, ...]:
        return (
            f"depth_whole_{self.pair}",
            f"depth_diff_{self.pair}",
            f"transactions_{self.pair}",
        )

    def _levels(self, rows) -> tuple[tuple[int, int], ...]:
        inst = self.instrument
        return tuple((inst.to_ticks(price), inst.to_lots(qty)) for price, qty in rows or ())

    def parse_message(self, message: dict) -> list[FeedEvent]:
        """Normalise one `{"room_name": ..., "message": {"data": ...}}` payload."""
        room = message.get("room_name")
        data = (message.get("message") or {}).get("data") or {}
        if room == f"depth_whole_{self.pair}":
            seq = int(data.get("sequenceId") or 0)
            self._whole_seq = seq
            return [
                DepthSnapshot(
                    bids=tuple(lv for lv in self._levels(data.get("bids")) if lv[1] > 0),
                    asks=tuple(lv for lv in self._levels(data.get("asks")) if lv[1] > 0),
                    last_update_id=seq,
                    ts_ns=int(data.get("timestamp") or 0) * 1_000_000,
                )
            ]
        if room == f"depth_diff_{self.pair}":
            seq = int(data.get("s") or 0)
            if self._whole_seq < 0 or seq <= self._whole_seq:
                return []
            return [
                DepthDelta(
                    bids=self._levels(data.get("b")),
                    asks=self._levels(data.get("a")),
                    first_id=seq,
                    final_id=seq,
                    ts_ns=int(data.get("t") or 0) * 1_000_000,
                )
            ]
        if room == f"transactions_{self.pair}":
            out: list[FeedEvent] = []
            for row in data.get("transactions") or ():
                side = row.get("side")
                qty = self.instrument.to_lots(row.get("amount") or 0)
                if side not in ("buy", "sell") or qty <= 0:
                    continue
                out.append(
                    TradeTick(
                        price=self.instrument.to_ticks(row["price"]),
                        qty=qty,
                        aggressor=Side.BUY if side == "buy" else Side.SELL,
                        trade_id=int(row.get("transaction_id") or 0),
                        ts_ns=int(row.get("executed_at") or 0) * 1_000_000,
                    )
                )
            return out
        return []

    @staticmethod
    def decode_frame(frame: str) -> tuple[str, dict | None]:
        """Split an engine.io frame into its kind and, for events, the payload."""
        if frame.startswith("42"):
            try:
                name, body = json.loads(frame[2:])[:2]
            except (ValueError, TypeError):
                return "event", None
            return ("message", body) if name == "message" else ("event", None)
        if frame == "2":
            return "ping", None
        if frame.startswith("0"):
            return "open", None
        if frame.startswith("40"):
            return "connected", None
        return "other", None

    async def stream(self) -> AsyncIterator[FeedEvent]:
        attempt = 0
        while True:
            try:
                yield FeedStatus("connecting", WS_URL)
                self._whole_seq = -1
                async with websockets.connect(
                    WS_URL, ping_interval=None, max_queue=2**16, ssl=ssl_context()
                ) as ws:
                    live = False
                    async for raw in ws:
                        kind, body = self.decode_frame(raw)
                        if kind == "open":
                            await ws.send("40")
                        elif kind == "connected":
                            for room in self.rooms:
                                await ws.send("42" + json.dumps(["join-room", room]))
                        elif kind == "ping":
                            await ws.send("3")
                        elif kind == "message" and body is not None:
                            for event in self.parse_message(body):
                                if not live and isinstance(event, DepthSnapshot):
                                    live = True
                                    attempt = 0
                                    yield FeedStatus("live", f"bitbank {self.pair}")
                                if live:
                                    yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect any broken stream
                attempt += 1
                delay = min(self.max_reconnect_delay, 2.0 ** min(attempt, 5))
                log.warning("bitbank feed dropped (%s); reconnecting in %.0fs", exc, delay)
                yield FeedStatus("disconnected", f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(delay)
