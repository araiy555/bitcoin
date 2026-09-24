"""Hyperliquid public perpetual book and trades.

Hyperliquid pushes the whole top of book (20 levels a side) on every change
rather than deltas, so each `l2Book` message becomes a :class:`DepthSnapshot`.
Trades arrive in batches; `side` is the aggressor ("B" bought, "A" sold).

Prices on Hyperliquid have no fixed tick. A perp price may carry at most five
significant figures and at most ``6 - szDecimals`` decimals, so the step
depends on the price level. The instrument is fixed for the length of a
recording, so the tick is taken from the price when recording starts; every
price the venue can print at that level is an exact multiple of it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator
from decimal import Decimal

import websockets

from ..core.types import Instrument, Side
from ..net import ssl_context
from .base import DepthSnapshot, Feed, FeedEvent, FeedStatus, TradeTick

log = logging.getLogger(__name__)

INFO_URL = "https://api.hyperliquid.xyz/info"
WS_URL = "wss://api.hyperliquid.xyz/ws"
MAX_PERP_DECIMALS = 6
SIG_FIGS = 5


def coin_for(symbol: str) -> str:
    """`ADAUSDT` → `ADA`: Hyperliquid names perps by coin alone."""
    return symbol.upper().removesuffix("USDT").removesuffix("USDC").removesuffix("USD")


def tick_for(price: float, sz_decimals: int) -> Decimal:
    """The price step Hyperliquid enforces at this price level."""
    if price <= 0:
        raise ValueError("price must be positive")
    by_sig_figs = math.floor(math.log10(price)) - (SIG_FIGS - 1)
    by_decimals = -(MAX_PERP_DECIMALS - sz_decimals)
    # Whole-number prices are always accepted, so the step never exceeds 1.
    return Decimal(1).scaleb(min(max(by_sig_figs, by_decimals), 0))


def instrument_for(coin: str, sz_decimals: int, price: float) -> Instrument:
    return Instrument(
        symbol=coin.upper(),
        tick_size=tick_for(price, sz_decimals),
        lot_size=Decimal(1).scaleb(-sz_decimals),
        base=coin.upper(),
        quote="USDC",
    )


async def fetch_instrument(coin: str) -> Instrument:
    """Size decimals from `meta`, the price level from `allMids`."""
    import aiohttp

    from ..net import make_session

    coin = coin.upper()
    async with make_session() as session:
        async with session.post(
            INFO_URL, json={"type": "meta"}, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            resp.raise_for_status()
            meta = await resp.json()
        async with session.post(
            INFO_URL, json={"type": "allMids"}, timeout=aiohttp.ClientTimeout(total=15)
        ) as resp:
            resp.raise_for_status()
            mids = await resp.json()
    asset = next((a for a in meta.get("universe", []) if a.get("name") == coin), None)
    if asset is None:
        raise RuntimeError(f"{coin} は Hyperliquid の無期限先物にありません")
    if coin not in mids:
        raise RuntimeError(f"{coin} の現在値を Hyperliquid から取得できません")
    return instrument_for(coin, int(asset["szDecimals"]), float(mids[coin]))


class HyperliquidFeed(Feed):
    """Live Hyperliquid L2 snapshots and trades for one perpetual."""

    def __init__(self, instrument: Instrument, *, max_reconnect_delay: float = 30.0) -> None:
        super().__init__(instrument)
        self.coin = instrument.symbol.upper()
        self.max_reconnect_delay = max_reconnect_delay
        self._book_seq = 0

    def _levels(self, rows) -> tuple[tuple[int, int], ...]:
        inst = self.instrument
        out = []
        for row in rows:
            lots = inst.to_lots(row["sz"])
            if lots > 0:
                out.append((inst.to_ticks(row["px"]), lots))
        return tuple(out)

    def parse_message(self, payload: dict) -> list[FeedEvent]:
        """Normalise one documented `l2Book` or `trades` message."""
        channel = payload.get("channel")
        data = payload.get("data")
        if channel == "l2Book" and isinstance(data, dict):
            if str(data.get("coin", "")).upper() != self.coin:
                return []
            levels = data.get("levels") or ((), ())
            if len(levels) != 2:
                return []
            self._book_seq += 1
            return [
                DepthSnapshot(
                    bids=self._levels(levels[0]),
                    asks=self._levels(levels[1]),
                    last_update_id=self._book_seq,
                    ts_ns=int(data.get("time") or 0) * 1_000_000,
                )
            ]
        if channel == "trades" and isinstance(data, list):
            out: list[FeedEvent] = []
            for row in data:
                if str(row.get("coin", "")).upper() != self.coin:
                    continue
                side = row.get("side")
                if side not in ("A", "B"):
                    continue
                qty = self.instrument.to_lots(row["sz"])
                if qty <= 0:
                    continue
                out.append(
                    TradeTick(
                        price=self.instrument.to_ticks(row["px"]),
                        qty=qty,
                        aggressor=Side.BUY if side == "B" else Side.SELL,
                        trade_id=int(row.get("tid") or 0),
                        ts_ns=int(row.get("time") or 0) * 1_000_000,
                    )
                )
            return out
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
                    for kind in ("l2Book", "trades"):
                        await ws.send(
                            json.dumps(
                                {
                                    "method": "subscribe",
                                    "subscription": {"type": kind, "coin": self.coin},
                                }
                            )
                        )
                    live = False
                    async for raw in ws:
                        payload = json.loads(raw)
                        if payload.get("channel") == "error":
                            raise RuntimeError(str(payload.get("data")))
                        events = self.parse_message(payload)
                        for event in events:
                            if not live and isinstance(event, DepthSnapshot):
                                live = True
                                attempt = 0
                                yield FeedStatus("live", f"hyperliquid {self.coin}")
                            if live:
                                yield event
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect any broken stream
                attempt += 1
                delay = min(self.max_reconnect_delay, 2.0 ** min(attempt, 5))
                log.warning("hyperliquid feed dropped (%s); reconnecting in %.0fs", exc, delay)
                yield FeedStatus("disconnected", f"{type(exc).__name__}: {exc}")
                await asyncio.sleep(delay)
