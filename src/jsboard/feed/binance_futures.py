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

  ---------------------------------------------------------------------
  The socket does not always send what it agrees to send
  ---------------------------------------------------------------------

Measured against fstream.binance.com: the venue accepts a subscription for
aggTrade, markPrice and forceOrder, echoes all of them back from
LIST_SUBSCRIPTIONS as active, and then delivers none of them. Depth and
bookTicker arrive normally throughout. Spot, asked the same way by the same
client, answers everything. So this is not a malformed request and not a
quiet market — markPrice is a fixed 1Hz heartbeat, and twelve silent seconds
of it cannot happen on a live symbol.

There is another host that does serve those streams. It is testnet: its depth
update ids sit 10.8 *trillion* away from production's, and its price runs
55bps from spot where a real perp basis is a few. Recording it would poison
every feature downstream while looking perfectly healthy, so it is not used,
and `probe_same_market.py` exists to keep that conclusion checkable.

REST is unaffected, so trades and mark price fall back to polling.
`/fapi/v1/aggTrades` takes `fromId`, which makes the fallback lossless rather
than a sample: chained polls return every trade, with exchange timestamps
intact. What polling costs is *receive-time* precision — about one poll
interval. Lead-lag measured on exchange time is unaffected; lead-lag measured
on arrival is not available for perp trades under fallback.

Liquidations have no REST equivalent here, so `forceOrder` is simply lost
while the stream is withheld. That is a real hole, not a papered-over one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
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

FALLBACK_MODES = ("auto", "always", "never")


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
        rest_fallback: str = "auto",
        fallback_after_s: float = 10.0,
        trade_poll_interval: float = 1.0,
        mark_poll_interval: float = 1.0,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(instrument)
        if depth_ms not in VALID_DEPTH_MS:
            raise ValueError(f"futures depth diffs come at {VALID_DEPTH_MS}ms")
        if snapshot_limit not in VALID_DEPTH_LIMITS:
            raise ValueError(f"snapshot_limit must be one of {VALID_DEPTH_LIMITS}")
        if rest_fallback not in FALLBACK_MODES:
            raise ValueError(f"rest_fallback must be one of {FALLBACK_MODES}")
        self.depth_ms = depth_ms
        self.snapshot_limit = snapshot_limit
        self.open_interest_interval = open_interest_interval
        self.max_reconnect_delay = max_reconnect_delay
        # "auto" waits `fallback_after_s` for the socket to prove it will send
        # trades and mark prices, then polls instead. "always" skips the wait,
        # "never" accepts the silence. The wait exists because the withholding
        # is not universal — the same code should stay on the socket wherever
        # the socket works.
        self.rest_fallback = rest_fallback
        self.fallback_after_s = fallback_after_s
        self.trade_poll_interval = trade_poll_interval
        self.mark_poll_interval = mark_poll_interval
        self._session = session
        self._owns_session = session is None
        # Everything sourced from REST lands here — open interest always, and
        # trades and mark price when the socket withholds them. One queue keeps
        # the main loop a single ordered sequence regardless of origin.
        self._rest_queue: asyncio.Queue[FeedEvent] = asyncio.Queue()
        self._used_weight: int = 0

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

    async def _get(self, path: str, **params):
        session = await self._ensure_session()
        async with session.get(
            f"{REST_BASE}{path}",
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            # Binance reports the rate-limit budget already spent this minute.
            # Reading it beats asserting a per-endpoint weight from memory:
            # if a poll interval is too aggressive, this is where it shows.
            used = resp.headers.get("X-MBX-USED-WEIGHT-1M")
            if used is not None:
                with contextlib.suppress(ValueError):
                    self._used_weight = int(used)
            return await resp.json()

    async def fetch_open_interest(self) -> OpenInterest:
        payload = await self._get("/fapi/v1/openInterest", symbol=self._symbol)
        stamped = int(payload.get("time", 0)) * 1_000_000
        lots = self.instrument.to_lots(payload["openInterest"])
        return OpenInterest(lots=lots, ts_ns=stamped) if stamped else OpenInterest(lots=lots)

    async def fetch_agg_trades(self, from_id: int | None = None, limit: int = 1000) -> list[dict]:
        """Aggregated trades, resumable by id.

        `fromId` is what makes the REST path a substitute rather than a
        sample: chaining polls from the last id seen returns every trade in
        between, however long the gap.
        """
        params = {"symbol": self._symbol, "limit": limit}
        if from_id is not None:
            params["fromId"] = from_id
        return await self._get("/fapi/v1/aggTrades", **params)

    async def fetch_premium_index(self) -> MarkPrice:
        payload = await self._get("/fapi/v1/premiumIndex", symbol=self._symbol)
        return self._parse_premium_index(payload)

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

    def _parse_premium_index(self, data: dict) -> MarkPrice:
        """The REST spelling of a mark-price update.

        Not quite the same payload as the stream: REST names its rate field
        `lastFundingRate`, where the stream sends `r`, documented as the
        *predicted* rate for the upcoming settlement. Whether the two mean the
        same thing is not settled here — the value is recorded as-is, and
        anything reading `funding_rate` across a fallback boundary should know
        the two legs may not be the same quantity.
        """
        inst = self.instrument
        return MarkPrice(
            mark=inst.to_ticks(data["markPrice"]),
            index=inst.to_ticks(data["indexPrice"]),
            funding_rate=float(data["lastFundingRate"]),
            next_funding_ns=int(data["nextFundingTime"]) * 1_000_000,
            ts_ns=int(data["time"]) * 1_000_000,
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
        tasks: list[asyncio.Task] = [
            snapshot_task,
            asyncio.create_task(self._poll_open_interest()),
        ]
        prev_u: int | None = None

        # Fallback bookkeeping. `opened` is the reference for the wait, and
        # `ws_extras_seen` records whether the socket ever proved it will send
        # trades or mark prices on this connection.
        opened = time.monotonic()
        ws_extras_seen = False
        polling = False
        if self.rest_fallback == "always":
            tasks += self._start_rest_pollers()
            polling = True

        try:
            async for raw in ws:
                # REST-sourced events arrive out of band; drain what is ready.
                while not self._rest_queue.empty():
                    yield self._rest_queue.get_nowait()

                if (
                    not polling
                    and self.rest_fallback == "auto"
                    and not ws_extras_seen
                    and time.monotonic() - opened >= self.fallback_after_s
                ):
                    tasks += self._start_rest_pollers()
                    polling = True
                    yield FeedStatus(
                        "degraded",
                        f"no trades or mark price in {self.fallback_after_s:.0f}s; polling REST",
                    )

                msg = json.loads(raw)
                data = msg.get("data", msg)
                kind = data.get("e")

                if kind in ("aggTrade", "markPriceUpdate"):
                    ws_extras_seen = True
                    # Once polling, the socket's copy is a duplicate of what
                    # REST already returns. Dropping it here is cheaper than
                    # de-duplicating two sources downstream.
                    if polling:
                        continue
                    yield (
                        self._parse_trade(data)
                        if kind == "aggTrade"
                        else self._parse_mark_price(data)
                    )
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
                        tasks.append(snapshot_task)
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
                    tasks.append(snapshot_task)
                    continue

                prev_u = delta.final_id
                yield delta
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                elif not task.cancelled():
                    # Retrieve it so a failed poller does not surface later as
                    # an unretrieved-exception warning during teardown.
                    task.exception()

    # ------------------------------------------------------- REST fallback

    def _start_rest_pollers(self) -> list[asyncio.Task]:
        log.info("%s: polling REST for trades and mark price", self._symbol)
        return [
            asyncio.create_task(self._poll_agg_trades()),
            asyncio.create_task(self._poll_mark_price()),
        ]

    async def _poll_open_interest(self) -> None:
        while True:
            try:
                self._rest_queue.put_nowait(await self.fetch_open_interest())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed poll is not fatal
                log.debug("open interest poll failed: %s", exc)
            await asyncio.sleep(self.open_interest_interval)

    async def _poll_agg_trades(self) -> None:
        """Every trade since the last one seen, not the most recent N.

        The first poll has no id to resume from, so it takes the tail of the
        tape and anchors there. Everything after chains by id, which is what
        makes a slow poll lossless — a delayed or failed round trip widens the
        next response instead of dropping what it missed.
        """
        from_id: int | None = None
        while True:
            try:
                rows = await self.fetch_agg_trades(from_id=from_id)
                if from_id is None and rows:
                    # Nothing before this poll belongs to the recording.
                    rows = rows[-1:]
                for row in rows:
                    self._rest_queue.put_nowait(self._parse_trade(row))
                if rows:
                    from_id = int(rows[-1]["a"]) + 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed poll is not fatal
                log.debug("aggTrades poll failed: %s", exc)
            await asyncio.sleep(self.trade_poll_interval)

    async def _poll_mark_price(self) -> None:
        while True:
            try:
                self._rest_queue.put_nowait(await self.fetch_premium_index())
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed poll is not fatal
                log.debug("premiumIndex poll failed: %s", exc)
            await asyncio.sleep(self.mark_poll_interval)
