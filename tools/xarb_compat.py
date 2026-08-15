"""Compatibility feed helpers for the six-market xarb tools.

The Binance Spot partial-depth stream uses ``bids``/``asks`` while the futures
partial-depth payload uses ``b``/``a``.  The research module originally shared
one futures-shaped parser, so Spot connected successfully but produced no
valid books.  Keep the compatibility shim in tools until the scanner is folded
into the main CLI.
"""

from __future__ import annotations

import asyncio
import json
import time

import websockets

from jsboard.net import ssl_context
from jsboard.research import xarb_scan as scan

_ORIGINAL_BINANCE_BOOK = scan._binance_book


async def binance_book_compatible(source: str, url: str, out: asyncio.Queue) -> None:
    """Parse Binance Spot partial depth and delegate futures to existing code."""
    if source != "binance_spot":
        await _ORIGINAL_BINANCE_BOOK(source, url, out)
        return

    while True:
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**14,
                ssl=ssl_context(),
            ) as ws:
                async for raw in ws:
                    recv = time.monotonic_ns()
                    data = json.loads(raw)
                    bids = scan._clean_levels(data.get("bids") or data.get("b"), reverse=True)
                    asks = scan._clean_levels(data.get("asks") or data.get("a"), reverse=False)
                    ts = int(data.get("E") or 0) * 1_000_000
                    book = scan.Book(source, bids, asks, ts, recv)
                    if book.valid():
                        await out.put(book)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[xarb-scan] Binance Spot reconnect: {type(exc).__name__}: {exc}")
            await asyncio.sleep(1.0)
