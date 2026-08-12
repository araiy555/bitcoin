"""Is the futures market-data path `/market/ws/` rather than `/ws/`?

Earlier probing found that on fstream.binance.com the venue accepted a
subscription for aggTrade, markPrice and forceOrder, echoed all of them back
from LIST_SUBSCRIPTIONS as active, and then delivered none of them — while
depth and bookTicker arrived normally on the same socket. That was read as
the venue withholding data. A likelier reading is that the path is stale:
`/ws/` still serves some streams while market data has moved to
`/market/ws/`, which would produce exactly this partial silence.

The claim is cheap to settle, so it gets settled rather than argued. Each
row below is one connection to one stream, so nothing is inferred from a
combined subscription:

    /market/ws/btcusdt@aggTrade        the trades that never arrived
    /market/ws/btcusdt@markPrice@1s    a fixed 1Hz heartbeat — silence here
                                       cannot mean "a quiet market"
    /market/ws/!forceOrder@arr         liquidations across all symbols, which
                                       fire far more often than one symbol's
    /ws/btcusdt@aggTrade               the old path, as a control
    /ws/btcusdt@depth@100ms            the stream that did work, as a control

If the /market rows deliver and the /ws rows stay silent, the path is the
whole story and the REST fallback becomes a backup rather than the norm.

    python tools/probe_market_path.py          # 15s per stream
    python tools/probe_market_path.py 30
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys

import websockets

from jsboard.net import ssl_context

HOST = "wss://fstream.binance.com"

SYMBOL = "wifusdt"
FOUR = "/".join(
    (
        f"{SYMBOL}@depth@100ms",
        f"{SYMBOL}@aggTrade",
        f"{SYMBOL}@markPrice@1s",
        f"{SYMBOL}@forceOrder",
    )
)

# The first row is the exact URL BinanceFuturesFeed builds. The previous run
# of this probe tested the /market combined path with aggTrade and markPrice
# and no depth, then the feed was switched to /market carrying all four — so
# depth on that path was never actually observed. An hour of live quoting
# later, no book had been built. Test what the code sends, not a near-miss.
CASES = [
    ("market  combined ALL FOUR", f"{HOST}/market/stream?streams={FOUR}"),
    ("market  depth alone (ws)", f"{HOST}/market/ws/{SYMBOL}@depth@100ms"),
    ("market  depth alone (stream)", f"{HOST}/market/stream?streams={SYMBOL}@depth@100ms"),
    ("market  aggTrade", f"{HOST}/market/ws/{SYMBOL}@aggTrade"),
    ("ws      depth      [control]", f"{HOST}/ws/{SYMBOL}@depth@100ms"),
    ("stream  combined   [control]", f"{HOST}/stream?streams={FOUR}"),
]


def label(msg) -> str:
    if isinstance(msg, list):
        head = msg[0] if msg else {}
        return f"{head.get('e', '?')}[arr]" if isinstance(head, dict) else "?[arr]"
    if isinstance(msg, dict):
        # A combined stream wraps its payload; a single one does not.
        inner = msg.get("data", msg)
        return str(inner.get("e", "(no 'e')")) if isinstance(inner, dict) else "?"
    return "?"


async def probe(name: str, url: str, seconds: float) -> None:
    counts: collections.Counter[str] = collections.Counter()
    sample = ""
    try:
        async with websockets.connect(url, ping_interval=20, ssl=ssl_context()) as ws:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + seconds
            while True:
                left = deadline - loop.time()
                if left <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=left)
                except TimeoutError:
                    break
                msg = json.loads(raw)
                if isinstance(msg, dict) and "id" in msg:
                    continue
                kind = label(msg)
                if not counts:
                    sample = raw[:120]
                counts[kind] += 1
    except Exception as exc:  # noqa: BLE001 - the failure is the result
        print(f"  {name:<30} FAILED  {type(exc).__name__}: {exc}")
        return

    if not counts:
        print(f"  {name:<30} [silent]")
        return
    summary = "  ".join(f"{k}={n:,}" for k, n in counts.most_common())
    print(f"  {name:<30} {summary}")
    print(f"  {'':<30} {sample}")


async def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    print(f"One connection per stream, {seconds:.0f}s each.\n")
    for name, url in CASES:
        await probe(name, url, seconds)
    print(
        "\nRead it as: /market rows carrying data while /ws rows stay silent\n"
        "means the path was stale, not that the venue withholds anything."
    )


if __name__ == "__main__":
    asyncio.run(main())
