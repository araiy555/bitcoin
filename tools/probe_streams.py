"""Map which futures streams actually deliver.

Where this stands: the venue accepts the subscription and LIST_SUBSCRIPTIONS
echoes all four streams back, yet only depth carries data. Spot, asked the
same way by the same client, answers normally. So the request is well formed
and the client is fine — something on the futures side is delivering a
subset, and guessing at why has run out of road.

This subscribes to a wide spread of futures streams on one connection and
reports which produced data. The set is chosen so the pattern in the answer
is itself informative:

  bookTicker, kline, miniTicker   is it only depth, or a broader set?
  markPrice@1s vs markPrice       does the cadence matter?
  !markPrice@arr@1s               do all-symbol streams behave differently
  !forceOrder@arr                 from per-symbol ones?
  ethusdt@aggTrade                is it the symbol or the stream type?
  depth20@100ms                   partial depth vs the diff stream

A second host is tried with the same set. If the alternate delivers what the
primary withholds, this is an edge-node problem and the fix is a host, not a
protocol change.

    python tools/probe_streams.py            # both hosts, 20s each
    python tools/probe_streams.py 40
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys

import websockets

from jsboard.net import ssl_context

HOSTS = [
    "wss://fstream.binance.com/ws",
    "wss://fstream.binancefuture.com/ws",
]

STREAMS = [
    "btcusdt@depth@100ms",
    "btcusdt@depth20@100ms",
    "btcusdt@aggTrade",
    "btcusdt@bookTicker",
    "btcusdt@markPrice@1s",
    "btcusdt@markPrice",
    "btcusdt@kline_1m",
    "btcusdt@miniTicker",
    "btcusdt@forceOrder",
    "!markPrice@arr@1s",
    "!forceOrder@arr",
    "ethusdt@aggTrade",
]


def label(msg) -> str:
    """Name the payload by event type, marking all-symbol arrays apart.

    Per-symbol and all-symbol mark price share an event type, so without the
    array marker the two subscriptions would be indistinguishable here — and
    telling them apart is one of the things this run is for.
    """
    if isinstance(msg, list):
        head = msg[0] if msg else {}
        inner = head.get("e", "?") if isinstance(head, dict) else "?"
        return f"{inner}[arr]"
    if isinstance(msg, dict):
        return str(msg.get("e", "(no 'e' field)"))
    return "(not an object)"


async def probe(url: str, seconds: float) -> None:
    print(f"\n=== {url}  ({seconds:.0f}s)")
    counts: collections.Counter[str] = collections.Counter()
    control: list[str] = []

    try:
        async with websockets.connect(url, ping_interval=20, ssl=ssl_context()) as ws:
            await ws.send(json.dumps({"method": "SUBSCRIBE", "params": STREAMS, "id": 1}))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + seconds
            while True:
                now = loop.time()
                if now >= deadline:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(1.0, deadline - now))
                except TimeoutError:
                    continue
                msg = json.loads(raw)
                if isinstance(msg, dict) and "id" in msg:
                    control.append(raw)
                    continue
                counts[label(msg)] += 1
    except Exception as exc:  # noqa: BLE001 - the failure is the result here
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return

    for line in control:
        print(f"    control: {line[:200]}")

    print(f"    requested {len(STREAMS)} streams, got {len(counts)} kinds of data:")
    if counts:
        for kind, n in counts.most_common():
            print(f"      {n:6,}  {kind}")
    else:
        print("      (none)")


async def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    for url in HOSTS:
        await probe(url, seconds)
    print(
        "\nRead it as: which kinds are missing says whether this is specific\n"
        "to trades, to per-symbol streams, or to this host. A host that\n"
        "delivers what the other withholds makes it an edge-node problem."
    )


if __name__ == "__main__":
    asyncio.run(main())
