"""What actually arrives on the futures socket.

The capture recorded depth diffs at the right rate but zero trades and zero
mark prices, which cannot be true of BTCUSDT perp. Either the extra streams
are not being subscribed, or they are arriving under names the feed does not
dispatch on. Reading the adapter cannot distinguish those; looking at the
wire can.

Each variant below isolates one suspicion, so a single run says which:

    all4    exactly the URL BinanceFuturesFeed builds
    two     depth + aggTrade only, the shape that works on spot
    trade   aggTrade alone, on the single-stream path
    mark    markPrice alone, on the single-stream path

If `all4` shows only depthUpdate while `trade` and `mark` show data, the
stream names are fine and the combined subscription is the problem. If
`trade` is also empty, the stream name itself is wrong.

    python tools/probe_fstream.py            # all variants, 12s each
    python tools/probe_fstream.py all4 30    # one variant, longer
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys

import websockets

from jsboard.net import ssl_context

SYMBOL = "btcusdt"
COMBINED = "wss://fstream.binance.com/stream?streams="
SINGLE = "wss://fstream.binance.com/ws/"

VARIANTS = {
    "all4": COMBINED
    + "/".join(
        (
            f"{SYMBOL}@depth@100ms",
            f"{SYMBOL}@aggTrade",
            f"{SYMBOL}@markPrice@1s",
            f"{SYMBOL}@forceOrder",
        )
    ),
    "two": COMBINED + f"{SYMBOL}@depth@100ms/{SYMBOL}@aggTrade",
    "trade": SINGLE + f"{SYMBOL}@aggTrade",
    "mark": SINGLE + f"{SYMBOL}@markPrice@1s",
}


async def probe(name: str, url: str, seconds: float) -> None:
    print(f"\n=== {name} ({seconds:.0f}s)")
    print(f"    {url}")
    by_event: collections.Counter[str] = collections.Counter()
    by_stream: collections.Counter[str] = collections.Counter()
    first_of_kind: dict[str, str] = {}
    frames = 0

    try:
        async with websockets.connect(url, ping_interval=20, ssl=ssl_context()) as ws:
            deadline = asyncio.get_running_loop().time() + seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except TimeoutError:
                    break
                frames += 1
                msg = json.loads(raw)
                # A combined stream wraps each payload; a single one does not.
                by_stream[msg.get("stream", "(unwrapped)")] += 1
                data = msg.get("data", msg)
                kind = data.get("e", "(no 'e' field)")
                by_event[kind] += 1
                if kind not in first_of_kind:
                    first_of_kind[kind] = json.dumps(data)[:160]
    except Exception as exc:  # noqa: BLE001 - the failure is the result here
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return

    if not frames:
        print("    nothing arrived")
        return

    print(f"    {frames} frames")
    print("    by stream name:")
    for stream, n in by_stream.most_common():
        print(f"      {n:6,}  {stream}")
    print("    by event type:")
    for kind, n in by_event.most_common():
        print(f"      {n:6,}  {kind}")
        print(f"              {first_of_kind[kind]}")


async def main() -> None:
    args = sys.argv[1:]
    seconds = 12.0
    names = list(VARIANTS)
    if args:
        if args[0] in VARIANTS:
            names = [args[0]]
            if len(args) > 1:
                seconds = float(args[1])
        else:
            seconds = float(args[0])

    for name in names:
        await probe(name, VARIANTS[name], seconds)

    print(
        "\nRead it as: 'all4' missing aggTrade/markPriceUpdate while 'trade'\n"
        "and 'mark' have them means the combined subscription is at fault,\n"
        "not the stream names."
    )


if __name__ == "__main__":
    asyncio.run(main())
