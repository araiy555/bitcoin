"""Ask Binance what it actually subscribed us to.

`probe_fstream.py` established that on the futures socket only the depth
stream delivers: `markPrice@1s` is a fixed 1Hz heartbeat, so twelve silent
seconds means the subscription never took, not that the market was quiet.
The URL form is not the cause either — the two-stream shape that works on
spot behaves the same way here.

So stop inferring from silence and use the socket's control channel. Binance
answers SUBSCRIBE with either `{"result": null, "id": n}` or an error object,
and answers LIST_SUBSCRIPTIONS with the streams it believes are active. One
of those two replies names the problem outright.

The spot leg runs the same script against `stream.binance.com` as a control.
Spot trades were recorded fine, so if spot answers and futures does not, the
difference is on the venue side rather than in this client.

    python tools/probe_subscribe.py           # futures, then spot
    python tools/probe_subscribe.py futures
"""

from __future__ import annotations

import asyncio
import collections
import json
import sys

import websockets

from jsboard.net import ssl_context

SYMBOL = "btcusdt"

VENUES = {
    "futures": (
        "wss://fstream.binance.com/ws",
        [
            f"{SYMBOL}@depth@100ms",
            f"{SYMBOL}@aggTrade",
            f"{SYMBOL}@markPrice@1s",
            f"{SYMBOL}@forceOrder",
        ],
    ),
    "spot": (
        "wss://stream.binance.com:9443/ws",
        [f"{SYMBOL}@depth@100ms", f"{SYMBOL}@aggTrade"],
    ),
}


async def probe(name: str, url: str, streams: list[str], seconds: float) -> None:
    print(f"\n=== {name}")
    print(f"    {url}")
    print(f"    requesting: {', '.join(streams)}")

    counts: collections.Counter[str] = collections.Counter()
    control: list[str] = []
    listed = False

    try:
        async with websockets.connect(url, ping_interval=20, ssl=ssl_context()) as ws:
            await ws.send(json.dumps({"method": "SUBSCRIBE", "params": streams, "id": 1}))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + seconds
            ask_at = loop.time() + 4.0

            while True:
                now = loop.time()
                if now >= deadline:
                    break
                if not listed and now >= ask_at:
                    await ws.send(json.dumps({"method": "LIST_SUBSCRIPTIONS", "id": 2}))
                    listed = True
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=min(1.0, deadline - now))
                except TimeoutError:
                    continue

                msg = json.loads(raw)
                # Control replies carry an `id` and no event type; data does
                # the reverse. Keeping them apart is the whole point here.
                if isinstance(msg, dict) and "id" in msg:
                    control.append(raw)
                    continue
                counts[msg.get("e", "(no 'e' field)")] += 1
    except Exception as exc:  # noqa: BLE001 - the failure is the result here
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return

    print("    control replies:")
    if control:
        for line in control:
            print(f"      {line}")
    else:
        print("      (none — the socket never acknowledged the request)")

    print(f"    data in {seconds:.0f}s:")
    if counts:
        for kind, n in counts.most_common():
            print(f"      {n:6,}  {kind}")
    else:
        print("      (none)")


async def main() -> None:
    args = sys.argv[1:]
    names = [a for a in args if a in VENUES] or list(VENUES)
    seconds = next((float(a) for a in args if a not in VENUES), 12.0)

    for name in names:
        url, streams = VENUES[name]
        await probe(name, url, streams, seconds)

    print(
        "\nRead it as: an error in the futures control reply names the cause.\n"
        "A LIST_SUBSCRIPTIONS answer that omits aggTrade/markPrice means the\n"
        "venue accepted the request and dropped those streams. Spot answering\n"
        "normally rules out this client."
    )


if __name__ == "__main__":
    asyncio.run(main())
