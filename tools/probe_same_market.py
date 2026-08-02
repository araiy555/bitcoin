"""Is the alternate futures host the same market as the primary?

`fstream.binancefuture.com` delivers the trades and mark prices that
`fstream.binance.com` withholds, which makes it the obvious fix. But
`binancefuture.com` is also the domain Binance's futures *testnet* lives on,
and testnet prices drift far from production. Recording testnet as though it
were real would corrupt every feature and every backtest downstream while
looking entirely healthy — so the host does not get switched on the strength
of "it returns data".

Both hosts do deliver depth, and that gives a test that does not depend on
judgement: depth diffs carry exchange-assigned update ids. Two hosts serving
the same book emit the same sequence. A separate market cannot coincidentally
land in the same id range, and its prices cannot track to a basis point.

So this checks two things at once:

  update ids   do the two hosts' `u` ranges overlap?
  prices       does the alternate's book agree with the primary's, and with
               spot, to within a few basis points?

Spot is included as an outside anchor: if both futures hosts agreed with each
other but not with spot, that would mean something different again.

    python tools/probe_same_market.py         # 15s
    python tools/probe_same_market.py 30
"""

from __future__ import annotations

import asyncio
import json
import sys

import aiohttp
import websockets

from jsboard.net import make_session, ssl_context

PRIMARY = "wss://fstream.binance.com/ws"
ALTERNATE = "wss://fstream.binancefuture.com/ws"
SPOT_REST = "https://api.binance.com/api/v3/ticker/bookTicker"
SYMBOL = "btcusdt"


class Observed:
    def __init__(self) -> None:
        self.first_u: int | None = None
        self.last_u: int | None = None
        self.deltas = 0
        self.bids: list[float] = []
        self.asks: list[float] = []
        self.error: str = ""

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        n = len(self.bids)
        return (sum(self.bids) / n + sum(self.asks) / len(self.asks)) / 2.0


async def watch(url: str, seconds: float) -> Observed:
    """Collect depth ids and top-of-book from one host."""
    seen = Observed()
    try:
        async with websockets.connect(url, ping_interval=20, ssl=ssl_context()) as ws:
            await ws.send(
                json.dumps(
                    {
                        "method": "SUBSCRIBE",
                        "params": [f"{SYMBOL}@depth@100ms", f"{SYMBOL}@bookTicker"],
                        "id": 1,
                    }
                )
            )
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
                if not isinstance(msg, dict) or "id" in msg:
                    continue
                kind = msg.get("e")
                if kind == "depthUpdate":
                    u = int(msg["u"])
                    seen.first_u = u if seen.first_u is None else seen.first_u
                    seen.last_u = u
                    seen.deltas += 1
                elif kind == "bookTicker":
                    seen.bids.append(float(msg["b"]))
                    seen.asks.append(float(msg["a"]))
    except Exception as exc:  # noqa: BLE001 - the failure is the result here
        seen.error = f"{type(exc).__name__}: {exc}"
    return seen


async def spot_mid() -> float | None:
    session = make_session()
    try:
        async with session.get(
            SPOT_REST,
            params={"symbol": SYMBOL.upper()},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
        return (float(payload["bidPrice"]) + float(payload["askPrice"])) / 2.0
    except Exception as exc:  # noqa: BLE001
        print(f"    spot anchor unavailable: {type(exc).__name__}: {exc}")
        return None
    finally:
        await session.close()


def report(name: str, seen: Observed) -> None:
    if seen.error:
        print(f"  {name:<10} FAILED: {seen.error}")
        return
    mid = seen.mid
    ids = f"u={seen.first_u} … {seen.last_u}" if seen.first_u else "u=(none)"
    price = f"mid={mid:,.2f}" if mid else "mid=(no book ticker)"
    print(f"  {name:<10} deltas={seen.deltas:<6,} {ids}  {price}")


async def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    print(f"Watching both futures hosts for {seconds:.0f}s, plus a spot anchor.\n")

    primary, alternate, spot = await asyncio.gather(
        watch(PRIMARY, seconds), watch(ALTERNATE, seconds), spot_mid()
    )

    report("primary", primary)
    report("alternate", alternate)
    if spot is not None:
        print(f"  {'spot':<10} mid={spot:,.2f}")

    print()
    if primary.error or alternate.error:
        print("  A host failed; nothing to compare.")
        return

    # --- update ids ---------------------------------------------------
    if primary.first_u and alternate.first_u:
        lo = max(primary.first_u, alternate.first_u)
        hi = min(primary.last_u, alternate.last_u)
        if lo <= hi:
            print("  update ids: ranges OVERLAP — the same book, same sequence")
        else:
            gap = abs(alternate.first_u - primary.last_u)
            print(f"  update ids: DISJOINT (off by {gap:,}) — not the same book")
    else:
        print("  update ids: not enough depth data to compare")

    # --- prices -------------------------------------------------------
    pm, am = primary.mid, alternate.mid
    if pm and am:
        diff_bps = abs(am - pm) / pm * 10_000
        print(f"  perp vs perp: {diff_bps:,.2f} bps apart")
    if am and spot:
        basis_bps = (am - spot) / spot * 10_000
        print(f"  alternate vs spot: {basis_bps:+,.2f} bps basis")

    print(
        "\n  Same market looks like: overlapping ids, sub-basis-point agreement\n"
        "  between the hosts, and a basis against spot in the single digits.\n"
        "  Testnet looks like: disjoint ids and a price that is simply wrong."
    )


if __name__ == "__main__":
    asyncio.run(main())
