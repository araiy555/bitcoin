"""Can REST supply what the futures socket refuses to push?

Established so far: fstream.binance.com is production (its book agrees with
spot to a few basis points), it accepts subscriptions for aggTrade, markPrice
and forceOrder, lists them as active, and then sends nothing for any of them.
Depth and bookTicker arrive normally. The alternate host that does serve them
is testnet and was rejected. REST, meanwhile, already works — the capture
pulled depth snapshots and open interest from fapi without trouble.

So the question is narrow: does REST carry the same information, and how much
is lost by polling for it.

The important detail is that `/fapi/v1/aggTrades` takes `fromId`, so polling
returns *every* trade since the last one seen rather than a sample. Exchange
timestamps come through exactly. What polling costs is receive-time
precision, not data — which matters for lead-lag measured on arrival, and not
at all for lead-lag measured on exchange time.

Checks, in order:

  premiumIndex   mark, index, funding rate — the markPrice stream's contents
  aggTrades      two polls chained by fromId, to prove nothing falls between
  bookTicker     a price sanity check against spot, because everything here
                 gets the same testnet scepticism the WS hosts got

    python tools/probe_rest_fallback.py
"""

from __future__ import annotations

import asyncio
import time

import aiohttp

from jsboard.net import make_session

FAPI = "https://fapi.binance.com"
SPOT = "https://api.binance.com"
SYMBOL = "BTCUSDT"


async def get(session: aiohttp.ClientSession, url: str, **params):
    async with session.get(
        url, params=params, timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def main() -> None:
    session = make_session()
    try:
        # --- what the markPrice stream would have carried ----------------
        print("=== /fapi/v1/premiumIndex")
        try:
            pi = await get(session, f"{FAPI}/fapi/v1/premiumIndex", symbol=SYMBOL)
            mark = float(pi["markPrice"])
            index = float(pi["indexPrice"])
            rate = float(pi["lastFundingRate"])
            until = (int(pi["nextFundingTime"]) - int(pi["time"])) / 3_600_000
            print(f"    mark   {mark:,.2f}")
            print(f"    index  {index:,.2f}")
            print(f"    funding {rate * 100:+.4f}%  next in {until:.2f}h")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            mark = None

        # --- every trade, not a sample -----------------------------------
        print("\n=== /fapi/v1/aggTrades  (two polls, chained by fromId)")
        try:
            first = await get(session, f"{FAPI}/fapi/v1/aggTrades", symbol=SYMBOL, limit=1000)
            span_ms = int(first[-1]["T"]) - int(first[0]["T"])
            print(f"    poll 1: {len(first):,} trades over {span_ms / 1000:.2f}s")
            print(f"            ids {first[0]['a']:,} … {first[-1]['a']:,}")

            await asyncio.sleep(3.0)
            next_id = int(first[-1]["a"]) + 1
            second = await get(
                session, f"{FAPI}/fapi/v1/aggTrades", symbol=SYMBOL, fromId=next_id, limit=1000
            )
            print(f"    poll 2: {len(second):,} trades from id {next_id:,}")
            if second:
                got = int(second[0]["a"])
                if got == next_id:
                    print("            continuous — no trade fell between the polls")
                else:
                    print(f"            GAP: asked for {next_id:,}, got {got:,}")
                rate_s = len(second) / 3.0
                print(f"            ~{rate_s:,.0f} trades/s at this moment")
                print(f"            a 3s poll needs limit>={rate_s * 3:,.0f}; max is 1000")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")

        # --- is this the real market? ------------------------------------
        print("\n=== price sanity (the same scepticism the WS hosts got)")
        try:
            perp = await get(session, f"{FAPI}/fapi/v1/ticker/bookTicker", symbol=SYMBOL)
            spot = await get(session, f"{SPOT}/api/v3/ticker/bookTicker", symbol=SYMBOL)
            perp_mid = (float(perp["bidPrice"]) + float(perp["askPrice"])) / 2
            spot_mid = (float(spot["bidPrice"]) + float(spot["askPrice"])) / 2
            basis = (perp_mid - spot_mid) / spot_mid * 10_000
            print(f"    perp REST {perp_mid:,.2f}")
            print(f"    spot REST {spot_mid:,.2f}")
            print(f"    basis     {basis:+,.2f} bps   (single digits = production)")
            if mark:
                print(f"    mark vs perp book: {(mark - perp_mid) / perp_mid * 10_000:+,.2f} bps")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {type(exc).__name__}: {exc}")

        # --- what polling costs ------------------------------------------
        print("\n=== poll latency (10 round trips)")
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            try:
                await get(session, f"{FAPI}/fapi/v1/premiumIndex", symbol=SYMBOL)
            except Exception:  # noqa: BLE001
                continue
            times.append((time.perf_counter() - t0) * 1000)
        if times:
            times.sort()
            print(f"    median {times[len(times) // 2]:.0f}ms   worst {times[-1]:.0f}ms")
            print("    this is the cost in receive-time precision;")
            print("    exchange timestamps are unaffected.")
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
