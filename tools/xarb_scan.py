#!/usr/bin/env python3
"""Run the six-market executable BTC cross-venue scanner.

Research-only: public market data, no authentication, no order entry.

OKX perpetual book sizes are reported in contracts.  Before scanning, this
launcher fetches BTC-USDT-SWAP instrument metadata and converts contract size
to BTC using ctVal so all six books are comparable in base-asset quantity.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp
import websockets

from jsboard.net import make_session, ssl_context
from jsboard.research import xarb_scan as scan


async def okx_contract_value_btc() -> float:
    url = "https://www.okx.com/api/v5/public/instruments"
    params = {"instType": "SWAP", "instId": "BTC-USDT-SWAP"}
    async with make_session() as session, session.get(
        url, params=params, timeout=aiohttp.ClientTimeout(total=15)
    ) as resp:
        resp.raise_for_status()
        payload = await resp.json()
    rows = payload.get("data") or []
    if not rows:
        raise RuntimeError("OKX BTC-USDT-SWAP metadata is empty")
    row = rows[0]
    ct_val = float(row.get("ctVal") or 0)
    ct_ccy = row.get("ctValCcy") or ""
    if ct_val <= 0:
        raise RuntimeError("OKX BTC-USDT-SWAP ctVal is missing")
    if ct_ccy != "BTC":
        raise RuntimeError(
            f"OKX BTC-USDT-SWAP ctValCcy={ct_ccy!r}; expected 'BTC', refusing to guess"
        )
    return ct_val


def okx_levels(rows, *, reverse: bool, qty_multiplier: float):
    levels = []
    for row in rows or ():
        try:
            price = float(row[0])
            qty = float(row[1]) * qty_multiplier
        except (TypeError, ValueError, IndexError):
            continue
        if price > 0 and qty > 0:
            levels.append((price, qty))
    levels.sort(key=lambda x: x[0], reverse=reverse)
    return tuple(levels)


async def okx_books_normalised(out: asyncio.Queue) -> None:
    contract_btc = await okx_contract_value_btc()
    print(f"[xarb-scan] OKX BTC-USDT-SWAP ctVal = {contract_btc:g} BTC/contract")

    subscriptions = [
        {"channel": "books5", "instId": "BTC-USDT"},
        {"channel": "books5", "instId": "BTC-USDT-SWAP"},
    ]
    source_for = {
        "BTC-USDT": "okx_spot",
        "BTC-USDT-SWAP": "okx_perp",
    }

    while True:
        try:
            async with websockets.connect(
                scan.OKX_WS,
                ping_interval=20,
                ping_timeout=20,
                max_queue=2**14,
                ssl=ssl_context(),
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": subscriptions}))
                async for raw in ws:
                    recv = time.monotonic_ns()
                    payload = json.loads(raw)
                    if payload.get("event") == "error":
                        raise RuntimeError(payload.get("msg") or "OKX subscribe failed")
                    arg = payload.get("arg") or {}
                    if arg.get("channel") != "books5":
                        continue
                    inst_id = arg.get("instId")
                    source = source_for.get(inst_id)
                    if source is None:
                        continue
                    multiplier = contract_btc if inst_id == "BTC-USDT-SWAP" else 1.0
                    for row in payload.get("data") or ():
                        book = scan.Book(
                            source=source,
                            bids=okx_levels(
                                row.get("bids"), reverse=True, qty_multiplier=multiplier
                            ),
                            asks=okx_levels(
                                row.get("asks"), reverse=False, qty_multiplier=multiplier
                            ),
                            exchange_ts_ns=int(row.get("ts") or 0) * 1_000_000,
                            receive_ts_ns=recv,
                        )
                        if book.valid():
                            await out.put(book)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[xarb-scan] OKX reconnect: {type(exc).__name__}: {exc}")
            await asyncio.sleep(1.0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BTC cross-venue executable Net-edge scanner")
    p.add_argument("--duration", type=float, default=3600.0, help="seconds")
    p.add_argument(
        "--notional",
        type=float,
        nargs="+",
        default=[100, 500, 1000, 5000, 10000],
        help="quote notionals to test in USDT/USD-equivalent",
    )
    p.add_argument("--min-net-bps", type=float, default=2.0)
    p.add_argument("--max-age-ms", type=float, default=250.0)
    p.add_argument("--max-skew-ms", type=float, default=250.0)
    p.add_argument("--latency-buffer-bps", type=float, default=1.0)
    p.add_argument("--fill-buffer-bps", type=float, default=1.0)
    p.add_argument("--rebalance-buffer-bps", type=float, default=1.0)
    p.add_argument("--safety-buffer-bps", type=float, default=1.0)
    p.add_argument("--out", default="xarb-scan.jsonl")

    # Research assumptions only. Override these with the user's actual fee tier.
    for source, default in scan.DEFAULT_FEES_BPS.items():
        p.add_argument(
            f"--{source.replace('_', '-')}-fee-bps",
            type=float,
            default=default,
            help=f"one-way taker fee bps for {source}; default {default:g}",
        )
    return p


async def amain(args: argparse.Namespace) -> int:
    scan._okx_books = okx_books_normalised

    fees = {
        source: getattr(args, f"{source}_fee_bps")
        for source in scan.DEFAULT_FEES_BPS
    }
    print("BTC cross-venue scanner")
    print("  markets : Binance/Bybit/OKX x spot/perp")
    print("  orders  : NONE (public market-data research only)")
    print(f"  duration: {args.duration:g}s")
    print(f"  sizes   : {', '.join(f'${x:g}' for x in args.notional)}")
    print(f"  min Net : {args.min_net_bps:g} bps")
    print("  fees    : " + ", ".join(f"{k}={v:g}" for k, v in fees.items()) + " bps")
    print(
        "  buffers : "
        f"latency={args.latency_buffer_bps:g}, fill={args.fill_buffer_bps:g}, "
        f"rebalance={args.rebalance_buffer_bps:g}, safety={args.safety_buffer_bps:g} bps"
    )
    print(f"  output  : {args.out}\n")

    result = await scan.run_live(
        duration_s=args.duration,
        notionals=tuple(args.notional),
        fees_bps=fees,
        min_net_bps=args.min_net_bps,
        max_age_ms=args.max_age_ms,
        max_skew_ms=args.max_skew_ms,
        latency_buffer_bps=args.latency_buffer_bps,
        fill_buffer_bps=args.fill_buffer_bps,
        rebalance_buffer_bps=args.rebalance_buffer_bps,
        safety_buffer_bps=args.safety_buffer_bps,
        out_path=args.out,
    )
    print("\n=== xarb-scan summary ===")
    print(scan.format_summary(result))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
