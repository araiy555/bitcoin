#!/usr/bin/env python3
"""Run the six-market executable cross-venue scanner without changing jsboard CLI."""

from __future__ import annotations

import argparse
import asyncio

from jsboard.research.xarb_scan import DEFAULT_FEES_BPS, format_summary, run_live


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

    for source, default in DEFAULT_FEES_BPS.items():
        p.add_argument(
            f"--{source.replace('_', '-')}-fee-bps",
            type=float,
            default=default,
            help=f"one-way taker fee bps for {source}; default {default:g}",
        )
    return p


async def amain(args: argparse.Namespace) -> int:
    fees = {
        source: getattr(args, f"{source}_fee_bps")
        for source in DEFAULT_FEES_BPS
    }
    print("BTC cross-venue scanner")
    print("  markets : Binance/Bybit/OKX x spot/perp")
    print("  orders  : NONE (public market-data research only)")
    print(f"  duration: {args.duration:g}s")
    print(f"  sizes   : {', '.join(f'${x:g}' for x in args.notional)}")
    print(f"  min Net : {args.min_net_bps:g} bps")
    print(f"  output  : {args.out}\n")

    result = await run_live(
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
    print(format_summary(result))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
