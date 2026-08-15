#!/usr/bin/env python3
"""Short diagnostic for the six-market cross-venue scanner.

Reports which sources actually produce valid books and the best executable
gross/net two-leg edge seen, even when it is still negative after costs.
No authentication and no order entry.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from pathlib import Path

# When this file is executed as ``python tools/xarb_diag.py`` Python puts the
# tools/ directory, not the repository root, on sys.path.  Add the root so the
# sibling launcher can be imported reliably.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsboard.research import xarb_scan as scan
from tools.xarb_scan import okx_books_normalised

EXPECTED = {
    "binance_spot",
    "binance_perp",
    "bybit_spot",
    "bybit_perp",
    "okx_spot",
    "okx_perp",
}


async def run(duration_s: float, notional: float) -> int:
    q: asyncio.Queue[scan.Book] = asyncio.Queue(maxsize=100_000)
    tasks = [
        asyncio.create_task(scan._binance_book("binance_spot", scan.BINANCE_SPOT_WS, q)),
        asyncio.create_task(scan._binance_book("binance_perp", scan.BINANCE_PERP_WS, q)),
        asyncio.create_task(scan._bybit_book("bybit_spot", q)),
        asyncio.create_task(scan._bybit_book("bybit_perp", q)),
        asyncio.create_task(okx_books_normalised(q)),
    ]

    books: dict[str, scan.Book] = {}
    counts = {name: 0 for name in EXPECTED}
    best_gross = None
    best_net = None
    started = time.monotonic()
    deadline = started + duration_s

    try:
        while time.monotonic() < deadline:
            left = deadline - time.monotonic()
            try:
                book = await asyncio.wait_for(q.get(), timeout=min(1.0, max(0.01, left)))
            except TimeoutError:
                continue
            books[book.source] = book
            counts[book.source] = counts.get(book.source, 0) + 1
            now_ns = time.monotonic_ns()

            for buy_source, buy_book in tuple(books.items()):
                for sell_source, sell_book in tuple(books.items()):
                    if buy_source == sell_source:
                        continue
                    opp = scan.evaluate_cross(
                        buy_book,
                        sell_book,
                        quote_notional=notional,
                        fees_bps=scan.DEFAULT_FEES_BPS,
                        now_ns=now_ns,
                        max_age_ms=250.0,
                        max_skew_ms=250.0,
                        latency_buffer_bps=1.0,
                        fill_buffer_bps=1.0,
                        rebalance_buffer_bps=1.0,
                        safety_buffer_bps=1.0,
                    )
                    if opp is None:
                        continue
                    if best_gross is None or opp.gross_edge_bps > best_gross.gross_edge_bps:
                        best_gross = opp
                    if best_net is None or opp.net_edge_bps > best_net.net_edge_bps:
                        best_net = opp
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    active = sorted(name for name, n in counts.items() if n > 0)
    missing = sorted(EXPECTED - set(active))
    print("\n=== xarb diagnostic ===")
    print("active  : " + (", ".join(active) if active else "none"))
    print("missing : " + (", ".join(missing) if missing else "none"))
    print("counts  : " + ", ".join(f"{k}={counts[k]}" for k in sorted(counts)))

    if best_gross is None:
        print("best    : no comparable fresh two-book sample")
        return 2

    print(
        "best gross: "
        f"{best_gross.buy_source} -> {best_gross.sell_source} "
        f"${notional:g} gross={best_gross.gross_edge_bps:+.3f}bps "
        f"fees={best_gross.fee_bps:.3f}bps net={best_gross.net_edge_bps:+.3f}bps "
        f"skew={best_gross.book_skew_ms:.1f}ms"
    )
    if best_net is not None:
        print(
            "best net  : "
            f"{best_net.buy_source} -> {best_net.sell_source} "
            f"${notional:g} gross={best_net.gross_edge_bps:+.3f}bps "
            f"fees={best_net.fee_bps:.3f}bps net={best_net.net_edge_bps:+.3f}bps "
            f"skew={best_net.book_skew_ms:.1f}ms"
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--notional", type=float, default=1000.0)
    args = p.parse_args()
    return asyncio.run(run(args.duration, args.notional))


if __name__ == "__main__":
    raise SystemExit(main())
