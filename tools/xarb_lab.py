#!/usr/bin/env python3
"""Stable entrypoint for the one-shot six-market xarb research lab.

This entrypoint patches the runtime precheck so stream freshness is measured at
precheck completion, before websocket shutdown latency, and so update-on-change
books are not incorrectly rejected for a one-second quiet interval.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import xarb_lab_runtime as runtime


async def fixed_precheck(duration_s: float = 20.0) -> dict:
    """Validate all 6 book + trade streams before a long capture.

    Freshness is evaluated *before* cancelling sockets. A public orderbook
    stream is update-on-change, so a 1 second quiet interval is not itself a
    broken feed; five seconds without a valid BTC book is treated as stale.
    """
    out: asyncio.Queue = asyncio.Queue(maxsize=300_000)
    tasks = await runtime.start_streams(out)
    book_counts, trade_counts, bad_books = Counter(), Counter(), Counter()
    last_books = {}
    deadline = time.monotonic() + duration_s
    check_end_ns = time.monotonic_ns()
    try:
        while time.monotonic() < deadline:
            left = max(0.01, deadline - time.monotonic())
            try:
                kind, obj = await asyncio.wait_for(out.get(), timeout=min(1.0, left))
            except TimeoutError:
                continue
            if kind == "book":
                book_counts[obj.source] += 1
                if not obj.valid():
                    bad_books[obj.source] += 1
                else:
                    last_books[obj.source] = obj
            else:
                trade_counts[obj.source] += 1
            check_end_ns = time.monotonic_ns()
    finally:
        # Freeze the freshness reference before websocket close handshakes.
        check_end_ns = time.monotonic_ns()
        await runtime.stop_tasks(tasks)

    missing_books = sorted(runtime.SOURCE_SET - set(book_counts))
    missing_trades = sorted(runtime.SOURCE_SET - set(trade_counts))
    stale = sorted(
        source
        for source, book in last_books.items()
        if (check_end_ns - book.receive_ts_ns) / 1e6 > 5_000.0
    )
    passed = not missing_books and not missing_trades and not bad_books and not stale
    result = {
        "passed": passed,
        "duration_s": duration_s,
        "book_counts": {s: book_counts[s] for s in runtime.SOURCES},
        "trade_counts": {s: trade_counts[s] for s in runtime.SOURCES},
        "missing_books": missing_books,
        "missing_trades": missing_trades,
        "bad_books": dict(bad_books),
        "stale_at_end": stale,
        "stale_threshold_ms": 5_000.0,
    }

    print("\n=== xarb-lab precheck ===")
    for source in runtime.SOURCES:
        print(f"{source:14s} books={book_counts[source]:6d} trades={trade_counts[source]:7d}")
    print("missing books :", ", ".join(missing_books) if missing_books else "none")
    print("missing trades:", ", ".join(missing_trades) if missing_trades else "none")
    print("bad books     :", dict(bad_books) if bad_books else "none")
    print("stale at end  :", ", ".join(stale) if stale else "none")
    print("PRECHECK      :", "PASS" if passed else "FAIL")
    return result


runtime.precheck = fixed_precheck


if __name__ == "__main__":
    raise SystemExit(runtime.main())
