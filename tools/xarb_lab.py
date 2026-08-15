#!/usr/bin/env python3
"""Stable entrypoint for the one-shot six-market xarb research lab."""

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

LAB_VERSION = "2026-08-15-precheck-v3"


async def fixed_precheck(duration_s: float = 20.0) -> dict:
    """Validate that every required public book and trade stream is alive.

    For an update-on-change order book, lack of a final update immediately at
    the end of the window is not itself a feed failure. Precheck therefore
    requires at least one valid book and one trade from every source during the
    window. Per-sample freshness/skew is still enforced later by the analyzer.
    """
    out: asyncio.Queue = asyncio.Queue(maxsize=300_000)
    tasks = await runtime.start_streams(out)
    book_counts, trade_counts, bad_books = Counter(), Counter(), Counter()
    valid_book_seen = set()
    deadline = time.monotonic() + duration_s
    try:
        while time.monotonic() < deadline:
            left = max(0.01, deadline - time.monotonic())
            try:
                kind, obj = await asyncio.wait_for(out.get(), timeout=min(1.0, left))
            except TimeoutError:
                continue
            if kind == "book":
                book_counts[obj.source] += 1
                if obj.valid():
                    valid_book_seen.add(obj.source)
                else:
                    bad_books[obj.source] += 1
            else:
                trade_counts[obj.source] += 1
    finally:
        await runtime.stop_tasks(tasks)

    missing_books = sorted(runtime.SOURCE_SET - valid_book_seen)
    missing_trades = sorted(runtime.SOURCE_SET - {s for s, n in trade_counts.items() if n > 0})
    passed = not missing_books and not missing_trades
    result = {
        "passed": passed,
        "version": LAB_VERSION,
        "duration_s": duration_s,
        "book_counts": {s: book_counts[s] for s in runtime.SOURCES},
        "trade_counts": {s: trade_counts[s] for s in runtime.SOURCES},
        "missing_books": missing_books,
        "missing_trades": missing_trades,
        "bad_books": dict(bad_books),
        "stale_at_end": [],
    }

    print(f"\n=== xarb-lab precheck ({LAB_VERSION}) ===")
    for source in runtime.SOURCES:
        print(f"{source:14s} books={book_counts[source]:6d} trades={trade_counts[source]:7d}")
    print("missing books :", ", ".join(missing_books) if missing_books else "none")
    print("missing trades:", ", ".join(missing_trades) if missing_trades else "none")
    print("bad books     :", dict(bad_books) if bad_books else "none")
    print("stale at end  : not used for precheck; analyzer enforces freshness/skew")
    print("PRECHECK      :", "PASS" if passed else "FAIL")
    return result


runtime.precheck = fixed_precheck


if __name__ == "__main__":
    print(f"[xarb-lab] version {LAB_VERSION}")
    raise SystemExit(runtime.main())
