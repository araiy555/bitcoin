#!/usr/bin/env python3
"""Run the six-market scanner with the Binance Spot compatibility parser."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import xarb_scan as launcher
from tools.xarb_compat import binance_book_compatible

launcher.scan._binance_book = binance_book_compatible

if __name__ == "__main__":
    raise SystemExit(launcher.main())
