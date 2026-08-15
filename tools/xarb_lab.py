#!/usr/bin/env python3
"""Stable entrypoint for the one-shot six-market xarb research lab."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.xarb_lab_runtime import main


if __name__ == "__main__":
    raise SystemExit(main())
