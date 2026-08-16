"""Compatibility shim for running scripts directly from the tools directory.

When Python executes ``python tools/jane_fusion.py``, ``sys.path[0]`` is the
``tools`` directory itself, so ``from tools import jane_lab`` would otherwise
fail because the repository root is not on the import path.  Expose jane_lab
from this local shim so the documented direct-script command works as-is.
"""

import jane_lab as jane_lab

__all__ = ["jane_lab"]
