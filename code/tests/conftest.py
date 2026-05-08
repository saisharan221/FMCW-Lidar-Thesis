"""Make `ptv3_fmcw` and `scripts` importable from the test suite.

`pytest` runs from the `code/` directory so `code/` itself is what
needs to be on `sys.path`. We add it here so `python -m pytest
code/tests/` from the repo root works without any extra wiring.
"""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
