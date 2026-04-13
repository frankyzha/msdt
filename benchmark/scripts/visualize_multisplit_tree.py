"""Backward-compatible wrapper for numeric-label MSPLIT visualization."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.visualize_multisplit_tree_n import main


if __name__ == "__main__":
    main()
