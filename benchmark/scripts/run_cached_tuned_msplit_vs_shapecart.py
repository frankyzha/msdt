#!/usr/bin/env python3
"""Compatibility wrapper for the canonical cached Optuna benchmark runner."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.benchmark_cached_optuna_msplit_vs_shapecart import main


if __name__ == "__main__":
    raise SystemExit(main())
