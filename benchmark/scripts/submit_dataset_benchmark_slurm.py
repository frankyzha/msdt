#!/usr/bin/env python3
"""Compatibility wrapper for the cached Optuna Slurm submitter."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.submit_cached_optuna_benchmark_slurm import main


if __name__ == "__main__":
    raise SystemExit(main())
