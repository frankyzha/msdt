#!/usr/bin/env python3
"""Submit the cached Optuna benchmark for one dataset to the Fitz nodes."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.benchmark_slurm_submit_common import submit_single_dataset_benchmark


def main() -> int:
    return submit_single_dataset_benchmark(
        script_name="benchmark_cached_optuna_msplit_vs_shapecart.py",
        run_prefix="cached_optuna",
        description=(
            "Submit the cached MSPLIT-vs-ShapeCART Optuna benchmark for one dataset "
            "to the Fitz nodes and return the Slurm submission details."
        ),
        default_log_subdir="cached_optuna_msplit_vs_shapecart",
    )


if __name__ == "__main__":
    raise SystemExit(main())
