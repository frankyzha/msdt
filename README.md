# msdt

This repository contains the current MSPLIT research codebase centered on two
tree-learning variants:

- `linear`: the lightweight linear selector in
  `SPLIT-ICML/split/src/libgosdt/src/msplit_linear.cpp`
- `nonlinear`: the default reference-guided atomized selector in
  `SPLIT-ICML/split/src/libgosdt/src/msplit_nonlinear.cpp`

## Repository Layout

- `SPLIT-ICML/split/`
  Active local `split` package, C++ solver, Python bindings, and tests.
- `Empowering-DTs-via-Shape-Functions/`
  Vendored ShapeCART baseline used by benchmark scripts.
- `datasets/`
  Dataset bundles and benchmark metadata used by the current experiments.
- `scripts/`
  Runnable experiment entrypoints:
  - `benchmark_teacher_guided_atomcolor_cached.py`
  - `run_cached_depth_benchmarks_msplit_linear_nonlinear_shapecart.py`
  - `visualize_multisplit_tree.py`
  - `analyze_coupon_linear_nonlinear_shapecart.py`
  - `run_msplit_cache_worker.py`
  - `tune_msplit_cached_optuna.py`
- `dataset.py`, `experiment_utils.py`, `lightgbm_binning.py`, `tree_artifact_utils.py`
  Shared Python helpers used by the scripts.

## What Was Removed

The repository has been pruned to remove:

- historical cluster submission scripts and run logs
- retired one-off diagnostics and benchmark experiments
- accidental tracked build trees and temporary artifacts
- duplicated local scratch datasets and stale output files

## Quick Start

Build the local `split` extension, then run one of the benchmark scripts under
`scripts/`. For example:

```bash
PYTHONPATH=SPLIT-ICML/split/build-fast-py:SPLIT-ICML/split/src \
python scripts/benchmark_teacher_guided_atomcolor_cached.py \
  --dataset electricity \
  --depth 6 \
  --lookahead-depth 3
```
