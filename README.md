# msdt

This repository contains the current MSPLIT research codebase centered on two
tree-learning variants:

- `linear`: the lightweight linear selector in
  `algorithm/msplit/src/libgosdt/src/msplit_linear.cpp`
- `nonlinear`: the default reference-guided atomized selector in
  `algorithm/msplit/src/libgosdt/src/msplit_nonlinear.cpp`

## Repository Layout

- `algorithm/msplit/`
  Active local `split` package, C++ solver, Python bindings, and tests.
- `algorithm/shapecart/`
  ShapeCART source tree used by the benchmark scripts.
- `benchmark/datasets/`
  Dataset bundles and benchmark metadata.
- `benchmark/cache/`
  Cached LightGBM binning artifacts used by the MSPLIT benchmark workflow.
- `benchmark/artifacts/`
  Benchmark outputs such as run directories, plots, tree visualizations, and summaries.
- `benchmark/scripts/`
  Runnable experiment entrypoints:
  - `benchmark_teacher_guided_atomcolor_cached.py`
  - `run_cached_depth_benchmarks_msplit_linear_nonlinear_shapecart.py`
  - `visualize_multisplit_tree_n.py`
  - `visualize_multisplit_tree_color.py`
  - `visualize_multisplit_tree.py` (compatibility wrapper)
  - `analyze_coupon_linear_nonlinear_shapecart.py`
  - `run_msplit_cache_worker.py`
  - `tune_msplit_cached_optuna.py`
- `benchmark/scripts/dataset.py`, `benchmark/scripts/experiment_utils.py`,
  `benchmark/scripts/lightgbm_binning.py`, `benchmark/scripts/tree_artifact_utils.py`,
  `benchmark/scripts/cache_utils.py`, `benchmark/scripts/benchmark_paths.py`
  Shared Python helpers used by the benchmark scripts.

## What Was Removed

The repository has been pruned to remove:

- historical cluster submission scripts and run logs
- retired one-off diagnostics and benchmark experiments
- accidental tracked build trees and temporary artifacts
- duplicated local scratch datasets and stale output files

## Quick Start

Build the local `split` extension, then run one of the benchmark scripts under
`benchmark/scripts/`. For example:

```bash
python3 benchmark/scripts/benchmark_teacher_guided_atomcolor_cached.py \
  --dataset electricity \
  --depth 6 \
  --lookahead-depth 3
```
