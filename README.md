# msdt

This repository is the active MSPLIT research workspace. The main solver lives
in the local `split` package under `algorithm/msplit/`, with benchmark code and
datasets organized under `benchmark/`.

## Solver Layout

- `algorithm/msplit/src/libgosdt/src/msplit_nonlinear.cpp`
  Active nonlinear selector entrypoint.
- `algorithm/msplit/src/libgosdt/src/msplit_atomized.cpp`
  Default reference-guided atomized selector used by the active nonlinear path.
- `algorithm/msplit/src/libgosdt/src/msplit_linear.cpp`
  Alternate linear selector kept in-tree for comparison and development.
- `algorithm/msplit/src/split/MSPLIT.py`
  Python estimator wrapper for the native MSPLIT solver.
- `split/__init__.py`
  Repository-local import shim so `import split` resolves to the in-repo package.

## Repository Layout

- `algorithm/msplit/`
  Native solver, pybind module, Python package, and tests.
- `algorithm/shapecart/`
  ShapeCART source used by benchmark comparisons.
- `benchmark/datasets/`
  Versioned benchmark datasets and metadata.
- `benchmark/cache/`
  Cached preprocessing artifacts for repeated benchmark runs.
- `benchmark/artifacts/`
  Generated outputs such as summaries, plots, and tree visualizations.
- `benchmark/scripts/`
  Main experiment entrypoints and shared helpers.

## Build The Local Package

The native extension is built from `algorithm/msplit/` with `scikit-build-core`
and CMake. A local editable install is the simplest setup:

```bash
python3 -m pip install -e algorithm/msplit
```

The build expects the native dependencies declared in
`algorithm/msplit/CMakeLists.txt`, notably TBB, GMP, and GLPK.

## Common Workflows

Run the benchmark driver:

```bash
python3 benchmark/scripts/benchmark_cached_msplit.py \
  --dataset electricity \
  --depth 6 \
  --lookahead-depth 3
```

Run the comparison benchmark sweep:

```bash
python3 benchmark/scripts/benchmark_cached_fixed_config_msplit_vs_shapecart.py
```

Run the coupon subset comparison study:

```bash
python3 benchmark/scripts/analyze_coupon_msplit_linear_nonlinear_vs_shapecart.py
```

Run the MSPLIT tests:

```bash
python3 -m pytest algorithm/msplit/tests
```

## Notes

- The maintained visualization entrypoints are
  `visualize_multisplit_tree_n.py` and `visualize_multisplit_tree_color.py`.
- The maintained benchmark entrypoints under `benchmark/scripts/` are
  `benchmark_cached_msplit.py`,
  `benchmark_cached_fixed_config_msplit_vs_shapecart.py`,
  and `benchmark_cached_optuna_msplit_vs_shapecart.py`.
- Generated benchmark artifacts are intentionally ignored so the repository
  stays focused on code, datasets, and reproducible entrypoints.
