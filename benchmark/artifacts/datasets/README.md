This directory is for checked-in dataset-specific benchmark snapshots that are kept only as historical reference.

Current status:

- `coupon_legacy_msplit_atomized/` is a legacy aggregate `coupon` snapshot from an older one-off benchmark flow that reported `msplit_atomized`, `shapecart`, and `xgboost` at depths 2-6.
- `electricity/` was removed because it was stale and incomplete: it mixed an old depth-3 summary with a timeout `last_error.json` from a runner that is no longer the canonical benchmark entrypoint.

Do not treat this directory as the current benchmark source of truth.

For current results, regenerate artifacts with the maintained scripts under `benchmark/scripts/`:

- `benchmark/scripts/benchmark_teacher_guided_atomcolor_cached.py`
- `benchmark/scripts/run_cached_depth_benchmarks_msplit_linear_nonlinear_shapecart.py`
- `benchmark/scripts/analyze_coupon_linear_nonlinear_shapecart.py`
