This directory is for checked-in dataset-specific benchmark snapshots that are kept only as historical reference.

Current status:

- `coupon_legacy_msplit_atomized/` is a legacy aggregate `coupon` snapshot from an older one-off benchmark flow that reported `msplit_atomized`, `shapecart`, and `xgboost` at depths 2-6.
- `electricity/` was removed because it was stale and incomplete: it mixed an old depth-3 summary with a timeout `last_error.json` from a runner that is no longer the canonical benchmark entrypoint.

Do not treat this directory as the current benchmark source of truth.

For current results, regenerate artifacts with the maintained scripts under `benchmark/scripts/`:

- `benchmark/scripts/benchmark_cached_msplit.py`
- `benchmark/scripts/benchmark_cached_optuna_msplit_vs_shapecart.py`
- `benchmark/scripts/benchmark_cached_gridcv_msplit_vs_shapecart.py`
- `benchmark/scripts/benchmark_cached_fixed_config_msplit_vs_shapecart.py`
- `benchmark/scripts/analyze_coupon_msplit_linear_nonlinear_vs_shapecart.py`
