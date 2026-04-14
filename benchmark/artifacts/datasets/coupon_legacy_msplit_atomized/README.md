This folder is a legacy aggregate `coupon` benchmark snapshot.

What it contains:

- `summary_by_depth.csv`: older depth-2..6 results for `msplit_atomized`, `shapecart`, and `xgboost`
- `plots/`: plots generated from that same summary

Why it is legacy:

- It comes from an older one-off benchmark flow, not from the current cached benchmark scripts.
- The MSPLIT numbers here are for the older `msplit_atomized` snapshot that was checked in alongside `benchmark/datasets/coupon/run.log`.
- The aggregate `coupon` task is heterogeneous, so this snapshot should not be used as the main reference for current coupon performance questions.

For current coupon analysis, use `benchmark/scripts/analyze_coupon_msplit_linear_nonlinear_vs_shapecart.py`, which evaluates the overall dataset plus the five coupon-type subsets with the maintained cached benchmark workflow.
