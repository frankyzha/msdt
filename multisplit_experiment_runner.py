"""Shared runner for LightGBM-binned multi-split SPLIT experiments."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import tarfile
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except Exception:  # pragma: no cover - non-POSIX environments
    fcntl = None

import matplotlib

# Force non-interactive backend so this runs cleanly in headless/batch sessions.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from experiment_utils import DATASET_LOADERS, PAPER_SOTA, encode_binary_target, make_preprocessor
from tree_artifact_utils import build_msplit_artifact, write_artifact_json

# Fallback import path if the package is not installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_SRC = PROJECT_ROOT / "SPLIT-ICML" / "split" / "src"
if str(SPLIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPLIT_SRC))

from split import MSPLIT, SPLIT
from split._binarizer import NumericBinarizer

try:
    import optuna
    from optuna.trial import TrialState
except Exception:  # pragma: no cover - optional dependency for legacy helpers
    optuna = None
    TrialState = None

# Require the native C++ backend so runs do not silently fall back to Python DP.
_MSPLIT_MODULE = importlib.import_module("split.MSPLIT")
if getattr(_MSPLIT_MODULE, "_cpp_msplit_fit", None) is None:
    raise RuntimeError(
        "split C++ backend (_libgosdt/msplit_fit) is unavailable. "
        "Install SPLIT with native extension support before running experiments."
    )


SEED_COLUMNS = [
    "dataset",
    "depth_budget",
    "lookahead_depth_budget",
    "seed",
    "status",
    "msplit_variant",
    "error",
    "class_prevalence",
    "train_accuracy",
    "accuracy",
    "balanced_accuracy",
    "trivial_accuracy",
    "fit_time_sec",
    "objective",
    "n_leaves",
    "n_internal",
    "max_arity",
    "exact_internal_nodes",
    "greedy_internal_nodes",
    "dp_subproblem_calls",
    "dp_cache_hits",
    "dp_unique_states",
    "dp_cache_profile_enabled",
    "dp_cache_lookup_calls",
    "dp_cache_miss_no_bucket",
    "dp_cache_miss_bucket_present",
    "dp_cache_miss_depth_mismatch_only",
    "dp_cache_miss_indices_mismatch",
    "dp_cache_depth_match_candidates",
    "dp_cache_bucket_entries_scanned",
    "dp_cache_bucket_max_size",
    "greedy_subproblem_calls",
    "profiling_greedy_complete_calls_by_depth",
    "greedy_cache_hits",
    "greedy_unique_states",
    "greedy_cache_entries_peak",
    "greedy_cache_bytes_peak",
    "native_n_classes",
    "native_teacher_class_count",
    "native_binary_mode",
    "atomized_features_prepared",
    "atomized_coarse_candidates",
    "atomized_coarse_pruned_candidates",
    "atomized_coarse_prune_rate",
    "atomized_coarse_survivor_rate",
    "atomized_final_candidates",
    "greedy_cache_clears",
    "nominee_unique_total",
    "nominee_child_interval_lookups",
    "nominee_child_interval_unique",
    "nominee_exactified_total",
    "nominee_incumbent_updates",
    "nominee_threatening_samples",
    "nominee_threatening_sum",
    "nominee_threatening_max",
    "nominee_exact_child_eval_sec",
    "nominee_debr_sec",
    "rush_total_time_sec",
    "rush_refinement_child_time_sec",
    "rush_refinement_child_time_fraction",
    "rush_refinement_child_calls",
    "rush_refinement_recursive_calls",
    "rush_refinement_recursive_unique_states",
    "rush_refinement_depth_logs",
    "interval_refinements_attempted",
    "debr_refine_calls",
    "debr_refine_improved",
    "debr_total_moves",
    "debr_bridge_policy_calls",
    "debr_descent_moves",
    "debr_bridge_moves",
    "debr_simplify_moves",
    "debr_total_hard_gain",
    "debr_total_soft_gain",
    "debr_total_delta_j",
    "debr_total_component_delta",
    "debr_final_geo_wins",
    "debr_final_block_wins",
    "expensive_child_calls",
    "expensive_child_sec",
    "expensive_child_exactify_calls",
    "expensive_child_exactify_sec",
    "fast100_exactify_nodes_allowed",
    "fast100_exactify_nodes_skipped_small_support",
    "fast100_exactify_nodes_skipped_dominant_gain",
    "depth1_skipped_by_low_global_ambiguity",
    "depth1_skipped_by_large_gap",
    "depth1_exactify_challenger_nodes",
    "depth1_exactified_nodes",
    "depth1_exactified_features_mean",
    "depth1_exactified_features_max",
    "depth1_teacher_replaced_runnerup",
    "depth1_teacher_rejected_by_uhat_gate",
    "depth1_exactify_set_size_mean",
    "depth1_exactify_set_size_max",
    "fast100_skipped_by_ub_lb_separation",
    "fast100_widen_forbidden_depth_gt0_attempts",
    "fast100_frontier_size_mean",
    "fast100_frontier_size_max",
    "fast100_stopped_midloop_separation",
    "fast100_M_depth0_mean",
    "fast100_M_depth0_max",
    "fast100_M_depth1_mean",
    "fast100_M_depth1_max",
    "fast100_cf_exactify_nodes_depth0",
    "fast100_cf_exactify_nodes_depth1",
    "fast100_cf_skipped_agreement",
    "fast100_cf_skipped_small_regret",
    "fast100_cf_skipped_low_impact",
    "fast100_cf_frontier_size_mean",
    "fast100_cf_frontier_size_max",
    "fast100_cf_exactified_features_mean",
    "fast100_cf_exactified_features_max",
    "rootsafe_exactified_features",
    "rootsafe_root_winner_changed_vs_proxy",
    "rootsafe_root_candidates_K",
    "fast100_used_lgb_prior_tiebreak",
    "gini_dp_calls_root",
    "gini_dp_calls_depth1",
    "gini_teacher_chosen_depth1",
    "gini_tiebreak_used_in_shortlist",
    "gini_dp_sec",
    "gini_root_k0",
    "gini_endpoints_added_root",
    "gini_endpoints_added_depth1",
    "gini_endpoints_features_touched_root",
    "gini_endpoints_features_touched_depth1",
    "gini_endpoints_added_per_feature_max",
    "gini_endpoint_sec",
    "used_max_bins",
    "used_min_samples_leaf",
    "used_min_child_size",
    "used_max_branching",
    "used_reg",
    "tree_artifact_path",
]

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

FIXED_TRAIN_FRACTION = 0.70
FIXED_VAL_FRACTION = 0.10
FIXED_TEST_FRACTION = 0.20
FIXED_VAL_WITHIN_TRAIN = FIXED_VAL_FRACTION / (1.0 - FIXED_TEST_FRACTION)


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    description: str
    solver: str
    binning_backend: str
    run_name_fmt: str
    results_root_default: str
    stable_results_dir_default: str
    depth_log_filename: str
    plot_filename: str
    plot_vs_paper_filename: str
    stable_csv_filename: str
    stable_plot_filename: str
    stable_log_filename: str
    plot_color: str
    plot_label: str
    depth_log_title: str
    start_message: str
    study_prefix: str
    artifact_pipeline_name: str


PIPELINES: dict[str, PipelineSpec] = {
    "lightgbm": PipelineSpec(
        name="lightgbm",
        description="Run LightGBM-binned multi-split SPLIT experiments on OpenML datasets.",
        solver="msplit",
        binning_backend="lightgbm",
        run_name_fmt="run_lightgbm_%Y%m%d_%H%M%S",
        results_root_default="results/runs_lightgbm",
        stable_results_dir_default="results/lightgbm",
        depth_log_filename="multisplit_lightgbm_depth_vs_accuracy.log",
        plot_filename="multisplit_lightgbm_dp_accuracy.png",
        plot_vs_paper_filename="multisplit_lightgbm_dp_vs_paper_accuracy.png",
        stable_csv_filename="multisplit_lightgbm_dp_results.csv",
        stable_plot_filename="multisplit_lightgbm_dp_accuracy.png",
        stable_log_filename="multisplit_lightgbm_depth_vs_accuracy.log",
        plot_color="#0a9396",
        plot_label="MSPLIT + LightGBM bins",
        depth_log_title="MSPLIT (LightGBM bins + C++ DP/lookahead) depth vs. accuracy",
        start_message="Starting LightGBM multisplit experiment run.",
        study_prefix="lightgbm",
        artifact_pipeline_name="lightgbm",
    ),
    "lightgbm_split": PipelineSpec(
        name="lightgbm_split",
        description="Run LightGBM-binned binary SPLIT experiments on OpenML datasets.",
        solver="split",
        binning_backend="lightgbm",
        run_name_fmt="run_lightgbm_split_%Y%m%d_%H%M%S",
        results_root_default="results/runs_lightgbm_split",
        stable_results_dir_default="results/lightgbm_split",
        depth_log_filename="split_lightgbm_depth_vs_accuracy.log",
        plot_filename="split_lightgbm_accuracy.png",
        plot_vs_paper_filename="split_lightgbm_vs_paper_accuracy.png",
        stable_csv_filename="split_lightgbm_results.csv",
        stable_plot_filename="split_lightgbm_accuracy.png",
        stable_log_filename="split_lightgbm_depth_vs_accuracy.log",
        plot_color="#2a9d8f",
        plot_label="SPLIT + LightGBM bins",
        depth_log_title="SPLIT (LightGBM bins + GOSDT) depth vs. accuracy",
        start_message="Starting LightGBM SPLIT experiment run.",
        study_prefix="lightgbm_split",
        artifact_pipeline_name="lightgbm_split",
    ),
}

WORKER_DATASET_PAYLOAD: dict[str, dict[str, Any]] | None = None
WORKER_ARGS: argparse.Namespace | None = None
WORKER_TUNED_PARAMS: dict[tuple[str, int], dict[str, Any]] | None = None
WORKER_PIPELINE: PipelineSpec | None = None
WORKER_PREPROCESSED_CACHE: dict[tuple[str, int], dict[str, Any]] = {}
WORKER_BINNER_CACHE: dict[tuple[Any, ...], Any] = {}


def _get_pipeline(pipeline_name: str) -> PipelineSpec:
    if pipeline_name not in PIPELINES:
        raise ValueError(f"unknown pipeline '{pipeline_name}', expected one of {sorted(PIPELINES)}")
    return PIPELINES[pipeline_name]


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(message: str, log_fp) -> None:
    stamped = f"[{_iso_now()}] {message}"
    print(stamped, flush=True)
    if log_fp is not None:
        log_fp.write(stamped + "\n")
        log_fp.flush()


def _add_lightgbm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lgb-n-estimators", type=int, default=10000)
    parser.add_argument("--lgb-num-leaves", type=int, default=31)
    parser.add_argument("--lgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgb-feature-fraction", type=float, default=1.0)
    parser.add_argument("--lgb-bagging-fraction", type=float, default=1.0)
    parser.add_argument("--lgb-bagging-freq", type=int, default=0)
    parser.add_argument("--lgb-max-depth", type=int, default=-1)
    parser.add_argument("--lgb-min-data-in-bin", type=int, default=1)
    parser.add_argument("--lgb-min-data-in-leaf", type=int, default=2)
    parser.add_argument("--lgb-lambda-l2", type=float, default=0.0)
    parser.add_argument("--lgb-num-threads", type=int, default=1)
    parser.add_argument(
        "--lgb-ensemble-runs",
        type=int,
        default=int(os.environ.get("LGB_ENSEMBLE_RUNS", "1")),
        help="Number of LightGBM binning fits to union thresholds across.",
    )
    parser.add_argument(
        "--lgb-ensemble-feature-fraction",
        type=float,
        default=float(os.environ.get("LGB_ENSEMBLE_FEATURE_FRACTION", "0.8")),
        help="Feature subsampling fraction used on stochastic ensemble runs (>1 run).",
    )
    parser.add_argument(
        "--lgb-ensemble-bagging-fraction",
        type=float,
        default=float(os.environ.get("LGB_ENSEMBLE_BAGGING_FRACTION", "0.8")),
        help="Row subsampling fraction used on stochastic ensemble runs (>1 run).",
    )
    parser.add_argument(
        "--lgb-ensemble-bagging-freq",
        type=int,
        default=int(os.environ.get("LGB_ENSEMBLE_BAGGING_FREQ", "1")),
        help="Bagging frequency used on stochastic ensemble runs (>1 run).",
    )
    parser.add_argument(
        "--lgb-threshold-dedup-eps",
        type=float,
        default=float(os.environ.get("LGB_THRESHOLD_DEDUP_EPS", "1e-9")),
        help="Absolute tolerance for deduplicating near-identical LightGBM thresholds in union mode.",
    )
    parser.add_argument(
        "--lgb-device-type",
        choices=["cpu", "gpu", "cuda"],
        default=str(os.environ.get("LGB_DEVICE_TYPE", "gpu")).strip().lower(),
        help="LightGBM backend for preprocessing binning.",
    )
    parser.add_argument(
        "--lgb-gpu-platform-id",
        type=int,
        default=int(os.environ.get("LGB_GPU_PLATFORM_ID", "0")),
        help="GPU platform id for LightGBM when device is gpu/cuda.",
    )
    parser.add_argument(
        "--lgb-gpu-device-id",
        type=int,
        default=int(os.environ.get("LGB_GPU_DEVICE_ID", "0")),
        help="GPU device id for LightGBM when device is gpu/cuda.",
    )
    parser.add_argument(
        "--lgb-gpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=str(os.environ.get("LGB_GPU_FALLBACK", "1")).strip().lower() not in {"0", "false", "no"},
        help="If GPU binning fails, retry LightGBM binning on CPU.",
    )
    parser.add_argument(
        "--lgb-max-gpu-jobs",
        type=int,
        default=int(os.environ.get("LGB_MAX_GPU_JOBS", "1")),
        help="Max concurrent LightGBM GPU fits across all worker processes.",
    )
    parser.add_argument(
        "--lgb-gpu-lock-dir",
        type=str,
        default=str(os.environ.get("LGB_GPU_LOCK_DIR", "/tmp/msdt_lgb_gpu_lock")),
        help="Shared lock directory used to enforce --lgb-max-gpu-jobs.",
    )


def _parse_args(pipeline: PipelineSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=pipeline.description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_LOADERS.keys()),
        default=["electricity", "eye-movements", "eye-state"],
        help="Datasets to run.",
    )
    parser.add_argument(
        "--depth-budgets",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="Tree depth budgets.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random seeds used for train/test splits.",
    )
    parser.add_argument("--max-bins", type=int, default=1024)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--min-child-size", type=int, default=2)
    parser.add_argument(
        "--min-split-size",
        type=int,
        default=0,
        help="Minimum rows required to consider splitting a node (0 lets the solver derive 2 * min_child_size).",
    )
    parser.add_argument(
        "--leaf-frac",
        type=float,
        default=0.002,
        help=(
            "Optional fixed min-support fraction over fit rows. When set, min_child_size is "
            "derived as max(2, ceil(leaf_frac * n_fit)) while LightGBM min_samples_leaf stays independent."
        ),
    )
    parser.add_argument(
        "--max-branching",
        type=int,
        default=0,
        help="0 means no cap in the C++ solver.",
    )
    parser.add_argument("--reg", type=float, default=1e-4)
    parser.add_argument("--time-limit", type=float, default=3000.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--lookahead-cap", type=int, default=3)
    parser.add_argument(
        "--msplit-variant",
        choices=["optimal_dp", "rush_dp"],
        default=str(os.environ.get("MSPLIT_VARIANT", "rush_dp")).strip().lower(),
        help="MSPLIT interval partitioner variant (only used for msplit solver pipelines).",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help="Number of parallel trial processes over (dataset, depth, seed).",
    )
    parser.add_argument(
        "--threads-per-trial",
        type=int,
        default=1,
        help="Thread cap inside each trial process (BLAS/OpenMP).",
    )
    parser.add_argument(
        "--optuna-enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Optuna hyperparameter optimization per (dataset, depth).",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=5,
        help="Number of Optuna trials per study.",
    )
    parser.add_argument(
        "--optuna-val-size",
        type=float,
        default=FIXED_VAL_WITHIN_TRAIN,
        help="Deprecated override; runtime now enforces a fixed stratified 70/10/20 train/val/test split.",
    )
    parser.add_argument(
        "--paper-split-protocol",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated no-op; the strict 70/10/20 split protocol is always enabled.",
    )
    parser.add_argument(
        "--optuna-seed",
        type=int,
        default=0,
        help="Base random seed for Optuna samplers and objective splits.",
    )
    parser.add_argument(
        "--optuna-timeout-sec",
        type=float,
        default=0.0,
        help="Optional timeout per Optuna study in seconds (0 disables timeout).",
    )
    parser.add_argument(
        "--optuna-warmstart-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm-start each Optuna study from historical best_params.csv files.",
    )
    parser.add_argument(
        "--optuna-warmstart-root",
        type=str,
        default="results",
        help="Root directory to scan recursively for historical best_params.csv.",
    )
    parser.add_argument(
        "--optuna-warmstart-max-per-study",
        type=int,
        default=8,
        help="Max warm-start trials enqueued per (dataset, depth) study.",
    )
    parser.add_argument(
        "--optuna-seed-candidates",
        type=int,
        default=4,
        help="Extra hand-crafted candidate trials enqueued per study.",
    )
    parser.add_argument(
        "--optuna-leaf-frac-grid",
        nargs="+",
        type=float,
        default=[0.01, 0.02, 0.05],
        help="Categorical Optuna candidate values for leaf_frac (fraction of fit rows).",
    )
    parser.add_argument(
        "--cpu-utilization-target",
        type=float,
        default=0.9,
        help="Fraction of requested CPU budget to actively use (0,1].",
    )
    parser.add_argument(
        "--cpu-affinity",
        type=str,
        default=str(os.environ.get("MSDT_CPU_AFFINITY", "")).strip(),
        help="Optional CPU set to pin this process to, e.g. '0-7' or '0,2,4,6'.",
    )
    parser.add_argument(
        "--cpu-nice",
        type=int,
        default=int(os.environ.get("MSDT_CPU_NICE", "0")),
        help="Optional niceness delta applied at startup (positive lowers priority).",
    )
    parser.add_argument(
        "--optuna-max-active-studies",
        type=int,
        default=0,
        help="Upper bound on concurrent Optuna studies (0 means auto).",
    )
    parser.add_argument(
        "--optuna-max-concurrent-trials",
        type=int,
        default=0,
        help="Upper bound on total concurrent Optuna trial evaluations across all studies (0 means auto).",
    )

    if pipeline.binning_backend == "lightgbm":
        _add_lightgbm_args(parser)

    parser.add_argument(
        "--results-root",
        type=str,
        default=pipeline.results_root_default,
        help="Parent directory that will contain per-run folders.",
    )
    parser.add_argument(
        "--stable-results-dir",
        type=str,
        default=pipeline.stable_results_dir_default,
        help="Directory for compatibility files with fixed names.",
    )
    parser.add_argument(
        "--openml-cache-dir",
        type=str,
        default="results/openml_cache",
        help="Persistent cache directory for OpenML downloads.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run folder name. If omitted, uses timestamp.",
    )
    parser.add_argument(
        "--tree-artifacts-dir",
        type=str,
        default=None,
        help="Directory where per-(dataset,depth,seed) tree artifacts are written.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run folder and skip completed trials.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Optional cap on newly executed trials (0 means no cap).",
    )
    parser.add_argument(
        "--package-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a tar.gz bundle of the run artifacts.",
    )
    parser.add_argument(
        "--include-paper-sota",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay paper SOTA curve in a second plot.",
    )
    parser.add_argument(
        "--copy-to",
        type=str,
        default=None,
        help="Optional destination directory to copy final artifacts for download.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately on non-timeout errors.",
    )

    args = parser.parse_args()
    args.pipeline_name = pipeline.name
    args.uses_lightgbm_binning = pipeline.binning_backend == "lightgbm"
    if args.uses_lightgbm_binning:
        argv_tokens = sys.argv[1:]

        def _arg_was_set(flag: str) -> bool:
            return any(token == flag or token.startswith(flag + "=") for token in argv_tokens)

        max_bins_set = _arg_was_set("--max-bins")
        lgb_num_leaves_set = _arg_was_set("--lgb-num-leaves")
        if max_bins_set ^ lgb_num_leaves_set:
            if max_bins_set:
                args.lgb_num_leaves = int(args.max_bins)
            else:
                args.max_bins = int(args.lgb_num_leaves)
    args.depth_budgets = sorted(set(args.depth_budgets))
    args.seeds = sorted(set(args.seeds))
    args.datasets = list(dict.fromkeys(args.datasets))
    args.parallel_trials = max(1, int(args.parallel_trials))
    args.threads_per_trial = max(1, int(args.threads_per_trial))
    args.optuna_trials = max(1, int(args.optuna_trials))
    args.optuna_timeout_sec = max(0.0, float(args.optuna_timeout_sec))
    args.optuna_warmstart_max_per_study = max(0, int(args.optuna_warmstart_max_per_study))
    args.optuna_seed_candidates = max(0, int(args.optuna_seed_candidates))
    args.optuna_max_active_studies = max(0, int(args.optuna_max_active_studies))
    args.optuna_max_concurrent_trials = max(0, int(args.optuna_max_concurrent_trials))
    args.cpu_utilization_target = min(1.0, max(0.05, float(args.cpu_utilization_target)))
    args.cpu_affinity = str(getattr(args, "cpu_affinity", "")).strip()
    args.cpu_nice = int(getattr(args, "cpu_nice", 0))
    if args.leaf_frac is not None:
        args.leaf_frac = float(args.leaf_frac)
        if not 0.0 < args.leaf_frac <= 1.0:
            raise ValueError(f"--leaf-frac must be in (0, 1], got {args.leaf_frac}")
        if math.isclose(args.leaf_frac, 0.005, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("--leaf-frac=0.005 is disabled; use >= 0.01.")
    leaf_grid = sorted({float(v) for v in (args.optuna_leaf_frac_grid or [])})
    if not leaf_grid:
        raise ValueError("--optuna-leaf-frac-grid must contain at least one value.")
    for frac in leaf_grid:
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"--optuna-leaf-frac-grid values must be in (0, 1], got {frac}")
        if math.isclose(frac, 0.005, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("--optuna-leaf-frac-grid cannot include 0.005.")
    args.optuna_leaf_frac_grid = leaf_grid
    variant_aliases = {
        "optimal_dp": "optimal_dp",
        "optimal": "optimal_dp",
        "dp": "optimal_dp",
        "rush_dp": "rush_dp",
        "rush": "rush_dp",
        "rushdp": "rush_dp",
    }
    variant_raw = str(getattr(args, "msplit_variant", "optimal_dp")).strip().lower()
    if variant_raw not in variant_aliases:
        raise ValueError(
            f"--msplit-variant must be one of {sorted({'optimal_dp', 'rush_dp'})}, got {args.msplit_variant!r}"
        )
    args.msplit_variant = variant_aliases[variant_raw]
    if not 0.0 < float(args.test_size) < 1.0:
        raise ValueError(f"--test-size must be in (0, 1), got {args.test_size}")
    if not 0.0 < float(args.optuna_val_size) < 1.0:
        raise ValueError(f"--optuna-val-size must be in (0, 1), got {args.optuna_val_size}")

    if not math.isclose(float(args.test_size), FIXED_TEST_FRACTION, rel_tol=0.0, abs_tol=1e-12):
        warnings.warn(
            f"Overriding --test-size={args.test_size} to fixed 70/10/20 protocol value {FIXED_TEST_FRACTION}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if not math.isclose(float(args.optuna_val_size), FIXED_VAL_WITHIN_TRAIN, rel_tol=0.0, abs_tol=1e-12):
        warnings.warn(
            (
                f"Overriding --optuna-val-size={args.optuna_val_size} to fixed train-relative validation "
                f"fraction {FIXED_VAL_WITHIN_TRAIN} (70/10/20 protocol)."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    args.test_size = float(FIXED_TEST_FRACTION)
    args.optuna_val_size = float(FIXED_VAL_WITHIN_TRAIN)
    args.paper_split_protocol = True

    if args.uses_lightgbm_binning:
        # Keep these fixed and out of Optuna; early stopping determines effective rounds.
        args.lgb_n_estimators = 10000
        args.lgb_max_depth = -1
        args.lgb_num_threads = max(1, int(args.lgb_num_threads))
        args.lgb_max_gpu_jobs = max(1, int(args.lgb_max_gpu_jobs))
        args.lgb_ensemble_runs = max(1, int(args.lgb_ensemble_runs))
        args.lgb_ensemble_feature_fraction = min(1.0, max(1e-6, float(args.lgb_ensemble_feature_fraction)))
        args.lgb_ensemble_bagging_fraction = min(1.0, max(1e-6, float(args.lgb_ensemble_bagging_fraction)))
        args.lgb_ensemble_bagging_freq = max(0, int(args.lgb_ensemble_bagging_freq))
        args.lgb_threshold_dedup_eps = max(0.0, float(args.lgb_threshold_dedup_eps))
        args.lgb_min_data_in_leaf = max(
            2,
            int(getattr(args, "lgb_min_data_in_leaf", getattr(args, "min_child_size", 2))),
        )
        args.lgb_lambda_l2 = max(0.0, float(getattr(args, "lgb_lambda_l2", 0.0)))

    return args


def _compute_core_budgets(args: argparse.Namespace) -> dict[str, int]:
    core_budget = max(1, int(args.parallel_trials) * int(args.threads_per_trial))
    effective_budget = max(1, int(math.floor(core_budget * float(args.cpu_utilization_target))))
    effective_parallel_trials = max(1, effective_budget // max(1, int(args.threads_per_trial)))
    return {
        "core_budget": int(core_budget),
        "effective_budget": int(effective_budget),
        "effective_parallel_trials": int(effective_parallel_trials),
    }


def _parse_cpu_affinity_spec(spec: str) -> list[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            if len(bounds) != 2:
                raise ValueError(f"Invalid CPU affinity range token: {token!r}")
            start = int(bounds[0])
            end = int(bounds[1])
            if start < 0 or end < 0:
                raise ValueError(f"CPU ids must be >= 0, got: {token!r}")
            if end < start:
                raise ValueError(f"CPU affinity range end < start: {token!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"CPU ids must be >= 0, got: {token!r}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError(f"Empty CPU affinity spec: {spec!r}")
    return sorted(cpus)


def _apply_process_runtime_controls(args: argparse.Namespace, log_fp) -> None:
    args.effective_cpu_affinity = []
    args.effective_cpu_nice = int(os.nice(0))

    affinity_spec = str(getattr(args, "cpu_affinity", "")).strip()
    if affinity_spec:
        requested = _parse_cpu_affinity_spec(affinity_spec)
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("--cpu-affinity is not supported on this platform.")
        os.sched_setaffinity(0, set(requested))
        args.effective_cpu_affinity = [int(cpu) for cpu in sorted(os.sched_getaffinity(0))]
        _log(
            f"cpu affinity applied: requested={requested}, effective={args.effective_cpu_affinity}",
            log_fp,
        )

    nice_delta = int(getattr(args, "cpu_nice", 0))
    if nice_delta != 0:
        try:
            os.nice(nice_delta)
        except OSError as exc:
            _log(f"cpu nice apply failed (delta={nice_delta}): {exc}", log_fp)
            raise
        args.effective_cpu_nice = int(os.nice(0))
        _log(
            f"cpu nice applied: delta={nice_delta}, effective_nice={args.effective_cpu_nice}",
            log_fp,
        )


def _compute_optuna_effective_parallel_trials(
    args: argparse.Namespace,
    pipeline: PipelineSpec,
) -> tuple[int, str]:
    base_slots = max(1, int(args.effective_parallel_trials))
    manual_cap = max(0, int(getattr(args, "optuna_max_concurrent_trials", 0)))
    if manual_cap > 0:
        return max(1, min(base_slots, manual_cap)), "manual_cap"

    # Memory guard for high-trial LightGBM tuning while preserving high throughput.
    if pipeline.binning_backend == "lightgbm" and int(args.optuna_trials) >= 20:
        auto_cap = max(1, int(math.floor(base_slots * 0.75)))
        return max(1, min(base_slots, auto_cap)), "auto_lightgbm_high_trials"

    return base_slots, "none"


def _set_thread_caps(threads: int) -> None:
    value = str(max(1, int(threads)))
    for env_name in THREAD_ENV_VARS:
        os.environ[env_name] = value


def _should_gate_lightgbm_gpu(args: argparse.Namespace, pipeline: PipelineSpec) -> bool:
    if pipeline.binning_backend != "lightgbm":
        return False
    if fcntl is None:
        return False
    device = str(getattr(args, "lgb_device_type", "cpu")).strip().lower()
    return device in {"gpu", "cuda"}


@contextlib.contextmanager
def _lightgbm_gpu_gate(args: argparse.Namespace, pipeline: PipelineSpec):
    if not _should_gate_lightgbm_gpu(args, pipeline):
        yield
        return

    max_jobs = max(1, int(getattr(args, "lgb_max_gpu_jobs", 1)))
    lock_dir = Path(str(getattr(args, "lgb_gpu_lock_dir", "/tmp/msdt_lgb_gpu_lock"))).resolve()
    lock_dir.mkdir(parents=True, exist_ok=True)
    slot_paths = [lock_dir / f"slot_{idx}.lock" for idx in range(max_jobs)]
    sleep_sec = 0.05
    lock_fd: int | None = None

    while lock_fd is None:
        for slot_path in slot_paths:
            fd = os.open(str(slot_path), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[union-attr]
                lock_fd = fd
                break
            except BlockingIOError:
                os.close(fd)
        if lock_fd is None:
            time.sleep(sleep_sec)

    try:
        yield
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)  # type: ignore[union-attr]
            finally:
                os.close(lock_fd)


def _worker_init(
    dataset_payload: dict[str, dict[str, Any]],
    args_dict: dict[str, Any],
    threads_per_trial: int,
    tuned_params: dict[tuple[str, int], dict[str, Any]] | None,
    pipeline_name: str,
) -> None:
    global WORKER_DATASET_PAYLOAD, WORKER_ARGS, WORKER_TUNED_PARAMS, WORKER_PIPELINE
    global WORKER_PREPROCESSED_CACHE, WORKER_BINNER_CACHE
    WORKER_DATASET_PAYLOAD = dataset_payload
    WORKER_ARGS = argparse.Namespace(**args_dict)
    WORKER_TUNED_PARAMS = tuned_params or {}
    WORKER_PIPELINE = _get_pipeline(pipeline_name)
    WORKER_PREPROCESSED_CACHE = {}
    WORKER_BINNER_CACHE = {}
    _set_thread_caps(threads_per_trial)


def _slice_rows(x, idx: np.ndarray):
    """Row slicing helper that works for numpy arrays and pandas objects."""
    if hasattr(x, "iloc"):
        return x.iloc[idx]
    return x[idx]


def _nearest_choice(value: float | int, choices: list[int] | list[float]) -> float:
    if not choices:
        raise ValueError("choices must be non-empty")
    return min(choices, key=lambda candidate: abs(float(candidate) - float(value)))


def _derive_min_support_from_leaf_frac(leaf_frac: float, n_fit: int) -> int:
    n_fit_i = max(1, int(n_fit))
    frac = float(leaf_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"leaf_frac must be positive and finite, got {leaf_frac!r}")
    return max(2, int(math.ceil(frac * n_fit_i)))


def _leaf_frac_from_legacy_m(m: int, n_fit: int) -> float:
    n_fit_i = max(1, int(n_fit))
    return max(1.0 / float(n_fit_i), min(1.0, float(max(2, int(m))) / float(n_fit_i)))


def _lgb_min_data_in_leaf_choices(min_child_size: int) -> list[int]:
    base = max(2, int(min_child_size))
    return sorted({max(2, base // 2), base, base * 2, base * 4})


def _lgb_min_data_in_leaf_grid(min_child_size: int) -> list[int]:
    base = max(2, int(min_child_size))
    return [max(2, base // 2), base, base * 2, base * 4]


def _prepare_eval_data(
    X,
    y_bin: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    eval_mode: str,
    dataset_name: str | None,
) -> dict[str, Any]:
    global WORKER_PREPROCESSED_CACHE

    cache_key: tuple[str, int] | None = None
    if eval_mode == "test" and dataset_name:
        cache_key = (str(dataset_name), int(seed))
        cached = WORKER_PREPROCESSED_CACHE.get(cache_key)
        if cached is not None:
            return cached

    all_idx = np.arange(y_bin.shape[0], dtype=np.int32)
    idx_train_all, idx_test = train_test_split(
        all_idx,
        test_size=FIXED_TEST_FRACTION,
        random_state=seed,
        stratify=y_bin,
    )
    y_train_all = y_bin[idx_train_all]
    idx_fit, idx_val = train_test_split(
        idx_train_all,
        test_size=FIXED_VAL_WITHIN_TRAIN,
        random_state=seed,
        stratify=y_train_all,
    )

    if eval_mode == "val":
        idx_fit_use, idx_eval_use = idx_fit, idx_val
    elif eval_mode == "test":
        idx_fit_use, idx_eval_use = idx_fit, idx_test
    else:
        raise ValueError(f"unsupported eval_mode={eval_mode!r}")

    X_fit = _slice_rows(X, idx_fit_use)
    X_val = _slice_rows(X, idx_val)
    X_eval = _slice_rows(X, idx_eval_use)
    y_fit = y_bin[idx_fit_use]
    y_val = y_bin[idx_val]
    y_eval = y_bin[idx_eval_use]

    preprocessor = make_preprocessor(X_fit)
    X_fit_proc = np.ascontiguousarray(preprocessor.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.ascontiguousarray(preprocessor.transform(X_val), dtype=np.float32)
    X_eval_proc = np.ascontiguousarray(preprocessor.transform(X_eval), dtype=np.float32)
    y_fit = np.ascontiguousarray(y_fit, dtype=np.int32)
    y_val = np.ascontiguousarray(y_val, dtype=np.int32)
    y_eval = np.ascontiguousarray(y_eval, dtype=np.int32)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"x{i}" for i in range(X_fit_proc.shape[1])]

    payload = {
        "X_fit_proc": X_fit_proc,
        "X_eval_proc": X_eval_proc,
        "X_binner_val_proc": X_val_proc,
        "y_fit": y_fit,
        "y_eval": y_eval,
        "y_binner_val": y_val,
        "feature_names": feature_names,
    }
    if cache_key is not None:
        WORKER_PREPROCESSED_CACHE[cache_key] = payload
    return payload


def _fit_binner(
    X_fit_proc: np.ndarray,
    y_fit: np.ndarray,
    X_binner_val_proc: np.ndarray,
    y_binner_val: np.ndarray,
    trial_params: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    pipeline: PipelineSpec,
):
    from lightgbm_binning import fit_lightgbm_binner

    with _lightgbm_gpu_gate(args=args, pipeline=pipeline):
        return fit_lightgbm_binner(
            X_fit_proc,
            y_fit,
            X_val=X_binner_val_proc,
            y_val=y_binner_val,
            max_bins=trial_params["max_bins"],
            min_samples_leaf=trial_params["min_samples_leaf"],
            random_state=seed,
            n_estimators=int(trial_params.get("lgb_n_estimators", 10000)),
            num_leaves=int(trial_params.get("lgb_num_leaves", args.lgb_num_leaves)),
            learning_rate=float(trial_params.get("lgb_learning_rate", args.lgb_learning_rate)),
            feature_fraction=float(trial_params.get("lgb_feature_fraction", args.lgb_feature_fraction)),
            bagging_fraction=float(trial_params.get("lgb_bagging_fraction", args.lgb_bagging_fraction)),
            bagging_freq=int(trial_params.get("lgb_bagging_freq", args.lgb_bagging_freq)),
            max_depth=int(trial_params.get("lgb_max_depth", -1)),
            min_data_in_bin=int(trial_params.get("lgb_min_data_in_bin", args.lgb_min_data_in_bin)),
            min_data_in_leaf=int(trial_params.get("lgb_min_data_in_leaf", args.lgb_min_data_in_leaf)),
            lambda_l2=float(trial_params.get("lgb_lambda_l2", args.lgb_lambda_l2)),
            early_stopping_rounds=100,
            num_threads=args.lgb_num_threads,
            device_type=args.lgb_device_type,
            gpu_platform_id=args.lgb_gpu_platform_id,
            gpu_device_id=args.lgb_gpu_device_id,
            gpu_fallback_to_cpu=args.lgb_gpu_fallback,
            ensemble_runs=args.lgb_ensemble_runs,
            ensemble_feature_fraction=args.lgb_ensemble_feature_fraction,
            ensemble_bagging_fraction=args.lgb_ensemble_bagging_fraction,
            ensemble_bagging_freq=args.lgb_ensemble_bagging_freq,
            threshold_dedup_eps=args.lgb_threshold_dedup_eps,
            collect_teacher_logit=True,
        )


def _binner_cache_key(
    dataset_name: str | None,
    seed: int,
    trial_params: dict[str, Any],
    args: argparse.Namespace,
    pipeline: PipelineSpec,
) -> tuple[Any, ...] | None:
    if dataset_name is None or pipeline.binning_backend != "lightgbm":
        return None
    return (
        str(dataset_name),
        int(seed),
        int(trial_params["max_bins"]),
        int(trial_params["min_samples_leaf"]),
        int(trial_params.get("lgb_n_estimators", 10000)),
        int(trial_params.get("lgb_num_leaves", getattr(args, "lgb_num_leaves", 31))),
        float(trial_params.get("lgb_learning_rate", getattr(args, "lgb_learning_rate", 0.05))),
        float(trial_params.get("lgb_feature_fraction", getattr(args, "lgb_feature_fraction", 1.0))),
        float(trial_params.get("lgb_bagging_fraction", getattr(args, "lgb_bagging_fraction", 1.0))),
        int(trial_params.get("lgb_bagging_freq", getattr(args, "lgb_bagging_freq", 0))),
        int(trial_params.get("lgb_min_data_in_bin", getattr(args, "lgb_min_data_in_bin", 1))),
        int(trial_params.get("lgb_min_data_in_leaf", getattr(args, "lgb_min_data_in_leaf", 2))),
        float(trial_params.get("lgb_lambda_l2", getattr(args, "lgb_lambda_l2", 0.0))),
        str(getattr(args, "lgb_device_type", "cpu")),
        int(getattr(args, "lgb_ensemble_runs", 1)),
        float(getattr(args, "lgb_ensemble_feature_fraction", 1.0)),
        float(getattr(args, "lgb_ensemble_bagging_fraction", 1.0)),
        int(getattr(args, "lgb_ensemble_bagging_freq", 0)),
        float(getattr(args, "lgb_threshold_dedup_eps", 1e-9)),
    )


def _fit_or_get_binner(
    *,
    dataset_name: str | None,
    X_fit_proc: np.ndarray,
    y_fit: np.ndarray,
    X_binner_val_proc: np.ndarray,
    y_binner_val: np.ndarray,
    trial_params: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
    pipeline: PipelineSpec,
):
    global WORKER_BINNER_CACHE
    cache_key = _binner_cache_key(dataset_name, seed, trial_params, args, pipeline)
    if cache_key is not None:
        cached = WORKER_BINNER_CACHE.get(cache_key)
        if cached is not None:
            return cached
    binner = _fit_binner(
        X_fit_proc,
        y_fit,
        X_binner_val_proc,
        y_binner_val,
        trial_params,
        seed=seed,
        args=args,
        pipeline=pipeline,
    )
    if cache_key is not None:
        WORKER_BINNER_CACHE[cache_key] = binner
    return binner


def _teacher_kwargs_from_binner(binner: Any) -> dict[str, Any]:
    return {
        "teacher_logit": getattr(binner, "teacher_train_logit", None),
        "teacher_boundary_gain": getattr(binner, "boundary_gain_per_feature", None),
        "teacher_boundary_cover": getattr(binner, "boundary_cover_per_feature", None),
        "teacher_boundary_value_jump": getattr(binner, "boundary_value_jump_per_feature", None),
    }


def _run_single_trial_task(task: tuple[str, int, int, float]) -> dict[str, Any]:
    dataset_name, depth_budget, seed, class_prevalence = task
    if WORKER_ARGS is None or WORKER_DATASET_PAYLOAD is None or WORKER_PIPELINE is None:
        raise RuntimeError("worker not initialized")

    lookahead = min(WORKER_ARGS.lookahead_cap, depth_budget - 1)
    row = {
        "dataset": dataset_name,
        "depth_budget": int(depth_budget),
        "lookahead_depth_budget": int(lookahead),
        "seed": int(seed),
        "status": "ok",
        "error": "",
        "class_prevalence": float(class_prevalence),
        "train_accuracy": np.nan,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "trivial_accuracy": np.nan,
        "fit_time_sec": np.nan,
        "objective": np.nan,
        "n_leaves": np.nan,
        "n_internal": np.nan,
        "max_arity": np.nan,
        "exact_internal_nodes": np.nan,
        "greedy_internal_nodes": np.nan,
        "dp_subproblem_calls": np.nan,
        "dp_cache_hits": np.nan,
        "dp_unique_states": np.nan,
        "dp_cache_profile_enabled": np.nan,
        "dp_cache_lookup_calls": np.nan,
        "dp_cache_miss_no_bucket": np.nan,
        "dp_cache_miss_bucket_present": np.nan,
        "dp_cache_miss_depth_mismatch_only": np.nan,
        "dp_cache_miss_indices_mismatch": np.nan,
        "dp_cache_depth_match_candidates": np.nan,
        "dp_cache_bucket_entries_scanned": np.nan,
        "dp_cache_bucket_max_size": np.nan,
        "greedy_subproblem_calls": np.nan,
        "greedy_cache_hits": np.nan,
        "greedy_unique_states": np.nan,
        "greedy_cache_entries_peak": np.nan,
        "greedy_cache_clears": np.nan,
        "rush_total_time_sec": np.nan,
        "rush_refinement_child_time_sec": np.nan,
        "rush_refinement_child_time_fraction": np.nan,
        "rush_refinement_child_calls": np.nan,
        "rush_refinement_recursive_calls": np.nan,
        "rush_refinement_recursive_unique_states": np.nan,
        "rush_refinement_depth_logs": "",
        "interval_refinements_attempted": np.nan,
        "debr_refine_calls": np.nan,
        "debr_refine_improved": np.nan,
        "debr_total_moves": np.nan,
        "debr_bridge_policy_calls": np.nan,
        "debr_descent_moves": np.nan,
        "debr_bridge_moves": np.nan,
        "debr_simplify_moves": np.nan,
        "debr_total_hard_gain": np.nan,
        "debr_total_soft_gain": np.nan,
        "debr_total_delta_j": np.nan,
        "debr_total_component_delta": np.nan,
        "debr_final_geo_wins": np.nan,
        "debr_final_block_wins": np.nan,
        "expensive_child_calls": np.nan,
        "expensive_child_sec": np.nan,
        "expensive_child_exactify_calls": np.nan,
        "expensive_child_exactify_sec": np.nan,
        "fast100_exactify_nodes_allowed": np.nan,
        "fast100_exactify_nodes_skipped_small_support": np.nan,
        "fast100_exactify_nodes_skipped_dominant_gain": np.nan,
        "depth1_skipped_by_low_global_ambiguity": np.nan,
        "depth1_skipped_by_large_gap": np.nan,
        "depth1_exactify_challenger_nodes": np.nan,
        "depth1_exactified_nodes": np.nan,
        "depth1_exactified_features_mean": np.nan,
        "depth1_exactified_features_max": np.nan,
        "depth1_teacher_replaced_runnerup": np.nan,
        "depth1_teacher_rejected_by_uhat_gate": np.nan,
        "depth1_exactify_set_size_mean": np.nan,
        "depth1_exactify_set_size_max": np.nan,
        "fast100_skipped_by_ub_lb_separation": np.nan,
        "fast100_widen_forbidden_depth_gt0_attempts": np.nan,
        "fast100_frontier_size_mean": np.nan,
        "fast100_frontier_size_max": np.nan,
        "fast100_stopped_midloop_separation": np.nan,
        "fast100_M_depth0_mean": np.nan,
        "fast100_M_depth0_max": np.nan,
        "fast100_M_depth1_mean": np.nan,
        "fast100_M_depth1_max": np.nan,
        "fast100_cf_exactify_nodes_depth0": np.nan,
        "fast100_cf_exactify_nodes_depth1": np.nan,
        "fast100_cf_skipped_agreement": np.nan,
        "fast100_cf_skipped_small_regret": np.nan,
        "fast100_cf_skipped_low_impact": np.nan,
        "fast100_cf_frontier_size_mean": np.nan,
        "fast100_cf_frontier_size_max": np.nan,
        "fast100_cf_exactified_features_mean": np.nan,
        "fast100_cf_exactified_features_max": np.nan,
        "rootsafe_exactified_features": np.nan,
        "rootsafe_root_winner_changed_vs_proxy": np.nan,
        "rootsafe_root_candidates_K": np.nan,
        "fast100_used_lgb_prior_tiebreak": np.nan,
        "gini_dp_calls_root": np.nan,
        "gini_dp_calls_depth1": np.nan,
        "gini_teacher_chosen_depth1": np.nan,
        "gini_tiebreak_used_in_shortlist": np.nan,
        "gini_dp_sec": np.nan,
        "gini_root_k0": np.nan,
        "gini_endpoints_added_root": np.nan,
        "gini_endpoints_added_depth1": np.nan,
        "gini_endpoints_features_touched_root": np.nan,
        "gini_endpoints_features_touched_depth1": np.nan,
        "gini_endpoints_added_per_feature_max": np.nan,
        "gini_endpoint_sec": np.nan,
        "used_max_bins": np.nan,
        "used_min_samples_leaf": np.nan,
        "used_min_child_size": np.nan,
        "used_max_branching": np.nan,
        "used_reg": np.nan,
        "msplit_variant": str(getattr(WORKER_ARGS, "msplit_variant", "optimal_dp")),
        "tree_artifact_path": "",
    }

    try:
        if dataset_name not in WORKER_DATASET_PAYLOAD:
            raise RuntimeError(f"missing dataset payload for '{dataset_name}'")
        payload = WORKER_DATASET_PAYLOAD[dataset_name]
        tuned = (WORKER_TUNED_PARAMS or {}).get((dataset_name, int(depth_budget)))
        metrics = _run_single_trial(
            X=payload["X"],
            y_bin=payload["y_bin"],
            dataset_name=dataset_name,
            class_labels=payload.get("class_labels"),
            target_name=payload.get("target_name", "target"),
            depth_budget=depth_budget,
            seed=seed,
            args=WORKER_ARGS,
            pipeline=WORKER_PIPELINE,
            param_overrides=tuned,
            eval_mode="test",
        )
        row.update(metrics)
    except (TimeoutError, RuntimeError) as exc:
        err_text = str(exc)
        if isinstance(exc, RuntimeError) and "time_limit" not in err_text.lower():
            row["status"] = "error"
            row["error"] = err_text
        else:
            row["status"] = "timeout"
            row["error"] = err_text
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
    return row


def _run_trials(
    dataset_payload: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    pending_trials: list[tuple[str, int, int, float]],
    log_fp,
    tuned_params: dict[tuple[str, int], dict[str, Any]] | None,
    pipeline: PipelineSpec,
) -> list[dict[str, Any]]:
    if not pending_trials:
        return []

    if args.effective_parallel_trials <= 1 or len(pending_trials) == 1:
        _worker_init(dataset_payload, vars(args).copy(), args.threads_per_trial, tuned_params, pipeline.name)
        return [_run_single_trial_task(task) for task in pending_trials]

    workers = min(args.effective_parallel_trials, len(pending_trials))
    worker_log = (
        f"executing {len(pending_trials)} pending trials across {len(dataset_payload)} dataset(s) with "
        f"parallel_workers={workers}, threads_per_trial={args.threads_per_trial}, "
    )
    if pipeline.binning_backend == "lightgbm":
        worker_log += f"lgb_threads={args.lgb_num_threads}, "
    worker_log += f"effective_core_budget={args.effective_core_budget}/{args.core_budget}"
    _log(worker_log, log_fp)

    try:
        mp_context = multiprocessing.get_context("fork")
    except ValueError:
        mp_context = multiprocessing.get_context()

    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        initializer=_worker_init,
        initargs=(dataset_payload, vars(args).copy(), args.threads_per_trial, tuned_params, pipeline.name),
    ) as executor:
        future_to_task = {executor.submit(_run_single_trial_task, task): task for task in pending_trials}
        for future in concurrent.futures.as_completed(future_to_task):
            row = future.result()
            results.append(row)
            if args.fail_fast and row.get("status") == "error":
                for pending in future_to_task:
                    pending.cancel()
                raise RuntimeError(
                    "Fail-fast triggered by trial error: "
                    f"dataset={row.get('dataset')}, depth={row.get('depth_budget')}, seed={row.get('seed')}, "
                    f"error={row.get('error')}"
                )
    return results


def _run_single_trial(
    X,
    y_bin: np.ndarray,
    dataset_name: str | None,
    class_labels: np.ndarray | None,
    target_name: str | None,
    depth_budget: int,
    seed: int,
    args: argparse.Namespace,
    pipeline: PipelineSpec,
    param_overrides: dict[str, Any] | None,
    eval_mode: str,
):
    split_payload = _prepare_eval_data(X, y_bin, seed=seed, args=args, eval_mode=eval_mode, dataset_name=dataset_name)
    X_fit_proc = split_payload["X_fit_proc"]
    X_eval_proc = split_payload["X_eval_proc"]
    X_binner_val_proc = split_payload["X_binner_val_proc"]
    y_fit = split_payload["y_fit"]
    y_eval = split_payload["y_eval"]
    y_binner_val = split_payload["y_binner_val"]
    feature_names = split_payload["feature_names"]
    trial_params = _resolve_trial_params(args, depth_budget, param_overrides, n_fit=int(y_fit.shape[0]))

    binner = _fit_or_get_binner(
        dataset_name=dataset_name,
        X_fit_proc=X_fit_proc,
        y_fit=y_fit,
        X_binner_val_proc=X_binner_val_proc,
        y_binner_val=y_binner_val,
        trial_params=trial_params,
        seed=seed,
        args=args,
        pipeline=pipeline,
    )
    Z_fit = binner.transform(X_fit_proc)
    Z_eval = binner.transform(X_eval_proc)

    if pipeline.solver == "split":
        return _run_binary_split_trial(
            Z_fit=Z_fit,
            Z_eval=Z_eval,
            y_fit=y_fit,
            y_eval=y_eval,
            feature_names=feature_names,
            depth_budget=depth_budget,
            seed=seed,
            trial_params=trial_params,
            args=args,
            eval_mode=eval_mode,
        )

    model, msplit_variant = _build_msplit_model(
        trial_params=trial_params,
        depth_budget=depth_budget,
        seed=seed,
        args=args,
    )

    start = time.time()
    model.fit(Z_fit, y_fit, **_teacher_kwargs_from_binner(binner))
    fit_time = time.time() - start
    objective = _extract_objective(model)

    y_pred_train = model.predict(Z_fit).astype(np.int32)
    train_accuracy = float(np.mean(y_pred_train == y_fit))
    y_pred = model.predict(Z_eval).astype(np.int32)
    accuracy = float(np.mean(y_pred == y_eval))
    balanced_acc = float(balanced_accuracy_score(y_eval, y_pred))
    if eval_mode == "val":
        return {
            "train_accuracy": train_accuracy,
            "objective_accuracy": accuracy,
            "objective_balanced_accuracy": balanced_acc,
            "fit_time_sec": fit_time,
        }

    trivial_acc = float(max(np.mean(y_eval == 0), np.mean(y_eval == 1)))

    n_leaves = 0
    n_internal = 0
    max_arity = 0
    stack = [model.tree_]
    while stack:
        node = stack.pop()
        if hasattr(node, "children"):
            n_internal += 1
            arity = int(getattr(node, "group_count", len(node.children)))
            max_arity = max(max_arity, arity)
            for child in node.children.values():
                stack.append(child)
        else:
            n_leaves += 1

    artifact_path_text = ""
    artifact_dir_raw = str(getattr(args, "tree_artifacts_dir", "") or "").strip()
    if artifact_dir_raw:
        artifact_dir = Path(artifact_dir_raw)
        ds_name = str(dataset_name or "dataset")
        cls = np.asarray(class_labels if class_labels is not None else [0, 1], dtype=object)
        payload = build_msplit_artifact(
            dataset=ds_name,
            pipeline=pipeline.artifact_pipeline_name,
            target_name=str(target_name or "target"),
            class_labels=cls,
            feature_names=feature_names,
            accuracy=accuracy,
            seed=int(seed),
            test_size=float(args.test_size),
            depth_budget=int(depth_budget),
            lookahead=int(trial_params["lookahead_depth_budget"]),
            time_limit=float(args.time_limit),
            max_bins=int(trial_params["max_bins"]),
            min_samples_leaf=int(trial_params["min_samples_leaf"]),
            min_child_size=int(trial_params["min_child_size"]),
            max_branching=int(trial_params["max_branching"]),
            reg=float(trial_params["reg"]),
            msplit_variant=str(msplit_variant),
            tree_root=model.tree_,
            binner=binner,
            z_train=Z_fit,
        )
        artifact_path = artifact_dir / ds_name / f"depth_{int(depth_budget)}" / f"seed_{int(seed)}.json"
        write_artifact_json(artifact_path, payload)
        artifact_path_text = str(artifact_path)

    return {
        "lookahead_depth_budget": int(trial_params["lookahead_depth_budget"]),
        "train_accuracy": train_accuracy,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "trivial_accuracy": trivial_acc,
        "fit_time_sec": fit_time,
        "objective": objective,
        "n_leaves": int(n_leaves),
        "n_internal": int(n_internal),
        "max_arity": int(max_arity),
        "exact_internal_nodes": int(getattr(model, "exact_internal_nodes_", 0)),
        "greedy_internal_nodes": int(getattr(model, "greedy_internal_nodes_", 0)),
        "dp_subproblem_calls": _optional_int_attr(model, "dp_subproblem_calls_"),
        "dp_cache_hits": _optional_int_attr(model, "dp_cache_hits_"),
        "dp_unique_states": _optional_int_attr(model, "dp_unique_states_"),
        "dp_cache_profile_enabled": _optional_int_attr(model, "dp_cache_profile_enabled_"),
        "dp_cache_lookup_calls": _optional_int_attr(model, "dp_cache_lookup_calls_"),
        "dp_cache_miss_no_bucket": _optional_int_attr(model, "dp_cache_miss_no_bucket_"),
        "dp_cache_miss_bucket_present": _optional_int_attr(model, "dp_cache_miss_bucket_present_"),
        "dp_cache_miss_depth_mismatch_only": _optional_int_attr(model, "dp_cache_miss_depth_mismatch_only_"),
        "dp_cache_miss_indices_mismatch": _optional_int_attr(model, "dp_cache_miss_indices_mismatch_"),
        "dp_cache_depth_match_candidates": _optional_int_attr(model, "dp_cache_depth_match_candidates_"),
        "dp_cache_bucket_entries_scanned": _optional_int_attr(model, "dp_cache_bucket_entries_scanned_"),
        "dp_cache_bucket_max_size": _optional_int_attr(model, "dp_cache_bucket_max_size_"),
        "greedy_subproblem_calls": _optional_int_attr(model, "greedy_subproblem_calls_"),
        "profiling_greedy_complete_calls_by_depth": _optional_json_attr(
            model, "profiling_greedy_complete_calls_by_depth_"
        ),
        "greedy_cache_hits": _optional_int_attr(model, "greedy_cache_hits_"),
        "greedy_unique_states": _optional_int_attr(model, "greedy_unique_states_"),
        "greedy_cache_entries_peak": _optional_int_attr(model, "greedy_cache_entries_peak_"),
        "greedy_cache_bytes_peak": _optional_int_attr(model, "greedy_cache_bytes_peak_"),
        "native_n_classes": _optional_int_attr(model, "native_n_classes_"),
        "native_teacher_class_count": _optional_int_attr(model, "native_teacher_class_count_"),
        "native_binary_mode": _optional_int_attr(model, "native_binary_mode_"),
        "atomized_features_prepared": _optional_int_attr(model, "atomized_features_prepared_"),
        "atomized_coarse_candidates": _optional_int_attr(model, "atomized_coarse_candidates_"),
        "atomized_coarse_pruned_candidates": _optional_int_attr(
            model, "atomized_coarse_pruned_candidates_"
        ),
        "atomized_coarse_prune_rate": (
            float(getattr(model, "atomized_coarse_pruned_candidates_", 0))
            / float(getattr(model, "atomized_coarse_candidates_", 0))
            if float(getattr(model, "atomized_coarse_candidates_", 0)) > 0.0
            else np.nan
        ),
        "atomized_coarse_survivor_rate": (
            1.0
            - (
                float(getattr(model, "atomized_coarse_pruned_candidates_", 0))
                / float(getattr(model, "atomized_coarse_candidates_", 0))
            )
            if float(getattr(model, "atomized_coarse_candidates_", 0)) > 0.0
            else np.nan
        ),
        "atomized_final_candidates": _optional_int_attr(model, "atomized_final_candidates_"),
        "greedy_feature_survivor_histogram": _optional_json_attr(
            model, "greedy_feature_survivor_histogram_"
        ),
        "nominee_unique_total": _optional_int_attr(model, "nominee_unique_total_"),
        "nominee_child_interval_lookups": _optional_int_attr(model, "nominee_child_interval_lookups_"),
        "nominee_child_interval_unique": _optional_int_attr(model, "nominee_child_interval_unique_"),
        "nominee_exactified_total": _optional_int_attr(model, "nominee_exactified_total_"),
        "nominee_incumbent_updates": _optional_int_attr(model, "nominee_incumbent_updates_"),
        "nominee_threatening_samples": _optional_int_attr(model, "nominee_threatening_samples_"),
        "nominee_threatening_sum": _optional_float_attr(model, "nominee_threatening_sum_"),
        "nominee_threatening_max": _optional_int_attr(model, "nominee_threatening_max_"),
        "nominee_exact_child_eval_sec": _optional_float_attr(
            model, "profiling_recursive_child_eval_sec_"
        ),
        "nominee_debr_sec": _optional_float_attr(model, "profiling_refine_sec_"),
        "debr_refine_calls": _optional_int_attr(model, "debr_refine_calls_"),
        "debr_refine_improved": _optional_int_attr(model, "debr_refine_improved_"),
        "debr_total_moves": _optional_int_attr(model, "debr_total_moves_"),
        "debr_bridge_policy_calls": _optional_int_attr(model, "debr_bridge_policy_calls_"),
        "debr_descent_moves": _optional_int_attr(model, "debr_descent_moves_"),
        "debr_bridge_moves": _optional_int_attr(model, "debr_bridge_moves_"),
        "debr_simplify_moves": _optional_int_attr(model, "debr_simplify_moves_"),
        "debr_total_hard_gain": _optional_float_attr(model, "debr_total_hard_gain_"),
        "debr_total_soft_gain": _optional_float_attr(model, "debr_total_soft_gain_"),
        "debr_total_delta_j": _optional_float_attr(model, "debr_total_delta_j_"),
        "debr_total_component_delta": _optional_int_attr(model, "debr_total_component_delta_"),
        "debr_final_geo_wins": _optional_int_attr(model, "debr_final_geo_wins_"),
        "debr_final_block_wins": _optional_int_attr(model, "debr_final_block_wins_"),
        "greedy_cache_clears": _optional_int_attr(model, "greedy_cache_clears_"),
        "rush_total_time_sec": _optional_float_attr(model, "rush_total_time_sec_"),
        "rush_refinement_child_time_sec": _optional_float_attr(model, "rush_refinement_child_time_sec_"),
        "rush_refinement_child_time_fraction": _optional_float_attr(model, "rush_refinement_child_time_fraction_"),
        "rush_refinement_child_calls": _optional_int_attr(model, "rush_refinement_child_calls_"),
        "rush_refinement_recursive_calls": _optional_int_attr(model, "rush_refinement_recursive_calls_"),
        "rush_refinement_recursive_unique_states": _optional_int_attr(
            model, "rush_refinement_recursive_unique_states_"
        ),
        "rush_refinement_depth_logs": _optional_json_attr(model, "rush_refinement_depth_logs_"),
        "interval_refinements_attempted": _optional_int_attr(model, "interval_refinements_attempted_"),
        "expensive_child_calls": _optional_int_attr(model, "expensive_child_calls_"),
        "expensive_child_sec": _optional_float_attr(model, "expensive_child_sec_"),
        "expensive_child_exactify_calls": _optional_int_attr(model, "expensive_child_exactify_calls_"),
        "expensive_child_exactify_sec": _optional_float_attr(model, "expensive_child_exactify_sec_"),
        "fast100_exactify_nodes_allowed": _optional_int_attr(
            model, "fast100_exactify_nodes_allowed_"
        ),
        "fast100_exactify_nodes_skipped_small_support": _optional_int_attr(
            model, "fast100_exactify_nodes_skipped_small_support_"
        ),
        "fast100_exactify_nodes_skipped_dominant_gain": _optional_int_attr(
            model, "fast100_exactify_nodes_skipped_dominant_gain_"
        ),
        "depth1_skipped_by_low_global_ambiguity": _optional_int_attr(
            model, "depth1_skipped_by_low_global_ambiguity_"
        ),
        "depth1_skipped_by_large_gap": _optional_int_attr(
            model, "depth1_skipped_by_large_gap_"
        ),
        "depth1_exactify_challenger_nodes": _optional_int_attr(
            model, "depth1_exactify_challenger_nodes_"
        ),
        "depth1_exactified_nodes": _optional_int_attr(
            model, "depth1_exactified_nodes_"
        ),
        "depth1_exactified_features_mean": _optional_float_attr(
            model, "depth1_exactified_features_mean_"
        ),
        "depth1_exactified_features_max": _optional_int_attr(
            model, "depth1_exactified_features_max_"
        ),
        "depth1_teacher_replaced_runnerup": _optional_int_attr(
            model, "depth1_teacher_replaced_runnerup_"
        ),
        "depth1_teacher_rejected_by_uhat_gate": _optional_int_attr(
            model, "depth1_teacher_rejected_by_uhat_gate_"
        ),
        "depth1_exactify_set_size_mean": _optional_float_attr(
            model, "depth1_exactify_set_size_mean_"
        ),
        "depth1_exactify_set_size_max": _optional_int_attr(
            model, "depth1_exactify_set_size_max_"
        ),
        "fast100_skipped_by_ub_lb_separation": _optional_int_attr(
            model, "fast100_skipped_by_ub_lb_separation_"
        ),
        "fast100_widen_forbidden_depth_gt0_attempts": _optional_int_attr(
            model, "fast100_widen_forbidden_depth_gt0_attempts_"
        ),
        "fast100_frontier_size_mean": _optional_float_attr(
            model, "fast100_frontier_size_mean_"
        ),
        "fast100_frontier_size_max": _optional_int_attr(
            model, "fast100_frontier_size_max_"
        ),
        "fast100_stopped_midloop_separation": _optional_int_attr(
            model, "fast100_stopped_midloop_separation_"
        ),
        "fast100_M_depth0_mean": _optional_float_attr(model, "fast100_M_depth0_mean_"),
        "fast100_M_depth0_max": _optional_int_attr(model, "fast100_M_depth0_max_"),
        "fast100_M_depth1_mean": _optional_float_attr(model, "fast100_M_depth1_mean_"),
        "fast100_M_depth1_max": _optional_int_attr(model, "fast100_M_depth1_max_"),
        "fast100_cf_exactify_nodes_depth0": _optional_int_attr(
            model, "fast100_cf_exactify_nodes_depth0_"
        ),
        "fast100_cf_exactify_nodes_depth1": _optional_int_attr(
            model, "fast100_cf_exactify_nodes_depth1_"
        ),
        "fast100_cf_skipped_agreement": _optional_int_attr(
            model, "fast100_cf_skipped_agreement_"
        ),
        "fast100_cf_skipped_small_regret": _optional_int_attr(
            model, "fast100_cf_skipped_small_regret_"
        ),
        "fast100_cf_skipped_low_impact": _optional_int_attr(
            model, "fast100_cf_skipped_low_impact_"
        ),
        "fast100_cf_frontier_size_mean": _optional_float_attr(
            model, "fast100_cf_frontier_size_mean_"
        ),
        "fast100_cf_frontier_size_max": _optional_int_attr(
            model, "fast100_cf_frontier_size_max_"
        ),
        "fast100_cf_exactified_features_mean": _optional_float_attr(
            model, "fast100_cf_exactified_features_mean_"
        ),
        "fast100_cf_exactified_features_max": _optional_int_attr(
            model, "fast100_cf_exactified_features_max_"
        ),
        "rootsafe_exactified_features": _optional_int_attr(
            model, "rootsafe_exactified_features_"
        ),
        "rootsafe_root_winner_changed_vs_proxy": _optional_int_attr(
            model, "rootsafe_root_winner_changed_vs_proxy_"
        ),
        "rootsafe_root_candidates_K": _optional_int_attr(
            model, "rootsafe_root_candidates_K_"
        ),
        "fast100_used_lgb_prior_tiebreak": _optional_int_attr(
            model, "fast100_used_lgb_prior_tiebreak_"
        ),
        "gini_dp_calls_root": _optional_int_attr(
            model, "gini_dp_calls_root_"
        ),
        "gini_dp_calls_depth1": _optional_int_attr(
            model, "gini_dp_calls_depth1_"
        ),
        "gini_teacher_chosen_depth1": _optional_int_attr(
            model, "gini_teacher_chosen_depth1_"
        ),
        "gini_tiebreak_used_in_shortlist": _optional_int_attr(
            model, "gini_tiebreak_used_in_shortlist_"
        ),
        "gini_dp_sec": _optional_float_attr(
            model, "gini_dp_sec_"
        ),
        "gini_root_k0": _optional_int_attr(
            model, "gini_root_k0_"
        ),
        "gini_endpoints_added_root": _optional_int_attr(
            model, "gini_endpoints_added_root_"
        ),
        "gini_endpoints_added_depth1": _optional_int_attr(
            model, "gini_endpoints_added_depth1_"
        ),
        "gini_endpoints_features_touched_root": _optional_int_attr(
            model, "gini_endpoints_features_touched_root_"
        ),
        "gini_endpoints_features_touched_depth1": _optional_int_attr(
            model, "gini_endpoints_features_touched_depth1_"
        ),
        "gini_endpoints_added_per_feature_max": _optional_int_attr(
            model, "gini_endpoints_added_per_feature_max_"
        ),
        "gini_endpoint_sec": _optional_float_attr(
            model, "gini_endpoint_sec_"
        ),
        "used_max_bins": int(trial_params["max_bins"]),
        "used_min_samples_leaf": int(trial_params["min_samples_leaf"]),
        "used_min_child_size": int(trial_params["min_child_size"]),
        "used_max_branching": int(trial_params["max_branching"]),
        "used_reg": float(trial_params["reg"]),
        "msplit_variant": str(msplit_variant),
        "tree_artifact_path": artifact_path_text,
    }


def _build_msplit_model(
    trial_params: dict[str, Any],
    depth_budget: int,
    seed: int,
    args: argparse.Namespace,
):
    msplit_variant = str(getattr(args, "msplit_variant", "rush_dp")).strip().lower()
    model = MSPLIT(
        lookahead_depth_budget=trial_params["lookahead_depth_budget"],
        full_depth_budget=depth_budget,
        reg=trial_params["reg"],
        min_child_size=trial_params["min_child_size"],
        min_split_size=trial_params["min_split_size"],
        max_branching=trial_params["max_branching"],
        time_limit=args.time_limit,
        verbose=False,
        random_state=seed,
        use_cpp_solver=True,
    )
    return model, msplit_variant


def _optional_int_attr(obj: Any, name: str) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return np.nan
    try:
        return float(int(value))
    except Exception:
        return np.nan


def _optional_float_attr(obj: Any, name: str) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return np.nan
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _optional_json_attr(obj: Any, name: str) -> str:
    value = getattr(obj, name, None)
    if value is None:
        return ""
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return ""


def _extract_objective(model: Any) -> float:
    candidates = [
        getattr(model, "objective_", None),
        getattr(getattr(model, "result_", None), "model_loss", None),
        getattr(getattr(getattr(model, "clf", None), "result_", None), "model_loss", None),
        getattr(model, "upper_bound_", None),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            out = float(value)
        except Exception:
            continue
        if np.isfinite(out):
            return out
    return np.nan


def _binarize_binned_features(
    Z_fit: np.ndarray,
    Z_eval: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = feature_names or [f"x{i}" for i in range(Z_fit.shape[1])]
    encoder = NumericBinarizer()
    encoder.fit(Z_fit, columns=columns)
    fit_bin = encoder.transform(Z_fit).astype(bool)
    eval_bin = encoder.transform(Z_eval).astype(bool)
    bin_names = encoder.get_feature_names_out()
    return (
        pd.DataFrame(fit_bin, columns=bin_names),
        pd.DataFrame(eval_bin, columns=bin_names),
    )


def _count_binary_tree(tree_dict: dict[str, Any] | None) -> tuple[int, int, int]:
    if not tree_dict:
        return 0, 0, 0
    if "prediction" in tree_dict:
        return 1, 0, 0
    left = tree_dict.get("True")
    right = tree_dict.get("False")
    leaves_l, internal_l, arity_l = _count_binary_tree(left)
    leaves_r, internal_r, arity_r = _count_binary_tree(right)
    return leaves_l + leaves_r, internal_l + internal_r + 1, max(2, arity_l, arity_r)


def _run_binary_split_trial(
    Z_fit: np.ndarray,
    Z_eval: np.ndarray,
    y_fit: np.ndarray,
    y_eval: np.ndarray,
    feature_names: list[str],
    depth_budget: int,
    seed: int,
    trial_params: dict[str, Any],
    args: argparse.Namespace,
    eval_mode: str,
) -> dict[str, Any]:
    X_fit_bin, X_eval_bin = _binarize_binned_features(Z_fit, Z_eval, feature_names)

    model = SPLIT(
        lookahead_depth_budget=trial_params["lookahead_depth_budget"],
        full_depth_budget=depth_budget,
        reg=trial_params["reg"],
        time_limit=max(1, int(math.ceil(float(args.time_limit)))),
        verbose=False,
        allow_small_reg=True,
        greedy_postprocess=False,
        binarize=False,
    )

    start = time.time()
    model.fit(X_fit_bin, y_fit)
    fit_time = time.time() - start
    objective = _extract_objective(model)

    y_pred_train = np.asarray(model.predict(X_fit_bin), dtype=np.int32)
    train_accuracy = float(np.mean(y_pred_train == y_fit))
    y_pred = np.asarray(model.predict(X_eval_bin), dtype=np.int32)
    accuracy = float(np.mean(y_pred == y_eval))
    balanced_acc = float(balanced_accuracy_score(y_eval, y_pred))
    if eval_mode == "val":
        return {
            "train_accuracy": train_accuracy,
            "objective_accuracy": accuracy,
            "objective_balanced_accuracy": balanced_acc,
            "fit_time_sec": fit_time,
        }

    trivial_acc = float(max(np.mean(y_eval == 0), np.mean(y_eval == 1)))
    tree_dict = model.tree_to_dict()
    n_leaves, n_internal, max_arity = _count_binary_tree(tree_dict)

    return {
        "lookahead_depth_budget": int(trial_params["lookahead_depth_budget"]),
        "train_accuracy": train_accuracy,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "trivial_accuracy": trivial_acc,
        "fit_time_sec": fit_time,
        "objective": objective,
        "n_leaves": int(n_leaves),
        "n_internal": int(n_internal),
        "max_arity": int(max_arity),
        "exact_internal_nodes": 0,
        "greedy_internal_nodes": 0,
        "dp_subproblem_calls": np.nan,
        "dp_cache_hits": np.nan,
        "dp_unique_states": np.nan,
        "greedy_subproblem_calls": np.nan,
        "greedy_cache_hits": np.nan,
        "greedy_unique_states": np.nan,
        "used_max_bins": int(trial_params["max_bins"]),
        "used_min_samples_leaf": int(trial_params["min_samples_leaf"]),
        "used_min_child_size": int(trial_params["min_child_size"]),
        "used_max_branching": int(trial_params["max_branching"]),
        "used_reg": float(trial_params["reg"]),
        "msplit_variant": "n/a",
        "tree_artifact_path": "",
    }


def _resolve_trial_params(
    args: argparse.Namespace,
    depth_budget: int,
    param_overrides: dict[str, Any] | None,
    n_fit: int | None = None,
) -> dict[str, Any]:
    is_lightgbm = bool(getattr(args, "uses_lightgbm_binning", False))
    lookahead_upper = max(1, int(depth_budget) - 1)
    params = {
        "lookahead_depth_budget": int(min(args.lookahead_cap, lookahead_upper)),
        "max_bins": int(args.max_bins),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_child_size": int(args.min_child_size),
        "min_split_size": int(args.min_split_size),
        "leaf_frac": float(args.leaf_frac) if getattr(args, "leaf_frac", None) is not None else None,
        "max_branching": int(args.max_branching),
        "reg": float(args.reg),
    }
    if is_lightgbm:
        params["lgb_n_estimators"] = 10000
        params["lgb_num_leaves"] = int(getattr(args, "lgb_num_leaves", 31))
        params["lgb_learning_rate"] = float(getattr(args, "lgb_learning_rate", 0.05))
        params["lgb_feature_fraction"] = float(getattr(args, "lgb_feature_fraction", 1.0))
        params["lgb_bagging_fraction"] = float(getattr(args, "lgb_bagging_fraction", 1.0))
        params["lgb_bagging_freq"] = int(getattr(args, "lgb_bagging_freq", 0))
        params["lgb_max_depth"] = -1
        params["lgb_min_data_in_bin"] = int(getattr(args, "lgb_min_data_in_bin", 1))
        params["lgb_min_data_in_leaf"] = int(
            getattr(args, "lgb_min_data_in_leaf", getattr(args, "min_child_size", 2))
        )
        params["lgb_lambda_l2"] = float(getattr(args, "lgb_lambda_l2", 0.0))
    if param_overrides:
        for key, value in param_overrides.items():
            if key == "lookahead_cap":
                params["lookahead_depth_budget"] = int(value)
            elif key in params:
                params[key] = value

    params["lookahead_depth_budget"] = max(1, min(int(params["lookahead_depth_budget"]), lookahead_upper))
    params["max_bins"] = max(2, int(params["max_bins"]))
    leaf_frac_raw = params.get("leaf_frac", None)
    if getattr(args, "leaf_frac", None) is not None:
        leaf_frac_raw = getattr(args, "leaf_frac")
    leaf_frac: float | None = None
    if leaf_frac_raw is not None:
        try:
            leaf_frac_candidate = float(leaf_frac_raw)
        except (TypeError, ValueError):
            leaf_frac_candidate = float("nan")
        if np.isfinite(leaf_frac_candidate) and 0.0 < leaf_frac_candidate <= 1.0:
            leaf_frac = leaf_frac_candidate

    requested_min_samples_leaf = int(params["min_samples_leaf"])
    requested_min_child_size = int(params["min_child_size"])
    min_samples_leaf = max(2, requested_min_samples_leaf)
    if requested_min_child_size > 0:
        min_child_size = max(2, requested_min_child_size)
    elif leaf_frac is not None and n_fit is not None and int(n_fit) > 0:
        min_child_size = _derive_min_support_from_leaf_frac(float(leaf_frac), int(n_fit))
    else:
        min_child_size = 2
    min_split_size = int(params["min_split_size"])
    if min_split_size <= 0:
        min_split_size = max(2, 2 * min_child_size)
    params["min_samples_leaf"] = int(min_samples_leaf)
    params["min_child_size"] = int(min_child_size)
    params["min_split_size"] = int(min_split_size)
    if leaf_frac is None and n_fit is not None and int(n_fit) > 0:
        params["leaf_frac"] = float(_leaf_frac_from_legacy_m(min_child_size, int(n_fit)))
    else:
        params["leaf_frac"] = float(leaf_frac) if leaf_frac is not None else None

    max_branching = int(params["max_branching"])
    if max_branching < 0:
        max_branching = 0
    if max_branching > 0:
        max_branching = min(max_branching, int(params["max_bins"]))
        max_branching = max(2, max_branching)
    params["max_branching"] = max_branching

    params["reg"] = max(0.0, float(params["reg"]))
    if is_lightgbm:
        params["lgb_n_estimators"] = 10000
        params["lgb_max_depth"] = -1
        params["lgb_num_leaves"] = max(8, int(params.get("lgb_num_leaves", getattr(args, "lgb_num_leaves", 31))))
        params["lgb_learning_rate"] = max(
            5e-3,
            min(1e-1, float(params.get("lgb_learning_rate", getattr(args, "lgb_learning_rate", 0.05)))),
        )
        params["lgb_feature_fraction"] = min(
            1.0, max(0.8, float(params.get("lgb_feature_fraction", getattr(args, "lgb_feature_fraction", 1.0))))
        )
        params["lgb_bagging_fraction"] = min(
            1.0, max(0.8, float(params.get("lgb_bagging_fraction", getattr(args, "lgb_bagging_fraction", 1.0))))
        )
        bagging_freq_choices = [0, 1, 5]
        params["lgb_bagging_freq"] = int(
            _nearest_choice(
                int(params.get("lgb_bagging_freq", getattr(args, "lgb_bagging_freq", 0))),
                bagging_freq_choices,
            )
        )
        if int(params["lgb_bagging_freq"]) == 0:
            params["lgb_bagging_fraction"] = 1.0
        min_data_choices = [1, 2, 3, 5, 8, 16]
        params["lgb_min_data_in_bin"] = int(
            _nearest_choice(
                int(params.get("lgb_min_data_in_bin", getattr(args, "lgb_min_data_in_bin", 1))),
                min_data_choices,
            )
        )
        min_data_in_leaf_choices = _lgb_min_data_in_leaf_choices(int(params["min_child_size"]))
        params["lgb_min_data_in_leaf"] = int(
            _nearest_choice(
                int(
                    params.get(
                        "lgb_min_data_in_leaf",
                        getattr(args, "lgb_min_data_in_leaf", params["min_child_size"]),
                    )
                ),
                min_data_in_leaf_choices,
            )
        )
        params["lgb_lambda_l2"] = float(
            _nearest_choice(
                float(params.get("lgb_lambda_l2", getattr(args, "lgb_lambda_l2", 0.0))),
                [0.0, 0.1],
            )
        )
    return params


def _optuna_space_limits(
    args: argparse.Namespace,
    depth_budget: int,
    n_fit: int | None = None,
    minority_count: int | None = None,
) -> dict[str, Any]:
    is_lightgbm = bool(getattr(args, "uses_lightgbm_binning", False))
    lookahead_upper = max(1, min(int(args.lookahead_cap), int(depth_budget) - 1))
    fixed_max_bins = max(2, int(args.max_bins))
    max_bins_upper = fixed_max_bins
    fixed_max_branching = max(0, int(args.max_branching))
    max_bins_lower = fixed_max_bins
    min_leaf_upper = max(2, int(args.min_samples_leaf) * 2)
    min_child_upper = max(2, int(args.min_child_size) * 2)
    if n_fit is not None and int(n_fit) > 0:
        n_fit_i = int(n_fit)
        max_reasonable = max(2, n_fit_i // 2)
        # Keep the adaptive ranges wide enough to capture larger-structure optima.
        min_leaf_upper = max(min_leaf_upper, int(math.ceil(0.25 * n_fit_i)))
        min_child_upper = max(min_child_upper, int(math.ceil(0.35 * n_fit_i)))
        if minority_count is not None and int(minority_count) > 0:
            # For strongly imbalanced datasets, allow larger minimum-node constraints.
            minority_i = int(minority_count)
            minority_frac = float(minority_i) / float(max(1, n_fit_i))
            if minority_frac < 0.2:
                imbalance_cap = int(math.ceil(0.9 * minority_i))
                min_leaf_upper = max(min_leaf_upper, imbalance_cap)
                min_child_upper = max(min_child_upper, imbalance_cap)
        min_leaf_upper = max(2, min(min_leaf_upper, max_reasonable))
        min_child_upper = max(2, min(min_child_upper, max_reasonable))
    if fixed_max_branching > 0:
        max_branch_choices = [min(max_bins_upper, max(2, fixed_max_branching))]
    else:
        max_branch_choices = [0] + list(range(2, max_bins_upper + 1))
    limits: dict[str, Any] = {
        "lookahead_upper": int(lookahead_upper),
        "max_bins_lower": int(max_bins_lower),
        "max_bins_upper": int(max_bins_upper),
        "min_leaf_upper": int(min_leaf_upper),
        "min_child_upper": int(min_child_upper),
        "leaf_frac_choices": [float(v) for v in getattr(args, "optuna_leaf_frac_grid", [0.01, 0.02, 0.05])],
        "max_branch_choices": max_branch_choices,
        "reg_min": 1e-6,
        "reg_max": 5e-2,
    }
    if is_lightgbm:
        limits.update(
            {
                "lgb_num_leaves_min": 8,
                "lgb_num_leaves_max": max(256, int(getattr(args, "lgb_num_leaves", 31))),
                "lgb_learning_rate_min": 5e-3,
                "lgb_learning_rate_max": 1e-1,
                "lgb_feature_fraction_min": 8e-1,
                "lgb_feature_fraction_max": 1.0,
                "lgb_bagging_fraction_min": 8e-1,
                "lgb_bagging_fraction_max": 1.0,
                "lgb_bagging_freq_choices": [0, 1, 5],
                "lgb_min_data_in_bin_choices": [1, 2, 3, 5, 8, 16],
                "lgb_lambda_l2_choices": [0.0, 0.1],
            }
        )
    return limits


def _sanitize_optuna_params(
    params: dict[str, Any],
    args: argparse.Namespace,
    depth_budget: int,
    n_fit: int | None = None,
    minority_count: int | None = None,
) -> dict[str, Any]:
    is_lightgbm = bool(getattr(args, "uses_lightgbm_binning", False))
    limits = _optuna_space_limits(args, depth_budget, n_fit=n_fit, minority_count=minority_count)
    lookahead = int(params.get("lookahead_cap", 1))
    max_bins = int(params.get("max_bins", int(args.max_bins)))
    min_samples_leaf_raw = int(params.get("min_samples_leaf", int(args.min_samples_leaf)))
    min_child_size_raw = int(params.get("min_child_size", int(args.min_child_size)))
    leaf_frac_raw = params.get("leaf_frac", getattr(args, "leaf_frac", None))
    if getattr(args, "leaf_frac", None) is not None:
        leaf_frac_raw = getattr(args, "leaf_frac")
    max_branching = int(params.get("max_branching", int(args.max_branching)))
    reg = float(params.get("reg", float(args.reg)))

    lookahead = max(1, min(lookahead, int(limits["lookahead_upper"])))
    max_bins = max(int(limits.get("max_bins_lower", 2)), min(max_bins, int(limits["max_bins_upper"])))
    leaf_frac: float | None = None
    if leaf_frac_raw is not None:
        try:
            leaf_frac_candidate = float(leaf_frac_raw)
        except (TypeError, ValueError):
            leaf_frac_candidate = float("nan")
        if np.isfinite(leaf_frac_candidate) and 0.0 < leaf_frac_candidate <= 1.0:
            leaf_frac = leaf_frac_candidate

    min_samples_leaf = max(2, min(int(min_samples_leaf_raw), int(limits["min_leaf_upper"])))
    if int(min_child_size_raw) > 0:
        min_child_size = max(2, min(int(min_child_size_raw), int(limits["min_child_upper"])))
    elif leaf_frac is not None and n_fit is not None and int(n_fit) > 0:
        min_child_size = max(
            2,
            min(_derive_min_support_from_leaf_frac(float(leaf_frac), int(n_fit)), int(limits["min_child_upper"])),
        )
    else:
        min_child_size = 2
    if leaf_frac is None and n_fit is not None and int(n_fit) > 0:
        leaf_frac = _leaf_frac_from_legacy_m(min_child_size, int(n_fit))

    fixed_max_branching = max(0, int(args.max_branching))
    if fixed_max_branching > 0:
        max_branching = max(2, min(fixed_max_branching, max_bins))
    elif max_branching <= 0:
        max_branching = 0
    else:
        max_branching = max(2, min(max_branching, max_bins))

    lgb_num_leaves = int(params.get("lgb_num_leaves", int(getattr(args, "lgb_num_leaves", 31))))
    lgb_learning_rate = float(params.get("lgb_learning_rate", float(getattr(args, "lgb_learning_rate", 0.05))))
    lgb_feature_fraction = float(params.get("lgb_feature_fraction", float(getattr(args, "lgb_feature_fraction", 1.0))))
    lgb_bagging_fraction = float(params.get("lgb_bagging_fraction", float(getattr(args, "lgb_bagging_fraction", 1.0))))
    lgb_bagging_freq = int(params.get("lgb_bagging_freq", int(getattr(args, "lgb_bagging_freq", 0))))
    lgb_min_data_in_bin = int(params.get("lgb_min_data_in_bin", int(getattr(args, "lgb_min_data_in_bin", 1))))
    lgb_min_data_in_leaf = int(
        params.get("lgb_min_data_in_leaf", int(getattr(args, "lgb_min_data_in_leaf", min_child_size)))
    )
    lgb_lambda_l2 = float(params.get("lgb_lambda_l2", float(getattr(args, "lgb_lambda_l2", 0.0))))

    reg = max(0.0, min(reg, float(limits["reg_max"])))

    out = {
        "lookahead_cap": int(lookahead),
        "max_bins": int(max_bins),
        "min_samples_leaf": int(min_samples_leaf),
        "min_child_size": int(min_child_size),
        "leaf_frac": float(leaf_frac) if leaf_frac is not None else None,
        "min_split_size": max(2, int(params.get("min_split_size", 0))) if int(params.get("min_split_size", 0)) > 0 else max(2, 2 * int(min_child_size)),
        "max_branching": int(max_branching),
        "reg": float(reg),
    }
    if is_lightgbm:
        lgb_num_leaves_min = int(limits.get("lgb_num_leaves_min", 8))
        lgb_num_leaves_max = int(limits.get("lgb_num_leaves_max", 256))
        lgb_bagging_freq_choices = [int(v) for v in limits.get("lgb_bagging_freq_choices", [])]
        lgb_min_data_choices = [int(v) for v in limits.get("lgb_min_data_in_bin_choices", [])]
        lgb_lambda_l2_choices = [float(v) for v in limits.get("lgb_lambda_l2_choices", [0.0, 0.1])]

        lgb_num_leaves = max(lgb_num_leaves_min, lgb_num_leaves)
        if lgb_bagging_freq_choices:
            lgb_bagging_freq = int(_nearest_choice(lgb_bagging_freq, lgb_bagging_freq_choices))
        else:
            lgb_bagging_freq = max(0, lgb_bagging_freq)
        if lgb_min_data_choices:
            lgb_min_data_in_bin = int(_nearest_choice(lgb_min_data_in_bin, lgb_min_data_choices))
        else:
            lgb_min_data_in_bin = max(1, lgb_min_data_in_bin)
        min_data_in_leaf_choices = _lgb_min_data_in_leaf_choices(min_child_size)
        lgb_min_data_in_leaf = int(_nearest_choice(lgb_min_data_in_leaf, min_data_in_leaf_choices))
        lgb_lambda_l2 = float(_nearest_choice(lgb_lambda_l2, lgb_lambda_l2_choices))

        lgb_learning_rate = max(float(limits.get("lgb_learning_rate_min", 5e-3)), lgb_learning_rate)
        lgb_learning_rate = min(float(limits.get("lgb_learning_rate_max", 1e-1)), lgb_learning_rate)
        lgb_feature_fraction = max(float(limits.get("lgb_feature_fraction_min", 0.8)), lgb_feature_fraction)
        lgb_feature_fraction = min(float(limits.get("lgb_feature_fraction_max", 1.0)), lgb_feature_fraction)
        lgb_bagging_fraction = max(float(limits.get("lgb_bagging_fraction_min", 0.8)), lgb_bagging_fraction)
        lgb_bagging_fraction = min(float(limits.get("lgb_bagging_fraction_max", 1.0)), lgb_bagging_fraction)
        if int(lgb_bagging_freq) == 0:
            lgb_bagging_fraction = 1.0

        out.update(
            {
                "lgb_n_estimators": 10000,
                "lgb_num_leaves": int(lgb_num_leaves),
                "lgb_learning_rate": float(lgb_learning_rate),
                "lgb_feature_fraction": float(lgb_feature_fraction),
                "lgb_bagging_fraction": float(lgb_bagging_fraction),
                "lgb_bagging_freq": int(lgb_bagging_freq),
                "lgb_max_depth": -1,
                "lgb_min_data_in_bin": int(lgb_min_data_in_bin),
                "lgb_min_data_in_leaf": int(lgb_min_data_in_leaf),
                "lgb_lambda_l2": float(lgb_lambda_l2),
            }
        )
    return out


def _optuna_param_signature(params: dict[str, Any]) -> tuple[Any, ...]:
    leaf_frac_raw = params.get("leaf_frac", None)
    if leaf_frac_raw is None:
        leaf_frac_sig = -1.0
    else:
        leaf_frac_sig = round(float(leaf_frac_raw), 12)
    try:
        min_leaf = int(params.get("min_samples_leaf", -1))
    except (TypeError, ValueError):
        min_leaf = -1
    try:
        min_child = int(params.get("min_child_size", -1))
    except (TypeError, ValueError):
        min_child = -1
    support_sig = int(max(min_leaf, min_child))
    return (
        int(params["lookahead_cap"]),
        int(params["max_bins"]),
        leaf_frac_sig,
        support_sig,
        int(params["max_branching"]),
        round(float(params["reg"]), 12),
        int(params.get("lgb_num_leaves", -1)),
        round(float(params.get("lgb_learning_rate", -1.0)), 12),
        round(float(params.get("lgb_feature_fraction", -1.0)), 12),
        round(float(params.get("lgb_bagging_fraction", -1.0)), 12),
        int(params.get("lgb_bagging_freq", -1)),
        int(params.get("lgb_min_data_in_bin", -1)),
        int(params.get("lgb_min_data_in_leaf", -1)),
        round(float(params.get("lgb_lambda_l2", -1.0)), 12),
    )


def _build_optuna_seed_candidates(
    args: argparse.Namespace,
    depth_budget: int,
    dataset_name: str,
) -> list[dict[str, Any]]:
    lookahead_upper = max(1, min(int(args.lookahead_cap), int(depth_budget) - 1))
    max_bins_upper = max(2, int(args.max_bins))
    max_branch_cap = max(2, min(4, max_bins_upper))

    seed_candidates: list[dict[str, Any]] = [
        {
            "lookahead_cap": 1,
            "max_bins": min(6, max_bins_upper),
            "leaf_frac": 0.02,
            "max_branching": 0,
            "reg": 3e-4,
        },
        {
            "lookahead_cap": min(2, lookahead_upper),
            "max_bins": min(5, max_bins_upper),
            "leaf_frac": 0.01,
            "max_branching": min(3, max_bins_upper),
            "reg": 8e-4,
        },
        {
            "lookahead_cap": lookahead_upper,
            "max_bins": min(4, max_bins_upper),
            "leaf_frac": 0.05,
            "max_branching": 2,
            "reg": 2e-3,
        },
    ]
    if lookahead_upper >= 3:
        seed_candidates.append(
            {
                "lookahead_cap": 3,
                "max_bins": min(6, max_bins_upper),
                "leaf_frac": 0.01,
                "max_branching": max_branch_cap,
                "reg": 1.5e-3,
            }
        )
    if str(dataset_name) == "electricity":
        seed_candidates.append(
            {
                "lookahead_cap": min(2, lookahead_upper),
                "max_bins": min(6, max_bins_upper),
                "leaf_frac": 0.01,
                "max_branching": max_branch_cap,
                "reg": 5e-4,
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in seed_candidates:
        sanitized = _sanitize_optuna_params(raw, args, depth_budget)
        sig = _optuna_param_signature(sanitized)
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(sanitized)

    limit = max(0, int(getattr(args, "optuna_seed_candidates", 0)))
    if limit > 0:
        return deduped[:limit]
    return []


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _matches_pipeline_warmstart_path(path: Path, pipeline: PipelineSpec) -> bool:
    text = str(path).lower()
    if pipeline.name == "lightgbm_split":
        return "lightgbm_split" in text
    if pipeline.name == "lightgbm":
        return "lightgbm" in text and "lightgbm_split" not in text
    if pipeline.binning_backend == "lightgbm":
        return "lightgbm" in text
    return True


def _build_optuna_warmstart_index(
    args: argparse.Namespace,
    pipeline: PipelineSpec,
    run_dir: Path,
    log_fp,
    respect_warmstart_toggle: bool = True,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    if respect_warmstart_toggle and not bool(getattr(args, "optuna_warmstart_enable", True)):
        return {}

    warm_root_raw = str(getattr(args, "optuna_warmstart_root", "results")).strip()
    warm_root_path = Path(warm_root_raw)
    warm_root = warm_root_path.resolve() if warm_root_path.is_absolute() else (PROJECT_ROOT / warm_root_path).resolve()
    if not warm_root.exists():
        _log(f"optuna warm-start root not found: {warm_root}", log_fp)
        return {}

    csv_paths = sorted(p for p in warm_root.rglob("best_params.csv") if p.is_file())
    scored_rows: dict[tuple[str, int], list[tuple[float, dict[str, Any]]]] = {}
    scanned = 0
    for csv_path in csv_paths:
        if _path_is_within(csv_path, run_dir):
            continue
        if not _matches_pipeline_warmstart_path(csv_path, pipeline):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        required = {
            "dataset",
            "depth_budget",
            "best_objective_accuracy",
            "lookahead_depth_budget",
            "max_bins",
            "max_branching",
            "reg",
        }
        if not required.issubset(set(df.columns)):
            continue
        scanned += 1
        for _, row in df.iterrows():
            dataset_name = str(row["dataset"])
            depth_budget = int(row["depth_budget"])
            if dataset_name not in args.datasets or depth_budget not in args.depth_budgets:
                continue
            score = float(row["best_objective_accuracy"])
            if not np.isfinite(score):
                continue
            params = {
                "lookahead_cap": int(row["lookahead_depth_budget"]),
                "max_bins": int(row["max_bins"]),
                "max_branching": int(row["max_branching"]),
                "reg": float(row["reg"]),
            }
            if "leaf_frac" in df.columns and not pd.isna(row.get("leaf_frac")):
                params["leaf_frac"] = float(row["leaf_frac"])
            if "min_samples_leaf" in df.columns and not pd.isna(row.get("min_samples_leaf")):
                params["min_samples_leaf"] = int(row["min_samples_leaf"])
            if "min_child_size" in df.columns and not pd.isna(row.get("min_child_size")):
                params["min_child_size"] = int(row["min_child_size"])
            if "lgb_n_estimators" in df.columns and not pd.isna(row.get("lgb_n_estimators")):
                params["lgb_n_estimators"] = int(row["lgb_n_estimators"])
            if "lgb_num_leaves" in df.columns and not pd.isna(row.get("lgb_num_leaves")):
                params["lgb_num_leaves"] = int(row["lgb_num_leaves"])
            if "lgb_learning_rate" in df.columns and not pd.isna(row.get("lgb_learning_rate")):
                params["lgb_learning_rate"] = float(row["lgb_learning_rate"])
            if "lgb_feature_fraction" in df.columns and not pd.isna(row.get("lgb_feature_fraction")):
                params["lgb_feature_fraction"] = float(row["lgb_feature_fraction"])
            if "lgb_bagging_fraction" in df.columns and not pd.isna(row.get("lgb_bagging_fraction")):
                params["lgb_bagging_fraction"] = float(row["lgb_bagging_fraction"])
            if "lgb_bagging_freq" in df.columns and not pd.isna(row.get("lgb_bagging_freq")):
                params["lgb_bagging_freq"] = int(row["lgb_bagging_freq"])
            if "lgb_max_depth" in df.columns and not pd.isna(row.get("lgb_max_depth")):
                params["lgb_max_depth"] = int(row["lgb_max_depth"])
            if "lgb_min_data_in_bin" in df.columns and not pd.isna(row.get("lgb_min_data_in_bin")):
                params["lgb_min_data_in_bin"] = int(row["lgb_min_data_in_bin"])
            if "lgb_min_data_in_leaf" in df.columns and not pd.isna(row.get("lgb_min_data_in_leaf")):
                params["lgb_min_data_in_leaf"] = int(row["lgb_min_data_in_leaf"])
            if "lgb_lambda_l2" in df.columns and not pd.isna(row.get("lgb_lambda_l2")):
                params["lgb_lambda_l2"] = float(row["lgb_lambda_l2"])
            sanitized = _sanitize_optuna_params(params, args, depth_budget)
            key = (dataset_name, depth_budget)
            scored_rows.setdefault(key, []).append((score, sanitized))

    max_per_study = max(0, int(getattr(args, "optuna_warmstart_max_per_study", 0)))
    warm_index: dict[tuple[str, int], list[dict[str, Any]]] = {}
    total_candidates = 0
    for key, entries in scored_rows.items():
        entries.sort(key=lambda item: item[0], reverse=True)
        seen: set[tuple[Any, ...]] = set()
        selected: list[dict[str, Any]] = []
        for _, params in entries:
            sig = _optuna_param_signature(params)
            if sig in seen:
                continue
            seen.add(sig)
            selected.append(params)
            if max_per_study > 0 and len(selected) >= max_per_study:
                break
        if selected:
            warm_index[key] = selected
            total_candidates += len(selected)

    _log(
        (
            f"optuna warm-start: scanned_files={scanned}, study_keys={len(warm_index)}, "
            f"candidates={total_candidates}, max_per_study={max_per_study}"
        ),
        log_fp,
    )
    return warm_index


def _sample_optuna_params(
    trial,
    args: argparse.Namespace,
    depth_budget: int,
    n_fit: int | None = None,
    minority_count: int | None = None,
) -> dict[str, Any]:
    limits = _optuna_space_limits(args, depth_budget, n_fit=n_fit, minority_count=minority_count)
    fixed_lookahead = max(1, min(int(args.lookahead_cap), int(limits["lookahead_upper"])))
    fixed_max_bins = max(2, int(args.max_bins))
    max_bins_high = int(limits["max_bins_upper"])
    fixed_max_branching = max(0, int(args.max_branching))
    leaf_frac_fixed = getattr(args, "leaf_frac", None)
    if leaf_frac_fixed is not None:
        leaf_frac = float(leaf_frac_fixed)
    else:
        leaf_frac = float(
            trial.suggest_categorical(
                "leaf_frac",
                [float(v) for v in limits.get("leaf_frac_choices", [0.01, 0.02, 0.05])],
            )
        )
    if n_fit is None or int(n_fit) <= 0:
        fallback_m = max(2, int(args.min_samples_leaf), int(args.min_child_size))
        derived_m = int(fallback_m)
    else:
        derived_m = _derive_min_support_from_leaf_frac(leaf_frac, int(n_fit))
    sampled = {
        "lookahead_cap": fixed_lookahead,
        "max_bins": fixed_max_bins,
        "leaf_frac": float(leaf_frac),
        "min_samples_leaf": int(derived_m),
        "min_child_size": int(derived_m),
        "reg": trial.suggest_float("reg", float(limits["reg_min"]), float(limits["reg_max"]), log=True),
    }
    if fixed_max_branching > 0:
        sampled["max_branching"] = int(min(max_bins_high, max(2, fixed_max_branching)))
    else:
        sampled["max_branching"] = trial.suggest_categorical("max_branching", list(limits["max_branch_choices"]))
    if bool(getattr(args, "uses_lightgbm_binning", False)):
        sampled["lgb_num_leaves"] = trial.suggest_int(
            "lgb_num_leaves",
            int(limits.get("lgb_num_leaves_min", 8)),
            int(limits.get("lgb_num_leaves_max", 256)),
        )
        sampled["lgb_learning_rate"] = trial.suggest_float(
            "lgb_learning_rate",
            float(limits.get("lgb_learning_rate_min", 5e-3)),
            float(limits.get("lgb_learning_rate_max", 1e-1)),
            log=True,
        )
        sampled["lgb_feature_fraction"] = trial.suggest_float(
            "lgb_feature_fraction",
            float(limits.get("lgb_feature_fraction_min", 8e-1)),
            float(limits.get("lgb_feature_fraction_max", 1.0)),
        )
        sampled["lgb_bagging_freq"] = trial.suggest_categorical(
            "lgb_bagging_freq",
            [int(v) for v in limits.get("lgb_bagging_freq_choices", [int(getattr(args, "lgb_bagging_freq", 0))])],
        )
        if int(sampled["lgb_bagging_freq"]) == 0:
            sampled["lgb_bagging_fraction"] = 1.0
        else:
            sampled["lgb_bagging_fraction"] = trial.suggest_float(
                "lgb_bagging_fraction",
                float(limits.get("lgb_bagging_fraction_min", 8e-1)),
                float(limits.get("lgb_bagging_fraction_max", 1.0)),
            )
        sampled["lgb_min_data_in_bin"] = trial.suggest_categorical(
            "lgb_min_data_in_bin",
            [int(v) for v in limits.get("lgb_min_data_in_bin_choices", [int(getattr(args, "lgb_min_data_in_bin", 1))])],
        )
        leaf_choice = trial.suggest_categorical(
            "lgb_min_data_in_leaf_choice",
            ["half", "base", "double", "quadruple"],
        )
        leaf_grid = _lgb_min_data_in_leaf_grid(int(sampled["min_child_size"]))
        leaf_choice_idx = {"half": 0, "base": 1, "double": 2, "quadruple": 3}[str(leaf_choice)]
        sampled["lgb_min_data_in_leaf"] = int(leaf_grid[leaf_choice_idx])
        sampled["lgb_lambda_l2"] = trial.suggest_categorical(
            "lgb_lambda_l2",
            [float(v) for v in limits.get("lgb_lambda_l2_choices", [0.0, 0.1])],
        )
    return _sanitize_optuna_params(
        sampled,
        args,
        depth_budget,
        n_fit=n_fit,
        minority_count=minority_count,
    )


def _study_seed(dataset_name: str, depth_budget: int, base_seed: int) -> int:
    dataset_bias = sum(ord(ch) for ch in str(dataset_name))
    return int(base_seed) + int(depth_budget) * 1000 + dataset_bias


def _plan_optuna_parallelism(
    num_studies: int,
    effective_slots: int,
    trials_per_study: int,
    max_active_override: int,
) -> tuple[int, int]:
    if num_studies <= 0:
        return 1, 1

    max_active = min(num_studies, max(1, int(effective_slots)))
    if max_active_override > 0:
        max_active = min(max_active, int(max_active_override))

    best_active = 1
    best_jobs = 1
    best_score = 1
    for active in range(1, max_active + 1):
        jobs = max(1, min(int(trials_per_study), int(effective_slots) // active))
        score = active * jobs
        if score > best_score or (score == best_score and active > best_active):
            best_active = active
            best_jobs = jobs
            best_score = score

    return int(best_active), int(best_jobs)


def _run_optuna_val_trial_preprocessed(
    X_fit_proc: np.ndarray,
    X_eval_proc: np.ndarray,
    y_fit: np.ndarray,
    y_eval: np.ndarray,
    depth_budget: int,
    seed: int,
    args: argparse.Namespace,
    pipeline: PipelineSpec,
    param_overrides: dict[str, Any] | None,
) -> dict[str, float]:
    trial_params = _resolve_trial_params(args, depth_budget, param_overrides, n_fit=int(y_fit.shape[0]))

    binner = _fit_or_get_binner(
        dataset_name=None,
        X_fit_proc=X_fit_proc,
        y_fit=y_fit,
        X_binner_val_proc=X_eval_proc,
        y_binner_val=y_eval,
        trial_params=trial_params,
        seed=seed,
        args=args,
        pipeline=pipeline,
    )
    Z_fit = binner.transform(X_fit_proc)
    Z_eval = binner.transform(X_eval_proc)

    if pipeline.solver == "split":
        feature_names = [f"x{i}" for i in range(Z_fit.shape[1])]
        split_result = _run_binary_split_trial(
            Z_fit=Z_fit,
            Z_eval=Z_eval,
            y_fit=y_fit,
            y_eval=y_eval,
            feature_names=feature_names,
            depth_budget=depth_budget,
            seed=seed,
            trial_params=trial_params,
            args=args,
            eval_mode="val",
        )
        return {
            "objective_accuracy": float(split_result["objective_accuracy"]),
            "objective_balanced_accuracy": float(split_result["objective_balanced_accuracy"]),
            "fit_time_sec": float(split_result["fit_time_sec"]),
        }

    model, _ = _build_msplit_model(
        trial_params=trial_params,
        depth_budget=depth_budget,
        seed=seed,
        args=args,
    )

    start = time.time()
    model.fit(Z_fit, y_fit, **_teacher_kwargs_from_binner(binner))
    fit_time = time.time() - start

    y_pred = model.predict(Z_eval).astype(np.int32)
    accuracy = float(np.mean(y_pred == y_eval))
    balanced_acc = float(balanced_accuracy_score(y_eval, y_pred))
    return {
        "objective_accuracy": accuracy,
        "objective_balanced_accuracy": balanced_acc,
        "fit_time_sec": fit_time,
    }


def _prepare_optuna_study_payloads(
    dataset_payload: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    pipeline: PipelineSpec,
    log_fp,
) -> dict[tuple[str, int], dict[str, str]]:
    if not args.optuna_enable:
        return {}

    optuna_payload_dir = run_dir / "optuna" / "study_payloads"
    optuna_payload_dir.mkdir(parents=True, exist_ok=True)
    payload_paths: dict[tuple[str, int], dict[str, str]] = {}
    manifest_rows: list[dict[str, Any]] = []

    for dataset_name in args.datasets:
        payload = dataset_payload[dataset_name]
        for depth_budget in args.depth_budgets:
            depth_budget_i = int(depth_budget)
            study_seed = _study_seed(dataset_name, depth_budget_i, args.optuna_seed)
            split_payload = _prepare_eval_data(
                payload["X"],
                payload["y_bin"],
                seed=study_seed,
                args=args,
                eval_mode="val",
                dataset_name=None,
            )
            X_fit_proc = np.ascontiguousarray(split_payload["X_fit_proc"], dtype=np.float32)
            X_eval_proc = np.ascontiguousarray(split_payload["X_eval_proc"], dtype=np.float32)
            y_fit = np.ascontiguousarray(split_payload["y_fit"], dtype=np.int32)
            y_eval = np.ascontiguousarray(split_payload["y_eval"], dtype=np.int32)

            stem = f"{dataset_name}_d{depth_budget_i}"
            x_fit_path = optuna_payload_dir / f"{stem}_x_fit.npy"
            x_eval_path = optuna_payload_dir / f"{stem}_x_eval.npy"
            y_fit_path = optuna_payload_dir / f"{stem}_y_fit.npy"
            y_eval_path = optuna_payload_dir / f"{stem}_y_eval.npy"
            np.save(x_fit_path, X_fit_proc, allow_pickle=False)
            np.save(x_eval_path, X_eval_proc, allow_pickle=False)
            np.save(y_fit_path, y_fit, allow_pickle=False)
            np.save(y_eval_path, y_eval, allow_pickle=False)

            paths = {
                "x_fit_proc": str(x_fit_path),
                "x_eval_proc": str(x_eval_path),
                "y_fit": str(y_fit_path),
                "y_eval": str(y_eval_path),
            }
            payload_paths[(dataset_name, depth_budget_i)] = paths
            manifest_rows.append(
                {
                    "dataset": dataset_name,
                    "depth_budget": depth_budget_i,
                    "study_seed": int(study_seed),
                    "x_fit_proc_path": str(x_fit_path),
                    "x_eval_proc_path": str(x_eval_path),
                    "y_fit_path": str(y_fit_path),
                    "y_eval_path": str(y_eval_path),
                    "n_fit": int(y_fit.shape[0]),
                    "n_eval": int(y_eval.shape[0]),
                    "n_features": int(X_fit_proc.shape[1]),
                }
            )

    manifest_path = run_dir / "optuna" / "study_payloads_manifest.csv"
    pd.DataFrame(manifest_rows).sort_values(["dataset", "depth_budget"]).to_csv(manifest_path, index=False)
    _log(
        (
            f"optuna fixed validation splits prepared for {len(manifest_rows)} studies "
            f"(pipeline={pipeline.name}, dir={optuna_payload_dir})"
        ),
        log_fp,
    )
    return payload_paths


def _make_optuna_journal_storage(storage_path: str):
    if optuna is None:
        raise RuntimeError("Optuna is not installed.")
    return optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(str(storage_path)))


def _optuna_objective_preprocessed(
    trial,
    *,
    args: argparse.Namespace,
    depth_budget: int,
    split_seed: int,
    pipeline: PipelineSpec,
    study_n_fit: int,
    study_minority: int,
    X_fit_proc: np.ndarray,
    X_eval_proc: np.ndarray,
    y_fit: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    params = _sample_optuna_params(
        trial,
        args,
        depth_budget,
        n_fit=study_n_fit,
        minority_count=study_minority,
    )
    try:
        result = _run_optuna_val_trial_preprocessed(
            X_fit_proc=X_fit_proc,
            X_eval_proc=X_eval_proc,
            y_fit=y_fit,
            y_eval=y_eval,
            depth_budget=depth_budget,
            seed=split_seed,
            args=args,
            pipeline=pipeline,
            param_overrides=params,
        )
        trial.set_user_attr("fit_time_sec", float(result["fit_time_sec"]))
        trial.set_user_attr("status", "ok")
        return float(result["objective_accuracy"])
    except (TimeoutError, RuntimeError) as exc:
        err_text = str(exc)
        timeout_like = isinstance(exc, TimeoutError) or "time_limit" in err_text.lower()
        if not timeout_like:
            trial.set_user_attr("status", "error")
            trial.set_user_attr("error", err_text)
            return 0.0
        trial.set_user_attr("status", "timeout")
        trial.set_user_attr("error", err_text)
        raise optuna.TrialPruned(err_text)
    except Exception as exc:  # pragma: no cover - defensive fallback
        trial.set_user_attr("status", "error")
        trial.set_user_attr("error", repr(exc))
        return 0.0


def _run_optuna_study_worker_chunk(
    *,
    study_name: str,
    storage_path: str,
    args: argparse.Namespace,
    depth_budget: int,
    split_seed: int,
    pipeline: PipelineSpec,
    study_payload_paths: dict[str, str],
    study_n_fit: int,
    study_minority: int,
    n_trials: int,
    timeout_sec: float | None,
) -> dict[str, int]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed.")
    trials_i = max(0, int(n_trials))
    if trials_i <= 0:
        return {"n_trials": 0}

    _set_thread_caps(args.threads_per_trial)
    X_fit_proc = np.load(str(study_payload_paths["x_fit_proc"]), mmap_mode="r")
    X_eval_proc = np.load(str(study_payload_paths["x_eval_proc"]), mmap_mode="r")
    y_fit = np.load(str(study_payload_paths["y_fit"]), mmap_mode="r")
    y_eval = np.load(str(study_payload_paths["y_eval"]), mmap_mode="r")

    storage = _make_optuna_journal_storage(storage_path)
    study = optuna.load_study(study_name=study_name, storage=storage)

    def _objective(trial):
        return _optuna_objective_preprocessed(
            trial,
            args=args,
            depth_budget=depth_budget,
            split_seed=split_seed,
            pipeline=pipeline,
            study_n_fit=study_n_fit,
            study_minority=study_minority,
            X_fit_proc=X_fit_proc,
            X_eval_proc=X_eval_proc,
            y_fit=y_fit,
            y_eval=y_eval,
        )

    study.optimize(
        _objective,
        n_trials=trials_i,
        timeout=timeout_sec,
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    return {"n_trials": trials_i}


def _run_optuna_single_study(
    dataset_name: str,
    depth_budget: int,
    args: argparse.Namespace,
    jobs_per_study: int,
    pipeline: PipelineSpec,
    allow_process_parallel: bool = True,
    X=None,
    y_bin: np.ndarray | None = None,
    study_payload_paths: dict[str, str] | None = None,
    warmstart_params: list[dict[str, Any]] | None = None,
    seed_candidate_params: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if optuna is None or TrialState is None:
        raise RuntimeError("Optuna is not installed. Run `pip install optuna` (or install requirements.txt).")
    _set_thread_caps(args.threads_per_trial)

    base_seed = _study_seed(dataset_name, depth_budget, args.optuna_seed)
    sampler = optuna.samplers.TPESampler(seed=base_seed)
    study_name = f"{pipeline.study_prefix}_{dataset_name}_d{depth_budget}"
    process_parallel = bool(allow_process_parallel and study_payload_paths and int(jobs_per_study) > 1)
    study_storage_path: str | None = None
    storage = None
    if process_parallel:
        payload_root = Path(str(study_payload_paths["x_fit_proc"])).resolve().parent.parent
        payload_root.mkdir(parents=True, exist_ok=True)
        safe_dataset = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(dataset_name))
        study_storage_file = payload_root / f"study_{pipeline.study_prefix}_{safe_dataset}_d{int(depth_budget)}.journal"
        if study_storage_file.exists():
            study_storage_file.unlink()
        study_storage_path = str(study_storage_file)
        storage = _make_optuna_journal_storage(study_storage_path)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
    )
    max_enqueued = max(0, int(args.optuna_trials) - 1)
    enqueued = 0
    seen_signatures: set[tuple[Any, ...]] = set()

    split_seed = int(base_seed)
    X_fit_proc: np.ndarray | None = None
    X_eval_proc: np.ndarray | None = None
    y_fit: np.ndarray | None = None
    y_eval: np.ndarray | None = None
    if study_payload_paths:
        X_fit_proc = np.load(str(study_payload_paths["x_fit_proc"]), mmap_mode="r")
        X_eval_proc = np.load(str(study_payload_paths["x_eval_proc"]), mmap_mode="r")
        y_fit = np.load(str(study_payload_paths["y_fit"]), mmap_mode="r")
        y_eval = np.load(str(study_payload_paths["y_eval"]), mmap_mode="r")
    elif X is not None and y_bin is not None:
        split_payload = _prepare_eval_data(
            X,
            y_bin,
            seed=split_seed,
            args=args,
            eval_mode="val",
            dataset_name=None,
        )
        X_fit_proc = np.ascontiguousarray(split_payload["X_fit_proc"], dtype=np.float32)
        X_eval_proc = np.ascontiguousarray(split_payload["X_eval_proc"], dtype=np.float32)
        y_fit = np.ascontiguousarray(split_payload["y_fit"], dtype=np.int32)
        y_eval = np.ascontiguousarray(split_payload["y_eval"], dtype=np.int32)
    else:
        raise RuntimeError("missing Optuna study payload")

    study_n_fit = int(y_fit.shape[0])  # type: ignore[union-attr]
    pos_count = int(np.sum(y_fit, dtype=np.int64))  # type: ignore[arg-type]
    neg_count = max(0, study_n_fit - pos_count)
    study_minority = min(pos_count, neg_count)

    for params in (warmstart_params or []) + (seed_candidate_params or []):
        if enqueued >= max_enqueued:
            break
        sanitized = _sanitize_optuna_params(
            dict(params),
            args,
            depth_budget,
            n_fit=study_n_fit,
            minority_count=study_minority,
        )
        if getattr(args, "leaf_frac", None) is not None:
            sanitized = dict(sanitized)
            sanitized.pop("leaf_frac", None)
        sig = _optuna_param_signature(sanitized)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        study.enqueue_trial(sanitized)
        enqueued += 1

    timeout = float(args.optuna_timeout_sec) if float(args.optuna_timeout_sec) > 0 else None
    if process_parallel and study_payload_paths and study_storage_path is not None:
        worker_count = max(1, min(int(jobs_per_study), int(args.optuna_trials)))
        total_trials = int(args.optuna_trials)
        base = total_trials // worker_count
        rem = total_trials % worker_count
        trial_chunks = [base + (1 if idx < rem else 0) for idx in range(worker_count)]
        trial_chunks = [int(v) for v in trial_chunks if int(v) > 0]
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = multiprocessing.get_context()
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            futures = [
                executor.submit(
                    _run_optuna_study_worker_chunk,
                    study_name=study_name,
                    storage_path=study_storage_path,
                    args=args,
                    depth_budget=int(depth_budget),
                    split_seed=int(split_seed),
                    pipeline=pipeline,
                    study_payload_paths=dict(study_payload_paths),
                    study_n_fit=int(study_n_fit),
                    study_minority=int(study_minority),
                    n_trials=int(chunk_n),
                    timeout_sec=timeout,
                )
                for chunk_n in trial_chunks
            ]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
        storage = _make_optuna_journal_storage(study_storage_path)
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        def _objective(trial):
            return _optuna_objective_preprocessed(
                trial,
                args=args,
                depth_budget=depth_budget,
                split_seed=split_seed,
                pipeline=pipeline,
                study_n_fit=study_n_fit,
                study_minority=study_minority,
                X_fit_proc=X_fit_proc,  # type: ignore[arg-type]
                X_eval_proc=X_eval_proc,  # type: ignore[arg-type]
                y_fit=y_fit,  # type: ignore[arg-type]
                y_eval=y_eval,  # type: ignore[arg-type]
            )

        study.optimize(
            _objective,
            n_trials=int(args.optuna_trials),
            timeout=timeout,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=False,
        )

    state_counts: dict[str, int] = {}
    for st in TrialState:
        state_counts[st.name] = int(sum(1 for t in study.trials if t.state == st))
    status_counts = {
        "ok": int(sum(1 for t in study.trials if str(t.user_attrs.get("status", "")) == "ok")),
        "timeout": int(sum(1 for t in study.trials if str(t.user_attrs.get("status", "")) == "timeout")),
        "error": int(sum(1 for t in study.trials if str(t.user_attrs.get("status", "")) == "error")),
    }

    if state_counts.get("COMPLETE", 0) > 0:
        best_params_raw = dict(study.best_params)
        best_value = float(study.best_value)
    else:
        best_params_raw = {}
        best_value = float("nan")

    resolved = _resolve_trial_params(args, depth_budget, best_params_raw, n_fit=study_n_fit)
    return {
        "dataset": dataset_name,
        "depth_budget": int(depth_budget),
        "study_seed": int(base_seed),
        "n_enqueued": int(enqueued),
        "best_value": float(best_value),
        "n_trials": int(len(study.trials)),
        "n_complete": int(state_counts.get("COMPLETE", 0)),
        "n_fail": int(status_counts["error"]),
        "n_timeout": int(status_counts["timeout"]),
        "n_pruned": int(state_counts.get("PRUNED", 0)),
        "best_params_raw": best_params_raw,
        "resolved_params": resolved,
        "trials_df": study.trials_dataframe(),
    }


def _run_optuna_tuning(
    dataset_payload: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    log_fp,
    pipeline: PipelineSpec,
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path | None]:
    if not args.optuna_enable:
        return {}, None
    if optuna is None:
        raise RuntimeError("Optuna is not installed.")

    study_tasks = [(dataset_name, int(depth_budget)) for dataset_name in args.datasets for depth_budget in args.depth_budgets]
    optuna_slots = max(1, int(getattr(args, "optuna_effective_parallel_trials", args.effective_parallel_trials)))
    active_studies, jobs_per_study = _plan_optuna_parallelism(
        num_studies=len(study_tasks),
        effective_slots=optuna_slots,
        trials_per_study=int(args.optuna_trials),
        max_active_override=int(args.optuna_max_active_studies),
    )
    _log(
        (
            f"optuna scheduling: studies={len(study_tasks)}, active_studies={active_studies}, "
            f"jobs_per_study={jobs_per_study}, effective_slots={optuna_slots}, "
            f"effective_core_budget={args.effective_core_budget}/{args.core_budget}"
        ),
        log_fp,
    )

    optuna_dir = run_dir / "optuna"
    optuna_dir.mkdir(parents=True, exist_ok=True)
    warmstart_index = _build_optuna_warmstart_index(
        args=args,
        pipeline=pipeline,
        run_dir=run_dir,
        log_fp=log_fp,
    )
    study_payload_paths = _prepare_optuna_study_payloads(
        dataset_payload=dataset_payload,
        args=args,
        run_dir=run_dir,
        pipeline=pipeline,
        log_fp=log_fp,
    )
    results: list[dict[str, Any]] = []

    if active_studies <= 1 or len(study_tasks) == 1:
        for dataset_name, depth_budget in study_tasks:
            results.append(
                _run_optuna_single_study(
                    dataset_name=dataset_name,
                    depth_budget=depth_budget,
                    args=args,
                    jobs_per_study=jobs_per_study,
                    pipeline=pipeline,
                    allow_process_parallel=True,
                    study_payload_paths=study_payload_paths.get((dataset_name, int(depth_budget))),
                    warmstart_params=warmstart_index.get((dataset_name, int(depth_budget))),
                    seed_candidate_params=_build_optuna_seed_candidates(args, int(depth_budget), dataset_name),
                )
            )
    else:
        workers = min(active_studies, len(study_tasks))
        try:
            mp_context = multiprocessing.get_context("fork")
        except ValueError:
            mp_context = multiprocessing.get_context()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as executor:
            future_map = {}
            for dataset_name, depth_budget in study_tasks:
                fut = executor.submit(
                    _run_optuna_single_study,
                    dataset_name=dataset_name,
                    depth_budget=depth_budget,
                    args=args,
                    jobs_per_study=jobs_per_study,
                    pipeline=pipeline,
                    allow_process_parallel=False,
                    study_payload_paths=study_payload_paths.get((dataset_name, int(depth_budget))),
                    warmstart_params=warmstart_index.get((dataset_name, int(depth_budget))),
                    seed_candidate_params=_build_optuna_seed_candidates(args, int(depth_budget), dataset_name),
                )
                future_map[fut] = (dataset_name, depth_budget)
            for fut in concurrent.futures.as_completed(future_map):
                results.append(fut.result())

    tuned_params: dict[tuple[str, int], dict[str, Any]] = {}
    best_rows: list[dict[str, Any]] = []
    for payload in sorted(results, key=lambda p: (p["dataset"], p["depth_budget"])):
        dataset_name = str(payload["dataset"])
        depth_budget = int(payload["depth_budget"])
        tuned_params[(dataset_name, depth_budget)] = dict(payload["resolved_params"])

        trial_csv = optuna_dir / f"trials_{dataset_name}_d{depth_budget}.csv"
        payload["trials_df"].to_csv(trial_csv, index=False)

        best_rows.append(
            {
                "dataset": dataset_name,
                "depth_budget": depth_budget,
                "study_seed": int(payload["study_seed"]),
                "n_enqueued": int(payload.get("n_enqueued", 0)),
                "best_objective_accuracy": float(payload["best_value"]),
                "n_trials": int(payload["n_trials"]),
                "n_complete": int(payload["n_complete"]),
                "n_fail": int(payload["n_fail"]),
                "n_timeout": int(payload["n_timeout"]),
                "n_pruned": int(payload["n_pruned"]),
                "lookahead_depth_budget": int(payload["resolved_params"]["lookahead_depth_budget"]),
                "max_bins": int(payload["resolved_params"]["max_bins"]),
                "leaf_frac": (
                    float(payload["resolved_params"]["leaf_frac"])
                    if payload["resolved_params"].get("leaf_frac", None) is not None
                    else np.nan
                ),
                "min_samples_leaf": int(payload["resolved_params"]["min_samples_leaf"]),
                "min_child_size": int(payload["resolved_params"]["min_child_size"]),
                "max_branching": int(payload["resolved_params"]["max_branching"]),
                "reg": float(payload["resolved_params"]["reg"]),
                "lgb_n_estimators": int(
                    payload["resolved_params"].get("lgb_n_estimators", int(getattr(args, "lgb_n_estimators", 10000)))
                ),
                "lgb_num_leaves": int(
                    payload["resolved_params"].get("lgb_num_leaves", int(getattr(args, "lgb_num_leaves", 31)))
                ),
                "lgb_learning_rate": float(
                    payload["resolved_params"].get("lgb_learning_rate", float(getattr(args, "lgb_learning_rate", 0.05)))
                ),
                "lgb_feature_fraction": float(
                    payload["resolved_params"].get(
                        "lgb_feature_fraction",
                        float(getattr(args, "lgb_feature_fraction", 1.0)),
                    )
                ),
                "lgb_bagging_fraction": float(
                    payload["resolved_params"].get(
                        "lgb_bagging_fraction",
                        float(getattr(args, "lgb_bagging_fraction", 1.0)),
                    )
                ),
                "lgb_bagging_freq": int(
                    payload["resolved_params"].get("lgb_bagging_freq", int(getattr(args, "lgb_bagging_freq", 0)))
                ),
                "lgb_max_depth": int(
                    payload["resolved_params"].get("lgb_max_depth", int(getattr(args, "lgb_max_depth", -1)))
                ),
                "lgb_min_data_in_bin": int(
                    payload["resolved_params"].get(
                        "lgb_min_data_in_bin",
                        int(getattr(args, "lgb_min_data_in_bin", 1)),
                    )
                ),
                "lgb_min_data_in_leaf": int(
                    payload["resolved_params"].get(
                        "lgb_min_data_in_leaf",
                        int(getattr(args, "lgb_min_data_in_leaf", 2)),
                    )
                ),
                "lgb_lambda_l2": float(
                    payload["resolved_params"].get(
                        "lgb_lambda_l2",
                        float(getattr(args, "lgb_lambda_l2", 0.0)),
                    )
                ),
            }
        )

    best_df = pd.DataFrame(best_rows).sort_values(["dataset", "depth_budget"]).reset_index(drop=True)
    best_csv = optuna_dir / "best_params.csv"
    best_df.to_csv(best_csv, index=False)
    _log(f"optuna best params saved: {best_csv}", log_fp)
    return tuned_params, best_csv


def _load_history_default_params(
    args: argparse.Namespace,
    run_dir: Path,
    log_fp,
    pipeline: PipelineSpec,
) -> tuple[dict[tuple[str, int], dict[str, Any]], Path | None]:
    if not bool(getattr(args, "optuna_warmstart_enable", True)):
        _log("historical default params disabled by optuna_warmstart_enable=False", log_fp)
        return {}, None
    warmstart_index = _build_optuna_warmstart_index(
        args=args,
        pipeline=pipeline,
        run_dir=run_dir,
        log_fp=log_fp,
        respect_warmstart_toggle=False,
    )
    if not warmstart_index:
        _log("historical default params: none found (using static defaults)", log_fp)
        return {}, None

    tuned_params: dict[tuple[str, int], dict[str, Any]] = {}
    best_rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        for depth_budget in args.depth_budgets:
            key = (str(dataset_name), int(depth_budget))
            candidates = warmstart_index.get(key, [])
            if not candidates:
                continue
            resolved = _resolve_trial_params(args, int(depth_budget), dict(candidates[0]))
            resolved["lookahead_depth_budget"] = int(min(args.lookahead_cap, max(1, int(depth_budget) - 1)))
            tuned_params[key] = dict(resolved)

            row = {
                "dataset": str(dataset_name),
                "depth_budget": int(depth_budget),
                "lookahead_depth_budget": int(resolved["lookahead_depth_budget"]),
                "max_bins": int(resolved["max_bins"]),
                "leaf_frac": float(resolved["leaf_frac"]) if resolved.get("leaf_frac", None) is not None else np.nan,
                "min_samples_leaf": int(resolved["min_samples_leaf"]),
                "min_child_size": int(resolved["min_child_size"]),
                "max_branching": int(resolved["max_branching"]),
                "reg": float(resolved["reg"]),
            }
            if bool(getattr(args, "uses_lightgbm_binning", False)):
                row.update(
                    {
                        "lgb_n_estimators": int(
                            resolved.get("lgb_n_estimators", int(getattr(args, "lgb_n_estimators", 10000)))
                        ),
                        "lgb_num_leaves": int(resolved.get("lgb_num_leaves", int(getattr(args, "lgb_num_leaves", 31)))),
                        "lgb_learning_rate": float(
                            resolved.get("lgb_learning_rate", float(getattr(args, "lgb_learning_rate", 0.05)))
                        ),
                        "lgb_feature_fraction": float(
                            resolved.get("lgb_feature_fraction", float(getattr(args, "lgb_feature_fraction", 1.0)))
                        ),
                        "lgb_bagging_fraction": float(
                            resolved.get("lgb_bagging_fraction", float(getattr(args, "lgb_bagging_fraction", 1.0)))
                        ),
                        "lgb_bagging_freq": int(
                            resolved.get("lgb_bagging_freq", int(getattr(args, "lgb_bagging_freq", 0)))
                        ),
                        "lgb_max_depth": int(resolved.get("lgb_max_depth", int(getattr(args, "lgb_max_depth", -1)))),
                        "lgb_min_data_in_bin": int(
                            resolved.get("lgb_min_data_in_bin", int(getattr(args, "lgb_min_data_in_bin", 1)))
                        ),
                        "lgb_min_data_in_leaf": int(
                            resolved.get("lgb_min_data_in_leaf", int(getattr(args, "lgb_min_data_in_leaf", 2)))
                        ),
                        "lgb_lambda_l2": float(
                            resolved.get("lgb_lambda_l2", float(getattr(args, "lgb_lambda_l2", 0.0)))
                        ),
                    }
                )
            best_rows.append(row)

    if not tuned_params:
        _log("historical default params: no matching dataset/depth rows found", log_fp)
        return {}, None

    history_dir = run_dir / "optuna"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_csv = history_dir / "history_default_params.csv"
    pd.DataFrame(best_rows).sort_values(["dataset", "depth_budget"]).to_csv(history_csv, index=False)
    _log(
        f"historical default params loaded for {len(tuned_params)} studies; saved {history_csv}",
        log_fp,
    )
    return tuned_params, history_csv


def _load_existing_seed_rows(seed_csv_path: Path):
    if not seed_csv_path.exists():
        return [], set()

    seed_df = pd.read_csv(seed_csv_path)
    if list(seed_df.columns) != SEED_COLUMNS:
        for col in SEED_COLUMNS:
            if col not in seed_df.columns:
                seed_df[col] = np.nan
        seed_df = seed_df.reindex(columns=SEED_COLUMNS)
        seed_df.to_csv(seed_csv_path, index=False)
    rows = seed_df.to_dict(orient="records")
    done = set()
    for row in rows:
        if pd.isna(row.get("dataset")) or pd.isna(row.get("depth_budget")) or pd.isna(row.get("seed")):
            continue
        done.add((str(row["dataset"]), int(row["depth_budget"]), int(row["seed"])))
    return rows, done


def _append_seed_row(seed_csv_path: Path, row: dict[str, Any]) -> None:
    row_df = pd.DataFrame([row], columns=SEED_COLUMNS)
    header = not seed_csv_path.exists()
    row_df.to_csv(seed_csv_path, mode="a", header=header, index=False)


def _build_summary(seed_df: pd.DataFrame, datasets: list[str], depth_budgets: list[int]) -> pd.DataFrame:
    summary_rows = []

    ok_df = seed_df[seed_df["status"] == "ok"].copy()
    for dataset_name in datasets:
        for depth_budget in depth_budgets:
            subset = ok_df[(ok_df["dataset"] == dataset_name) & (ok_df["depth_budget"] == depth_budget)]
            if subset.empty:
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "depth_budget": depth_budget,
                        "lookahead_depth_budget": min(depth_budget - 1, int(seed_df["lookahead_depth_budget"].max()))
                        if not seed_df.empty
                        else min(depth_budget - 1, 2),
                        "n_success": 0,
                        "mean_train_accuracy": np.nan,
                        "mean_accuracy": np.nan,
                        "mean_balanced_accuracy": np.nan,
                        "mean_trivial_accuracy": np.nan,
                        "std_accuracy": np.nan,
                        "mean_fit_time_sec": np.nan,
                        "mean_objective": np.nan,
                        "mean_n_leaves": np.nan,
                        "mean_n_internal": np.nan,
                        "mean_max_arity": np.nan,
                        "mean_exact_internal_nodes": np.nan,
                        "mean_greedy_internal_nodes": np.nan,
                        "mean_dp_subproblem_calls": np.nan,
                        "mean_dp_cache_hits": np.nan,
                        "mean_dp_unique_states": np.nan,
                        "mean_greedy_subproblem_calls": np.nan,
                        "mean_greedy_cache_hits": np.nan,
                        "mean_greedy_unique_states": np.nan,
                        "mean_nominee_unique_total": np.nan,
                        "mean_nominee_child_interval_lookups": np.nan,
                        "mean_nominee_child_interval_unique": np.nan,
                        "mean_nominee_exactified_total": np.nan,
                        "mean_nominee_incumbent_updates": np.nan,
                        "mean_nominee_threatening_samples": np.nan,
                        "mean_nominee_threatening_sum": np.nan,
                        "mean_nominee_threatening_max": np.nan,
                        "mean_nominee_exact_child_eval_sec": np.nan,
                        "mean_nominee_debr_sec": np.nan,
                    }
                )
                continue

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "depth_budget": depth_budget,
                    "lookahead_depth_budget": int(subset["lookahead_depth_budget"].iloc[0]),
                    "n_success": int(subset.shape[0]),
                    "mean_train_accuracy": float(np.nanmean(subset["train_accuracy"])),
                    "mean_accuracy": float(np.nanmean(subset["accuracy"])),
                    "mean_balanced_accuracy": float(np.nanmean(subset["balanced_accuracy"])),
                    "mean_trivial_accuracy": float(np.nanmean(subset["trivial_accuracy"])),
                    "std_accuracy": float(np.nanstd(subset["accuracy"], ddof=1)) if subset.shape[0] > 1 else 0.0,
                    "mean_fit_time_sec": float(np.nanmean(subset["fit_time_sec"])),
                    "mean_objective": float(np.nanmean(subset["objective"])),
                    "mean_n_leaves": float(np.nanmean(subset["n_leaves"])),
                    "mean_n_internal": float(np.nanmean(subset["n_internal"])),
                    "mean_max_arity": float(np.nanmean(subset["max_arity"])),
                    "mean_exact_internal_nodes": float(np.nanmean(subset["exact_internal_nodes"])),
                    "mean_greedy_internal_nodes": float(np.nanmean(subset["greedy_internal_nodes"])),
                    "mean_dp_subproblem_calls": float(np.nanmean(subset["dp_subproblem_calls"])),
                    "mean_dp_cache_hits": float(np.nanmean(subset["dp_cache_hits"])),
                    "mean_dp_unique_states": float(np.nanmean(subset["dp_unique_states"])),
                    "mean_greedy_subproblem_calls": float(np.nanmean(subset["greedy_subproblem_calls"])),
                    "mean_greedy_cache_hits": float(np.nanmean(subset["greedy_cache_hits"])),
                    "mean_greedy_unique_states": float(np.nanmean(subset["greedy_unique_states"])),
                    "mean_nominee_unique_total": float(np.nanmean(subset["nominee_unique_total"])),
                    "mean_nominee_child_interval_lookups": float(
                        np.nanmean(subset["nominee_child_interval_lookups"])
                    ),
                    "mean_nominee_child_interval_unique": float(
                        np.nanmean(subset["nominee_child_interval_unique"])
                    ),
                    "mean_nominee_exactified_total": float(np.nanmean(subset["nominee_exactified_total"])),
                    "mean_nominee_incumbent_updates": float(
                        np.nanmean(subset["nominee_incumbent_updates"])
                    ),
                    "mean_nominee_threatening_samples": float(
                        np.nanmean(subset["nominee_threatening_samples"])
                    ),
                    "mean_nominee_threatening_sum": float(np.nanmean(subset["nominee_threatening_sum"])),
                    "mean_nominee_threatening_max": float(np.nanmean(subset["nominee_threatening_max"])),
                    "mean_nominee_exact_child_eval_sec": float(
                        np.nanmean(subset["nominee_exact_child_eval_sec"])
                    ),
                    "mean_nominee_debr_sec": float(np.nanmean(subset["nominee_debr_sec"])),
                }
            )

    return pd.DataFrame(summary_rows).sort_values(["dataset", "depth_budget"]).reset_index(drop=True)


def _plot_accuracy(
    summary_df: pd.DataFrame,
    datasets: list[str],
    depth_budgets: list[int],
    out_path: Path,
    include_paper_sota: bool,
    pipeline: PipelineSpec,
    dpi: int = 220,
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.4 * len(datasets), 4.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset_name in zip(axes, datasets):
        subset = summary_df[summary_df["dataset"] == dataset_name]
        depths = subset["depth_budget"].to_numpy(dtype=int)
        ours = subset["mean_accuracy"].to_numpy(dtype=float)

        ax.plot(
            depths,
            ours * 100.0,
            marker="o",
            linewidth=2,
            color=pipeline.plot_color,
            label=pipeline.plot_label,
        )

        if include_paper_sota and dataset_name in PAPER_SOTA:
            paper = np.array([PAPER_SOTA[dataset_name].get(int(d), np.nan) for d in depths], dtype=float)
            ax.plot(
                depths,
                paper * 100.0,
                marker="s",
                linestyle="--",
                linewidth=1.8,
                color="#ae2012",
                label="Paper SOTA",
            )

        ax.set_title(dataset_name)
        ax.set_xlabel("Depth budget")
        ax.set_xticks(depth_budgets)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

    axes[0].set_ylabel("Test accuracy (%)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _format_depth_log_config(args: argparse.Namespace, pipeline: PipelineSpec) -> str:
    parts = [
        f"SEEDS={args.seeds}",
        f"TEST_SIZE={args.test_size}",
        f"MAX_BINS={args.max_bins}",
        f"LEAF_FRAC={getattr(args, 'leaf_frac', None)}",
        f"MIN_SAMPLES_LEAF={args.min_samples_leaf}",
        f"MIN_CHILD_SIZE={args.min_child_size}",
        f"MAX_BRANCHING={args.max_branching}",
        f"REG={args.reg}",
        f"LOOKAHEAD_CAP={args.lookahead_cap}",
        f"MSPLIT_VARIANT={getattr(args, 'msplit_variant', 'rush_dp')}",
        f"TIME_LIMIT={args.time_limit}",
    ]
    if pipeline.binning_backend == "lightgbm":
        parts.extend(
            [
                f"LGB_N_EST={args.lgb_n_estimators}",
                f"LGB_NUM_LEAVES={args.lgb_num_leaves}",
                f"LGB_LR={args.lgb_learning_rate}",
                f"LGB_FEATURE_FRAC={args.lgb_feature_fraction}",
                f"LGB_BAGGING_FRAC={args.lgb_bagging_fraction}",
                f"LGB_BAGGING_FREQ={args.lgb_bagging_freq}",
                f"LGB_MAX_DEPTH={args.lgb_max_depth}",
                f"LGB_MIN_DATA_IN_BIN={args.lgb_min_data_in_bin}",
                f"LGB_MIN_DATA_IN_LEAF={args.lgb_min_data_in_leaf}",
                f"LGB_LAMBDA_L2={args.lgb_lambda_l2}",
                f"LGB_THREADS={args.lgb_num_threads}",
                f"LGB_ENSEMBLE_RUNS={args.lgb_ensemble_runs}",
                f"LGB_ENSEMBLE_FEATURE_FRAC={args.lgb_ensemble_feature_fraction}",
                f"LGB_ENSEMBLE_BAGGING_FRAC={args.lgb_ensemble_bagging_fraction}",
                f"LGB_ENSEMBLE_BAGGING_FREQ={args.lgb_ensemble_bagging_freq}",
                f"LGB_THRESHOLD_DEDUP_EPS={args.lgb_threshold_dedup_eps}",
                f"LGB_DEVICE={args.lgb_device_type}",
                f"LGB_GPU_PLATFORM_ID={args.lgb_gpu_platform_id}",
                f"LGB_GPU_DEVICE_ID={args.lgb_gpu_device_id}",
                f"LGB_GPU_FALLBACK={args.lgb_gpu_fallback}",
            ]
        )
    parts.extend(
        [
            f"PARALLEL_TRIALS={args.parallel_trials}",
            f"THREADS_PER_TRIAL={args.threads_per_trial}",
            f"CPU_TARGET={args.cpu_utilization_target}",
            f"CPU_AFFINITY={getattr(args, 'cpu_affinity', '')}",
            f"CPU_NICE={getattr(args, 'cpu_nice', 0)}",
            f"EFFECTIVE_PARALLEL={args.effective_parallel_trials}",
            f"VAL_SIZE={args.optuna_val_size}",
        ]
    )
    return ", ".join(parts)


def _write_depth_log(
    summary_df: pd.DataFrame,
    out_path: Path,
    datasets: list[str],
    args: argparse.Namespace,
    pipeline: PipelineSpec,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{pipeline.depth_log_title}\n")
        f.write(f"CONFIG: {_format_depth_log_config(args, pipeline)}\n\n")
        for dataset_name in datasets:
            f.write(f"Dataset: {dataset_name}\n")
            subset = summary_df[summary_df["dataset"] == dataset_name]
            for _, row in subset.iterrows():
                depth = int(row["depth_budget"])
                mean_acc = float(row["mean_accuracy"])
                mean_train_acc = float(row.get("mean_train_accuracy", np.nan))
                mean_bal_acc = float(row["mean_balanced_accuracy"])
                mean_trivial_acc = float(row["mean_trivial_accuracy"])
                mean_obj = float(row.get("mean_objective", np.nan))
                std_acc = float(row["std_accuracy"])
                n_success = int(row["n_success"])

                if dataset_name in PAPER_SOTA and depth in PAPER_SOTA[dataset_name]:
                    paper_acc = float(PAPER_SOTA[dataset_name][depth])
                    delta = mean_acc - paper_acc
                    paper_text = (
                        f", paper={paper_acc:.4f} ({paper_acc*100:.2f}%), "
                        f"delta={delta:+.4f} ({delta*100:+.2f} pp)"
                    )
                else:
                    paper_text = ""

                f.write(
                    f"Depth {depth}: n_success={n_success}, "
                    f"train_acc={mean_train_acc:.4f} ({mean_train_acc*100:.2f}%), "
                    f"ours={mean_acc:.4f} ({mean_acc*100:.2f}%), "
                    f"bal_acc={mean_bal_acc:.4f} ({mean_bal_acc*100:.2f}%), "
                    f"trivial_acc={mean_trivial_acc:.4f} ({mean_trivial_acc*100:.2f}%), "
                    f"obj={mean_obj:.6f}, "
                    f"std={std_acc:.4f}, "
                    f"mean_leaves={float(row['mean_n_leaves']):.2f}, "
                    f"mean_max_arity={float(row['mean_max_arity']):.2f}, "
                    f"exact_nodes={float(row['mean_exact_internal_nodes']):.2f}, "
                    f"greedy_nodes={float(row['mean_greedy_internal_nodes']):.2f}, "
                    f"dp_calls={float(row['mean_dp_subproblem_calls']):.2f}, "
                    f"dp_unique={float(row['mean_dp_unique_states']):.2f}, "
                    f"greedy_calls={float(row['mean_greedy_subproblem_calls']):.2f}, "
                    f"greedy_unique={float(row['mean_greedy_unique_states']):.2f}"
                    f"{paper_text}\n"
                )
            f.write("\n")


def _configure_openml_cache(cache_dir: Path, log_fp) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import openml

        if hasattr(openml, "config") and hasattr(openml.config, "set_root_cache_directory"):
            openml.config.set_root_cache_directory(str(cache_dir))
            _log(f"OpenML cache directory set to {cache_dir}", log_fp)
        else:
            _log("OpenML cache directory API not available; using default openml cache.", log_fp)
    except Exception as exc:  # pragma: no cover - defensive logging only
        _log(f"[warning] Unable to configure OpenML cache directory: {exc}", log_fp)


def _safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_manifest(
    manifest_path: Path,
    run_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    artifact_paths: dict[str, Path],
    start_time: float,
    end_time: float,
) -> None:
    artifacts = []
    for label, path in artifact_paths.items():
        if not path.exists():
            continue
        artifacts.append(
            {
                "label": label,
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
            }
        )

    manifest = {
        "created_at": _iso_now(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "command": " ".join(sys.argv),
        "elapsed_sec": end_time - start_time,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "artifacts": artifacts,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _create_artifact_bundle(archive_path: Path, files: list[Path]) -> None:
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in files:
            if path.exists():
                tar.add(path, arcname=path.name)


def _load_dataset_payload(args: argparse.Namespace, log_fp) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    dataset_payload: dict[str, dict[str, Any]] = {}
    class_prevalence_by_dataset: dict[str, float] = {}

    for dataset_name in args.datasets:
        loader = DATASET_LOADERS[dataset_name]
        _log(f"Loading dataset={dataset_name}", log_fp)
        X, y = loader()
        y_bin = encode_binary_target(y, dataset_name)
        y_labels = np.array(sorted(np.unique(np.asarray(y)).tolist(), key=lambda v: str(v)), dtype=object)
        target_name = getattr(y, "name", None) or "target"
        dataset_payload[dataset_name] = {
            "X": X,
            "y_bin": y_bin,
            "class_labels": y_labels,
            "target_name": str(target_name),
        }
        class_prevalence = float(max(np.mean(y_bin == 0), np.mean(y_bin == 1)))
        class_prevalence_by_dataset[dataset_name] = class_prevalence
        _log(f"dataset={dataset_name}, class_prevalence(max class)={class_prevalence:.4f}", log_fp)

    return dataset_payload, class_prevalence_by_dataset


def _run_internal(pipeline: PipelineSpec) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = _parse_args(pipeline)
    budgets = _compute_core_budgets(args)
    args.core_budget = int(budgets["core_budget"])
    args.effective_core_budget = int(budgets["effective_budget"])
    args.effective_parallel_trials = int(budgets["effective_parallel_trials"])
    optuna_slots, optuna_slots_reason = _compute_optuna_effective_parallel_trials(args, pipeline)
    args.optuna_effective_parallel_trials = int(optuna_slots)
    args.optuna_slots_reason = str(optuna_slots_reason)

    run_name = args.run_name or datetime.now().strftime(pipeline.run_name_fmt)
    runs_root = (PROJECT_ROOT / args.results_root).resolve()
    run_dir = runs_root / run_name

    if run_dir.exists() and not args.resume:
        raise RuntimeError(f"Run directory already exists: {run_dir}. Use --resume or pick --run-name.")

    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.tree_artifacts_dir:
        args.tree_artifacts_dir = str(run_dir / "tree_artifacts")
    Path(args.tree_artifacts_dir).mkdir(parents=True, exist_ok=True)
    stable_results_dir = (PROJECT_ROOT / args.stable_results_dir).resolve()
    stable_results_dir.mkdir(parents=True, exist_ok=True)

    stdout_log_path = run_dir / "run_stdout.log"
    with stdout_log_path.open("a", encoding="utf-8") as log_fp:
        start_time = time.time()
        _log(pipeline.start_message, log_fp)
        _log(f"run_name={run_name}", log_fp)
        _log(f"run_dir={run_dir}", log_fp)

        _apply_process_runtime_controls(args, log_fp)
        _set_thread_caps(args.threads_per_trial)
        parallel_msg = (
            f"parallel settings: parallel_trials={args.parallel_trials}, "
            f"threads_per_trial={args.threads_per_trial}, "
        )
        if pipeline.binning_backend == "lightgbm":
            parallel_msg += (
                f"lgb_threads={args.lgb_num_threads}, "
                f"lgb_ensemble_runs={args.lgb_ensemble_runs}, "
                f"lgb_device={args.lgb_device_type}, "
                f"lgb_max_gpu_jobs={args.lgb_max_gpu_jobs}, "
            )
        parallel_msg += (
            f"core_budget={args.core_budget}, effective_core_budget={args.effective_core_budget}, "
            f"effective_parallel_trials={args.effective_parallel_trials}, cpu_target={args.cpu_utilization_target}, "
            f"cpu_affinity={getattr(args, 'effective_cpu_affinity', [])}, "
            f"cpu_nice={getattr(args, 'effective_cpu_nice', int(os.nice(0)))}"
        )
        _log(parallel_msg, log_fp)
        if args.optuna_enable and int(args.optuna_effective_parallel_trials) < int(args.effective_parallel_trials):
            _log(
                (
                    f"optuna concurrency cap: using {args.optuna_effective_parallel_trials}/"
                    f"{args.effective_parallel_trials} trial slots "
                    f"(reason={args.optuna_slots_reason})"
                ),
                log_fp,
            )
        _configure_openml_cache((PROJECT_ROOT / args.openml_cache_dir).resolve(), log_fp)

        config_path = run_dir / "config.json"
        config_payload = {
            "created_at": _iso_now(),
            "command": " ".join(sys.argv),
            "config": vars(args),
        }
        config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")

        seed_csv_path = run_dir / "seed_results.csv"
        summary_csv_path = run_dir / "summary_results.csv"
        depth_log_path = run_dir / pipeline.depth_log_filename
        plot_path = run_dir / pipeline.plot_filename
        plot_vs_paper_path = run_dir / pipeline.plot_vs_paper_filename

        seed_rows, completed_keys = _load_existing_seed_rows(seed_csv_path) if args.resume else ([], set())
        if seed_rows:
            _log(f"Resuming from existing seed results: {seed_csv_path} (rows={len(seed_rows)})", log_fp)

        dataset_payload, class_prevalence_by_dataset = _load_dataset_payload(args, log_fp)

        optuna_tuned_params, optuna_best_csv = _run_optuna_tuning(
            dataset_payload=dataset_payload,
            args=args,
            run_dir=run_dir,
            log_fp=log_fp,
            pipeline=pipeline,
        )
        history_defaults, history_default_csv = _load_history_default_params(
            args=args,
            run_dir=run_dir,
            log_fp=log_fp,
            pipeline=pipeline,
        )
        tuned_params: dict[tuple[str, int], dict[str, Any]] = dict(history_defaults)
        tuned_params.update(optuna_tuned_params)

        pending_trials: list[tuple[str, int, int, float]] = []
        for dataset_name in args.datasets:
            class_prevalence = class_prevalence_by_dataset[dataset_name]
            for depth_budget in args.depth_budgets:
                key_2d = (str(dataset_name), int(depth_budget))
                resolved = _resolve_trial_params(args, depth_budget, tuned_params.get((dataset_name, int(depth_budget))))
                if key_2d in optuna_tuned_params:
                    param_source = "optuna"
                elif key_2d in history_defaults:
                    param_source = "history_defaults"
                else:
                    param_source = "static_defaults"
                _log(
                    (
                        f"dataset={dataset_name}, depth={depth_budget}, "
                        f"lookahead={resolved['lookahead_depth_budget']}, seeds={args.seeds}, "
                        f"param_source={param_source}"
                    ),
                    log_fp,
                )
                for seed in args.seeds:
                    key = (dataset_name, int(depth_budget), int(seed))
                    if key in completed_keys:
                        _log(
                            f"[resume] skipping completed trial dataset={dataset_name}, depth={depth_budget}, seed={seed}",
                            log_fp,
                        )
                        continue
                    pending_trials.append((dataset_name, int(depth_budget), int(seed), class_prevalence))

        if args.max_trials > 0 and len(pending_trials) > args.max_trials:
            pending_trials = pending_trials[: args.max_trials]
            _log(f"Reached --max-trials={args.max_trials}. Stopping early.", log_fp)

        result_rows = _run_trials(
            dataset_payload=dataset_payload,
            args=args,
            pending_trials=pending_trials,
            log_fp=log_fp,
            tuned_params=tuned_params,
            pipeline=pipeline,
        )

        for row in result_rows:
            key = (str(row["dataset"]), int(row["depth_budget"]), int(row["seed"]))
            seed_rows.append(row)
            completed_keys.add(key)
            _append_seed_row(seed_csv_path, row)
            _log(
                (
                    f"trial dataset={row['dataset']}, depth={row['depth_budget']}, seed={row['seed']}, "
                    f"status={row['status']}, acc={row['accuracy']}, obj={row.get('objective', np.nan)}"
                ),
                log_fp,
            )
            if args.fail_fast and row.get("status") == "error":
                raise RuntimeError(
                    "Fail-fast triggered by trial error: "
                    f"dataset={row.get('dataset')}, depth={row.get('depth_budget')}, seed={row.get('seed')}, "
                    f"error={row.get('error')}"
                )

        seed_df = pd.DataFrame(seed_rows, columns=SEED_COLUMNS)
        seed_df.to_csv(seed_csv_path, index=False)

        summary_df = _build_summary(seed_df, datasets=args.datasets, depth_budgets=args.depth_budgets)
        summary_df.to_csv(summary_csv_path, index=False)

        _plot_accuracy(
            summary_df,
            datasets=args.datasets,
            depth_budgets=args.depth_budgets,
            out_path=plot_path,
            include_paper_sota=False,
            pipeline=pipeline,
        )

        if args.include_paper_sota:
            _plot_accuracy(
                summary_df,
                datasets=args.datasets,
                depth_budgets=args.depth_budgets,
                out_path=plot_vs_paper_path,
                include_paper_sota=True,
                pipeline=pipeline,
            )

        _write_depth_log(summary_df, depth_log_path, args.datasets, args, pipeline)

        stable_csv = stable_results_dir / pipeline.stable_csv_filename
        stable_plot = stable_results_dir / pipeline.stable_plot_filename
        stable_log = stable_results_dir / pipeline.stable_log_filename

        _safe_copy(summary_csv_path, stable_csv)
        _safe_copy(plot_path, stable_plot)
        _safe_copy(depth_log_path, stable_log)


        manifest_path = run_dir / "manifest.json"
        artifact_paths = {
            "seed_results_csv": seed_csv_path,
            "summary_results_csv": summary_csv_path,
            "depth_log": depth_log_path,
            "plot_accuracy": plot_path,
            "plot_accuracy_vs_paper": plot_vs_paper_path,
            "config_json": config_path,
            "run_stdout_log": stdout_log_path,
            "stable_summary_csv": stable_csv,
            "stable_plot_png": stable_plot,
            "stable_log_txt": stable_log,
        }
        if optuna_best_csv is not None and optuna_best_csv.exists():
            artifact_paths["optuna_best_params_csv"] = optuna_best_csv
        if history_default_csv is not None and history_default_csv.exists():
            artifact_paths["history_default_params_csv"] = history_default_csv
        optuna_payload_manifest = run_dir / "optuna" / "study_payloads_manifest.csv"
        if optuna_payload_manifest.exists():
            artifact_paths["optuna_study_payload_manifest_csv"] = optuna_payload_manifest

        end_time = time.time()
        _write_manifest(
            manifest_path=manifest_path,
            run_name=run_name,
            run_dir=run_dir,
            args=args,
            artifact_paths=artifact_paths,
            start_time=start_time,
            end_time=end_time,
        )

        archive_path = run_dir / f"{run_name}_artifacts.tar.gz"
        if args.package_artifacts:
            bundle_files = [
                config_path,
                manifest_path,
                seed_csv_path,
                summary_csv_path,
                depth_log_path,
                plot_path,
                plot_vs_paper_path,
                stdout_log_path,
            ]
            if optuna_best_csv is not None and optuna_best_csv.exists():
                bundle_files.append(optuna_best_csv)
                bundle_files.extend(sorted((run_dir / "optuna").glob("trials_*.csv")))
            if history_default_csv is not None and history_default_csv.exists():
                bundle_files.append(history_default_csv)
            if optuna_payload_manifest.exists():
                bundle_files.append(optuna_payload_manifest)
            _create_artifact_bundle(archive_path, bundle_files)
            _log(f"Packaged artifact bundle: {archive_path}", log_fp)

        if args.copy_to:
            copy_dir = Path(args.copy_to).expanduser().resolve()
            copy_dir.mkdir(parents=True, exist_ok=True)
            for path in [summary_csv_path, depth_log_path, plot_path, manifest_path]:
                if path.exists():
                    _safe_copy(path, copy_dir / path.name)
            if plot_vs_paper_path.exists():
                _safe_copy(plot_vs_paper_path, copy_dir / plot_vs_paper_path.name)
            if archive_path.exists():
                _safe_copy(archive_path, copy_dir / archive_path.name)
            _log(f"Copied selected artifacts to {copy_dir}", log_fp)

        _log("=== Summary Table ===", log_fp)
        summary_text = summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}")
        print(summary_text, flush=True)
        log_fp.write(summary_text + "\n")

        _log("=== Depth vs Accuracy (Exact) ===", log_fp)
        for dataset_name in args.datasets:
            _log(f"Dataset: {dataset_name}", log_fp)
            subset = summary_df[summary_df["dataset"] == dataset_name]
            for _, row in subset.iterrows():
                depth = int(row["depth_budget"])
                mean_acc = float(row["mean_accuracy"])
                mean_bal_acc = float(row["mean_balanced_accuracy"])
                mean_trivial_acc = float(row["mean_trivial_accuracy"])
                mean_obj = float(row.get("mean_objective", np.nan))
                msg = (
                    f"Depth {depth}: ours={mean_acc:.4f} ({mean_acc*100:.2f}%), "
                    f"bal_acc={mean_bal_acc:.4f} ({mean_bal_acc*100:.2f}%), "
                    f"trivial_acc={mean_trivial_acc:.4f} ({mean_trivial_acc*100:.2f}%), "
                    f"obj={mean_obj:.6f}, "
                    f"mean_leaves={float(row['mean_n_leaves']):.2f}, "
                    f"dp_calls={float(row['mean_dp_subproblem_calls']):.2f}, "
                    f"dp_unique={float(row['mean_dp_unique_states']):.2f}"
                )
                if dataset_name in PAPER_SOTA and depth in PAPER_SOTA[dataset_name]:
                    paper_acc = float(PAPER_SOTA[dataset_name][depth])
                    delta = mean_acc - paper_acc
                    msg += (
                        f", paper={paper_acc:.4f} ({paper_acc*100:.2f}%), "
                        f"delta={delta:+.4f} ({delta*100:+.2f} pp)"
                    )
                _log(msg, log_fp)

        _log(f"Saved run summary CSV: {summary_csv_path}", log_fp)
        _log(f"Saved run seed CSV: {seed_csv_path}", log_fp)
        _log(f"Saved run plot: {plot_path}", log_fp)
        _log(f"Saved run log: {depth_log_path}", log_fp)
        _log(f"Saved run manifest: {manifest_path}", log_fp)
        if plot_vs_paper_path.exists():
            _log(f"Saved run paper comparison plot: {plot_vs_paper_path}", log_fp)
        if archive_path.exists():
            _log(f"Saved run bundle: {archive_path}", log_fp)

        _log(f"Compatibility CSV: {stable_csv}", log_fp)
        _log(f"Compatibility plot: {stable_plot}", log_fp)
        _log(f"Compatibility log: {stable_log}", log_fp)


def run_main(pipeline_name: str) -> None:
    pipeline = _get_pipeline(pipeline_name)
    _run_internal(pipeline)


if __name__ == "__main__":
    run_main("lightgbm")
