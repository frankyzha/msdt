#!/usr/bin/env python3
"""Tune cached MSPLIT and ShapeCART with exhaustive grid search + 3-fold CV.

This runner does one job only:

1. Load an existing cached benchmark split from ``benchmark/cache``.
2. Evaluate every requested hyperparameter configuration with inner CV.
3. Refit the selected configuration on the cached fit split.
4. Write paper-ready tables plus tree artifacts for the selected trees.

The cache is required. If the requested cached LightGBM protocol artifact is
missing, the run fails instead of rebuilding it inside the benchmark.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.scripts.benchmark_cached_common import (
    DEFAULT_CV_FOLDS,
    DEFAULT_DEPTHS,
    DEFAULT_MSPLIT_EXACTIFY_TOP_K_VALUES,
    DEFAULT_MSPLIT_LOOKAHEAD_DEPTH_VALUES,
    DEFAULT_MSPLIT_MAX_BRANCHING_VALUES,
    DEFAULT_MSPLIT_REG_VALUES,
    DEFAULT_SEEDS,
    DEFAULT_SHAPE_BRANCHING_PENALTY_VALUES,
    DEFAULT_SHAPE_CRITERION_VALUES,
    DEFAULT_SHAPE_INNER_MAX_LEAF_VALUES,
    DEFAULT_SHAPE_INNER_MIN_LEAF_VALUES,
    DEFAULT_SHAPE_TAO_REG_VALUES,
    DEFAULT_SHARED_MIN_LEAF_VALUES,
    DEFAULT_SHARED_SPLIT_MULTIPLIERS,
    aggregate_results,
    best_depth_table,
    benchmark_timing_fields,
    cached_protocol_manifest_row,
    coerce_numeric_token,
    configure_timing_mode,
    fit_shapecart_once,
    json_safe,
    load_cached_protocol,
    resolve_search_jobs,
    timing_guard_scope,
    write_csv_tables,
    write_msplit_artifacts,
    write_shapecart_artifacts,
)
from benchmark.scripts.benchmark_paths import BENCHMARK_ARTIFACTS_ROOT
from benchmark.scripts.benchmark_cached_msplit import run_cached_msplit
from benchmark.scripts.cache_utils import (
    DEFAULT_CACHE_VERSION,
    DEFAULT_MAX_BINS,
    DEFAULT_MIN_CHILD_SIZE,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
)
from benchmark.scripts.experiment_utils import canonical_dataset_list


def _write_progress_tables(
    *,
    out_dir: Path,
    selected_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    cache_manifest_rows: list[dict[str, Any]],
) -> None:
    write_csv_tables(
        out_dir,
        {
            "selected_results.csv": selected_rows,
            "cv_grid_results.csv": grid_rows,
            "cv_fold_results.csv": fold_rows,
            "cache_manifest.csv": cache_manifest_rows,
        },
    )


def _subset_cache_for_msplit(
    *,
    cache: dict[str, np.ndarray],
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "Z_fit": np.asarray(cache["Z_fit"][fit_idx], dtype=np.int32),
        "Z_val": np.asarray(cache["Z_fit"][val_idx], dtype=np.int32),
        "Z_test": np.asarray(cache["Z_fit"][val_idx], dtype=np.int32),
        "y_fit": np.asarray(cache["y_fit"][fit_idx], dtype=np.int32),
        "y_val": np.asarray(cache["y_fit"][val_idx], dtype=np.int32),
        "y_test": np.asarray(cache["y_fit"][val_idx], dtype=np.int32),
        "teacher_logit": np.asarray(cache["teacher_logit"][fit_idx], dtype=np.float64),
        "teacher_boundary_gain": np.asarray(cache["teacher_boundary_gain"], dtype=np.float64),
        "teacher_boundary_cover": np.asarray(cache["teacher_boundary_cover"], dtype=np.float64),
        "teacher_boundary_value_jump": np.asarray(cache["teacher_boundary_value_jump"], dtype=np.float64),
    }


def _subset_cache_for_shapecart(
    *,
    cache: dict[str, np.ndarray],
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "X_fit_proc": np.asarray(cache["X_fit_proc"][fit_idx], dtype=np.float32),
        "X_val_proc": np.asarray(cache["X_fit_proc"][val_idx], dtype=np.float32),
        "X_test_proc": np.asarray(cache["X_fit_proc"][val_idx], dtype=np.float32),
        "y_fit": np.asarray(cache["y_fit"][fit_idx], dtype=np.int32),
        "y_val": np.asarray(cache["y_fit"][val_idx], dtype=np.int32),
        "y_test": np.asarray(cache["y_fit"][val_idx], dtype=np.int32),
    }


def _msplit_param_grid(args: argparse.Namespace, depth: int) -> list[dict[str, Any]]:
    lookahead_values = sorted({min(int(v), int(depth)) for v in args.msplit_lookahead_depth_values})
    combos = itertools.product(
        args.shared_min_leaf_values,
        args.shared_split_multipliers,
        lookahead_values,
        args.msplit_exactify_top_k_values,
        args.msplit_max_branching_values,
        args.msplit_reg_values,
    )
    rows: list[dict[str, Any]] = []
    for min_child_size, split_multiplier, lookahead_depth, exactify_top_k, max_branching, reg in combos:
        rows.append(
            {
                "min_child_size": int(min_child_size),
                "min_split_size": int(int(min_child_size) * int(split_multiplier)),
                "min_split_multiplier": int(split_multiplier),
                "lookahead_depth": int(lookahead_depth),
                "exactify_top_k": int(exactify_top_k),
                "max_branching": int(max_branching),
                "reg": float(reg),
            }
        )
    return rows


def _shapecart_param_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    combos = itertools.product(
        args.shared_min_leaf_values,
        args.shared_split_multipliers,
        args.shape_criterion_values,
        args.shape_inner_max_leaf_values,
        args.shape_inner_min_leaf_values,
        args.shape_branching_penalty_values if int(args.shape_k) > 2 else [0.0],
        args.shape_tao_reg_values if bool(args.shape_use_tao) else [0.0],
    )
    rows: list[dict[str, Any]] = []
    for (
        min_samples_leaf,
        split_multiplier,
        criterion,
        inner_max_leaf_nodes,
        inner_min_samples_leaf,
        branching_penalty,
        tao_reg,
    ) in combos:
        rows.append(
            {
                "min_samples_leaf": int(min_samples_leaf),
                "min_samples_split": int(int(min_samples_leaf) * int(split_multiplier)),
                "min_split_multiplier": int(split_multiplier),
                "criterion": str(criterion),
                "inner_max_depth": int(args.shape_inner_max_depth),
                "inner_max_leaf_nodes": int(inner_max_leaf_nodes),
                "inner_min_samples_leaf": inner_min_samples_leaf,
                "branching_penalty": float(branching_penalty),
                "tao_reg": float(tao_reg),
                "k": int(args.shape_k),
                "max_iter": int(args.shape_max_iter),
                "pairwise_candidates": float(args.shape_pairwise_candidates),
                "smart_init": bool(args.shape_smart_init),
                "random_pairs": bool(args.shape_random_pairs),
                "use_dpdt": bool(args.shape_use_dpdt),
                "use_tao": bool(args.shape_use_tao),
                "tao_n_runs": int(args.shape_tao_n_runs),
                "tao_pair_scale": float(args.shape_tao_pair_scale),
            }
        )
    return rows


def _select_best_grid_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No grid-search rows were produced.")
    rows_sorted = sorted(
        rows,
        key=lambda row: (
            -float(row["cv_val_accuracy_mean"]),
            float(row["cv_model_fit_time_mean"]),
            int(row["candidate_index"]),
            json.dumps(json_safe(row["params"]), sort_keys=True),
        ),
    )
    return rows_sorted[0]


def _make_split_list(y_fit: np.ndarray, *, cv_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(
        n_splits=int(cv_folds),
        shuffle=True,
        random_state=int(seed),
    )
    return [
        (np.asarray(fit_idx, dtype=np.int32), np.asarray(val_idx, dtype=np.int32))
        for fit_idx, val_idx in splitter.split(np.zeros(y_fit.shape[0], dtype=np.int32), y_fit)
    ]


def _evaluate_msplit_candidate(
    *,
    protocol,
    depth: int,
    candidate_index: int,
    params: dict[str, Any],
    split_list: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    fold_val_acc: list[float] = []
    fold_fit_times: list[float] = []
    for fold_index, (fit_idx, val_idx) in enumerate(split_list):
        fold_cache = _subset_cache_for_msplit(cache=protocol.cache, fit_idx=fit_idx, val_idx=val_idx)
        result = run_cached_msplit(
            cache=fold_cache,
            depth=int(depth),
            lookahead_depth=int(params["lookahead_depth"]),
            reg=float(params["reg"]),
            exactify_top_k=int(params["exactify_top_k"]),
            min_split_size=int(params["min_split_size"]),
            min_child_size=int(params["min_child_size"]),
            max_branching=int(params["max_branching"]),
        )
        if result.get("val_accuracy") is None:
            raise RuntimeError("Validation accuracy missing from MSPLIT fold result.")
        fold_rows.append(
            {
                "dataset": protocol.dataset,
                "seed": int(protocol.seed),
                "depth_budget": int(depth),
                "algorithm": "msplit",
                "candidate_index": int(candidate_index),
                "fold_index": int(fold_index),
                "val_accuracy": float(result["val_accuracy"]),
                "fit_seconds": float(result["fit_seconds"]),
                "params_json": json.dumps(json_safe(params), sort_keys=True),
            }
        )
        fold_val_acc.append(float(result["val_accuracy"]))
        fold_fit_times.append(float(result["fit_seconds"]))
    grid_row = {
        "algorithm": "msplit",
        "dataset": protocol.dataset,
        "seed": int(protocol.seed),
        "depth_budget": int(depth),
        "candidate_index": int(candidate_index),
        "cv_val_accuracy_mean": float(np.mean(fold_val_acc)),
        "cv_val_accuracy_std": float(np.std(fold_val_acc, ddof=0)),
        "cv_model_fit_time_mean": float(np.mean(fold_fit_times)),
        "cv_model_fit_time_std": float(np.std(fold_fit_times, ddof=0)),
        "params": params,
    }
    return grid_row, fold_rows


def _evaluate_shapecart_candidate(
    *,
    protocol,
    depth: int,
    candidate_index: int,
    params: dict[str, Any],
    split_list: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_rows: list[dict[str, Any]] = []
    fold_val_acc: list[float] = []
    fold_fit_times: list[float] = []
    algorithm_name = "shapecart_tao" if bool(params["use_tao"]) else "shapecart"
    for fold_index, (fit_idx, val_idx) in enumerate(split_list):
        fold_cache = _subset_cache_for_shapecart(cache=protocol.cache, fit_idx=fit_idx, val_idx=val_idx)
        result = fit_shapecart_once(
            cache=fold_cache,
            depth=int(depth),
            random_state=int(protocol.seed) + 100 * int(fold_index),
            params=params,
            include_model=False,
        )
        fold_rows.append(
            {
                "dataset": protocol.dataset,
                "seed": int(protocol.seed),
                "depth_budget": int(depth),
                "algorithm": algorithm_name,
                "candidate_index": int(candidate_index),
                "fold_index": int(fold_index),
                "val_accuracy": float(result["val_accuracy"]),
                "fit_seconds": float(result["fit_seconds"]),
                "params_json": json.dumps(json_safe(params), sort_keys=True),
            }
        )
        fold_val_acc.append(float(result["val_accuracy"]))
        fold_fit_times.append(float(result["fit_seconds"]))
    grid_row = {
        "algorithm": algorithm_name,
        "dataset": protocol.dataset,
        "seed": int(protocol.seed),
        "depth_budget": int(depth),
        "candidate_index": int(candidate_index),
        "cv_val_accuracy_mean": float(np.mean(fold_val_acc)),
        "cv_val_accuracy_std": float(np.std(fold_val_acc, ddof=0)),
        "cv_model_fit_time_mean": float(np.mean(fold_fit_times)),
        "cv_model_fit_time_std": float(np.std(fold_fit_times, ddof=0)),
        "params": params,
    }
    return grid_row, fold_rows


def _evaluate_grid(
    *,
    candidate_fn,
    protocol,
    depth: int,
    params_grid: list[dict[str, Any]],
    split_list: list[tuple[np.ndarray, np.ndarray]],
    search_jobs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    started = time.perf_counter()
    with timing_guard_scope(enabled=False):
        if int(search_jobs) == 1:
            outputs = [
                candidate_fn(
                    protocol=protocol,
                    depth=int(depth),
                    candidate_index=int(candidate_index),
                    params=params,
                    split_list=split_list,
                )
                for candidate_index, params in enumerate(params_grid)
            ]
        else:
            with ThreadPoolExecutor(max_workers=int(search_jobs)) as executor:
                futures = [
                    executor.submit(
                        candidate_fn,
                        protocol=protocol,
                        depth=int(depth),
                        candidate_index=int(candidate_index),
                        params=params,
                        split_list=split_list,
                    )
                    for candidate_index, params in enumerate(params_grid)
                ]
                outputs = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    grid_rows = [row for row, _fold_rows in outputs]
    fold_rows = [fold_row for _row, fold_list in outputs for fold_row in fold_list]
    grid_rows.sort(key=lambda row: int(row["candidate_index"]))
    fold_rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            int(row["seed"]),
            int(row["depth_budget"]),
            str(row["algorithm"]),
            int(row["candidate_index"]),
            int(row["fold_index"]),
        )
    )
    return grid_rows, fold_rows, float(elapsed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune cached MSPLIT and ShapeCART with exhaustive grid search + k-fold CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["electricity"])
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--search-jobs", type=int, default=None)
    parser.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS)
    parser.add_argument("--binner-min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--cache-min-child-size", type=int, default=DEFAULT_MIN_CHILD_SIZE)
    parser.add_argument("--cache-version", type=int, default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--timing-mode", choices=("fair", "fast"), default="fair")
    parser.add_argument("--msplit-build-dir", type=str, default="build-nonlinear-py")
    parser.add_argument("--shared-min-leaf-values", nargs="+", type=int, default=DEFAULT_SHARED_MIN_LEAF_VALUES)
    parser.add_argument("--shared-split-multipliers", nargs="+", type=int, default=DEFAULT_SHARED_SPLIT_MULTIPLIERS)
    parser.add_argument("--msplit-lookahead-depth-values", nargs="+", type=int, default=DEFAULT_MSPLIT_LOOKAHEAD_DEPTH_VALUES)
    parser.add_argument("--msplit-exactify-top-k-values", nargs="+", type=int, default=DEFAULT_MSPLIT_EXACTIFY_TOP_K_VALUES)
    parser.add_argument("--msplit-max-branching-values", nargs="+", type=int, default=DEFAULT_MSPLIT_MAX_BRANCHING_VALUES)
    parser.add_argument("--msplit-reg-values", nargs="+", default=DEFAULT_MSPLIT_REG_VALUES)
    parser.add_argument("--shape-criterion-values", nargs="+", default=DEFAULT_SHAPE_CRITERION_VALUES)
    parser.add_argument("--shape-inner-max-depth", type=int, default=6)
    parser.add_argument("--shape-inner-max-leaf-values", nargs="+", type=int, default=DEFAULT_SHAPE_INNER_MAX_LEAF_VALUES)
    parser.add_argument("--shape-inner-min-leaf-values", nargs="+", default=DEFAULT_SHAPE_INNER_MIN_LEAF_VALUES)
    parser.add_argument("--shape-k", type=int, default=3)
    parser.add_argument("--shape-max-iter", type=int, default=20)
    parser.add_argument("--shape-pairwise-candidates", type=float, default=0.0)
    parser.add_argument("--shape-smart-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shape-random-pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-use-dpdt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-use-tao", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shape-branching-penalty-values", nargs="+", default=DEFAULT_SHAPE_BRANCHING_PENALTY_VALUES)
    parser.add_argument("--shape-tao-reg-values", nargs="+", default=DEFAULT_SHAPE_TAO_REG_VALUES)
    parser.add_argument("--shape-tao-n-runs", type=int, default=10)
    parser.add_argument("--shape-tao-pair-scale", type=float, default=1.1)
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(BENCHMARK_ARTIFACTS_ROOT / "benchmarks" / "cached_gridcv_msplit_vs_shapecart"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if int(args.cv_folds) < 2:
        raise ValueError("--cv-folds must be at least 2")
    if int(args.shape_k) < 2:
        raise ValueError("--shape-k must be at least 2")
    args.msplit_reg_values = [float(coerce_numeric_token(v)) for v in args.msplit_reg_values]
    args.shape_inner_min_leaf_values = [coerce_numeric_token(v) for v in args.shape_inner_min_leaf_values]
    args.shape_branching_penalty_values = [float(coerce_numeric_token(v)) for v in args.shape_branching_penalty_values]
    args.shape_tao_reg_values = [float(coerce_numeric_token(v)) for v in args.shape_tao_reg_values]
    return args


def main() -> int:
    args = _parse_args()
    os.environ["MSPLIT_BUILD_DIR"] = str(args.msplit_build_dir)
    configure_timing_mode(args.timing_mode)

    datasets = canonical_dataset_list(args.datasets)
    depths = sorted(set(int(v) for v in args.depths))
    seeds = [int(v) for v in args.seeds]
    search_jobs = resolve_search_jobs(args.search_jobs, args.timing_mode)
    final_fit_guard_enabled = bool(args.timing_mode == "fair")

    out_root = Path(args.results_root)
    run_name = args.run_name or datetime.now().strftime("cached_gridcv_msplit_vs_shapecart_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "datasets": datasets,
        "depths": depths,
        "seeds": seeds,
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "cv_folds": int(args.cv_folds),
        "search_jobs": int(search_jobs),
        "max_bins": int(args.max_bins),
        "binner_min_samples_leaf": int(args.binner_min_samples_leaf),
        "cache_min_child_size": int(args.cache_min_child_size),
        "cache_version": int(args.cache_version),
        "timing_mode": str(args.timing_mode),
        "search_guard_enabled": False,
        "final_fit_guard_enabled": bool(final_fit_guard_enabled),
        "msplit_build_dir": str(args.msplit_build_dir),
        "shared_min_leaf_values": [int(v) for v in args.shared_min_leaf_values],
        "shared_split_multipliers": [int(v) for v in args.shared_split_multipliers],
        "msplit_lookahead_depth_values": [int(v) for v in args.msplit_lookahead_depth_values],
        "msplit_exactify_top_k_values": [int(v) for v in args.msplit_exactify_top_k_values],
        "msplit_max_branching_values": [int(v) for v in args.msplit_max_branching_values],
        "msplit_reg_values": [float(v) for v in args.msplit_reg_values],
        "shape_criterion_values": list(args.shape_criterion_values),
        "shape_inner_max_depth": int(args.shape_inner_max_depth),
        "shape_inner_max_leaf_values": [int(v) for v in args.shape_inner_max_leaf_values],
        "shape_inner_min_leaf_values": json_safe(args.shape_inner_min_leaf_values),
        "shape_k": int(args.shape_k),
        "shape_max_iter": int(args.shape_max_iter),
        "shape_pairwise_candidates": float(args.shape_pairwise_candidates),
        "shape_smart_init": bool(args.shape_smart_init),
        "shape_random_pairs": bool(args.shape_random_pairs),
        "shape_use_dpdt": bool(args.shape_use_dpdt),
        "shape_use_tao": bool(args.shape_use_tao),
        "shape_branching_penalty_values": [float(v) for v in args.shape_branching_penalty_values],
        "shape_tao_reg_values": [float(v) for v in args.shape_tao_reg_values],
        "shape_tao_n_runs": int(args.shape_tao_n_runs),
        "shape_tao_pair_scale": float(args.shape_tao_pair_scale),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    selected_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cache_manifest_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        for seed in seeds:
            protocol = load_cached_protocol(
                dataset=dataset,
                seed=int(seed),
                test_size=float(args.test_size),
                val_size=float(args.val_size),
                max_bins=int(args.max_bins),
                binner_min_samples_leaf=int(args.binner_min_samples_leaf),
                cache_min_child_size=int(args.cache_min_child_size),
                cache_version=int(args.cache_version),
            )
            cache_manifest_rows.append(cached_protocol_manifest_row(protocol))
            split_list = _make_split_list(
                np.asarray(protocol.cache["y_fit"], dtype=np.int32),
                cv_folds=int(args.cv_folds),
                seed=int(protocol.seed),
            )

            for depth in depths:
                print(f"[gridcv] dataset={dataset} seed={seed} depth={depth} -> MSPLIT", flush=True)
                msplit_grid_rows, msplit_fold_rows, msplit_search_seconds = _evaluate_grid(
                    candidate_fn=_evaluate_msplit_candidate,
                    protocol=protocol,
                    depth=int(depth),
                    params_grid=_msplit_param_grid(args, int(depth)),
                    split_list=split_list,
                    search_jobs=int(search_jobs),
                )
                best_msplit = _select_best_grid_row(msplit_grid_rows)
                with timing_guard_scope(enabled=bool(final_fit_guard_enabled)):
                    msplit_final = run_cached_msplit(
                        cache=protocol.cache,
                        depth=int(depth),
                        lookahead_depth=int(best_msplit["params"]["lookahead_depth"]),
                        reg=float(best_msplit["params"]["reg"]),
                        exactify_top_k=int(best_msplit["params"]["exactify_top_k"]),
                        min_split_size=int(best_msplit["params"]["min_split_size"]),
                        min_child_size=int(best_msplit["params"]["min_child_size"]),
                        max_branching=int(best_msplit["params"]["max_branching"]),
                    )
                msplit_tree_artifact_path, msplit_metrics_path = write_msplit_artifacts(
                    out_dir=out_dir,
                    protocol=protocol,
                    depth=int(depth),
                    best_params=best_msplit["params"],
                    final_result=msplit_final,
                )
                selected_rows.append(
                    {
                        "dataset": protocol.dataset,
                        "seed": int(protocol.seed),
                        "depth_budget": int(depth),
                        "algorithm": "msplit",
                        "train_accuracy": float(msplit_final["train_accuracy"]),
                        "val_accuracy": float(best_msplit["cv_val_accuracy_mean"]),
                        "test_accuracy": float(msplit_final["test_accuracy"]),
                        "search_time_sec": float(msplit_search_seconds),
                        "search_jobs": int(search_jobs),
                        "cv_folds": int(args.cv_folds),
                        "n_grid_candidates": int(len(msplit_grid_rows)),
                        "cv_val_accuracy_mean": float(best_msplit["cv_val_accuracy_mean"]),
                        "cv_val_accuracy_std": float(best_msplit["cv_val_accuracy_std"]),
                        "cv_model_fit_time_mean": float(best_msplit["cv_model_fit_time_mean"]),
                        "cv_model_fit_time_std": float(best_msplit["cv_model_fit_time_std"]),
                        "n_internal_nodes": int(msplit_final["n_internal"]),
                        "n_leaves": int(msplit_final["n_leaves"]),
                        "selected_params_json": json.dumps(json_safe(best_msplit["params"]), sort_keys=True),
                        "objective": float(msplit_final["objective"]),
                        "cache_path": str(protocol.cache_path),
                        "cache_requested_path": str(protocol.requested_cache_path),
                        "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
                        "tree_artifact_path": msplit_tree_artifact_path,
                        "model_metrics_path": msplit_metrics_path,
                        **benchmark_timing_fields(
                            algorithm="msplit",
                            model_fit_time_sec=float(msplit_final["fit_seconds"]),
                            shared_cache_build_seconds=float(protocol.cache_build_seconds),
                        ),
                    }
                )
                for row in msplit_grid_rows:
                    grid_rows.append(
                        {
                            "dataset": protocol.dataset,
                            "seed": int(protocol.seed),
                            "depth_budget": int(depth),
                            "algorithm": "msplit",
                            "candidate_index": int(row["candidate_index"]),
                            "cv_val_accuracy_mean": float(row["cv_val_accuracy_mean"]),
                            "cv_val_accuracy_std": float(row["cv_val_accuracy_std"]),
                            "cv_model_fit_time_mean": float(row["cv_model_fit_time_mean"]),
                            "cv_model_fit_time_std": float(row["cv_model_fit_time_std"]),
                            "params_json": json.dumps(json_safe(row["params"]), sort_keys=True),
                        }
                    )
                fold_rows.extend(msplit_fold_rows)

                print(f"[gridcv] dataset={dataset} seed={seed} depth={depth} -> ShapeCART", flush=True)
                shapecart_grid_rows, shapecart_fold_rows, shapecart_search_seconds = _evaluate_grid(
                    candidate_fn=_evaluate_shapecart_candidate,
                    protocol=protocol,
                    depth=int(depth),
                    params_grid=_shapecart_param_grid(args),
                    split_list=split_list,
                    search_jobs=int(search_jobs),
                )
                best_shapecart = _select_best_grid_row(shapecart_grid_rows)
                shapecart_params = dict(best_shapecart["params"])
                with timing_guard_scope(enabled=bool(final_fit_guard_enabled)):
                    shapecart_final = fit_shapecart_once(
                        cache=protocol.cache,
                        depth=int(depth),
                        random_state=int(protocol.seed),
                        params=shapecart_params,
                        include_model=True,
                    )
                shapecart_tree_artifact_path, shapecart_metrics_path = write_shapecart_artifacts(
                    out_dir=out_dir,
                    protocol=protocol,
                    depth=int(depth),
                    final_params=shapecart_params,
                    final_result=shapecart_final,
                )
                selected_rows.append(
                    {
                        "dataset": protocol.dataset,
                        "seed": int(protocol.seed),
                        "depth_budget": int(depth),
                        "algorithm": "shapecart_tao" if bool(args.shape_use_tao) else "shapecart",
                        "train_accuracy": float(shapecart_final["train_accuracy"]),
                        "val_accuracy": float(best_shapecart["cv_val_accuracy_mean"]),
                        "test_accuracy": float(shapecart_final["test_accuracy"]),
                        "search_time_sec": float(shapecart_search_seconds),
                        "search_jobs": int(search_jobs),
                        "cv_folds": int(args.cv_folds),
                        "n_grid_candidates": int(len(shapecart_grid_rows)),
                        "cv_val_accuracy_mean": float(best_shapecart["cv_val_accuracy_mean"]),
                        "cv_val_accuracy_std": float(best_shapecart["cv_val_accuracy_std"]),
                        "cv_model_fit_time_mean": float(best_shapecart["cv_model_fit_time_mean"]),
                        "cv_model_fit_time_std": float(best_shapecart["cv_model_fit_time_std"]),
                        "n_internal_nodes": int(shapecart_final["n_internal_nodes"]),
                        "n_leaves": int(shapecart_final["n_leaves"]),
                        "selected_params_json": json.dumps(json_safe(shapecart_params), sort_keys=True),
                        "objective": np.nan,
                        "cache_path": str(protocol.cache_path),
                        "cache_requested_path": str(protocol.requested_cache_path),
                        "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
                        "tree_artifact_path": shapecart_tree_artifact_path,
                        "model_metrics_path": shapecart_metrics_path,
                        **benchmark_timing_fields(
                            algorithm="shapecart",
                            model_fit_time_sec=float(shapecart_final["fit_seconds"]),
                            shared_cache_build_seconds=float(protocol.cache_build_seconds),
                        ),
                    }
                )
                for row in shapecart_grid_rows:
                    grid_rows.append(
                        {
                            "dataset": protocol.dataset,
                            "seed": int(protocol.seed),
                            "depth_budget": int(depth),
                            "algorithm": "shapecart_tao" if bool(args.shape_use_tao) else "shapecart",
                            "candidate_index": int(row["candidate_index"]),
                            "cv_val_accuracy_mean": float(row["cv_val_accuracy_mean"]),
                            "cv_val_accuracy_std": float(row["cv_val_accuracy_std"]),
                            "cv_model_fit_time_mean": float(row["cv_model_fit_time_mean"]),
                            "cv_model_fit_time_std": float(row["cv_model_fit_time_std"]),
                            "params_json": json.dumps(json_safe(row["params"]), sort_keys=True),
                        }
                    )
                fold_rows.extend(shapecart_fold_rows)
                _write_progress_tables(
                    out_dir=out_dir,
                    selected_rows=selected_rows,
                    grid_rows=grid_rows,
                    fold_rows=fold_rows,
                    cache_manifest_rows=cache_manifest_rows,
                )

    _write_progress_tables(
        out_dir=out_dir,
        selected_rows=selected_rows,
        grid_rows=grid_rows,
        fold_rows=fold_rows,
        cache_manifest_rows=cache_manifest_rows,
    )

    summary_df = aggregate_results(selected_rows)
    write_csv_tables(out_dir, {"summary_by_depth.csv": summary_df})
    best_depth_df = best_depth_table(summary_df)
    write_csv_tables(out_dir, {"best_depth_by_dataset.csv": best_depth_df})

    print(f"[done] wrote results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
