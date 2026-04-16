#!/usr/bin/env python3
"""Support helpers for the nested-CV Optuna benchmark runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
import json
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold

from benchmark.scripts.benchmark_cached_common import (
    CachedProtocol,
    DEFAULT_CV_FOLDS,
    DEFAULT_DEPTHS,
    DEFAULT_MSPLIT_EXACTIFY_TOP_K_VALUES,
    DEFAULT_MSPLIT_LOOKAHEAD_DEPTH_VALUES,
    DEFAULT_MSPLIT_MAX_BRANCHING_VALUES,
    DEFAULT_MSPLIT_REG_VALUES,
    DEFAULT_SHAPE_BRANCHING_PENALTY_VALUES,
    DEFAULT_SHAPE_CRITERION_VALUES,
    DEFAULT_SHAPE_INNER_MAX_LEAF_VALUES,
    DEFAULT_SHAPE_INNER_MIN_LEAF_VALUES,
    DEFAULT_SHAPE_TAO_REG_VALUES,
    DEFAULT_SHARED_MIN_LEAF_VALUES,
    DEFAULT_SHARED_SPLIT_MULTIPLIERS,
    coerce_numeric_token,
    json_safe,
    write_csv_tables,
)
from benchmark.scripts.benchmark_paths import BENCHMARK_ARTIFACTS_ROOT
from benchmark.scripts.cache_utils import (
    DEFAULT_LIGHTGBM_BINNING_KWARGS,
    DEFAULT_LGB_NUM_THREADS,
    DEFAULT_CACHE_VERSION,
    DEFAULT_MAX_BINS,
    DEFAULT_MIN_CHILD_SIZE,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
)
from benchmark.scripts.experiment_utils import (
    DATASET_LOADERS,
    canonical_dataset_list,
    encode_target,
    make_preprocessor,
)
from benchmark.scripts.lightgbm_binning import fit_lightgbm_binner, serialize_lightgbm_binner

DEFAULT_OUTER_FOLDS = 3
DEFAULT_FOLD_SEED = 42


@dataclass(frozen=True)
class FoldRun:
    dataset: str
    fold_index: int
    outer_folds: int
    final_protocol: CachedProtocol
    tune_protocols: list[CachedProtocol]
    manifest_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MSPLIT and ShapeCART with nested cross-validation and Bayesian optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["electricity"])
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Deprecated compatibility argument. Outer-fold evaluation now uses --fold-seed instead.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=DEFAULT_TEST_SIZE,
        help="Deprecated compatibility argument. Ignored by the nested-CV runner.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=DEFAULT_VAL_SIZE,
        help="Deprecated compatibility argument. Ignored by the nested-CV runner.",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="Inner CV folds used for Optuna scoring.")
    parser.add_argument("--search-jobs", type=int, default=None)
    parser.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS)
    parser.add_argument("--binner-min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--cache-min-child-size", type=int, default=DEFAULT_MIN_CHILD_SIZE)
    parser.add_argument("--cache-version", type=int, default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--timing-mode", choices=("fair", "fast"), default="fair")
    parser.add_argument("--msplit-build-dir", type=str, default="build-nonlinear-py")
    parser.add_argument("--msplit-worker-limit", type=int, default=1)
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
        default=str(BENCHMARK_ARTIFACTS_ROOT / "benchmarks" / "cached_optuna_msplit_vs_shapecart"),
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--persist-optuna-study",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Persist Optuna studies to SQLite. Disabled by default to avoid per-trial I/O overhead.",
    )
    parser.add_argument(
        "--save-model-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write tree artifacts/metrics for selected models after the benchmark completes.",
    )
    parser.add_argument(
        "--save-trials",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write full per-trial Optuna tables. Disabled by default to minimize benchmark overhead.",
    )
    parser.add_argument(
        "--save-cache-manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write fold/cache manifest tables. Disabled by default to minimize benchmark overhead.",
    )
    args = parser.parse_args()
    if int(args.n_trials) < 1:
        raise ValueError("--n-trials must be at least 1")
    if int(args.outer_folds) < 2:
        raise ValueError("--outer-folds must be at least 2")
    if int(args.cv_folds) < 2:
        raise ValueError("--cv-folds must be at least 2")
    if int(args.shape_k) < 2:
        raise ValueError("--shape-k must be at least 2")
    args.msplit_reg_values = [float(coerce_numeric_token(v)) for v in args.msplit_reg_values]
    args.shape_inner_min_leaf_values = [coerce_numeric_token(v) for v in args.shape_inner_min_leaf_values]
    args.shape_branching_penalty_values = [float(coerce_numeric_token(v)) for v in args.shape_branching_penalty_values]
    args.shape_tao_reg_values = [float(coerce_numeric_token(v)) for v in args.shape_tao_reg_values]
    return args


def resolve_requested_run(args: argparse.Namespace) -> tuple[list[str], list[int], int, list[int]]:
    datasets = canonical_dataset_list(args.datasets)
    depths = sorted(set(int(v) for v in args.depths))
    fold_seed = int(args.fold_seed)
    deprecated_seed_args = [int(v) for v in args.seeds] if args.seeds else []
    if deprecated_seed_args and len(deprecated_seed_args) == 1 and int(args.fold_seed) == DEFAULT_FOLD_SEED:
        fold_seed = int(deprecated_seed_args[0])
    return datasets, depths, fold_seed, deprecated_seed_args


def build_run_config(
    *,
    args: argparse.Namespace,
    datasets: list[str],
    depths: list[int],
    fold_seed: int,
    search_jobs: int,
    final_fit_guard_enabled: bool,
) -> dict[str, Any]:
    return {
        "datasets": datasets,
        "depths": depths,
        "outer_folds": int(args.outer_folds),
        "fold_seed": int(fold_seed),
        "deprecated_seed_args": [int(v) for v in args.seeds] if args.seeds else [],
        "deprecated_test_size": float(args.test_size),
        "deprecated_val_size": float(args.val_size),
        "n_trials": int(args.n_trials),
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
        "msplit_worker_limit": int(args.msplit_worker_limit),
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
        "persist_optuna_study": bool(args.persist_optuna_study),
        "save_model_artifacts": bool(args.save_model_artifacts),
        "save_trials": bool(args.save_trials),
        "save_cache_manifest": bool(args.save_cache_manifest),
    }


def write_result_tables(
    *,
    out_dir: Path,
    selected_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]] | None = None,
    cache_manifest_rows: list[dict[str, Any]] | None = None,
) -> None:
    is_msplit = lambda row: str(row.get("algorithm", "")) == "msplit"
    is_shapecart = lambda row: str(row.get("algorithm", "")).startswith("shapecart")
    tables: dict[str, Any] = {
        "selected_results.csv": selected_rows,
        "msplit_selected_results.csv": [row for row in selected_rows if is_msplit(row)],
        "shapecart_selected_results.csv": [row for row in selected_rows if is_shapecart(row)],
    }
    if trial_rows is not None:
        tables["tuning_trials.csv"] = trial_rows
        tables["msplit_tuning_trials.csv"] = [row for row in trial_rows if is_msplit(row)]
        tables["shapecart_tuning_trials.csv"] = [row for row in trial_rows if is_shapecart(row)]
    if cache_manifest_rows is not None:
        tables["cache_manifest.csv"] = cache_manifest_rows
    write_csv_tables(out_dir, tables)


def append_progress_log(out_dir: Path, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    print(message, flush=True)
    with (out_dir / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line)


def checkpoint_benchmark_state(
    *,
    out_dir: Path,
    selected_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]] | None,
    cache_manifest_rows: list[dict[str, Any]] | None,
    aggregate_results,
    best_depth_table,
    write_csv_tables,
) -> None:
    write_result_tables(
        out_dir=out_dir,
        selected_rows=selected_rows,
        trial_rows=trial_rows,
        cache_manifest_rows=cache_manifest_rows,
    )
    if not selected_rows:
        return
    summary_df = aggregate_results(selected_rows)
    write_csv_tables(out_dir, {"summary_by_depth.csv": summary_df})
    write_csv_tables(out_dir, {"best_depth_by_dataset.csv": best_depth_table(summary_df)})


def study_storage_uri(*, out_dir: Path, search_jobs: int, persist_study: bool) -> str | None:
    if not persist_study or int(search_jobs) > 1:
        return None
    return f"sqlite:///{(out_dir / 'optuna_studies.sqlite3').resolve()}"


def iter_fold_runs(
    *,
    args: argparse.Namespace,
    datasets: list[str],
    fold_seed: int,
    out_dir: Path,
    collect_manifest: bool,
) -> Iterator[FoldRun]:
    for dataset in datasets:
        x, y = DATASET_LOADERS[dataset]()
        y_encoded, class_labels, _ = encode_target(y)
        y_encoded = np.asarray(y_encoded, dtype=np.int32)
        class_labels = np.asarray(class_labels, dtype=object)
        outer_splits = _make_split_list(y_encoded, cv_folds=int(args.outer_folds), seed=int(fold_seed))
        for outer_zero, (outer_train_idx, outer_test_idx) in enumerate(outer_splits, start=1):
            final_protocol = _build_protocol_from_indices(
                dataset=dataset,
                x=x,
                y_encoded=y_encoded,
                class_labels=class_labels,
                idx_fit=np.asarray(outer_train_idx, dtype=np.int32),
                idx_val=np.asarray(outer_test_idx, dtype=np.int32),
                idx_test=np.asarray(outer_test_idx, dtype=np.int32),
                split_seed=int(fold_seed),
                max_bins=int(args.max_bins),
                binner_min_samples_leaf=int(args.binner_min_samples_leaf),
                cache_min_child_size=int(args.cache_min_child_size),
                cache_version=int(args.cache_version),
                cache_label=str(out_dir / "_virtual_cache_refs" / f"{dataset}_outer_fold{outer_zero}_final.npz"),
                teacher_uses_validation=False,
            )
            manifest_rows: list[dict[str, Any]] = []
            if collect_manifest:
                manifest_rows.append(
                    _protocol_manifest_row(
                        protocol=final_protocol,
                        outer_fold_index=int(outer_zero),
                        outer_folds=int(args.outer_folds),
                        role="outer_final",
                    )
                )
            inner_splits = _make_split_list(
                y_encoded[np.asarray(outer_train_idx, dtype=np.int32)],
                cv_folds=int(args.cv_folds),
                seed=int(fold_seed) + 1000 * int(outer_zero),
            )
            tune_protocols: list[CachedProtocol] = []
            for inner_zero, (inner_fit_local, inner_val_local) in enumerate(inner_splits, start=1):
                inner_fit_idx = np.asarray(outer_train_idx[np.asarray(inner_fit_local, dtype=np.int32)], dtype=np.int32)
                inner_val_idx = np.asarray(outer_train_idx[np.asarray(inner_val_local, dtype=np.int32)], dtype=np.int32)
                protocol = _build_protocol_from_indices(
                    dataset=dataset,
                    x=x,
                    y_encoded=y_encoded,
                    class_labels=class_labels,
                    idx_fit=inner_fit_idx,
                    idx_val=inner_val_idx,
                    idx_test=inner_val_idx,
                    split_seed=int(fold_seed),
                    max_bins=int(args.max_bins),
                    binner_min_samples_leaf=int(args.binner_min_samples_leaf),
                    cache_min_child_size=int(args.cache_min_child_size),
                    cache_version=int(args.cache_version),
                    cache_label=str(
                        out_dir / "_virtual_cache_refs" / f"{dataset}_outer_fold{outer_zero}_inner_fold{inner_zero}.npz"
                    ),
                    teacher_uses_validation=True,
                )
                tune_protocols.append(protocol)
                if collect_manifest:
                    manifest_rows.append(
                        _protocol_manifest_row(
                            protocol=protocol,
                            outer_fold_index=int(outer_zero),
                            outer_folds=int(args.outer_folds),
                            role="inner_tuning",
                            inner_fold_index=int(inner_zero),
                        )
                    )
            yield FoldRun(
                dataset=dataset,
                fold_index=int(outer_zero),
                outer_folds=int(args.outer_folds),
                final_protocol=final_protocol,
                tune_protocols=tune_protocols,
                manifest_rows=manifest_rows,
            )


def _slice_rows(x, idx: np.ndarray):
    return x.iloc[idx] if hasattr(x, "iloc") else x[idx]


def _make_split_list(y_fit: np.ndarray, *, cv_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=int(seed))
    dummy = np.zeros(y_fit.shape[0], dtype=np.int32)
    return [(np.asarray(fit_idx, dtype=np.int32), np.asarray(val_idx, dtype=np.int32)) for fit_idx, val_idx in splitter.split(dummy, y_fit)]


def _build_protocol_from_indices(
    *,
    dataset: str,
    x,
    y_encoded: np.ndarray,
    class_labels: np.ndarray,
    idx_fit: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    split_seed: int,
    max_bins: int,
    binner_min_samples_leaf: int,
    cache_min_child_size: int,
    cache_version: int,
    cache_label: str,
    teacher_uses_validation: bool,
    lgb_num_threads: int = DEFAULT_LGB_NUM_THREADS,
) -> CachedProtocol:
    x_fit, x_val, x_test = (_slice_rows(x, idx) for idx in (idx_fit, idx_val, idx_test))
    y_fit = np.asarray(y_encoded[idx_fit], dtype=np.int32)
    y_val = np.asarray(y_encoded[idx_val], dtype=np.int32)
    y_test = np.asarray(y_encoded[idx_test], dtype=np.int32)
    pre = make_preprocessor(x_fit)
    x_fit_proc = np.asarray(pre.fit_transform(x_fit), dtype=np.float32)
    x_val_proc = np.asarray(pre.transform(x_val), dtype=np.float32)
    x_test_proc = np.asarray(pre.transform(x_test), dtype=np.float32)
    started = time.perf_counter()
    binner = fit_lightgbm_binner(
        x_fit_proc,
        y_fit,
        X_val=x_val_proc if teacher_uses_validation else None,
        y_val=y_val if teacher_uses_validation else None,
        max_bins=int(max_bins),
        min_samples_leaf=int(binner_min_samples_leaf),
        random_state=int(split_seed),
        min_data_in_leaf=int(binner_min_samples_leaf),
        num_threads=int(lgb_num_threads),
        **DEFAULT_LIGHTGBM_BINNING_KWARGS,
    )
    build_seconds = time.perf_counter() - started
    arrays = {
        "idx_fit": np.asarray(idx_fit, dtype=np.int32),
        "idx_val": np.asarray(idx_val, dtype=np.int32),
        "idx_test": np.asarray(idx_test, dtype=np.int32),
        "X_fit_proc": x_fit_proc,
        "X_val_proc": x_val_proc,
        "X_test_proc": x_test_proc,
        "y_fit": y_fit,
        "y_val": y_val,
        "y_test": y_test,
        "Z_fit": np.asarray(binner.transform(x_fit_proc), dtype=np.int32),
        "Z_val": np.asarray(binner.transform(x_val_proc), dtype=np.int32),
        "Z_test": np.asarray(binner.transform(x_test_proc), dtype=np.int32),
        "teacher_logit": np.asarray(getattr(binner, "teacher_train_logit"), dtype=np.float64),
        "teacher_boundary_gain": np.asarray(getattr(binner, "boundary_gain_per_feature"), dtype=np.float64),
        "teacher_boundary_cover": np.asarray(getattr(binner, "boundary_cover_per_feature"), dtype=np.float64),
        "teacher_boundary_value_jump": np.asarray(getattr(binner, "boundary_value_jump_per_feature"), dtype=np.float64),
        "feature_names": np.asarray(pre.get_feature_names_out(), dtype=str),
        "class_labels": np.asarray([str(label) for label in class_labels], dtype=str),
        **serialize_lightgbm_binner(binner),
    }
    n_total = int(len(y_encoded))
    cache_meta = {
        "dataset": dataset,
        "seed": int(split_seed),
        "test_size": float(len(idx_test) / max(1, n_total)),
        "val_size": float(len(idx_val) / max(1, n_total)),
        "max_bins": int(max_bins),
        "min_samples_leaf": int(binner_min_samples_leaf),
        "min_child_size": int(cache_min_child_size),
        "lgb_num_threads": int(lgb_num_threads),
        "build_seconds": float(build_seconds),
        "cache_file": cache_label,
        "cache_bytes": 0,
        "n_rows": int(n_total),
        "n_fit": int(len(idx_fit)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "n_features_preprocessed": int(x_fit_proc.shape[1]),
        "class_count": int(len(class_labels)),
        "class_labels": [str(label) for label in class_labels],
        "cache_version": int(cache_version),
    }
    virtual_path = Path(cache_label)
    return CachedProtocol(
        dataset=dataset,
        seed=int(split_seed),
        requested_test_size=float(cache_meta["test_size"]),
        requested_val_size=float(cache_meta["val_size"]),
        requested_max_bins=int(max_bins),
        requested_binner_min_samples_leaf=int(binner_min_samples_leaf),
        requested_cache_min_child_size=int(cache_min_child_size),
        cache_path=virtual_path,
        cache=arrays,
        cache_meta=cache_meta,
        cache_used_fallback=False,
        requested_cache_path=virtual_path,
    )


def _protocol_manifest_row(
    *,
    protocol: CachedProtocol,
    outer_fold_index: int,
    outer_folds: int,
    role: str,
    inner_fold_index: int | None = None,
) -> dict[str, Any]:
    row = {
        "dataset": protocol.dataset,
        "split_seed": int(protocol.seed),
        "outer_fold_index": int(outer_fold_index),
        "outer_folds": int(outer_folds),
        "role": str(role),
        "cache_path": str(protocol.cache_path),
        "cache_requested_path": str(protocol.requested_cache_path),
        "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
        "test_size": float(protocol.test_size),
        "val_size": float(protocol.val_size),
        "max_bins": int(protocol.max_bins),
        "binner_min_samples_leaf": int(protocol.binner_min_samples_leaf),
        "cache_min_child_size": int(protocol.cache_min_child_size),
        "cache_build_seconds": float(protocol.cache_build_seconds),
        "n_fit": int(protocol.n_fit),
        "n_val": int(protocol.n_val),
        "n_test": int(protocol.n_test),
    }
    if inner_fold_index is not None:
        row["inner_fold_index"] = int(inner_fold_index)
    return row
