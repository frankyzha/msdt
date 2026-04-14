#!/usr/bin/env python3
"""Tune cached MSPLIT and ShapeCART with Bayesian optimization.

This runner does one job only:

1. Load an existing cached benchmark split from ``benchmark/cache``.
2. Tune MSPLIT and ShapeCART independently for each ``(dataset, seed, depth)``.
3. Refit the selected configuration on the cached fit split.
4. Write paper-ready tables plus tree artifacts for the selected trees.

The cache is required. If the requested cached LightGBM protocol artifact is
missing, the run fails instead of rebuilding it inside the benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import optuna
except Exception as exc:  # pragma: no cover - CLI guard
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None

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
    json_safe,
    load_cached_protocol,
    resolve_search_jobs,
    timing_guard_scope,
    write_csv_tables,
    fit_shapecart_once,
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
    trial_rows: list[dict[str, Any]],
    cache_manifest_rows: list[dict[str, Any]],
) -> None:
    msplit_selected_rows = [row for row in selected_rows if str(row.get("algorithm", "")) == "msplit"]
    shapecart_selected_rows = [
        row for row in selected_rows if str(row.get("algorithm", "")).startswith("shapecart")
    ]
    msplit_trial_rows = [row for row in trial_rows if str(row.get("algorithm", "")) == "msplit"]
    shapecart_trial_rows = [
        row for row in trial_rows if str(row.get("algorithm", "")).startswith("shapecart")
    ]
    write_csv_tables(
        out_dir,
        {
            "selected_results.csv": selected_rows,
            "msplit_selected_results.csv": msplit_selected_rows,
            "shapecart_selected_results.csv": shapecart_selected_rows,
            "tuning_trials.csv": trial_rows,
            "msplit_tuning_trials.csv": msplit_trial_rows,
            "shapecart_tuning_trials.csv": shapecart_trial_rows,
            "cache_manifest.csv": cache_manifest_rows,
        },
    )


def _objective_result_or_failure(*, objective_fn, trial) -> float:
    try:
        value = float(objective_fn(trial))
        trial.set_user_attr("status", "ok")
        return value
    except Exception as exc:  # pragma: no cover - CLI guard
        trial.set_user_attr("status", "error")
        trial.set_user_attr("error", repr(exc))
        return -1.0


def _best_trial_payload(study: "optuna.study.Study") -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for trial in study.trials:
        if trial.user_attrs.get("status") != "ok":
            error = str(trial.user_attrs.get("error", "")).strip()
            if error:
                errors.append(f"trial {int(trial.number)}: {error}")
            continue
        if trial.value is None:
            continue
        candidates.append(
            {
                "number": int(trial.number),
                "val_accuracy": float(trial.value),
                "fit_seconds": float(trial.user_attrs.get("fit_seconds", math.inf)),
                "params": {str(k): json_safe(v) for k, v in trial.params.items()},
            }
        )
    if not candidates:
        if errors:
            raise RuntimeError("No successful trials were completed. Errors: " + " | ".join(errors[:5]))
        raise RuntimeError("No successful trials were completed.")
    candidates.sort(
        key=lambda row: (
            -float(row["val_accuracy"]),
            float(row["fit_seconds"]),
            int(row["number"]),
        )
    )
    return candidates[0]


def _study_trials_to_rows(
    *,
    study: "optuna.study.Study",
    dataset: str,
    seed: int,
    depth: int,
    algorithm: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        rows.append(
            {
                "dataset": dataset,
                "seed": int(seed),
                "depth_budget": int(depth),
                "algorithm": algorithm,
                "trial_number": int(trial.number),
                "status": str(trial.user_attrs.get("status", "unknown")),
                "val_accuracy": float(trial.value) if trial.value is not None else np.nan,
                "cv_val_accuracy_std": float(trial.user_attrs.get("cv_val_accuracy_std", np.nan)),
                "fit_seconds": float(trial.user_attrs.get("fit_seconds", np.nan)),
                "cv_model_fit_time_std": float(trial.user_attrs.get("cv_model_fit_time_std", np.nan)),
                "train_accuracy": float(trial.user_attrs.get("train_accuracy", np.nan)),
                "test_accuracy": float(trial.user_attrs.get("test_accuracy", np.nan)),
                "n_internal_nodes": float(trial.user_attrs.get("n_internal_nodes", np.nan)),
                "n_leaves": float(trial.user_attrs.get("n_leaves", np.nan)),
                "params_json": json.dumps({str(k): json_safe(v) for k, v in trial.params.items()}, sort_keys=True),
                "error": str(trial.user_attrs.get("error", "")),
            }
        )
    return rows


def _study_trial_by_number(study: "optuna.study.Study", trial_number: int):
    for trial in study.trials:
        if int(trial.number) == int(trial_number):
            return trial
    raise KeyError(f"Trial number {trial_number} not found in study.")


def _study_storage_uri(*, out_dir: Path, search_jobs: int) -> str | None:
    # SQLite is not reliable under Optuna's multi-threaded `n_jobs` path on the
    # cluster. Use in-memory studies for parallel search to avoid `database is
    # locked` failures; TPE still learns from completed trials within the run.
    if int(search_jobs) > 1:
        return None
    return f"sqlite:///{(out_dir / 'optuna_studies.sqlite3').resolve()}"


def _subset_cache_for_msplit(
    *,
    cache: dict[str, np.ndarray],
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "X_fit_proc": np.asarray(cache["X_fit_proc"][fit_idx], dtype=np.float32),
        "X_val_proc": np.asarray(cache["X_fit_proc"][val_idx], dtype=np.float32),
        "X_test_proc": np.asarray(cache["X_fit_proc"][val_idx], dtype=np.float32),
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
        "feature_names": np.asarray(cache["feature_names"], dtype=object),
        "class_labels": np.asarray(cache["class_labels"], dtype=object),
        "idx_fit": np.asarray(cache["idx_fit"][fit_idx], dtype=np.int32),
        "idx_val": np.asarray(cache["idx_fit"][val_idx], dtype=np.int32),
        "idx_test": np.asarray(cache["idx_fit"][val_idx], dtype=np.int32),
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


def _suggest_discrete_int(
    *,
    trial,
    name: str,
    values: list[int],
    lower_bound: int | None = None,
    upper_bound: int | None = None,
) -> int:
    allowed = sorted({int(v) for v in values})
    if lower_bound is not None:
        allowed = [int(v) for v in allowed if int(v) >= int(lower_bound)]
    if upper_bound is not None:
        allowed = [int(v) for v in allowed if int(v) <= int(upper_bound)]
    if not allowed:
        raise ValueError(f"No allowed values remain for {name}")
    return int(trial.suggest_categorical(name, allowed))


def _suggest_nonnegative_float_range(*, trial, name: str, values: list[float]) -> float:
    sorted_values = sorted(float(v) for v in values)
    allow_zero = any(float(v) == 0.0 for v in sorted_values)
    positive_values = [float(v) for v in sorted_values if float(v) > 0.0]
    if not positive_values:
        return 0.0
    if allow_zero:
        mode = trial.suggest_categorical(f"{name}_mode", ["zero", "positive"])
        if mode == "zero":
            return 0.0
    lo = float(min(positive_values))
    hi = float(max(positive_values))
    if math.isclose(lo, hi):
        return float(trial.suggest_categorical(f"{name}_positive", [float(lo)]))
    return float(trial.suggest_float(f"{name}_positive", lo, hi, log=True))


def _decode_nonnegative_float_param(best_params: dict[str, Any], name: str) -> float:
    mode = str(best_params.get(f"{name}_mode", "")).strip().lower()
    if mode == "zero":
        return 0.0
    if f"{name}_positive" in best_params:
        return float(best_params[f"{name}_positive"])
    if name in best_params:
        return float(best_params[name])
    return 0.0


def _suggest_shape_inner_min_samples_leaf(*, trial, values: list[int | float]) -> int | float:
    int_values = sorted(int(v) for v in values if isinstance(v, int))
    frac_values = sorted(float(v) for v in values if isinstance(v, float))
    if int_values and frac_values:
        mode = trial.suggest_categorical("inner_min_samples_leaf_mode", ["count", "fraction"])
        if mode == "count":
            return _suggest_discrete_int(
                trial=trial,
                name="inner_min_samples_leaf_count",
                values=int_values,
            )
        lo = float(min(frac_values))
        hi = float(max(frac_values))
        if math.isclose(lo, hi):
            return float(trial.suggest_categorical("inner_min_samples_leaf_fraction", [float(lo)]))
        return float(trial.suggest_float("inner_min_samples_leaf_fraction", lo, hi, log=True))
    if int_values:
        return _suggest_discrete_int(
            trial=trial,
            name="inner_min_samples_leaf_count",
            values=int_values,
        )
    lo = float(min(frac_values))
    hi = float(max(frac_values))
    if math.isclose(lo, hi):
        return float(trial.suggest_categorical("inner_min_samples_leaf_fraction", [float(lo)]))
    return float(trial.suggest_float("inner_min_samples_leaf_fraction", lo, hi, log=True))


def _tune_msplit(
    *,
    protocol,
    depth: int,
    split_list: list[tuple[np.ndarray, np.ndarray]],
    cv_folds: int,
    n_trials: int,
    search_jobs: int,
    final_fit_guard_enabled: bool,
    sampler_seed: int,
    study_storage_uri: str | None,
    study_name: str,
    shared_min_leaf_values: list[int],
    shared_split_multipliers: list[int],
    msplit_lookahead_depth_values: list[int],
    msplit_exactify_top_k_values: list[int],
    msplit_max_branching_values: list[int],
    msplit_reg_values: list[float],
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    study_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "sampler": sampler,
    }
    if study_storage_uri is not None:
        study_kwargs.update(
            {
                "storage": study_storage_uri,
                "study_name": study_name,
                "load_if_exists": True,
            }
        )
    study = optuna.create_study(**study_kwargs)

    def _objective(trial) -> float:
        min_child_size = _suggest_discrete_int(
            trial=trial,
            name="min_child_size",
            values=shared_min_leaf_values,
        )
        split_multiplier = _suggest_discrete_int(
            trial=trial,
            name="min_split_multiplier",
            values=shared_split_multipliers,
        )
        min_split_size = int(min_child_size * split_multiplier)
        lookahead_depth = _suggest_discrete_int(
            trial=trial,
            name="lookahead_depth",
            values=msplit_lookahead_depth_values,
            upper_bound=int(depth),
        )
        exactify_top_k = _suggest_discrete_int(
            trial=trial,
            name="exactify_top_k",
            values=msplit_exactify_top_k_values,
        )
        max_branching = _suggest_discrete_int(
            trial=trial,
            name="max_branching",
            values=msplit_max_branching_values,
        )
        reg = _suggest_nonnegative_float_range(
            trial=trial,
            name="reg",
            values=msplit_reg_values,
        )
        fold_val_acc: list[float] = []
        fold_fit_times: list[float] = []
        fold_train_acc: list[float] = []
        fold_n_internal: list[int] = []
        fold_n_leaves: list[int] = []
        fold_objectives: list[float] = []
        for fit_idx, val_idx in split_list:
            fold_cache = _subset_cache_for_msplit(cache=protocol.cache, fit_idx=fit_idx, val_idx=val_idx)
            result = run_cached_msplit(
                cache=fold_cache,
                depth=int(depth),
                lookahead_depth=int(lookahead_depth),
                reg=float(reg),
                exactify_top_k=int(exactify_top_k),
                min_split_size=int(min_split_size),
                min_child_size=int(min_child_size),
                max_branching=int(max_branching),
            )
            val_accuracy = result.get("val_accuracy")
            if val_accuracy is None:
                raise RuntimeError("Validation accuracy missing from MSPLIT result.")
            fold_val_acc.append(float(val_accuracy))
            fold_fit_times.append(float(result["fit_seconds"]))
            fold_train_acc.append(float(result["train_accuracy"]))
            fold_n_internal.append(int(result["n_internal"]))
            fold_n_leaves.append(int(result["n_leaves"]))
            fold_objectives.append(float(result["objective"]))
        trial.set_user_attr("fit_seconds", float(np.mean(fold_fit_times)))
        trial.set_user_attr("cv_model_fit_time_std", float(np.std(fold_fit_times, ddof=0)))
        trial.set_user_attr("train_accuracy", float(np.mean(fold_train_acc)))
        trial.set_user_attr("test_accuracy", np.nan)
        trial.set_user_attr("n_internal_nodes", float(np.mean(fold_n_internal)))
        trial.set_user_attr("n_leaves", float(np.mean(fold_n_leaves)))
        trial.set_user_attr("objective", float(np.mean(fold_objectives)))
        trial.set_user_attr("cv_val_accuracy_std", float(np.std(fold_val_acc, ddof=0)))
        return float(np.mean(fold_val_acc))

    started = time.perf_counter()
    with timing_guard_scope(enabled=False):
        study.optimize(
            lambda trial: _objective_result_or_failure(objective_fn=_objective, trial=trial),
            n_trials=int(n_trials),
            n_jobs=int(search_jobs),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    search_seconds = time.perf_counter() - started
    best = _best_trial_payload(study)
    best_trial = _study_trial_by_number(study, int(best["number"]))
    best_params = dict(best["params"])
    final_params = {
        "min_child_size": int(best_params["min_child_size"]),
        "min_split_multiplier": int(best_params["min_split_multiplier"]),
        "lookahead_depth": int(best_params["lookahead_depth"]),
        "exactify_top_k": int(best_params["exactify_top_k"]),
        "max_branching": int(best_params["max_branching"]),
        "reg": float(_decode_nonnegative_float_param(best_params, "reg")),
    }
    with timing_guard_scope(enabled=bool(final_fit_guard_enabled)):
        final_result = run_cached_msplit(
            cache=protocol.cache,
            depth=int(depth),
            lookahead_depth=int(final_params["lookahead_depth"]),
            reg=float(final_params["reg"]),
            exactify_top_k=int(final_params["exactify_top_k"]),
            min_split_size=int(final_params["min_child_size"]) * int(final_params["min_split_multiplier"]),
            min_child_size=int(final_params["min_child_size"]),
            max_branching=int(final_params["max_branching"]),
        )
    tree_artifact_path, metrics_path = write_msplit_artifacts(
        out_dir=out_dir,
        protocol=protocol,
        depth=int(depth),
        best_params=final_params,
        final_result=final_result,
    )
    selected_row = {
        "dataset": protocol.dataset,
        "seed": int(protocol.seed),
        "depth_budget": int(depth),
        "algorithm": "msplit",
        "train_accuracy": float(final_result["train_accuracy"]),
        "val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": float(final_result["test_accuracy"]),
        "search_time_sec": float(search_seconds),
        "search_jobs": int(search_jobs),
        "cv_folds": int(cv_folds),
        "cv_val_accuracy_mean": float(best["val_accuracy"]),
        "cv_val_accuracy_std": float(best_trial.user_attrs.get("cv_val_accuracy_std", np.nan)),
        "cv_model_fit_time_mean": float(best_trial.user_attrs.get("fit_seconds", np.nan)),
        "cv_model_fit_time_std": float(best_trial.user_attrs.get("cv_model_fit_time_std", np.nan)),
        "n_internal_nodes": int(final_result["n_internal"]),
        "n_leaves": int(final_result["n_leaves"]),
        "best_trial_number": int(best["number"]),
        "n_trials": int(n_trials),
        "selected_params_json": json.dumps(json_safe(final_params), sort_keys=True),
        "objective": float(final_result["objective"]),
        "cache_path": str(protocol.cache_path),
        "cache_requested_path": str(protocol.requested_cache_path),
        "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
        "tree_artifact_path": tree_artifact_path,
        "model_metrics_path": metrics_path,
        **benchmark_timing_fields(
            algorithm="msplit",
            model_fit_time_sec=float(final_result["fit_seconds"]),
            shared_cache_build_seconds=float(protocol.cache_build_seconds),
        ),
    }
    trial_rows = _study_trials_to_rows(
        study=study,
        dataset=protocol.dataset,
        seed=int(protocol.seed),
        depth=int(depth),
        algorithm="msplit",
    )
    return selected_row, trial_rows


def _tune_shapecart(
    *,
    protocol,
    depth: int,
    split_list: list[tuple[np.ndarray, np.ndarray]],
    cv_folds: int,
    n_trials: int,
    search_jobs: int,
    final_fit_guard_enabled: bool,
    sampler_seed: int,
    study_storage_uri: str | None,
    study_name: str,
    shared_min_leaf_values: list[int],
    shared_split_multipliers: list[int],
    shape_criterion_values: list[str],
    shape_inner_max_depth: int,
    shape_inner_max_leaf_values: list[int],
    shape_inner_min_leaf_values: list[int | float],
    shape_k: int,
    shape_max_iter: int,
    shape_pairwise_candidates: float,
    shape_smart_init: bool,
    shape_random_pairs: bool,
    shape_use_dpdt: bool,
    shape_use_tao: bool,
    shape_branching_penalty_values: list[float],
    shape_tao_reg_values: list[float],
    shape_tao_n_runs: int,
    shape_tao_pair_scale: float,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    study_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "sampler": sampler,
    }
    if study_storage_uri is not None:
        study_kwargs.update(
            {
                "storage": study_storage_uri,
                "study_name": study_name,
                "load_if_exists": True,
            }
        )
    study = optuna.create_study(**study_kwargs)

    def _objective(trial) -> float:
        min_samples_leaf = _suggest_discrete_int(
            trial=trial,
            name="min_samples_leaf",
            values=shared_min_leaf_values,
        )
        split_multiplier = _suggest_discrete_int(
            trial=trial,
            name="min_split_multiplier",
            values=shared_split_multipliers,
        )
        min_samples_split = int(min_samples_leaf * split_multiplier)
        criterion = str(trial.suggest_categorical("criterion", list(shape_criterion_values)))
        inner_max_leaf_nodes = _suggest_discrete_int(
            trial=trial,
            name="inner_max_leaf_nodes",
            values=shape_inner_max_leaf_values,
        )
        inner_min_samples_leaf = _suggest_shape_inner_min_samples_leaf(
            trial=trial,
            values=shape_inner_min_leaf_values,
        )
        branching_penalty = 0.0
        if int(shape_k) > 2:
            branching_penalty = _suggest_nonnegative_float_range(
                trial=trial,
                name="branching_penalty",
                values=shape_branching_penalty_values,
            )
        tao_reg = 0.0
        if shape_use_tao:
            tao_reg = _suggest_nonnegative_float_range(
                trial=trial,
                name="tao_reg",
                values=shape_tao_reg_values,
            )
        params = {
            "min_samples_leaf": int(min_samples_leaf),
            "min_samples_split": int(min_samples_split),
            "criterion": criterion,
            "inner_max_depth": int(shape_inner_max_depth),
            "inner_max_leaf_nodes": int(inner_max_leaf_nodes),
            "inner_min_samples_leaf": inner_min_samples_leaf,
            "k": int(shape_k),
            "max_iter": int(shape_max_iter),
            "pairwise_candidates": float(shape_pairwise_candidates),
            "smart_init": bool(shape_smart_init),
            "random_pairs": bool(shape_random_pairs),
            "use_dpdt": bool(shape_use_dpdt),
            "use_tao": bool(shape_use_tao),
            "branching_penalty": float(branching_penalty),
            "tao_reg": float(tao_reg),
            "tao_n_runs": int(shape_tao_n_runs),
            "tao_pair_scale": float(shape_tao_pair_scale),
        }
        fold_val_acc: list[float] = []
        fold_fit_times: list[float] = []
        fold_train_acc: list[float] = []
        fold_n_internal: list[int] = []
        fold_n_leaves: list[int] = []
        for fold_index, (fit_idx, val_idx) in enumerate(split_list):
            fold_cache = _subset_cache_for_shapecart(cache=protocol.cache, fit_idx=fit_idx, val_idx=val_idx)
            result = fit_shapecart_once(
                cache=fold_cache,
                depth=int(depth),
                random_state=int(protocol.seed) + 100 * int(fold_index),
                params=params,
                include_model=False,
            )
            fold_val_acc.append(float(result["val_accuracy"]))
            fold_fit_times.append(float(result["fit_seconds"]))
            fold_train_acc.append(float(result["train_accuracy"]))
            fold_n_internal.append(int(result["n_internal_nodes"]))
            fold_n_leaves.append(int(result["n_leaves"]))
        trial.set_user_attr("fit_seconds", float(np.mean(fold_fit_times)))
        trial.set_user_attr("cv_model_fit_time_std", float(np.std(fold_fit_times, ddof=0)))
        trial.set_user_attr("train_accuracy", float(np.mean(fold_train_acc)))
        trial.set_user_attr("test_accuracy", np.nan)
        trial.set_user_attr("n_internal_nodes", float(np.mean(fold_n_internal)))
        trial.set_user_attr("n_leaves", float(np.mean(fold_n_leaves)))
        trial.set_user_attr("cv_val_accuracy_std", float(np.std(fold_val_acc, ddof=0)))
        return float(np.mean(fold_val_acc))

    started = time.perf_counter()
    with timing_guard_scope(enabled=False):
        study.optimize(
            lambda trial: _objective_result_or_failure(objective_fn=_objective, trial=trial),
            n_trials=int(n_trials),
            n_jobs=int(search_jobs),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    search_seconds = time.perf_counter() - started
    best = _best_trial_payload(study)
    best_trial = _study_trial_by_number(study, int(best["number"]))
    best_params = dict(best["params"])
    if "inner_min_samples_leaf_count" in best_params:
        inner_min_samples_leaf = int(best_params["inner_min_samples_leaf_count"])
    elif "inner_min_samples_leaf_fraction" in best_params:
        inner_min_samples_leaf = float(best_params["inner_min_samples_leaf_fraction"])
    else:
        inner_min_samples_leaf = best_params["inner_min_samples_leaf"]
    final_params = {
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "min_samples_split": int(best_params["min_samples_leaf"]) * int(best_params["min_split_multiplier"]),
        "criterion": str(best_params["criterion"]),
        "inner_max_depth": int(shape_inner_max_depth),
        "inner_max_leaf_nodes": int(best_params["inner_max_leaf_nodes"]),
        "inner_min_samples_leaf": inner_min_samples_leaf,
        "k": int(shape_k),
        "max_iter": int(shape_max_iter),
        "pairwise_candidates": float(shape_pairwise_candidates),
        "smart_init": bool(shape_smart_init),
        "random_pairs": bool(shape_random_pairs),
        "use_dpdt": bool(shape_use_dpdt),
        "use_tao": bool(shape_use_tao),
        "branching_penalty": float(_decode_nonnegative_float_param(best_params, "branching_penalty")),
        "tao_reg": float(_decode_nonnegative_float_param(best_params, "tao_reg")),
        "tao_n_runs": int(shape_tao_n_runs),
        "tao_pair_scale": float(shape_tao_pair_scale),
    }
    with timing_guard_scope(enabled=bool(final_fit_guard_enabled)):
        final_result = fit_shapecart_once(
            cache=protocol.cache,
            depth=int(depth),
            random_state=int(protocol.seed),
            params=final_params,
            include_model=True,
        )
    tree_artifact_path, metrics_path = write_shapecart_artifacts(
        out_dir=out_dir,
        protocol=protocol,
        depth=int(depth),
        final_params=final_params,
        final_result=final_result,
    )
    algorithm_name = "shapecart_tao" if shape_use_tao else "shapecart"
    selected_row = {
        "dataset": protocol.dataset,
        "seed": int(protocol.seed),
        "depth_budget": int(depth),
        "algorithm": algorithm_name,
        "train_accuracy": float(final_result["train_accuracy"]),
        "val_accuracy": float(best["val_accuracy"]),
        "test_accuracy": float(final_result["test_accuracy"]),
        "search_time_sec": float(search_seconds),
        "search_jobs": int(search_jobs),
        "cv_folds": int(cv_folds),
        "cv_val_accuracy_mean": float(best["val_accuracy"]),
        "cv_val_accuracy_std": float(best_trial.user_attrs.get("cv_val_accuracy_std", np.nan)),
        "cv_model_fit_time_mean": float(best_trial.user_attrs.get("fit_seconds", np.nan)),
        "cv_model_fit_time_std": float(best_trial.user_attrs.get("cv_model_fit_time_std", np.nan)),
        "n_internal_nodes": int(final_result["n_internal_nodes"]),
        "n_leaves": int(final_result["n_leaves"]),
        "best_trial_number": int(best["number"]),
        "n_trials": int(n_trials),
        "selected_params_json": json.dumps(json_safe(final_params), sort_keys=True),
        "objective": np.nan,
        "cache_path": str(protocol.cache_path),
        "cache_requested_path": str(protocol.requested_cache_path),
        "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
        "tree_artifact_path": tree_artifact_path,
        "model_metrics_path": metrics_path,
        **benchmark_timing_fields(
            algorithm="shapecart",
            model_fit_time_sec=float(final_result["fit_seconds"]),
            shared_cache_build_seconds=float(protocol.cache_build_seconds),
        ),
    }
    trial_rows = _study_trials_to_rows(
        study=study,
        dataset=protocol.dataset,
        seed=int(protocol.seed),
        depth=int(depth),
        algorithm=algorithm_name,
    )
    return selected_row, trial_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune cached MSPLIT and ShapeCART with Bayesian optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=["electricity"])
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument("--search-jobs", type=int, default=None)
    parser.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS)
    parser.add_argument("--binner-min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--cache-min-child-size", type=int, default=DEFAULT_MIN_CHILD_SIZE)
    parser.add_argument("--cache-version", type=int, default=DEFAULT_CACHE_VERSION)
    parser.add_argument("--timing-mode", choices=("fair", "fast"), default="fair")
    parser.add_argument("--msplit-build-dir", type=str, default="build-nonlinear-py")
    parser.add_argument(
        "--shared-min-leaf-values",
        nargs="+",
        type=int,
        default=DEFAULT_SHARED_MIN_LEAF_VALUES,
        help="Integer search bounds for outer leaf sizes. Optuna samples continuously within this range.",
    )
    parser.add_argument(
        "--shared-split-multipliers",
        nargs="+",
        type=int,
        default=DEFAULT_SHARED_SPLIT_MULTIPLIERS,
        help="Integer bounds for deriving min split size from min leaf size.",
    )
    parser.add_argument("--msplit-lookahead-depth-values", nargs="+", type=int, default=DEFAULT_MSPLIT_LOOKAHEAD_DEPTH_VALUES)
    parser.add_argument("--msplit-exactify-top-k-values", nargs="+", type=int, default=DEFAULT_MSPLIT_EXACTIFY_TOP_K_VALUES)
    parser.add_argument("--msplit-max-branching-values", nargs="+", type=int, default=DEFAULT_MSPLIT_MAX_BRANCHING_VALUES)
    parser.add_argument(
        "--msplit-reg-values",
        nargs="+",
        default=DEFAULT_MSPLIT_REG_VALUES,
        help="Bounds for continuous MSPLIT regularization search. Include 0.0 to allow the zero-regime.",
    )
    parser.add_argument("--shape-criterion-values", nargs="+", default=DEFAULT_SHAPE_CRITERION_VALUES)
    parser.add_argument("--shape-inner-max-depth", type=int, default=6)
    parser.add_argument("--shape-inner-max-leaf-values", nargs="+", type=int, default=DEFAULT_SHAPE_INNER_MAX_LEAF_VALUES)
    parser.add_argument(
        "--shape-inner-min-leaf-values",
        nargs="+",
        default=DEFAULT_SHAPE_INNER_MIN_LEAF_VALUES,
        help="Bounds for count- or fraction-valued ShapeCART inner leaf sizes.",
    )
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
    args = parser.parse_args()

    if optuna is None:
        raise RuntimeError("Optuna is not installed in this environment.") from _OPTUNA_IMPORT_ERROR
    if float(args.test_size) <= 0.0 or float(args.val_size) <= 0.0:
        raise ValueError("--test-size and --val-size must both be positive")
    if float(args.test_size) + float(args.val_size) >= 1.0:
        raise ValueError("--test-size + --val-size must be < 1.0")
    if int(args.n_trials) < 1:
        raise ValueError("--n-trials must be at least 1")
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    os.environ["MSPLIT_BUILD_DIR"] = str(args.msplit_build_dir)
    configure_timing_mode(args.timing_mode)

    datasets = canonical_dataset_list(args.datasets)
    depths = sorted(set(int(v) for v in args.depths))
    seeds = [int(v) for v in args.seeds]
    search_jobs = resolve_search_jobs(args.search_jobs, args.timing_mode)
    final_fit_guard_enabled = bool(args.timing_mode == "fair")

    out_root = Path(args.results_root)
    run_name = args.run_name or datetime.now().strftime("cached_optuna_msplit_vs_shapecart_%Y%m%d_%H%M%S")
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
    trial_rows: list[dict[str, Any]] = []
    cache_manifest_rows: list[dict[str, Any]] = []
    study_storage_uri = _study_storage_uri(out_dir=out_dir, search_jobs=int(search_jobs))

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
                print(f"[optuna] dataset={dataset} seed={seed} depth={depth} -> MSPLIT", flush=True)
                msplit_row, msplit_trials = _tune_msplit(
                    protocol=protocol,
                    depth=int(depth),
                    split_list=split_list,
                    cv_folds=int(args.cv_folds),
                    n_trials=int(args.n_trials),
                    search_jobs=int(search_jobs),
                    final_fit_guard_enabled=bool(final_fit_guard_enabled),
                    sampler_seed=int(100000 + 1000 * int(seed) + 10 * int(depth) + 1),
                    study_storage_uri=study_storage_uri,
                    study_name=f"msplit::{dataset}::seed{int(seed)}::depth{int(depth)}",
                    shared_min_leaf_values=[int(v) for v in args.shared_min_leaf_values],
                    shared_split_multipliers=[int(v) for v in args.shared_split_multipliers],
                    msplit_lookahead_depth_values=[int(v) for v in args.msplit_lookahead_depth_values],
                    msplit_exactify_top_k_values=[int(v) for v in args.msplit_exactify_top_k_values],
                    msplit_max_branching_values=[int(v) for v in args.msplit_max_branching_values],
                    msplit_reg_values=[float(v) for v in args.msplit_reg_values],
                    out_dir=out_dir,
                )
                selected_rows.append(msplit_row)
                trial_rows.extend(msplit_trials)
                _write_progress_tables(
                    out_dir=out_dir,
                    selected_rows=selected_rows,
                    trial_rows=trial_rows,
                    cache_manifest_rows=cache_manifest_rows,
                )

                print(f"[optuna] dataset={dataset} seed={seed} depth={depth} -> ShapeCART", flush=True)
                shapecart_row, shapecart_trials = _tune_shapecart(
                    protocol=protocol,
                    depth=int(depth),
                    split_list=split_list,
                    cv_folds=int(args.cv_folds),
                    n_trials=int(args.n_trials),
                    search_jobs=int(search_jobs),
                    final_fit_guard_enabled=bool(final_fit_guard_enabled),
                    sampler_seed=int(200000 + 1000 * int(seed) + 10 * int(depth) + 2),
                    study_storage_uri=study_storage_uri,
                    study_name=f"shapecart::{dataset}::seed{int(seed)}::depth{int(depth)}",
                    shared_min_leaf_values=[int(v) for v in args.shared_min_leaf_values],
                    shared_split_multipliers=[int(v) for v in args.shared_split_multipliers],
                    shape_criterion_values=list(args.shape_criterion_values),
                    shape_inner_max_depth=int(args.shape_inner_max_depth),
                    shape_inner_max_leaf_values=[int(v) for v in args.shape_inner_max_leaf_values],
                    shape_inner_min_leaf_values=list(args.shape_inner_min_leaf_values),
                    shape_k=int(args.shape_k),
                    shape_max_iter=int(args.shape_max_iter),
                    shape_pairwise_candidates=float(args.shape_pairwise_candidates),
                    shape_smart_init=bool(args.shape_smart_init),
                    shape_random_pairs=bool(args.shape_random_pairs),
                    shape_use_dpdt=bool(args.shape_use_dpdt),
                    shape_use_tao=bool(args.shape_use_tao),
                    shape_branching_penalty_values=[float(v) for v in args.shape_branching_penalty_values],
                    shape_tao_reg_values=[float(v) for v in args.shape_tao_reg_values],
                    shape_tao_n_runs=int(args.shape_tao_n_runs),
                    shape_tao_pair_scale=float(args.shape_tao_pair_scale),
                    out_dir=out_dir,
                )
                selected_rows.append(shapecart_row)
                trial_rows.extend(shapecart_trials)

                _write_progress_tables(
                    out_dir=out_dir,
                    selected_rows=selected_rows,
                    trial_rows=trial_rows,
                    cache_manifest_rows=cache_manifest_rows,
                )

    _write_progress_tables(
        out_dir=out_dir,
        selected_rows=selected_rows,
        trial_rows=trial_rows,
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
