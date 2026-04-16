#!/usr/bin/env python3
"""Optuna tuning helpers for the nested-CV benchmark runner."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from benchmark.scripts.benchmark_cached_common import (
    CachedProtocol,
    benchmark_timing_fields,
    fit_shapecart_once,
    json_safe,
    timing_guard_scope,
    write_msplit_artifacts,
    write_shapecart_artifacts,
)
from benchmark.scripts.benchmark_cached_msplit import run_cached_msplit


def require_optuna():
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - CLI guard
        raise RuntimeError("Optuna is not installed in this environment.") from exc
    return optuna


def tune_msplit(
    *,
    tune_protocols: list[CachedProtocol],
    final_protocol: CachedProtocol,
    fold_index: int,
    outer_folds: int,
    depth: int,
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
    msplit_worker_limit: int,
    out_dir: Path,
    save_model_artifacts: bool,
    collect_trial_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    optuna = require_optuna()
    study = _create_study(optuna, sampler_seed, study_storage_uri, study_name)

    def objective(trial) -> float:
        min_child_size = _suggest_discrete_int(trial=trial, name="min_child_size", values=shared_min_leaf_values)
        params = {
            "lookahead_depth": _suggest_discrete_int(
                trial=trial,
                name="lookahead_depth",
                values=msplit_lookahead_depth_values,
                upper_bound=int(depth),
            ),
            "exactify_top_k": _suggest_discrete_int(trial=trial, name="exactify_top_k", values=msplit_exactify_top_k_values),
            "max_branching": _suggest_discrete_int(trial=trial, name="max_branching", values=msplit_max_branching_values),
            "min_child_size": int(min_child_size),
            "min_split_multiplier": _suggest_discrete_int(
                trial=trial,
                name="min_split_multiplier",
                values=shared_split_multipliers,
            ),
            "reg": _suggest_nonnegative_float_range(trial=trial, name="reg", values=msplit_reg_values),
        }
        results = [
            run_cached_msplit(
                cache=protocol.cache,
                depth=int(depth),
                lookahead_depth=int(params["lookahead_depth"]),
                reg=float(params["reg"]),
                exactify_top_k=int(params["exactify_top_k"]),
                min_split_size=int(params["min_child_size"]) * int(params["min_split_multiplier"]),
                min_child_size=int(params["min_child_size"]),
                max_branching=int(params["max_branching"]),
                worker_limit=int(msplit_worker_limit),
                include_tree=False,
                include_diagnostics=False,
            )
            for protocol in tune_protocols
        ]
        val_scores = [result.get("val_accuracy") for result in results]
        if any(score is None for score in val_scores):
            raise RuntimeError("Validation accuracy missing from MSPLIT result.")
        _set_trial_stats(
            trial,
            fit_times=[float(result["fit_seconds"]) for result in results],
            train_scores=[float(result["train_accuracy"]) for result in results],
            val_scores=[float(score) for score in val_scores],
            internal_nodes=[int(result["n_internal"]) for result in results],
            leaves=[int(result["n_leaves"]) for result in results],
            objective=float(np.mean([float(result["objective"]) for result in results])),
        )
        return float(np.mean(val_scores))

    study, search_seconds = _optimize_study(optuna, study, objective, n_trials=n_trials, search_jobs=search_jobs)
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
            cache=final_protocol.cache,
            depth=int(depth),
            lookahead_depth=int(final_params["lookahead_depth"]),
            reg=float(final_params["reg"]),
            exactify_top_k=int(final_params["exactify_top_k"]),
            min_split_size=int(final_params["min_child_size"]) * int(final_params["min_split_multiplier"]),
            min_child_size=int(final_params["min_child_size"]),
            max_branching=int(final_params["max_branching"]),
            worker_limit=int(msplit_worker_limit),
            include_tree=bool(save_model_artifacts),
            include_diagnostics=False,
        )
    tree_path, metrics_path = "", ""
    selected_row = _selected_row(
        final_protocol=final_protocol,
        algorithm="msplit",
        fold_index=fold_index,
        outer_folds=outer_folds,
        depth=depth,
        cv_folds=cv_folds,
        n_trials=n_trials,
        search_jobs=search_jobs,
        search_seconds=search_seconds,
        best=best,
        best_trial=best_trial,
        final_result=final_result,
        selected_params=final_params,
        tree_artifact_path=tree_path,
        metrics_path=metrics_path,
        n_internal_key="n_internal",
        objective=float(final_result["objective"]),
    )
    artifact_job = None
    if save_model_artifacts:
        artifact_job = {
            "kind": "msplit",
            "selected_row": selected_row,
            "protocol": final_protocol,
            "depth": int(depth),
            "params": final_params,
            "result": final_result,
            "partition_label": f"fold{int(fold_index)}_seed{int(final_protocol.seed)}",
            "metadata_extra": _artifact_metadata(fold_index, outer_folds, final_protocol),
            "out_dir": out_dir,
        }
    return (
        selected_row,
        _study_trials_to_rows(
            study=study,
            dataset=final_protocol.dataset,
            split_seed=int(final_protocol.seed),
            fold_index=int(fold_index),
            outer_folds=int(outer_folds),
            depth=int(depth),
            algorithm="msplit",
        ) if collect_trial_rows else [],
        artifact_job,
    )


def tune_shapecart(
    *,
    tune_protocols: list[CachedProtocol],
    final_protocol: CachedProtocol,
    fold_index: int,
    outer_folds: int,
    depth: int,
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
    save_model_artifacts: bool,
    collect_trial_rows: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    optuna = require_optuna()
    study = _create_study(optuna, sampler_seed, study_storage_uri, study_name)

    def objective(trial) -> float:
        branching_penalty = 0.0
        if int(shape_k) > 2:
            branching_penalty = _suggest_nonnegative_float_range(
                trial=trial,
                name="branching_penalty",
                values=shape_branching_penalty_values,
            )
        tao_reg = 0.0
        if shape_use_tao:
            tao_reg = _suggest_nonnegative_float_range(trial=trial, name="tao_reg", values=shape_tao_reg_values)
        params = {
            "min_samples_leaf": _suggest_discrete_int(trial=trial, name="min_samples_leaf", values=shared_min_leaf_values),
            "min_split_multiplier": _suggest_discrete_int(
                trial=trial,
                name="min_split_multiplier",
                values=shared_split_multipliers,
            ),
            "criterion": str(trial.suggest_categorical("criterion", list(shape_criterion_values))),
            "inner_max_depth": int(shape_inner_max_depth),
            "inner_max_leaf_nodes": _suggest_discrete_int(
                trial=trial,
                name="inner_max_leaf_nodes",
                values=shape_inner_max_leaf_values,
            ),
            "inner_min_samples_leaf": _suggest_shape_inner_min_samples_leaf(
                trial=trial,
                values=shape_inner_min_leaf_values,
            ),
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
        params["min_samples_split"] = int(params["min_samples_leaf"]) * int(params["min_split_multiplier"])
        results = [
            fit_shapecart_once(
                cache=protocol.cache,
                depth=int(depth),
                random_state=int(final_protocol.seed) + 100 * int(inner_fold_index),
                params=params,
                include_model=False,
            )
            for inner_fold_index, protocol in enumerate(tune_protocols)
        ]
        _set_trial_stats(
            trial,
            fit_times=[float(result["fit_seconds"]) for result in results],
            train_scores=[float(result["train_accuracy"]) for result in results],
            val_scores=[float(result["val_accuracy"]) for result in results],
            internal_nodes=[int(result["n_internal_nodes"]) for result in results],
            leaves=[int(result["n_leaves"]) for result in results],
        )
        return float(np.mean([float(result["val_accuracy"]) for result in results]))

    study, search_seconds = _optimize_study(optuna, study, objective, n_trials=n_trials, search_jobs=search_jobs)
    best = _best_trial_payload(study)
    best_trial = _study_trial_by_number(study, int(best["number"]))
    best_params = dict(best["params"])
    final_params = {
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "min_samples_split": int(best_params["min_samples_leaf"]) * int(best_params["min_split_multiplier"]),
        "criterion": str(best_params["criterion"]),
        "inner_max_depth": int(shape_inner_max_depth),
        "inner_max_leaf_nodes": int(best_params["inner_max_leaf_nodes"]),
        "inner_min_samples_leaf": _decode_shape_inner_min_samples_leaf(best_params),
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
            cache=final_protocol.cache,
            depth=int(depth),
            random_state=int(final_protocol.seed),
            params=final_params,
            include_model=bool(save_model_artifacts),
        )
    algorithm = "shapecart_tao" if shape_use_tao else "shapecart"
    tree_path, metrics_path = "", ""
    selected_row = _selected_row(
        final_protocol=final_protocol,
        algorithm=algorithm,
        fold_index=fold_index,
        outer_folds=outer_folds,
        depth=depth,
        cv_folds=cv_folds,
        n_trials=n_trials,
        search_jobs=search_jobs,
        search_seconds=search_seconds,
        best=best,
        best_trial=best_trial,
        final_result=final_result,
        selected_params=final_params,
        tree_artifact_path=tree_path,
        metrics_path=metrics_path,
        n_internal_key="n_internal_nodes",
        objective=np.nan,
    )
    artifact_job = None
    if save_model_artifacts:
        artifact_job = {
            "kind": "shapecart",
            "selected_row": selected_row,
            "protocol": final_protocol,
            "depth": int(depth),
            "params": final_params,
            "result": final_result,
            "partition_label": f"fold{int(fold_index)}_seed{int(final_protocol.seed)}",
            "metadata_extra": _artifact_metadata(fold_index, outer_folds, final_protocol),
            "out_dir": out_dir,
        }
    return (
        selected_row,
        _study_trials_to_rows(
            study=study,
            dataset=final_protocol.dataset,
            split_seed=int(final_protocol.seed),
            fold_index=int(fold_index),
            outer_folds=int(outer_folds),
            depth=int(depth),
            algorithm=algorithm,
        ) if collect_trial_rows else [],
        artifact_job,
    )


def flush_artifact_jobs(artifact_jobs: list[dict[str, Any]]) -> None:
    for job in artifact_jobs:
        if str(job["kind"]) == "msplit":
            tree_path, metrics_path = write_msplit_artifacts(
                out_dir=job["out_dir"],
                protocol=job["protocol"],
                depth=job["depth"],
                best_params=job["params"],
                final_result=job["result"],
                partition_label=job["partition_label"],
                metadata_extra=job["metadata_extra"],
            )
        else:
            tree_path, metrics_path = write_shapecart_artifacts(
                out_dir=job["out_dir"],
                protocol=job["protocol"],
                depth=job["depth"],
                final_params=job["params"],
                final_result=job["result"],
                partition_label=job["partition_label"],
                metadata_extra=job["metadata_extra"],
            )
        job["selected_row"]["tree_artifact_path"] = tree_path
        job["selected_row"]["model_metrics_path"] = metrics_path


def _create_study(optuna, sampler_seed: int, study_storage_uri: str | None, study_name: str):
    kwargs: dict[str, Any] = {"direction": "maximize", "sampler": optuna.samplers.TPESampler(seed=int(sampler_seed))}
    if study_storage_uri is not None:
        kwargs.update({"storage": study_storage_uri, "study_name": study_name, "load_if_exists": True})
    return optuna.create_study(**kwargs)


def _optimize_study(optuna, study, objective, *, n_trials: int, search_jobs: int):
    started = time.perf_counter()
    with timing_guard_scope(enabled=False):
        study.optimize(
            lambda trial: _objective_result_or_failure(objective_fn=objective, trial=trial),
            n_trials=int(n_trials),
            n_jobs=int(search_jobs),
            gc_after_trial=False,
            show_progress_bar=False,
        )
    return study, time.perf_counter() - started


def _objective_result_or_failure(*, objective_fn, trial) -> float:
    try:
        value = float(objective_fn(trial))
        trial.set_user_attr("status", "ok")
        return value
    except Exception as exc:  # pragma: no cover - CLI guard
        trial.set_user_attr("status", "error")
        trial.set_user_attr("error", repr(exc))
        return -1.0


def _best_trial_payload(study) -> dict[str, Any]:
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
    candidates.sort(key=lambda row: (-float(row["val_accuracy"]), float(row["fit_seconds"]), int(row["number"])))
    return candidates[0]


def _study_trial_by_number(study, trial_number: int):
    for trial in study.trials:
        if int(trial.number) == int(trial_number):
            return trial
    raise KeyError(f"Trial number {trial_number} not found in study.")


def _study_trials_to_rows(
    *,
    study,
    dataset: str,
    split_seed: int,
    fold_index: int,
    outer_folds: int,
    depth: int,
    algorithm: str,
) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "split_seed": int(split_seed),
            "fold_index": int(fold_index),
            "outer_folds": int(outer_folds),
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
        for trial in study.trials
    ]


def _selected_row(
    *,
    final_protocol: CachedProtocol,
    algorithm: str,
    fold_index: int,
    outer_folds: int,
    depth: int,
    cv_folds: int,
    n_trials: int,
    search_jobs: int,
    search_seconds: float,
    best: dict[str, Any],
    best_trial,
    final_result: dict[str, Any],
    selected_params: dict[str, Any],
    tree_artifact_path: str,
    metrics_path: str,
    n_internal_key: str,
    objective: float,
) -> dict[str, Any]:
    return {
        "dataset": final_protocol.dataset,
        "split_seed": int(final_protocol.seed),
        "fold_index": int(fold_index),
        "outer_folds": int(outer_folds),
        "depth_budget": int(depth),
        "algorithm": algorithm,
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
        "n_internal_nodes": int(final_result[n_internal_key]),
        "n_leaves": int(final_result["n_leaves"]),
        "best_trial_number": int(best["number"]),
        "n_trials": int(n_trials),
        "selected_params_json": json.dumps(json_safe(selected_params), sort_keys=True),
        "objective": objective,
        "cache_path": str(final_protocol.cache_path),
        "cache_requested_path": str(final_protocol.requested_cache_path),
        "cache_used_compatible_fallback": bool(final_protocol.cache_used_fallback),
        "tree_artifact_path": tree_artifact_path,
        "model_metrics_path": metrics_path,
        **benchmark_timing_fields(
            algorithm="msplit" if algorithm == "msplit" else "shapecart",
            model_fit_time_sec=float(final_result["fit_seconds"]),
            shared_cache_build_seconds=float(final_protocol.cache_build_seconds),
        ),
    }


def _artifact_metadata(fold_index: int, outer_folds: int, protocol: CachedProtocol) -> dict[str, Any]:
    return {
        "fold_index": int(fold_index),
        "outer_folds": int(outer_folds),
        "split_seed": int(protocol.seed),
    }


def _set_trial_stats(
    trial,
    *,
    fit_times: list[float],
    train_scores: list[float],
    val_scores: list[float],
    internal_nodes: list[int],
    leaves: list[int],
    objective: float | None = None,
) -> None:
    trial.set_user_attr("fit_seconds", float(np.mean(fit_times)))
    trial.set_user_attr("cv_model_fit_time_std", float(np.std(fit_times, ddof=0)))
    trial.set_user_attr("train_accuracy", float(np.mean(train_scores)))
    trial.set_user_attr("test_accuracy", np.nan)
    trial.set_user_attr("n_internal_nodes", float(np.mean(internal_nodes)))
    trial.set_user_attr("n_leaves", float(np.mean(leaves)))
    trial.set_user_attr("cv_val_accuracy_std", float(np.std(val_scores, ddof=0)))
    if objective is not None:
        trial.set_user_attr("objective", float(objective))


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
    if allow_zero and trial.suggest_categorical(f"{name}_mode", ["zero", "positive"]) == "zero":
        return 0.0
    lo = float(min(positive_values))
    hi = float(max(positive_values))
    if math.isclose(lo, hi):
        return float(trial.suggest_categorical(f"{name}_positive", [float(lo)]))
    return float(trial.suggest_float(f"{name}_positive", lo, hi, log=True))


def _decode_nonnegative_float_param(best_params: dict[str, Any], name: str) -> float:
    if str(best_params.get(f"{name}_mode", "")).strip().lower() == "zero":
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
        if trial.suggest_categorical("inner_min_samples_leaf_mode", ["count", "fraction"]) == "count":
            return _suggest_discrete_int(trial=trial, name="inner_min_samples_leaf_count", values=int_values)
        lo, hi = float(min(frac_values)), float(max(frac_values))
        if math.isclose(lo, hi):
            return float(trial.suggest_categorical("inner_min_samples_leaf_fraction", [float(lo)]))
        return float(trial.suggest_float("inner_min_samples_leaf_fraction", lo, hi, log=True))
    if int_values:
        return _suggest_discrete_int(trial=trial, name="inner_min_samples_leaf_count", values=int_values)
    lo, hi = float(min(frac_values)), float(max(frac_values))
    if math.isclose(lo, hi):
        return float(trial.suggest_categorical("inner_min_samples_leaf_fraction", [float(lo)]))
    return float(trial.suggest_float("inner_min_samples_leaf_fraction", lo, hi, log=True))


def _decode_shape_inner_min_samples_leaf(best_params: dict[str, Any]) -> int | float:
    if "inner_min_samples_leaf_count" in best_params:
        return int(best_params["inner_min_samples_leaf_count"])
    if "inner_min_samples_leaf_fraction" in best_params:
        return float(best_params["inner_min_samples_leaf_fraction"])
    return best_params["inner_min_samples_leaf"]
