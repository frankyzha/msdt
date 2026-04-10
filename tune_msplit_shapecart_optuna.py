#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import optuna
except Exception as exc:  # pragma: no cover - CLI guard
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_teacher_guided_atomcolor_cached import (
    _slice_rows,
    load_local_libgosdt,
    predict_tree,
    tree_stats as msplit_tree_stats,
)
from experiment_utils import DATASET_LOADERS, encode_binary_target, make_preprocessor
from lightgbm_binning import fit_lightgbm_binner

SHAPECART_ROOT = REPO_ROOT / "Empowering-DTs-via-Shape-Functions"
if str(SHAPECART_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPECART_ROOT))

from src.ShapeCARTClassifier import ShapeCARTClassifier  # type: ignore


N_TRIALS = 50
N_FOLDS = 3
DEFAULT_SEED = 0
# Paper-style protocol: 70% train, 30% holdout, then 70/30 split inside holdout.
OUTER_TRAIN_FRACTION = 0.70
INNER_TRAIN_FRACTION = 0.70
MAX_BINS = 1024
LOOKAHEAD_DEPTH_CAP = 3
BRANCHING_FACTOR = 3
TIME_LIMIT_SECONDS = 28800.0
LGB_NUM_THREADS = 3
LGB_LAMBDA_L2 = 0.0
MIN_CHILD_SIZE = 4
MIN_SPLIT_SIZE = 8
SHAPECART_INNER_MAX_DEPTH = 6
SHAPECART_INNER_MAX_LEAF_NODES = 32
SHAPECART_MAX_ITER = 20
SHAPECART_PAIRWISE_CANDIDATES = 0.0

REG_RANGE = (1e-5, 1e-3)


@dataclass(frozen=True)
class FoldPayload:
    X_fit_proc: np.ndarray
    X_val_proc: np.ndarray
    X_test_proc: np.ndarray
    Z_fit: np.ndarray
    Z_val: np.ndarray
    Z_test: np.ndarray
    y_fit: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    teacher_logit: np.ndarray
    teacher_boundary_gain: np.ndarray
    teacher_boundary_cover: np.ndarray
    teacher_boundary_value_jump: np.ndarray
    min_child_size: int
    min_split_size: int
    n_fit: int
    n_val: int
    n_test: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MSPLIT and ShapeCART for one dataset at one fixed depth."
    )
    parser.add_argument("--dataset", default="electricity", choices=sorted(DATASET_LOADERS.keys()))
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if optuna is None:
        raise RuntimeError("Optuna is not installed in this environment.") from _OPTUNA_IMPORT_ERROR
    if int(args.depth) < 1:
        raise ValueError("--depth must be at least 1")
    return args


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_pred, dtype=np.int32) == np.asarray(y_true, dtype=np.int32)))


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _mean_dict(dicts: list[dict[str, object]]) -> dict[str, float]:
    if not dicts:
        return {}
    keys = sorted(set().union(*(d.keys() for d in dicts)))
    out: dict[str, float] = {}
    for key in keys:
        vals: list[float] = []
        for d in dicts:
            value = d.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                vals.append(float(value))
        if vals:
            out[key] = float(np.mean(np.asarray(vals, dtype=np.float64)))
    return out


def _msplit_tree_depth(tree: dict[str, object]) -> int:
    if tree.get("type") == "leaf":
        return 0
    child_depths: list[int] = []
    for group in tree.get("groups", []):
        child = group.get("child") if isinstance(group, dict) else None
        if isinstance(child, dict):
            child_depths.append(_msplit_tree_depth(child))
    return 1 + max(child_depths, default=0)


def _shape_tree_stats(model) -> dict[str, int]:
    n_internal = 0
    n_leaves = 0
    used_features: set[int] = set()
    stack = [0]
    while stack:
        node_idx = int(stack.pop())
        if bool(model.is_leaf[node_idx]) or model.children[node_idx] is None:
            n_leaves += 1
            continue
        n_internal += 1
        node = model.nodes[node_idx]
        feature_key = getattr(node, "final_key", None)
        feature_dict = getattr(node, "feature_dict", {}) or {}
        if feature_key is not None:
            if feature_key in feature_dict:
                used_features.update(int(v) for v in feature_dict[feature_key])
            elif isinstance(feature_key, tuple):
                for part in feature_key:
                    if part in feature_dict:
                        used_features.update(int(v) for v in feature_dict[part])
                    else:
                        try:
                            used_features.add(int(part))
                        except Exception:
                            continue
            else:
                try:
                    used_features.add(int(feature_key))
                except Exception:
                    pass
        stack.extend(int(v) for v in (model.children[node_idx] or []))
    max_depth = int(max(model.depths)) if getattr(model, "depths", None) else 0
    return {
        "n_internal_nodes": int(n_internal),
        "n_leaves": int(n_leaves),
        "tree_depth": int(max_depth),
        "used_feature_count": int(len(used_features)),
    }


def _msplit_solver_stats(cpp_result: dict[str, object]) -> dict[str, float]:
    keys = {
        "greedy_internal_nodes",
        "greedy_subproblem_calls",
        "exact_dp_subproblem_calls_above_lookahead",
        "greedy_cache_hits",
        "greedy_unique_states",
        "greedy_cache_entries_peak",
        "greedy_cache_bytes_peak",
        "greedy_interval_evals",
        "debr_refine_calls",
        "debr_refine_improved",
        "debr_total_moves",
        "debr_bridge_policy_calls",
        "debr_refine_windowed_calls",
        "debr_refine_unwindowed_calls",
        "debr_refine_overlap_segments",
        "debr_refine_calls_with_overlap",
        "debr_refine_calls_without_overlap",
        "debr_candidate_total",
        "debr_candidate_legal",
        "debr_candidate_source_size_rejects",
        "debr_candidate_target_size_rejects",
        "debr_candidate_descent_eligible",
        "debr_candidate_descent_rejected",
        "debr_candidate_bridge_eligible",
        "debr_candidate_bridge_window_blocked",
        "debr_candidate_bridge_used_blocked",
        "debr_candidate_bridge_guide_rejected",
        "debr_candidate_cleanup_eligible",
        "debr_candidate_cleanup_primary_rejected",
        "debr_candidate_cleanup_complexity_rejected",
        "debr_candidate_score_rejected",
        "debr_descent_moves",
        "debr_bridge_moves",
        "debr_simplify_moves",
        "debr_total_hard_gain",
        "debr_total_soft_gain",
        "debr_total_delta_j",
        "debr_total_component_delta",
        "debr_final_geo_wins",
        "debr_final_block_wins",
        "family_compare_total",
        "family_compare_equivalent",
        "family1_both_wins",
        "family2_hard_loss_wins",
        "family2_hard_impurity_wins",
        "family2_both_wins",
        "family_metric_disagreement",
        "family_hard_loss_ties",
        "family_hard_impurity_ties",
        "family_joint_impurity_ties",
        "family_neither_both_wins",
        "family1_selected_by_equivalence",
        "family1_selected_by_dominance",
        "family2_selected_by_dominance",
        "family_sent_both",
        "atomized_features_prepared",
        "atomized_coarse_candidates",
        "atomized_final_candidates",
        "atomized_coarse_pruned_candidates",
        "atomized_compression_features_applied",
        "atomized_compression_features_collapsed_to_single_block",
        "atomized_compression_atoms_before_total",
        "atomized_compression_blocks_after_total",
        "atomized_compression_atoms_merged_total",
        "nominee_unique_total",
        "nominee_child_interval_lookups",
        "nominee_child_interval_unique",
        "nominee_exactified_total",
        "nominee_incumbent_updates",
        "nominee_threatening_samples",
        "nominee_threatening_sum",
        "nominee_threatening_max",
        "nominee_certificate_nodes",
        "nominee_certificate_exhausted_nodes",
        "nominee_exactified_until_certificate_total",
        "nominee_exactified_until_certificate_max",
        "nominee_elbow_prefix_total",
        "nominee_elbow_prefix_max",
        "heuristic_selector_nodes",
        "heuristic_selector_candidate_total",
        "heuristic_selector_candidate_pruned_total",
        "heuristic_selector_survivor_total",
        "heuristic_selector_leaf_optimal_nodes",
        "heuristic_selector_improving_split_nodes",
        "heuristic_selector_improving_split_retained_nodes",
        "heuristic_selector_improving_split_margin_sum",
        "heuristic_selector_improving_split_margin_max",
        "profiling_lp_solve_calls",
        "profiling_lp_solve_sec",
        "profiling_pricing_calls",
        "profiling_pricing_sec",
        "profiling_greedy_complete_calls",
        "profiling_greedy_complete_sec",
        "profiling_feature_prepare_sec",
        "profiling_candidate_nomination_sec",
        "profiling_candidate_shortlist_sec",
        "profiling_candidate_generation_sec",
        "profiling_recursive_child_eval_sec",
        "profiling_refine_calls",
        "profiling_refine_sec",
    }
    stats: dict[str, float] = {}
    for key in keys:
        value = cpp_result.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            stats[key] = float(value)

    comparisons = int(stats.get("family_compare_total", 0.0))
    if comparisons > 0:
        stats["family2_selected_rate"] = float(stats.get("debr_final_block_wins", 0.0)) / float(comparisons)
        stats["family_metric_disagreement_rate"] = float(stats.get("family_metric_disagreement", 0.0)) / float(comparisons)
        stats["family1_selected_by_equivalence_rate"] = float(stats.get("family1_selected_by_equivalence", 0.0)) / float(comparisons)
        stats["family1_selected_by_dominance_rate"] = float(stats.get("family1_selected_by_dominance", 0.0)) / float(comparisons)
        stats["family2_selected_by_dominance_rate"] = float(stats.get("family2_selected_by_dominance", 0.0)) / float(comparisons)
        stats["family_sent_both_rate"] = float(stats.get("family_sent_both", 0.0)) / float(comparisons)

    coarse_candidates = int(stats.get("atomized_coarse_candidates", 0.0))
    if coarse_candidates > 0:
        stats["atomized_coarse_prune_rate"] = float(stats.get("atomized_coarse_pruned_candidates", 0.0)) / float(coarse_candidates)
        stats["atomized_coarse_survivor_rate"] = 1.0 - stats["atomized_coarse_prune_rate"]

    nominee_unique_total = int(stats.get("nominee_unique_total", 0.0))
    if nominee_unique_total > 0:
        stats["nominee_exactified_rate"] = float(stats.get("nominee_exactified_total", 0.0)) / float(nominee_unique_total)

    heuristic_nodes = float(stats.get("heuristic_selector_nodes", 0.0))
    if heuristic_nodes > 0.0:
        total_candidates = float(stats.get("heuristic_selector_candidate_total", 0.0))
        if total_candidates > 0.0:
            stats["heuristic_selector_candidate_prune_rate"] = (
                float(stats.get("heuristic_selector_candidate_pruned_total", 0.0)) / total_candidates
            )
            stats["heuristic_selector_candidate_survivor_rate"] = (
                float(stats.get("heuristic_selector_survivor_total", 0.0)) / total_candidates
            )
        stats["heuristic_selector_leaf_optimal_rate"] = (
            float(stats.get("heuristic_selector_leaf_optimal_nodes", 0.0)) / heuristic_nodes
        )
        improving_nodes = float(stats.get("heuristic_selector_improving_split_nodes", 0.0))
        if improving_nodes > 0.0:
            stats["heuristic_selector_improving_split_retained_rate"] = (
                float(stats.get("heuristic_selector_improving_split_retained_nodes", 0.0)) / improving_nodes
            )
            stats["heuristic_selector_improving_split_margin_mean"] = (
                float(stats.get("heuristic_selector_improving_split_margin_sum", 0.0)) / improving_nodes
            )

    return stats


def _run_study(objective, *, seed: int):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _safe_objective(trial) -> float:
        try:
            return float(objective(trial))
        except Exception:
            return 0.0

    study.optimize(_safe_objective, n_trials=N_TRIALS)
    return study


def _fit_binner(
    X_fit_proc: np.ndarray,
    y_fit: np.ndarray,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    *,
    min_child_size: int,
    seed: int,
):
    return fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=MAX_BINS,
        min_samples_leaf=int(min_child_size),
        random_state=seed,
        n_estimators=10000,
        num_leaves=255,
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        min_data_in_bin=1,
        min_data_in_leaf=int(min_child_size),
        lambda_l2=LGB_LAMBDA_L2,
        early_stopping_rounds=100,
        num_threads=LGB_NUM_THREADS,
        device_type="cpu",
        collect_teacher_logit=True,
    )


def _make_fold_payload(
    X,
    y_bin: np.ndarray,
    *,
    idx_fit: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    seed: int,
) -> FoldPayload:
    X_fit = _slice_rows(X, idx_fit)
    X_val = _slice_rows(X, idx_val)
    X_test = _slice_rows(X, idx_test)
    y_fit = np.asarray(y_bin[idx_fit], dtype=np.int32)
    y_val = np.asarray(y_bin[idx_val], dtype=np.int32)
    y_test = np.asarray(y_bin[idx_test], dtype=np.int32)

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.asarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.asarray(pre.transform(X_val), dtype=np.float32)
    X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)

    binner = _fit_binner(
        X_fit_proc,
        y_fit,
        X_val_proc,
        y_val,
        min_child_size=MIN_CHILD_SIZE,
        seed=seed,
    )

    return FoldPayload(
        X_fit_proc=X_fit_proc,
        X_val_proc=X_val_proc,
        X_test_proc=X_test_proc,
        Z_fit=np.asarray(binner.transform(X_fit_proc), dtype=np.int32),
        Z_val=np.asarray(binner.transform(X_val_proc), dtype=np.int32),
        Z_test=np.asarray(binner.transform(X_test_proc), dtype=np.int32),
        y_fit=y_fit,
        y_val=y_val,
        y_test=y_test,
        teacher_logit=np.asarray(getattr(binner, "teacher_train_logit"), dtype=np.float64),
        teacher_boundary_gain=np.asarray(getattr(binner, "boundary_gain_per_feature"), dtype=np.float64),
        teacher_boundary_cover=np.asarray(getattr(binner, "boundary_cover_per_feature"), dtype=np.float64),
        teacher_boundary_value_jump=np.asarray(
            getattr(binner, "boundary_value_jump_per_feature"),
            dtype=np.float64,
        ),
        min_child_size=MIN_CHILD_SIZE,
        min_split_size=MIN_SPLIT_SIZE,
        n_fit=int(idx_fit.shape[0]),
        n_val=int(idx_val.shape[0]),
        n_test=int(idx_test.shape[0]),
    )


@lru_cache(maxsize=None)
def _prepare_folds(dataset_name: str) -> tuple[FoldPayload, ...]:
    X, y = DATASET_LOADERS[dataset_name]()
    y_bin = encode_binary_target(y, dataset_name)
    y_bin = np.asarray(y_bin, dtype=np.int32)
    all_idx = np.arange(y_bin.shape[0], dtype=np.int32)
    outer_splitter = StratifiedShuffleSplit(
        n_splits=N_FOLDS,
        train_size=OUTER_TRAIN_FRACTION,
        random_state=DEFAULT_SEED,
    )

    folds: list[FoldPayload] = []
    for fold_idx, (idx_fit, idx_holdout) in enumerate(outer_splitter.split(all_idx, y_bin)):
        idx_fit = np.asarray(idx_fit, dtype=np.int32)
        idx_holdout = np.asarray(idx_holdout, dtype=np.int32)
        y_holdout = y_bin[idx_holdout]

        inner_splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=INNER_TRAIN_FRACTION,
            random_state=DEFAULT_SEED + fold_idx + 1,
        )
        holdout_rows = np.zeros((idx_holdout.shape[0], 1), dtype=np.int8)
        idx_val_rel, idx_test_rel = next(inner_splitter.split(holdout_rows, y_holdout))
        idx_val = np.asarray(idx_holdout[idx_val_rel], dtype=np.int32)
        idx_test = np.asarray(idx_holdout[idx_test_rel], dtype=np.int32)
        folds.append(
            _make_fold_payload(
                X,
                y_bin,
                idx_fit=idx_fit,
                idx_val=idx_val,
                idx_test=idx_test,
                seed=DEFAULT_SEED + fold_idx,
            )
        )

    return tuple(folds)


def _fit_msplit_candidate(
    payload: FoldPayload,
    *,
    depth: int,
    regularization: float,
    exactify_top_k: int | None,
) -> dict[str, object]:
    libgosdt = load_local_libgosdt()
    z_fit = payload.Z_fit
    z_val = payload.Z_val
    z_test = payload.Z_test
    y_fit = payload.y_fit
    y_val = payload.y_val
    y_test = payload.y_test
    sample_weight = np.full(z_fit.shape[0], 1.0 / float(max(1, z_fit.shape[0])), dtype=np.float64)
    exactify_top_k_arg = 0 if exactify_top_k is None else int(exactify_top_k)

    started = time.perf_counter()
    cpp_result = libgosdt.msplit_fit(
        z_fit,
        y_fit,
        sample_weight,
        payload.teacher_logit,
        payload.teacher_boundary_gain,
        payload.teacher_boundary_cover,
        payload.teacher_boundary_value_jump,
        int(depth),
        min(int(depth), LOOKAHEAD_DEPTH_CAP),
        float(regularization),
        payload.min_split_size,
        payload.min_child_size,
        TIME_LIMIT_SECONDS,
        BRANCHING_FACTOR,
        exactify_top_k_arg,
    )
    fit_time_sec = time.perf_counter() - started
    tree = json.loads(str(cpp_result["tree"]))
    pred_val = predict_tree(tree, z_val)
    pred_test = predict_tree(tree, z_test)
    pred_fit = predict_tree(tree, z_fit)
    return {
        "train_accuracy": _accuracy(y_fit, pred_fit),
        "val_accuracy": _accuracy(y_val, pred_val),
        "test_accuracy": _accuracy(y_test, pred_test),
        "fit_time_sec": float(fit_time_sec),
        "tree_stats": {
            **msplit_tree_stats(tree),
            "tree_depth": int(_msplit_tree_depth(tree)),
        },
        "solver_stats": _msplit_solver_stats(cpp_result),
    }


def _fit_shapecart_candidate(
    payload: FoldPayload,
    *,
    depth: int,
    regularization: float,
) -> dict[str, object]:
    X_fit = payload.X_fit_proc
    X_val = payload.X_val_proc
    X_test = payload.X_test_proc
    y_fit = payload.y_fit
    y_val = payload.y_val
    y_test = payload.y_test

    model = ShapeCARTClassifier(
        max_depth=int(depth),
        min_samples_leaf=payload.min_child_size,
        min_samples_split=max(2, payload.min_split_size),
        inner_min_samples_leaf=payload.min_child_size,
        inner_min_samples_split=max(2, payload.min_split_size),
        inner_max_depth=SHAPECART_INNER_MAX_DEPTH,
        inner_max_leaf_nodes=SHAPECART_INNER_MAX_LEAF_NODES,
        max_iter=SHAPECART_MAX_ITER,
        k=BRANCHING_FACTOR,
        branching_penalty=float(regularization),
        random_state=DEFAULT_SEED,
        verbose=False,
        pairwise_candidates=SHAPECART_PAIRWISE_CANDIDATES,
        smart_init=True,
        random_pairs=False,
        use_dpdt=False,
        use_tao=False,
    )
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_time_sec = time.perf_counter() - started

    pred_fit = np.asarray(model.predict(X_fit), dtype=np.int32)
    pred_val = np.asarray(model.predict(X_val), dtype=np.int32)
    pred_test = np.asarray(model.predict(X_test), dtype=np.int32)
    return {
        "train_accuracy": _accuracy(y_fit, pred_fit),
        "val_accuracy": _accuracy(y_val, pred_val),
        "test_accuracy": _accuracy(y_test, pred_test),
        "fit_time_sec": float(fit_time_sec),
        "tree_stats": _shape_tree_stats(model),
    }


def _summarize_search(
    *,
    depth: int,
    study,
    best_params: dict[str, object],
    fold_results: list[dict[str, object]],
    search_time_sec: float,
) -> dict[str, object]:
    train_scores = [float(result["train_accuracy"]) for result in fold_results]
    val_scores = [float(result["val_accuracy"]) for result in fold_results]
    test_scores = [float(result["test_accuracy"]) for result in fold_results]
    fit_times = [float(result["fit_time_sec"]) for result in fold_results]
    final_fit_time_sec = float(np.sum(np.asarray(fit_times, dtype=np.float64)))
    tree_stats_by_fold = [dict(result["tree_stats"]) for result in fold_results]
    solver_stats_by_fold = [dict(result.get("solver_stats", {})) for result in fold_results]
    return {
        "depth": int(depth),
        "best_params": best_params,
        "train_accuracy": _mean(train_scores),
        "val_accuracy": _mean(val_scores),
        "test_accuracy": _mean(test_scores),
        "train_accuracy_by_fold": train_scores,
        "val_accuracy_by_fold": val_scores,
        "test_accuracy_by_fold": test_scores,
        "fit_time_sec_by_fold": fit_times,
        "fit_time_sec": final_fit_time_sec,
        "mean_fit_time_sec": _mean(fit_times),
        "tree_stats_by_fold": tree_stats_by_fold,
        "tree_stats_mean": _mean_dict(tree_stats_by_fold),
        "solver_stats_by_fold": solver_stats_by_fold,
        "solver_stats_mean": _mean_dict(solver_stats_by_fold),
        "search_time_sec": float(search_time_sec),
        "total_time_sec": float(search_time_sec + final_fit_time_sec),
        "optuna_trials": int(len(study.trials)),
    }


def _tune_msplit_payloads(payloads: tuple[FoldPayload, ...], depth: int) -> dict[str, object]:
    def _objective(trial) -> float:
        regularization = trial.suggest_float("regularization", REG_RANGE[0], REG_RANGE[1], log=True)
        fold_val_scores = []
        for payload in payloads:
            result = _fit_msplit_candidate(
                payload,
                depth=int(depth),
                regularization=float(regularization),
                exactify_top_k=None,
            )
            fold_val_scores.append(float(result["val_accuracy"]))
        return _mean(fold_val_scores)

    search_started = time.perf_counter()
    study = _run_study(_objective, seed=DEFAULT_SEED)
    search_time_sec = time.perf_counter() - search_started
    best_params = dict(study.best_params)
    resolved_best_params = {
        "regularization": float(best_params["regularization"]),
        "exactify_top_k": None,
    }
    fold_results = [
        _fit_msplit_candidate(
            payload,
            depth=int(depth),
            regularization=resolved_best_params["regularization"],
            exactify_top_k=None,
        )
        for payload in payloads
    ]
    return _summarize_search(
        depth=int(depth),
        study=study,
        best_params=resolved_best_params,
        fold_results=fold_results,
        search_time_sec=float(search_time_sec),
    )


def tune_msplit(dataset: str, depth: int) -> dict[str, object]:
    return _tune_msplit_payloads(_prepare_folds(dataset), depth)


def _tune_shapecart_payloads(payloads: tuple[FoldPayload, ...], depth: int) -> dict[str, object]:
    def _objective(trial) -> float:
        regularization = trial.suggest_float("regularization", REG_RANGE[0], REG_RANGE[1], log=True)
        fold_val_scores = []
        for payload in payloads:
            result = _fit_shapecart_candidate(
                payload,
                depth=int(depth),
                regularization=float(regularization),
            )
            fold_val_scores.append(float(result["val_accuracy"]))
        return _mean(fold_val_scores)

    search_started = time.perf_counter()
    study = _run_study(_objective, seed=DEFAULT_SEED + 1)
    search_time_sec = time.perf_counter() - search_started
    best_params = dict(study.best_params)
    resolved_best_params = {
        "regularization": float(best_params["regularization"]),
    }
    fold_results = [
        _fit_shapecart_candidate(
            payload,
            depth=int(depth),
            regularization=resolved_best_params["regularization"],
        )
        for payload in payloads
    ]
    return _summarize_search(
        depth=int(depth),
        study=study,
        best_params=resolved_best_params,
        fold_results=fold_results,
        search_time_sec=float(search_time_sec),
    )


def tune_shapecart(dataset: str, depth: int) -> dict[str, object]:
    return _tune_shapecart_payloads(_prepare_folds(dataset), depth)


def main() -> int:
    args = _parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    payloads = _prepare_folds(args.dataset)
    outer_holdout_fraction = 1.0 - OUTER_TRAIN_FRACTION
    result = {
        "dataset": args.dataset,
        "depth": int(args.depth),
        "fold_count": int(len(payloads)),
        "fit_fraction": float(OUTER_TRAIN_FRACTION),
        "val_fraction": float(outer_holdout_fraction * INNER_TRAIN_FRACTION),
        "test_fraction": float(outer_holdout_fraction * (1.0 - INNER_TRAIN_FRACTION)),
        "fold_sizes": [
            {
                "fit": payload.n_fit,
                "val": payload.n_val,
                "test": payload.n_test,
            }
            for payload in payloads
        ],
        "branching_factor": BRANCHING_FACTOR,
        "msplit": _tune_msplit_payloads(payloads, int(args.depth)),
        "shapecart": _tune_shapecart_payloads(payloads, int(args.depth)),
    }
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
