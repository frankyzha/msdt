#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils import DATASET_LOADERS, encode_binary_target, make_preprocessor

SPLIT_SRC = REPO_ROOT / "SPLIT-ICML" / "split" / "src"
if str(SPLIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPLIT_SRC))

CACHE_VERSION = 5
CACHE_REQUIRED_KEYS = {
    "X_fit_proc",
    "X_test_proc",
    "y_fit",
    "y_test",
    "Z_fit",
    "Z_test",
    "teacher_logit",
    "teacher_boundary_gain",
    "teacher_boundary_cover",
    "teacher_boundary_value_jump",
}


def _slice_rows(x, idx: np.ndarray):
    if hasattr(x, "iloc"):
        return x.iloc[idx]
    return x[idx]


def _protocol_split_indices(
    y_bin: np.ndarray,
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, np.ndarray]:
    total = float(test_size) + float(val_size)
    if total >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {total:.6f}")

    all_idx = np.arange(y_bin.shape[0], dtype=np.int32)
    train_size = 1.0 - float(test_size) - float(val_size)
    idx_fit, idx_holdout = train_test_split(
        all_idx,
        train_size=float(train_size),
        random_state=int(seed),
        stratify=y_bin,
    )
    y_holdout = y_bin[idx_holdout]
    holdout_size = float(test_size) + float(val_size)
    val_fraction_within_holdout = float(val_size) / holdout_size
    idx_val, idx_test = train_test_split(
        idx_holdout,
        train_size=float(val_fraction_within_holdout),
        random_state=int(seed) + 1,
        stratify=y_holdout,
    )
    idx_fit = np.asarray(idx_fit, dtype=np.int32)
    idx_val = np.asarray(idx_val, dtype=np.int32)
    idx_test = np.asarray(idx_test, dtype=np.int32)
    return {
        "idx_fit": np.asarray(idx_fit, dtype=np.int32),
        "idx_val": np.asarray(idx_val, dtype=np.int32),
        "idx_test": np.asarray(idx_test, dtype=np.int32),
    }


def default_cache_path(
    dataset: str,
    seed: int,
    test_size: float,
    val_size: float,
    max_bins: int,
    min_samples_leaf: int,
    min_child_size: int,
) -> Path:
    stem = (
        f"{dataset}_seed{int(seed)}_"
        f"test{str(float(test_size)).replace('.', 'p')}_"
        f"val{str(float(val_size)).replace('.', 'p')}_"
        f"bins{int(max_bins)}_leaf{int(min_samples_leaf)}_child{int(min_child_size)}"
    )
    if dataset == "compas":
        stem += "_raw"
    if CACHE_VERSION:
        stem += f"_v{CACHE_VERSION}"
    return REPO_ROOT / "results" / "cache" / "lightgbm_binner" / f"{stem}.npz"


def derive_min_child_size(*, leaf_frac: float, n_fit: int) -> int:
    frac = float(leaf_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"leaf_frac must be positive and finite, got {leaf_frac!r}")
    return max(2, int(math.ceil(frac * max(1, int(n_fit)))))


def derive_min_split_size(*, leaf_frac: float, n_fit: int) -> int:
    frac = float(leaf_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"leaf_frac must be positive and finite, got {leaf_frac!r}")
    return max(2, int(math.ceil((2.0 * frac) * max(1, int(n_fit)))))


def build_cache(
    *,
    dataset: str,
    depth: int,
    seed: int,
    test_size: float,
    val_size: float,
    max_bins: int,
    min_samples_leaf: int,
    min_child_size: int,
    lgb_num_threads: int,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    from lightgbm_binning import fit_lightgbm_binner

    print(f"[cache] loading dataset={dataset} seed={seed}", flush=True)
    X, y = DATASET_LOADERS[dataset]()
    y_bin = encode_binary_target(y, dataset)
    split_idx = _protocol_split_indices(
        y_bin=np.asarray(y_bin, dtype=np.int32),
        seed=int(seed),
        test_size=float(test_size),
        val_size=float(val_size),
    )
    idx_fit = split_idx["idx_fit"]
    idx_val = split_idx["idx_val"]
    idx_test = split_idx["idx_test"]

    X_fit = _slice_rows(X, idx_fit)
    X_val = _slice_rows(X, idx_val)
    X_test = _slice_rows(X, idx_test)
    y_fit = y_bin[idx_fit]
    y_val = y_bin[idx_val]
    y_test = y_bin[idx_test]

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.asarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.asarray(pre.transform(X_val), dtype=np.float32)
    X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)

    print("[cache] fitting LightGBM binner", flush=True)
    def _progress(message: str) -> None:
        print(f"[cache] {message}", flush=True)

    started = time.perf_counter()
    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=int(max_bins),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_estimators=10000,
        num_leaves=255,
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        min_data_in_bin=1,
        min_data_in_leaf=int(min_samples_leaf),
        lambda_l2=0.0,
        early_stopping_rounds=100,
        num_threads=int(lgb_num_threads),
        device_type="cpu",
        collect_teacher_logit=True,
        progress_callback=_progress,
    )
    build_seconds = time.perf_counter() - started
    print(f"[cache] binner done in {build_seconds:.3f}s", flush=True)

    Z_fit = binner.transform(X_fit_proc)
    Z_test = binner.transform(X_test_proc)
    arrays = {
        "idx_fit": np.asarray(idx_fit, dtype=np.int32),
        "idx_val": np.asarray(idx_val, dtype=np.int32),
        "idx_test": np.asarray(idx_test, dtype=np.int32),
        "X_fit_proc": np.asarray(X_fit_proc, dtype=np.float32),
        "X_val_proc": np.asarray(X_val_proc, dtype=np.float32),
        "X_test_proc": np.asarray(X_test_proc, dtype=np.float32),
        "y_fit": np.asarray(y_fit, dtype=np.int32),
        "y_val": np.asarray(y_val, dtype=np.int32),
        "y_test": np.asarray(y_test, dtype=np.int32),
        "Z_fit": np.asarray(Z_fit, dtype=np.int32),
        "Z_test": np.asarray(Z_test, dtype=np.int32),
        "teacher_logit": np.asarray(getattr(binner, "teacher_train_logit"), dtype=np.float64),
        "teacher_boundary_gain": np.asarray(getattr(binner, "boundary_gain_per_feature"), dtype=np.float64),
        "teacher_boundary_cover": np.asarray(getattr(binner, "boundary_cover_per_feature"), dtype=np.float64),
        "teacher_boundary_value_jump": np.asarray(
            getattr(binner, "boundary_value_jump_per_feature"), dtype=np.float64
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    meta = {
        "dataset": dataset,
        "depth": int(depth),
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "max_bins": int(max_bins),
        "min_samples_leaf": int(min_samples_leaf),
        "min_child_size": int(min_child_size),
        "lgb_num_threads": int(lgb_num_threads),
        "build_seconds": float(build_seconds),
        "cache_file": cache_path.name,
        "cache_bytes": int(cache_path.stat().st_size),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return arrays


def cache_is_complete(cache: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    missing = sorted(key for key in CACHE_REQUIRED_KEYS if key not in cache)
    return (len(missing) == 0), missing


def load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as npz:
        cache = {k: npz[k] for k in npz.files}
    return cache


def load_local_libgosdt():
    build_override = os.environ.get("MSPLIT_BUILD_DIR")
    if build_override:
        build_dir = Path(build_override)
        if not build_dir.is_absolute():
            build_dir = (REPO_ROOT / "SPLIT-ICML" / "split" / build_dir).resolve()
    else:
        build_dir = REPO_ROOT / "SPLIT-ICML" / "split" / "build-fast-py"
    candidates = sorted(build_dir.glob("_libgosdt*.so"))
    if candidates:
        so_path = candidates[0]
        spec = importlib.util.spec_from_file_location("split._libgosdt", so_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load _libgosdt from {so_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["split._libgosdt"] = module
        spec.loader.exec_module(module)
        return module

    try:
        from split import _libgosdt as live_module

        return live_module
    except Exception:
        pass

    raise FileNotFoundError(f"Could not find built _libgosdt extension under {build_dir}")


def _predict_tree_row(tree: dict[str, object], row: np.ndarray) -> int:
    cur = tree
    while cur.get("type") == "node":
        feature = int(cur["feature"])
        bin_id = int(row[feature])
        next_child = None
        nearest_group = None
        best_dist = None
        best_lo = None
        for group in cur.get("groups", []):
            spans = group.get("spans", [])
            for lo, hi in spans:
                if int(lo) <= bin_id <= int(hi):
                    next_child = group["child"]
                    break
            if next_child is not None:
                break
            if not spans:
                continue
            group_lo = min(int(lo) for lo, _ in spans)
            group_dist = min(
                0 if int(lo) <= bin_id <= int(hi) else min(abs(bin_id - int(lo)), abs(bin_id - int(hi)))
                for lo, hi in spans
            )
            if best_dist is None or group_dist < best_dist or (
                group_dist == best_dist and (best_lo is None or group_lo < best_lo)
            ):
                best_dist = group_dist
                best_lo = group_lo
                nearest_group = group
        if next_child is None:
            if nearest_group is None:
                return int(cur.get("fallback_prediction", 0))
            next_child = nearest_group["child"]
        cur = next_child
    return int(cur.get("prediction", 0))


def predict_tree(tree: dict[str, object], z: np.ndarray) -> np.ndarray:
    preds = np.empty(z.shape[0], dtype=np.int32)
    for i in range(z.shape[0]):
        preds[i] = _predict_tree_row(tree, z[i])
    return preds


def tree_stats(tree: dict[str, object]) -> dict[str, int]:
    if tree.get("type") == "leaf":
        return {"n_leaves": 1, "n_internal": 0, "max_arity": 0}

    n_leaves = 0
    n_internal = 1
    max_arity = int(len(tree.get("groups", [])))
    for group in tree.get("groups", []):
        child_stats = tree_stats(group["child"])
        n_leaves += int(child_stats["n_leaves"])
        n_internal += int(child_stats["n_internal"])
        max_arity = max(max_arity, int(child_stats["max_arity"]))
    return {
        "n_leaves": n_leaves,
        "n_internal": n_internal,
        "max_arity": max_arity,
    }


def root_has_noncontiguous_group(tree: dict[str, object]) -> bool:
    if tree.get("type") != "node":
        return False
    for group in tree.get("groups", []):
        if len(group.get("spans", [])) > 1:
            return True
    return False


def run_cached_msplit(
    *,
    cache: dict[str, np.ndarray],
    depth: int,
    lookahead_depth: int,
    reg: float,
    exactify_top_k: int | None,
    min_split_size: int,
    min_child_size: int,
    max_branching: int,
) -> dict[str, object]:
    print("[msplit] starting native atomized fit from cached LightGBM bins", flush=True)
    libgosdt = load_local_libgosdt()
    z_fit = np.asarray(cache["Z_fit"], dtype=np.int32)
    z_test = np.asarray(cache["Z_test"], dtype=np.int32)
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    y_test = np.asarray(cache["y_test"], dtype=np.int32)
    teacher_logit = np.asarray(cache["teacher_logit"], dtype=np.float64)
    sample_weight = np.full(z_fit.shape[0], 1.0 / float(max(1, z_fit.shape[0])), dtype=np.float64)

    started = time.perf_counter()
    cpp_result = libgosdt.msplit_fit(
        z_fit,
        y_fit,
        sample_weight,
        teacher_logit,
        np.asarray(cache["teacher_boundary_gain"], dtype=np.float64),
        np.asarray(cache["teacher_boundary_cover"], dtype=np.float64),
        np.asarray(cache["teacher_boundary_value_jump"], dtype=np.float64),
        int(depth),
        int(lookahead_depth),
        float(reg),
        int(min_split_size),
        int(min_child_size),
        28800.0,
        int(max_branching),
        int(exactify_top_k) if exactify_top_k is not None else 0,
    )
    fit_seconds = time.perf_counter() - started
    print(f"[msplit] fit done in {fit_seconds:.3f}s", flush=True)
    tree = json.loads(str(cpp_result["tree"]))
    pred_train = predict_tree(tree, z_fit)
    pred_test = predict_tree(tree, z_test)
    stats = tree_stats(tree)
    result = {
        "fit_seconds": float(fit_seconds),
        "train_accuracy": float(np.mean(pred_train == y_fit)),
        "test_accuracy": float(np.mean(pred_test == y_test)),
        "objective": float(cpp_result["objective"]),
        "root_feature": int(tree.get("feature", -1)),
        "root_group_count": int(tree.get("group_count", 0)),
        "root_has_noncontiguous_group": bool(root_has_noncontiguous_group(tree)),
        "n_leaves": int(stats["n_leaves"]),
        "n_internal": int(stats["n_internal"]),
        "max_arity": int(stats["max_arity"]),
        "greedy_internal_nodes": int(cpp_result.get("greedy_internal_nodes", 0)),
        "greedy_subproblem_calls": int(cpp_result.get("greedy_subproblem_calls", 0)),
        "greedy_cache_hits": int(cpp_result.get("greedy_cache_hits", 0)),
        "greedy_unique_states": int(cpp_result.get("greedy_unique_states", 0)),
        "greedy_interval_evals": int(cpp_result.get("greedy_interval_evals", 0)),
        "elapsed_time_sec": float(cpp_result.get("elapsed_time_sec", 0.0)),
        "debr_refine_calls": int(cpp_result.get("debr_refine_calls", 0)),
        "debr_refine_improved": int(cpp_result.get("debr_refine_improved", 0)),
        "debr_total_moves": int(cpp_result.get("debr_total_moves", 0)),
        "debr_total_hard_gain": float(cpp_result.get("debr_total_hard_gain", 0.0)),
        "debr_total_soft_gain": float(cpp_result.get("debr_total_soft_gain", 0.0)),
        "debr_total_delta_j": float(cpp_result.get("debr_total_delta_j", 0.0)),
        "debr_total_component_delta": int(cpp_result.get("debr_total_component_delta", 0)),
        "debr_final_geo_wins": int(cpp_result.get("debr_final_geo_wins", 0)),
        "debr_final_block_wins": int(cpp_result.get("debr_final_block_wins", 0)),
        "family_compare_total": int(cpp_result.get("family_compare_total", 0)),
        "family_compare_equivalent": int(cpp_result.get("family_compare_equivalent", 0)),
        "family1_both_wins": int(cpp_result.get("family1_both_wins", 0)),
        "family2_hard_loss_wins": int(cpp_result.get("family2_hard_loss_wins", 0)),
        "family2_hard_impurity_wins": int(cpp_result.get("family2_hard_impurity_wins", 0)),
        "family2_both_wins": int(cpp_result.get("family2_both_wins", 0)),
        "family_metric_disagreement": int(cpp_result.get("family_metric_disagreement", 0)),
        "family_hard_loss_ties": int(cpp_result.get("family_hard_loss_ties", 0)),
        "family_hard_impurity_ties": int(cpp_result.get("family_hard_impurity_ties", 0)),
        "family_joint_impurity_ties": int(cpp_result.get("family_joint_impurity_ties", 0)),
        "family_neither_both_wins": int(cpp_result.get("family_neither_both_wins", 0)),
        "family1_selected_by_equivalence": int(cpp_result.get("family1_selected_by_equivalence", 0)),
        "family1_selected_by_dominance": int(cpp_result.get("family1_selected_by_dominance", 0)),
        "family2_selected_by_dominance": int(cpp_result.get("family2_selected_by_dominance", 0)),
        "family_sent_both": int(cpp_result.get("family_sent_both", 0)),
        "family1_hard_loss_sum": float(cpp_result.get("family1_hard_loss_sum", 0.0)),
        "family2_hard_loss_sum": float(cpp_result.get("family2_hard_loss_sum", 0.0)),
        "family_hard_loss_delta_sum": float(cpp_result.get("family_hard_loss_delta_sum", 0.0)),
        "family1_hard_impurity_sum": float(cpp_result.get("family1_hard_impurity_sum", 0.0)),
        "family2_hard_impurity_sum": float(cpp_result.get("family2_hard_impurity_sum", 0.0)),
        "family_hard_impurity_delta_sum": float(cpp_result.get("family_hard_impurity_delta_sum", 0.0)),
        "family1_joint_impurity_sum": float(cpp_result.get("family1_joint_impurity_sum", 0.0)),
        "family2_joint_impurity_sum": float(cpp_result.get("family2_joint_impurity_sum", 0.0)),
        "family_joint_impurity_delta_sum": float(cpp_result.get("family_joint_impurity_delta_sum", 0.0)),
        "family1_soft_impurity_sum": float(cpp_result.get("family1_soft_impurity_sum", 0.0)),
        "family2_soft_impurity_sum": float(cpp_result.get("family2_soft_impurity_sum", 0.0)),
        "family_soft_impurity_delta_sum": float(cpp_result.get("family_soft_impurity_delta_sum", 0.0)),
        "family2_joint_impurity_wins": int(cpp_result.get("family2_joint_impurity_wins", 0)),
        "family1_hard_loss_inversion_traces": cpp_result.get("family1_hard_loss_inversion_traces", []),
        "teacher_available": bool(cpp_result.get("teacher_available", False)),
        "n_classes": int(cpp_result.get("n_classes", 0)),
        "teacher_class_count": int(cpp_result.get("teacher_class_count", 0)),
        "binary_mode": bool(cpp_result.get("binary_mode", True)),
        "atomized_features_prepared": int(cpp_result.get("atomized_features_prepared", 0)),
        "atomized_coarse_candidates": int(cpp_result.get("atomized_coarse_candidates", 0)),
        "atomized_coarse_pruned_candidates": int(
            cpp_result.get("atomized_coarse_pruned_candidates", 0)
        ),
        "atomized_final_candidates": int(cpp_result.get("atomized_final_candidates", 0)),
        "greedy_feature_survivor_histogram": cpp_result.get(
            "greedy_feature_survivor_histogram", []
        ),
        "greedy_candidate_count_histogram": cpp_result.get(
            "greedy_candidate_count_histogram", []
        ),
        "per_node_candidate_count": cpp_result.get(
            "per_node_candidate_count", []
        ),
        "per_node_total_weight": cpp_result.get("per_node_total_weight", []),
        "per_node_mu_node": cpp_result.get("per_node_mu_node", []),
        "per_node_candidate_upper_bounds": cpp_result.get(
            "per_node_candidate_upper_bounds", []
        ),
        "per_node_candidate_lower_bounds": cpp_result.get(
            "per_node_candidate_lower_bounds", []
        ),
        "per_node_candidate_hard_loss": cpp_result.get("per_node_candidate_hard_loss", []),
        "per_node_candidate_impurity_objective": cpp_result.get(
            "per_node_candidate_impurity_objective", []
        ),
        "per_node_candidate_hard_impurity": cpp_result.get(
            "per_node_candidate_hard_impurity", []
        ),
        "per_node_candidate_soft_impurity": cpp_result.get(
            "per_node_candidate_soft_impurity", []
        ),
        "per_node_candidate_boundary_penalty": cpp_result.get(
            "per_node_candidate_boundary_penalty", []
        ),
        "per_node_candidate_components": cpp_result.get(
            "per_node_candidate_components", []
        ),
        "nominee_certificate_nodes": int(cpp_result.get("nominee_certificate_nodes", 0)),
        "nominee_certificate_exhausted_nodes": int(
            cpp_result.get("nominee_certificate_exhausted_nodes", 0)
        ),
        "nominee_exactified_until_certificate_total": int(
            cpp_result.get("nominee_exactified_until_certificate_total", 0)
        ),
        "nominee_exactified_until_certificate_max": int(
            cpp_result.get("nominee_exactified_until_certificate_max", 0)
        ),
        "nominee_certificate_min_remaining_lower_bound_sum": float(
            cpp_result.get("nominee_certificate_min_remaining_lower_bound_sum", 0.0)
        ),
        "nominee_certificate_min_remaining_lower_bound_max": float(
            cpp_result.get("nominee_certificate_min_remaining_lower_bound_max", 0.0)
        ),
        "nominee_certificate_incumbent_exact_score_sum": float(
            cpp_result.get("nominee_certificate_incumbent_exact_score_sum", 0.0)
        ),
        "nominee_certificate_incumbent_exact_score_max": float(
            cpp_result.get("nominee_certificate_incumbent_exact_score_max", 0.0)
        ),
        "nominee_exactified_until_certificate_histogram": cpp_result.get(
            "nominee_exactified_until_certificate_histogram", []
        ),
        "nominee_certificate_stop_depth_histogram": cpp_result.get(
            "nominee_certificate_stop_depth_histogram", []
        ),
        "nominee_elbow_prefix_total": int(cpp_result.get("nominee_elbow_prefix_total", 0)),
        "nominee_elbow_prefix_max": int(cpp_result.get("nominee_elbow_prefix_max", 0)),
        "nominee_elbow_prefix_histogram": cpp_result.get(
            "nominee_elbow_prefix_histogram", []
        ),
        "atomized_feature_atom_count_histogram": cpp_result.get("atomized_feature_atom_count_histogram", []),
        "atomized_feature_block_atom_count_histogram": cpp_result.get(
            "atomized_feature_block_atom_count_histogram", []
        ),
        "atomized_feature_q_effective_histogram": cpp_result.get(
            "atomized_feature_q_effective_histogram", []
        ),
        "heuristic_selector_nodes": int(cpp_result.get("heuristic_selector_nodes", 0)),
        "heuristic_selector_candidate_total": int(
            cpp_result.get("heuristic_selector_candidate_total", 0)
        ),
        "heuristic_selector_candidate_pruned_total": int(
            cpp_result.get("heuristic_selector_candidate_pruned_total", 0)
        ),
        "heuristic_selector_survivor_total": int(
            cpp_result.get("heuristic_selector_survivor_total", 0)
        ),
        "heuristic_selector_leaf_optimal_nodes": int(
            cpp_result.get("heuristic_selector_leaf_optimal_nodes", 0)
        ),
        "heuristic_selector_improving_split_nodes": int(
            cpp_result.get("heuristic_selector_improving_split_nodes", 0)
        ),
        "heuristic_selector_improving_split_retained_nodes": int(
            cpp_result.get("heuristic_selector_improving_split_retained_nodes", 0)
        ),
        "heuristic_selector_improving_split_margin_sum": float(
            cpp_result.get("heuristic_selector_improving_split_margin_sum", 0.0)
        ),
        "heuristic_selector_improving_split_margin_max": float(
            cpp_result.get("heuristic_selector_improving_split_margin_max", 0.0)
        ),
        "heuristic_selector_nodes_by_depth": cpp_result.get("heuristic_selector_nodes_by_depth", []),
        "heuristic_selector_candidate_total_by_depth": cpp_result.get(
            "heuristic_selector_candidate_total_by_depth", []
        ),
        "heuristic_selector_candidate_pruned_total_by_depth": cpp_result.get(
            "heuristic_selector_candidate_pruned_total_by_depth", []
        ),
        "heuristic_selector_survivor_total_by_depth": cpp_result.get(
            "heuristic_selector_survivor_total_by_depth", []
        ),
        "heuristic_selector_leaf_optimal_nodes_by_depth": cpp_result.get(
            "heuristic_selector_leaf_optimal_nodes_by_depth", []
        ),
        "heuristic_selector_improving_split_nodes_by_depth": cpp_result.get(
            "heuristic_selector_improving_split_nodes_by_depth", []
        ),
        "heuristic_selector_improving_split_retained_nodes_by_depth": cpp_result.get(
            "heuristic_selector_improving_split_retained_nodes_by_depth", []
        ),
        "heuristic_selector_improving_split_margin_sum_by_depth": cpp_result.get(
            "heuristic_selector_improving_split_margin_sum_by_depth", []
        ),
        "heuristic_selector_improving_split_margin_max_by_depth": cpp_result.get(
            "heuristic_selector_improving_split_margin_max_by_depth", []
        ),
    }
    comparisons = int(result["family_compare_total"])
    if comparisons > 0:
        result["family2_selected_rate"] = float(result["debr_final_block_wins"]) / float(comparisons)
        result["family2_hard_loss_win_rate"] = float(result["family2_hard_loss_wins"]) / float(comparisons)
        result["family2_hard_impurity_win_rate"] = float(result["family2_hard_impurity_wins"]) / float(comparisons)
        result["family2_both_win_rate"] = float(result["family2_both_wins"]) / float(comparisons)
        result["family_metric_disagreement_rate"] = float(result["family_metric_disagreement"]) / float(comparisons)
        result["family_equivalent_rate"] = float(result["family_compare_equivalent"]) / float(comparisons)
        result["family_hard_loss_tie_rate"] = float(result["family_hard_loss_ties"]) / float(comparisons)
        result["family_hard_impurity_tie_rate"] = float(result["family_hard_impurity_ties"]) / float(comparisons)
        result["family_joint_impurity_tie_rate"] = float(result["family_joint_impurity_ties"]) / float(comparisons)
        result["family_neither_both_wins_rate"] = float(result["family_neither_both_wins"]) / float(comparisons)
        result["family1_selected_by_equivalence_rate"] = float(result["family1_selected_by_equivalence"]) / float(comparisons)
        result["family1_selected_by_dominance_rate"] = float(result["family1_selected_by_dominance"]) / float(comparisons)
        result["family2_selected_by_dominance_rate"] = float(result["family2_selected_by_dominance"]) / float(comparisons)
        result["family_sent_both_rate"] = float(result["family_sent_both"]) / float(comparisons)
        result["family1_mean_hard_loss"] = float(result["family1_hard_loss_sum"]) / float(comparisons)
        result["family2_mean_hard_loss"] = float(result["family2_hard_loss_sum"]) / float(comparisons)
        result["family1_mean_hard_impurity"] = float(result["family1_hard_impurity_sum"]) / float(comparisons)
        result["family2_mean_hard_impurity"] = float(result["family2_hard_impurity_sum"]) / float(comparisons)
        result["family_mean_hard_loss_delta"] = float(result["family_hard_loss_delta_sum"]) / float(comparisons)
        result["family_mean_hard_impurity_delta"] = float(result["family_hard_impurity_delta_sum"]) / float(comparisons)
        result["family2_joint_impurity_win_rate"] = float(result["family2_joint_impurity_wins"]) / float(comparisons)
        result["family1_mean_joint_impurity"] = float(result["family1_joint_impurity_sum"]) / float(comparisons)
        result["family2_mean_joint_impurity"] = float(result["family2_joint_impurity_sum"]) / float(comparisons)
        result["family_mean_joint_impurity_delta"] = float(result["family_joint_impurity_delta_sum"]) / float(comparisons)
        result["family1_mean_soft_impurity"] = float(result["family1_soft_impurity_sum"]) / float(comparisons)
        result["family2_mean_soft_impurity"] = float(result["family2_soft_impurity_sum"]) / float(comparisons)
        result["family_mean_soft_impurity_delta"] = float(result["family_soft_impurity_delta_sum"]) / float(comparisons)
    if result["atomized_coarse_candidates"] > 0:
        result["atomized_coarse_prune_rate"] = (
            float(result["atomized_coarse_pruned_candidates"]) /
            float(result["atomized_coarse_candidates"])
        )
        result["atomized_coarse_survivor_rate"] = (
            1.0 - result["atomized_coarse_prune_rate"]
        )
    if result["heuristic_selector_nodes"] > 0:
        nodes = float(result["heuristic_selector_nodes"])
        total_candidates = float(result["heuristic_selector_candidate_total"])
        if total_candidates > 0.0:
            result["heuristic_selector_candidate_prune_rate"] = (
                float(result["heuristic_selector_candidate_pruned_total"]) / total_candidates
            )
            result["heuristic_selector_candidate_survivor_rate"] = (
                float(result["heuristic_selector_survivor_total"]) / total_candidates
            )
        result["heuristic_selector_leaf_optimal_rate"] = (
            float(result["heuristic_selector_leaf_optimal_nodes"]) / nodes
        )
        improving_nodes = float(result["heuristic_selector_improving_split_nodes"])
        if improving_nodes > 0.0:
            result["heuristic_selector_improving_split_retained_rate"] = (
                float(result["heuristic_selector_improving_split_retained_nodes"]) / improving_nodes
            )
            result["heuristic_selector_improving_split_margin_mean"] = (
                float(result["heuristic_selector_improving_split_margin_sum"]) / improving_nodes
            )
    heuristic_nodes_by_depth = result.get("heuristic_selector_nodes_by_depth", [])
    if heuristic_nodes_by_depth:
        cand_total_by_depth = result.get("heuristic_selector_candidate_total_by_depth", [])
        cand_pruned_by_depth = result.get("heuristic_selector_candidate_pruned_total_by_depth", [])
        survivor_by_depth = result.get("heuristic_selector_survivor_total_by_depth", [])
        leaf_opt_by_depth = result.get("heuristic_selector_leaf_optimal_nodes_by_depth", [])
        improving_by_depth = result.get("heuristic_selector_improving_split_nodes_by_depth", [])
        retained_by_depth = result.get("heuristic_selector_improving_split_retained_nodes_by_depth", [])
        improving_margin_sum_by_depth = result.get("heuristic_selector_improving_split_margin_sum_by_depth", [])
        result["heuristic_selector_candidate_prune_rate_by_depth"] = [
            (float(cand_pruned_by_depth[i]) / float(cand_total_by_depth[i]))
            if i < len(cand_total_by_depth) and float(cand_total_by_depth[i]) > 0.0
            else 0.0
            for i in range(len(heuristic_nodes_by_depth))
        ]
        result["heuristic_selector_candidate_survivor_rate_by_depth"] = [
            (float(survivor_by_depth[i]) / float(cand_total_by_depth[i]))
            if i < len(cand_total_by_depth) and float(cand_total_by_depth[i]) > 0.0
            else 0.0
            for i in range(len(heuristic_nodes_by_depth))
        ]
        result["heuristic_selector_leaf_optimal_rate_by_depth"] = [
            (float(leaf_opt_by_depth[i]) / float(heuristic_nodes_by_depth[i]))
            if float(heuristic_nodes_by_depth[i]) > 0.0 and i < len(leaf_opt_by_depth)
            else 0.0
            for i in range(len(heuristic_nodes_by_depth))
        ]
        result["heuristic_selector_improving_split_retained_rate_by_depth"] = [
            (float(retained_by_depth[i]) / float(improving_by_depth[i]))
            if i < len(improving_by_depth) and float(improving_by_depth[i]) > 0.0
            else 0.0
            for i in range(len(heuristic_nodes_by_depth))
        ]
        result["heuristic_selector_improving_split_margin_mean_by_depth"] = [
            (float(improving_margin_sum_by_depth[i]) / float(improving_by_depth[i]))
            if i < len(improving_by_depth) and float(improving_by_depth[i]) > 0.0
            else 0.0
            for i in range(len(heuristic_nodes_by_depth))
        ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or reuse a cached electricity LightGBM binning artifact and benchmark MSPLIT directly."
    )
    parser.add_argument("--dataset", default="electricity", choices=sorted(DATASET_LOADERS.keys()))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.125)
    parser.add_argument("--max-bins", type=int, default=1024)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-split-size", type=int, default=0)
    parser.add_argument("--min-child-size", type=int, default=0)
    parser.add_argument(
        "--leaf-frac",
        type=float,
        default=0.001,
        help=(
            "If support sizes are not set explicitly, derive min_child_size as ceil(leaf_frac * n_fit) "
            "and min_split_size as ceil(2 * leaf_frac * n_fit)."
        ),
    )
    parser.add_argument("--proposal-atom-cap", type=int, default=0)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--reg", type=float, default=0.0005)
    parser.add_argument(
        "--exactify-top-k",
        type=int,
        default=None,
        help="If set, exactify at most this many shortlisted candidates per node above lookahead depth.",
    )
    parser.add_argument("--lgb-num-threads", type=int, default=3)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument(
        "--build-cache-only",
        action="store_true",
        help="Materialize or refresh the LightGBM cache and exit before fitting MSPLIT.",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    if args.exactify_top_k is not None and int(args.exactify_top_k) < 1:
        raise ValueError("--exactify-top-k must be a positive integer when specified")

    preview_X, preview_y = DATASET_LOADERS[args.dataset]()
    preview_y_bin = encode_binary_target(preview_y, args.dataset)
    split_idx = _protocol_split_indices(
        y_bin=np.asarray(preview_y_bin, dtype=np.int32),
        seed=int(args.seed),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
    )
    n_fit = int(split_idx["idx_fit"].shape[0])
    resolved_min_child_size = int(args.min_child_size)
    if resolved_min_child_size <= 0:
        resolved_min_child_size = derive_min_child_size(leaf_frac=float(args.leaf_frac), n_fit=n_fit)
    resolved_min_split_size = int(args.min_split_size)
    if resolved_min_split_size <= 0:
        resolved_min_split_size = derive_min_split_size(leaf_frac=float(args.leaf_frac), n_fit=n_fit)

    cache_path = args.cache_path
    if cache_path is None:
        cache_path = default_cache_path(
            args.dataset,
            args.seed,
            args.test_size,
            args.val_size,
            args.max_bins,
            args.min_samples_leaf,
            resolved_min_child_size,
        )

    cache_hit = cache_path.exists() and not args.force_rebuild_cache
    if cache_hit:
        print(f"[cache] hit: {cache_path}", flush=True)
        cache = load_cache(cache_path)
        cache_ok, missing = cache_is_complete(cache)
        if not cache_ok:
            print(f"[cache] stale cache detected, rebuilding (missing: {missing})", flush=True)
            cache_hit = False
            cache = {}
        cache_meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8")) if cache_path.with_suffix(".json").exists() else {}
    else:
        print(f"[cache] miss: {cache_path}", flush=True)
        cache_meta = {}

    if not cache_hit:
        cache = build_cache(
            dataset=args.dataset,
            depth=args.depth,
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
            max_bins=args.max_bins,
            min_samples_leaf=args.min_samples_leaf,
            min_child_size=resolved_min_child_size,
            lgb_num_threads=args.lgb_num_threads,
            cache_path=cache_path,
        )
        cache_meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))

    if args.build_cache_only:
        print(f"[cache] build-only complete: {cache_path}", flush=True)
        return 0

    result = run_cached_msplit(
        cache=cache,
        depth=args.depth,
        lookahead_depth=args.lookahead_depth,
        reg=args.reg,
        exactify_top_k=args.exactify_top_k,
        min_split_size=resolved_min_split_size,
        min_child_size=resolved_min_child_size,
        max_branching=args.max_branching,
    )
    result.update(
        {
            "dataset": args.dataset,
            "depth": int(args.depth),
            "full_depth_budget": int(args.depth),
            "cache_path": str(cache_path),
            "cache_hit": bool(cache_hit),
            "cache_build_seconds": float(cache_meta.get("build_seconds", 0.0)) if cache_meta else 0.0,
            "resolved_min_split_size": int(resolved_min_split_size),
            "resolved_min_child_size": int(resolved_min_child_size),
            "lookahead_depth": int(args.lookahead_depth),
            "n_fit": int(n_fit),
            "reg": float(args.reg),
            "exactify_top_k": (
                int(args.exactify_top_k) if args.exactify_top_k is not None else None
            ),
        }
    )
    print(json.dumps(result, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
