#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.scripts.benchmark_paths import (
    ensure_repo_import_paths,
    resolve_msplit_build_dir,
)
from benchmark.scripts.cache_utils import (
    DEFAULT_CACHE_SEED,
    DEFAULT_LGB_NUM_THREADS,
    DEFAULT_MAX_BINS,
    DEFAULT_MIN_CHILD_SIZE,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    build_cache,
    default_cache_path,
    resolve_compatible_cache,
    resolve_protocol_support_sizes,
)
from benchmark.scripts.msplit_benchmark_defaults import (
    DEFAULT_EXACTIFY_TOP_K,
    DEFAULT_LOOKAHEAD_DEPTH,
    DEFAULT_MAX_BRANCHING,
    DEFAULT_MIN_SPLIT_SIZE,
    DEFAULT_REG,
)
from benchmark.scripts.experiment_utils import DATASET_LOADERS

ensure_repo_import_paths(include_msplit_src=True)


@lru_cache(maxsize=1)
def load_local_libgosdt():
    build_override = os.environ.get("MSPLIT_BUILD_DIR")
    build_dir_candidates = (
        [build_override]
        if build_override
        else [
            # Prefer the maintained nonlinear build when available.
            "build-nonlinear-py",
            # Fall back to the historical fast-build name only if it exists locally.
            "build-fast-py",
        ]
    )
    for build_dir_name in build_dir_candidates:
        build_dir = resolve_msplit_build_dir(build_dir_name)
        candidates = sorted(build_dir.glob("_libgosdt*.so"))
        if not candidates:
            continue
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

    raise FileNotFoundError(
        "Could not find a built _libgosdt extension. "
        f"Tried build dirs: {build_dir_candidates}"
    )


@dataclass(frozen=True)
class _CompiledLeaf:
    prediction: int


@dataclass(frozen=True)
class _CompiledGroup:
    spans: tuple[tuple[int, int], ...]
    child: "_CompiledTree"
    min_lo: int | None


@dataclass(frozen=True)
class _CompiledNode:
    feature: int
    fallback_prediction: int
    groups: tuple[_CompiledGroup, ...]


_CompiledTree = _CompiledLeaf | _CompiledNode


def _compile_tree(tree: dict[str, object]) -> _CompiledTree:
    if tree.get("type") != "node":
        return _CompiledLeaf(prediction=int(tree.get("prediction", 0)))

    groups = []
    for group in tree.get("groups", []):
        spans = tuple((int(lo), int(hi)) for lo, hi in group.get("spans", []))
        groups.append(
            _CompiledGroup(
                spans=spans,
                child=_compile_tree(group["child"]),
                min_lo=min((lo for lo, _ in spans), default=None),
            )
        )
    return _CompiledNode(
        feature=int(tree["feature"]),
        fallback_prediction=int(tree.get("fallback_prediction", 0)),
        groups=tuple(groups),
    )


def _predict_tree_row(tree: _CompiledTree, row: np.ndarray) -> int:
    cur = tree
    while isinstance(cur, _CompiledNode):
        bin_id = int(row[cur.feature])
        next_child = None
        nearest_group = None
        best_dist = None
        best_lo = None
        for group in cur.groups:
            for lo, hi in group.spans:
                if lo <= bin_id <= hi:
                    next_child = group.child
                    break
            if next_child is not None:
                break
            if not group.spans:
                continue
            group_dist = min(
                min(abs(bin_id - lo), abs(bin_id - hi))
                for lo, hi in group.spans
            )
            if best_dist is None or group_dist < best_dist or (
                group_dist == best_dist and (best_lo is None or (group.min_lo is not None and group.min_lo < best_lo))
            ):
                best_dist = group_dist
                best_lo = group.min_lo
                nearest_group = group
        if next_child is not None:
            cur = next_child
            continue
        if nearest_group is None:
            return cur.fallback_prediction
        cur = nearest_group.child
    return cur.prediction


def predict_tree(tree: dict[str, object], z: np.ndarray) -> np.ndarray:
    compiled_tree = _compile_tree(tree)
    preds = np.empty(z.shape[0], dtype=np.int32)
    for i, row in enumerate(z):
        preds[i] = _predict_tree_row(compiled_tree, row)
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
    return any(len(group.get("spans", [])) > 1 for group in tree.get("groups", []))


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
    z_val = np.asarray(cache["Z_val"], dtype=np.int32) if "Z_val" in cache else None
    z_test = np.asarray(cache["Z_test"], dtype=np.int32)
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    y_val = np.asarray(cache["y_val"], dtype=np.int32) if "y_val" in cache else None
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
    pred_val = predict_tree(tree, z_val) if z_val is not None else None
    pred_test = predict_tree(tree, z_test)
    stats = tree_stats(tree)
    result = {
        "fit_seconds": float(fit_seconds),
        "train_accuracy": float(np.mean(pred_train == y_fit)),
        "val_accuracy": float(np.mean(pred_val == y_val)) if pred_val is not None and y_val is not None else None,
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
        "nominee_exactify_prefix_total": int(
            cpp_result.get(
                "nominee_exactify_prefix_total",
                cpp_result.get("nominee_elbow_prefix_total", 0),
            )
        ),
        "nominee_exactify_prefix_max": int(
            cpp_result.get(
                "nominee_exactify_prefix_max",
                cpp_result.get("nominee_elbow_prefix_max", 0),
            )
        ),
        "nominee_exactify_prefix_histogram": cpp_result.get(
            "nominee_exactify_prefix_histogram",
            cpp_result.get("nominee_elbow_prefix_histogram", []),
        ),
        "profiling_lp_solve_calls": int(cpp_result.get("profiling_lp_solve_calls", 0)),
        "profiling_lp_solve_sec": float(cpp_result.get("profiling_lp_solve_sec", 0.0)),
        "profiling_pricing_calls": int(cpp_result.get("profiling_pricing_calls", 0)),
        "profiling_pricing_sec": float(cpp_result.get("profiling_pricing_sec", 0.0)),
        "profiling_greedy_complete_calls": int(
            cpp_result.get("profiling_greedy_complete_calls", 0)
        ),
        "profiling_greedy_complete_sec": float(
            cpp_result.get("profiling_greedy_complete_sec", 0.0)
        ),
        "profiling_greedy_complete_calls_by_depth": cpp_result.get(
            "profiling_greedy_complete_calls_by_depth", []
        ),
        "profiling_feature_prepare_sec": float(
            cpp_result.get("profiling_feature_prepare_sec", 0.0)
        ),
        "profiling_candidate_nomination_sec": float(
            cpp_result.get("profiling_candidate_nomination_sec", 0.0)
        ),
        "profiling_candidate_shortlist_sec": float(
            cpp_result.get("profiling_candidate_shortlist_sec", 0.0)
        ),
        "profiling_candidate_generation_sec": float(
            cpp_result.get("profiling_candidate_generation_sec", 0.0)
        ),
        "profiling_recursive_child_eval_sec": float(
            cpp_result.get("profiling_recursive_child_eval_sec", 0.0)
        ),
        "profiling_refine_calls": int(cpp_result.get("profiling_refine_calls", 0)),
        "profiling_refine_sec": float(cpp_result.get("profiling_refine_sec", 0.0)),
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
        "above_lookahead_impurity_pairs_total": int(
            cpp_result.get("above_lookahead_impurity_pairs_total", 0)
        ),
        "above_lookahead_hardloss_pairs_total": int(
            cpp_result.get("above_lookahead_hardloss_pairs_total", 0)
        ),
        "above_lookahead_single_pairs_total": int(
            cpp_result.get("above_lookahead_single_pairs_total", 0)
        ),
        "above_lookahead_impurity_bucket_before_prune_total": int(
            cpp_result.get("above_lookahead_impurity_bucket_before_prune_total", 0)
        ),
        "above_lookahead_impurity_bucket_after_prune_total": int(
            cpp_result.get("above_lookahead_impurity_bucket_after_prune_total", 0)
        ),
        "above_lookahead_hardloss_bucket_before_prune_total": int(
            cpp_result.get("above_lookahead_hardloss_bucket_before_prune_total", 0)
        ),
        "above_lookahead_hardloss_bucket_after_prune_total": int(
            cpp_result.get("above_lookahead_hardloss_bucket_after_prune_total", 0)
        ),
        "above_lookahead_single_bucket_before_prune_total": int(
            cpp_result.get("above_lookahead_single_bucket_before_prune_total", 0)
        ),
        "above_lookahead_single_bucket_after_prune_total": int(
            cpp_result.get("above_lookahead_single_bucket_after_prune_total", 0)
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
        description="Build or reuse a cached benchmark LightGBM binning artifact and benchmark MSPLIT directly."
    )
    parser.add_argument("--dataset", default="electricity", choices=sorted(DATASET_LOADERS.keys()))
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=DEFAULT_LOOKAHEAD_DEPTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_CACHE_SEED)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--max-bins", type=int, default=DEFAULT_MAX_BINS)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--min-split-size", type=int, default=DEFAULT_MIN_SPLIT_SIZE)
    parser.add_argument("--min-child-size", type=int, default=DEFAULT_MIN_CHILD_SIZE)
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
    parser.add_argument("--max-branching", type=int, default=DEFAULT_MAX_BRANCHING)
    parser.add_argument("--reg", type=float, default=DEFAULT_REG)
    parser.add_argument(
        "--exactify-top-k",
        type=int,
        default=DEFAULT_EXACTIFY_TOP_K,
        help="If set, exactify at most this many shortlisted candidates per node above lookahead depth.",
    )
    parser.add_argument("--lgb-num-threads", type=int, default=DEFAULT_LGB_NUM_THREADS)
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

    n_fit, resolved_min_child_size, resolved_min_split_size = resolve_protocol_support_sizes(
        dataset=args.dataset,
        seed=int(args.seed),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        leaf_frac=float(args.leaf_frac),
        min_child_size=int(args.min_child_size),
        min_split_size=int(args.min_split_size),
    )

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

    cache_resolved_path, cache, cache_meta, cache_hit, cache_used_fallback = resolve_compatible_cache(
        cache_path,
        force_rebuild=args.force_rebuild_cache,
    )
    if not cache_hit:
        print(f"[cache] miss: {cache_path}", flush=True)
        cache_resolved_path = cache_path
        cache_meta = {}

    if not cache_hit:
        cache = build_cache(
            dataset=args.dataset,
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
            max_bins=args.max_bins,
            min_samples_leaf=args.min_samples_leaf,
            min_child_size=resolved_min_child_size,
            lgb_num_threads=args.lgb_num_threads,
            cache_path=cache_resolved_path,
        )
        cache_meta = json.loads(cache_resolved_path.with_suffix(".json").read_text(encoding="utf-8"))
    if n_fit is None:
        n_fit = int(np.asarray(cache["idx_fit"]).shape[0])

    if args.build_cache_only:
        print(f"[cache] build-only complete: {cache_resolved_path}", flush=True)
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
            "cache_path": str(cache_resolved_path),
            "cache_requested_path": str(cache_path),
            "cache_hit": bool(cache_hit),
            "cache_used_compatible_fallback": bool(cache_used_fallback),
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
