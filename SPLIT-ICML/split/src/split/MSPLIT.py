"""Multiway SPLIT-style tree solver with LightGBM bins and lookahead greedy completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import os
import json
import time
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from ._libgosdt import msplit_fit as _cpp_msplit_fit
except Exception:
    _cpp_msplit_fit = None


def _metric_keys(spec: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in spec.splitlines() if line.strip())


_NATIVE_INT_METRIC_KEYS = _metric_keys(
    """
    family_compare_total
    family_compare_equivalent
    family1_both_wins
    family2_hard_loss_wins
    family2_hard_impurity_wins
    family2_both_wins
    family_metric_disagreement
    family_hard_loss_ties
    family_hard_impurity_ties
    family_joint_impurity_ties
    family_neither_both_wins
    family1_selected_by_equivalence
    family1_selected_by_dominance
    family2_selected_by_dominance
    family_sent_both
    debr_refine_calls
    debr_refine_improved
    debr_total_moves
    debr_bridge_policy_calls
    debr_refine_windowed_calls
    debr_refine_unwindowed_calls
    debr_refine_overlap_segments
    debr_refine_calls_with_overlap
    debr_refine_calls_without_overlap
    debr_candidate_total
    debr_candidate_legal
    debr_candidate_source_size_rejects
    debr_candidate_target_size_rejects
    debr_candidate_descent_eligible
    debr_candidate_descent_rejected
    debr_candidate_bridge_eligible
    debr_candidate_bridge_window_blocked
    debr_candidate_bridge_used_blocked
    debr_candidate_bridge_guide_rejected
    debr_candidate_cleanup_eligible
    debr_candidate_cleanup_primary_rejected
    debr_candidate_cleanup_complexity_rejected
    debr_candidate_score_rejected
    debr_descent_moves
    debr_bridge_moves
    debr_simplify_moves
    debr_total_component_delta
    debr_final_geo_wins
    debr_final_block_wins
    atomized_features_prepared
    atomized_coarse_candidates
    atomized_final_candidates
    atomized_coarse_pruned_candidates
    atomized_compression_features_applied
    atomized_compression_features_collapsed_to_single_block
    atomized_compression_atoms_before_total
    atomized_compression_blocks_after_total
    atomized_compression_atoms_merged_total
    nominee_unique_total
    nominee_child_interval_lookups
    nominee_child_interval_unique
    nominee_exactified_total
    nominee_incumbent_updates
    nominee_threatening_samples
    nominee_threatening_max
    nominee_exactify_prefix_total
    nominee_exactify_prefix_max
    profiling_lp_solve_calls
    profiling_pricing_calls
    profiling_greedy_complete_calls
    heuristic_selector_nodes
    heuristic_selector_candidate_total
    heuristic_selector_candidate_pruned_total
    heuristic_selector_survivor_total
    heuristic_selector_leaf_optimal_nodes
    heuristic_selector_improving_split_nodes
    heuristic_selector_improving_split_retained_nodes
    profiling_refine_calls
    exact_internal_nodes
    greedy_internal_nodes
    dp_subproblem_calls
    dp_cache_hits
    dp_unique_states
    dp_cache_profile_enabled
    dp_cache_lookup_calls
    dp_cache_miss_no_bucket
    dp_cache_miss_bucket_present
    dp_cache_miss_depth_mismatch_only
    dp_cache_miss_indices_mismatch
    dp_cache_depth_match_candidates
    dp_cache_bucket_entries_scanned
    dp_cache_bucket_max_size
    greedy_subproblem_calls
    exact_dp_subproblem_calls_above_lookahead
    greedy_cache_hits
    greedy_unique_states
    greedy_cache_entries_peak
    greedy_interval_evals
    expensive_child_calls
    expensive_child_exactify_calls
    """
)

_NATIVE_FLOAT_METRIC_KEYS = _metric_keys(
    """
    family1_hard_loss_sum
    family2_hard_loss_sum
    family_hard_loss_delta_sum
    family1_hard_impurity_sum
    family2_hard_impurity_sum
    family_hard_impurity_delta_sum
    family1_soft_impurity_sum
    family2_soft_impurity_sum
    family_soft_impurity_delta_sum
    family1_joint_impurity_sum
    family2_joint_impurity_sum
    family_joint_impurity_delta_sum
    debr_total_hard_gain
    debr_total_soft_gain
    debr_total_delta_j
    nominee_threatening_sum
    profiling_lp_solve_sec
    profiling_pricing_sec
    profiling_greedy_complete_sec
    profiling_feature_prepare_sec
    profiling_candidate_nomination_sec
    profiling_candidate_shortlist_sec
    profiling_candidate_generation_sec
    profiling_recursive_child_eval_sec
    heuristic_selector_improving_split_margin_sum
    heuristic_selector_improving_split_margin_max
    profiling_refine_sec
    expensive_child_sec
    expensive_child_exactify_sec
    """
)

_NATIVE_LIST_METRIC_KEYS = _metric_keys(
    """
    debr_source_group_row_size_histogram
    debr_source_component_atom_size_histogram
    debr_source_component_row_size_histogram
    greedy_feature_survivor_histogram
    atomized_feature_atom_count_histogram
    atomized_feature_block_atom_count_histogram
    atomized_feature_q_effective_histogram
    greedy_feature_preserved_histogram
    greedy_candidate_count_histogram
    per_node_total_weight
    per_node_mu_node
    per_node_candidate_upper_bounds
    per_node_candidate_lower_bounds
    nominee_exactify_prefix_histogram
    profiling_greedy_complete_calls_by_depth
    """
)


def _to_python_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


@dataclass
class MultiLeaf:
    prediction: int
    loss: float
    n_samples: int
    class_counts: Tuple[int, ...]


@dataclass
class MultiNode:
    feature: int
    children: Dict[int, Union["MultiNode", MultiLeaf]]
    child_spans: Dict[int, Tuple[Tuple[int, int], ...]]
    fallback_bin: int
    fallback_prediction: int
    group_count: int
    n_samples: int


@dataclass
class BoundResult:
    lb: float
    ub: float
    tree: Union[MultiNode, MultiLeaf]


class MSPLIT(ClassifierMixin, BaseEstimator):
    """Reference-guided multiway tree solver.

    Each split is a discrete multiway routing over the ordered bins of one
    feature. The native solver generates both impurity and hard-loss candidates,
    merges them into one shortlist, prunes them with a LightGBM-guided reference
    floor, and exactifies only a small prefix. Near the root it exactifies a
    broader shortlist; below the lookahead horizon it switches to a much cheaper
    greedy completion. When ``exactify_top_k`` is omitted, the shortlist budget
    defaults to a ``sqrt(N)`` rule on the active candidate set. When
    ``exactify_top_k=K`` is specified, the solver exactifies at most ``K``
    candidates above the lookahead horizon.
    """

    def __init__(
        self,
        lookahead_depth_budget: int | None = None,
        lookahead_depth: int | None = None,
        full_depth_budget: int = 5,
        reg: float = 0.01,
        exactify_top_k: int | None = None,
        min_child_size: int = 5,
        min_split_size: int | None = None,
        max_branching: int = 0,
        time_limit: int = 100,
        verbose: bool = False,
        random_state: int = 0,
        use_cpp_solver: bool = True,
        **legacy_kwargs,
    ):
        del legacy_kwargs
        self.lookahead_depth_budget = lookahead_depth_budget
        self.lookahead_depth = lookahead_depth
        self.full_depth_budget = full_depth_budget
        self.reg = reg
        self.exactify_top_k = exactify_top_k
        self.min_child_size = min_child_size
        self.min_split_size = min_split_size
        self.max_branching = max_branching
        self.time_limit = time_limit
        self.verbose = verbose
        self.random_state = random_state
        self.use_cpp_solver = use_cpp_solver

    def _normalize_sample_weight(self, sample_weight) -> np.ndarray:
        if sample_weight is None:
            return np.full(self._n_train, 1.0 / float(self._n_train), dtype=np.float64)
        weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if weights.shape[0] != self._n_train:
            raise ValueError("sample_weight must have shape (n_samples,)")
        if np.any(weights < 0):
            raise ValueError("sample_weight must be non-negative")
        total_weight = float(np.sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("sample_weight must have a positive finite sum")
        return (weights / total_weight).astype(np.float64, copy=False)

    def _resolve_lookahead_depth(self) -> int:
        if self.lookahead_depth is not None:
            requested_depth = int(self.lookahead_depth)
        elif self.lookahead_depth_budget is not None:
            requested_depth = int(self.lookahead_depth_budget)
        else:
            requested_depth = max(1, (self.full_depth_budget + 1) // 2)
        return max(1, min(requested_depth, self.full_depth_budget))

    def _assign_scalar_metrics(
        self,
        cpp_result,
        metric_keys: tuple[str, ...],
        caster,
        *,
        default_value,
    ) -> None:
        for key in metric_keys:
            setattr(self, f"{key}_", caster(cpp_result.get(key, default_value)))

    def _assign_native_solver_metrics(self, cpp_result, *, class_count: int, objective_value: float) -> None:
        self.lower_bound_ = float(cpp_result.get("lowerbound", objective_value))
        self.upper_bound_ = float(cpp_result.get("upperbound", objective_value))
        self.native_n_classes_ = int(cpp_result.get("native_n_classes", class_count))
        self.native_teacher_class_count_ = int(cpp_result.get("native_teacher_class_count", 0))
        self.native_binary_mode_ = bool(cpp_result.get("native_binary_mode", int(class_count == 2)))

        self._assign_scalar_metrics(cpp_result, _NATIVE_INT_METRIC_KEYS, int, default_value=0)
        self._assign_scalar_metrics(cpp_result, _NATIVE_FLOAT_METRIC_KEYS, float, default_value=0.0)
        self._assign_scalar_metrics(cpp_result, _NATIVE_LIST_METRIC_KEYS, list, default_value=[])

        self.per_node_prepared_features_ = list(
            cpp_result.get("per_node_prepared_features", self.greedy_feature_preserved_histogram_)
        )
        self.per_node_candidate_count_ = list(
            cpp_result.get("per_node_candidate_count", self.greedy_candidate_count_histogram_)
        )

    def _reset_solver_diagnostics(self, *, class_count: int) -> None:
        self.native_n_classes_ = int(class_count)
        self.native_teacher_class_count_ = 0
        self.native_binary_mode_ = bool(class_count == 2)

        for key in _NATIVE_INT_METRIC_KEYS:
            setattr(self, f"{key}_", 0)
        for key in _NATIVE_FLOAT_METRIC_KEYS:
            setattr(self, f"{key}_", 0.0)
        for key in _NATIVE_LIST_METRIC_KEYS:
            setattr(self, f"{key}_", [])

        self.per_node_prepared_features_ = []
        self.per_node_candidate_count_ = []

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        teacher_logit=None,
        teacher_boundary_gain=None,
        teacher_boundary_cover=None,
        teacher_boundary_value_jump=None,
        **kwargs,
    ):
        del kwargs
        y_encoded, class_labels = self._encode_target(y)
        self.class_labels_ = class_labels
        self.classes_ = class_labels
        Z = check_array(X, ensure_2d=True, dtype=None)
        if not np.issubdtype(np.asarray(Z).dtype, np.integer):
            raise ValueError("MSPLIT expects LightGBM-binned integer features")
        Z = np.asarray(Z, dtype=np.int32)
        if (Z < 0).any():
            raise ValueError("Binned input must be non-negative integer values")
        self.feature_names_ = [f"x{i}" for i in range(Z.shape[1])]

        self._Z_train = np.asarray(Z, dtype=np.int32)
        self._y_train = np.asarray(y_encoded, dtype=np.int32)
        self._n_train = int(self._Z_train.shape[0])
        self._n_features = int(self._Z_train.shape[1])
        if self._n_train == 0:
            raise ValueError("Cannot fit MSPLIT on an empty dataset")
        self._w_train = self._normalize_sample_weight(sample_weight)

        if self.full_depth_budget < 1:
            raise ValueError("full_depth_budget must be at least 1")
        if self.min_child_size < 1:
            raise ValueError("min_child_size must be at least 1")
        if self.max_branching < 0:
            raise ValueError("max_branching must be >= 0 (0 means unlimited)")
        if self.reg < 0:
            raise ValueError("reg must be non-negative")
        if self.exactify_top_k is not None and int(self.exactify_top_k) < 1:
            raise ValueError("exactify_top_k must be a positive integer when specified")

        self.effective_lookahead_depth_ = self._resolve_lookahead_depth()
        if self.use_cpp_solver and _cpp_msplit_fit is not None:
            restore_cache_env = False
            previous_cache_env = os.environ.get("MSPLIT_GREEDY_CACHE_MAX_DEPTH")
            if previous_cache_env is None and self.full_depth_budget <= 2:
                # Tiny exact trees repeatedly revisit the same subproblems.
                # The native solver benefits from a shallow greedy cache here,
                # but we do not want to leak that process-wide setting into
                # larger fits, so set it only for the duration of this call.
                os.environ["MSPLIT_GREEDY_CACHE_MAX_DEPTH"] = str(self.full_depth_budget)
                restore_cache_env = True
            try:
                cpp_result = _cpp_msplit_fit(
                    self._Z_train,
                    self._y_train,
                    self._w_train,
                    teacher_logit,
                    teacher_boundary_gain,
                    teacher_boundary_cover,
                    teacher_boundary_value_jump,
                    int(self.full_depth_budget),
                    int(self.effective_lookahead_depth_),
                    float(self.reg),
                    int(getattr(self, "min_split_size", 0) or 0),
                    int(self.min_child_size),
                    float(self.time_limit),
                    int(self.max_branching),
                    int(self.exactify_top_k) if self.exactify_top_k is not None else 0,
                )
            finally:
                if restore_cache_env:
                    os.environ.pop("MSPLIT_GREEDY_CACHE_MAX_DEPTH", None)
            tree_obj = json.loads(str(cpp_result["tree"]))
            self.tree_ = self._dict_to_tree(tree_obj)
            objective_value = float(cpp_result.get("objective", 0.0))
            self.objective_ = objective_value
            self._assign_native_solver_metrics(
                cpp_result,
                class_count=len(class_labels),
                objective_value=objective_value,
            )
        else:
            if self.use_cpp_solver and _cpp_msplit_fit is None:
                warnings.warn(
                    "MSPLIT C++ solver unavailable; falling back to Python DP solver.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            warnings.warn(
                "Python fallback solver uses per-bin branching; adaptive bin-group arity optimization is available in C++ mode.",
                RuntimeWarning,
                stacklevel=2,
            )

            self._start_time = time.perf_counter()
            self._dp_cache: Dict[Tuple[bytes, int], BoundResult] = {}
            self._greedy_cache: Dict[Tuple[bytes, int], Tuple[float, Union[MultiNode, MultiLeaf]]] = {}
            self._exact_internal_nodes_count = 0
            self._greedy_internal_nodes_count = 0
            self._reset_solver_diagnostics(class_count=len(class_labels))

            root_indices = np.arange(self._n_train, dtype=np.int32)
            result = self._solve_subproblem(root_indices, self.full_depth_budget, current_depth=0)
            self.tree_ = result.tree
            self.lower_bound_ = float(result.lb)
            self.upper_bound_ = float(result.ub)
            self.objective_ = float(result.ub)
            self.exact_internal_nodes_ = int(self._exact_internal_nodes_count)
            self.greedy_internal_nodes_ = int(self._greedy_internal_nodes_count)
            self.dp_subproblem_calls_ = int(len(self._dp_cache))
            self.dp_unique_states_ = int(len(self._dp_cache))
            self.greedy_subproblem_calls_ = int(len(self._greedy_cache))
            self.greedy_unique_states_ = int(len(self._greedy_cache))
            self.greedy_cache_entries_peak_ = int(len(self._greedy_cache))

        self.tree = self._format_tree(self.tree_)
        self.n_features_in_ = self._n_features
        return self

    def predict(self, X):
        check_is_fitted(self, ["tree_", "classes_"])
        Z = check_array(X, ensure_2d=True, dtype=None)
        if not np.issubdtype(np.asarray(Z).dtype, np.integer):
            raise ValueError("MSPLIT expects LightGBM-binned integer features")
        Z = np.asarray(Z, dtype=np.int32)
        if (Z < 0).any():
            raise ValueError("Binned input must be non-negative integer values")
        preds = np.zeros(Z.shape[0], dtype=np.int32)
        for i in range(Z.shape[0]):
            preds[i] = self._predict_leaf(Z[i], self.tree_).prediction
        return self.classes_[preds]

    def predict_proba(self, X):
        check_is_fitted(self, ["tree_", "classes_"])
        Z = check_array(X, ensure_2d=True, dtype=None)
        if not np.issubdtype(np.asarray(Z).dtype, np.integer):
            raise ValueError("MSPLIT expects LightGBM-binned integer features")
        Z = np.asarray(Z, dtype=np.int32)
        if (Z < 0).any():
            raise ValueError("Binned input must be non-negative integer values")

        n_classes = int(len(self.classes_))
        proba = np.zeros((Z.shape[0], n_classes), dtype=np.float64)
        for i in range(Z.shape[0]):
            leaf = self._predict_leaf(Z[i], self.tree_)
            counts = np.asarray(leaf.class_counts, dtype=np.float64)
            if counts.size < n_classes:
                counts = np.pad(counts, (0, n_classes - counts.size))
            elif counts.size > n_classes:
                counts = counts[:n_classes]
            total = float(np.sum(counts))
            if total > 0.0:
                proba[i] = counts / total
            else:
                proba[i, int(leaf.prediction)] = 1.0
        return proba

    def _predict_leaf(self, row: np.ndarray, node: Union[MultiNode, MultiLeaf]) -> MultiLeaf:
        cur = node
        while isinstance(cur, MultiNode):
            feature_index = self._resolve_feature_index(cur.feature, row.shape[0])
            bin_id = int(row[feature_index])
            child = None
            for group_id in sorted(cur.children.keys()):
                spans = cur.child_spans.get(group_id, ())
                matched = False
                for lo, hi in spans:
                    if int(lo) <= bin_id <= int(hi):
                        matched = True
                        break
                if matched:
                    child = cur.children.get(group_id)
                    break
            if child is None:
                # Route to nearest known span if this bin was unseen during training.
                nearest_group = None
                best_dist = None
                best_lo = None
                for group_id in sorted(cur.children.keys()):
                    spans = cur.child_spans.get(group_id, ())
                    if not spans:
                        continue
                    group_dist = None
                    group_lo = None
                    for lo, hi in spans:
                        lo_i = int(lo)
                        hi_i = int(hi)
                        if group_lo is None or lo_i < group_lo:
                            group_lo = lo_i
                        if lo_i <= bin_id <= hi_i:
                            dist = 0
                        elif bin_id < lo_i:
                            dist = lo_i - bin_id
                        else:
                            dist = bin_id - hi_i
                        if group_dist is None or dist < group_dist:
                            group_dist = dist
                    if group_dist is None:
                        continue
                    if (
                        best_dist is None
                        or group_dist < best_dist
                        or (group_dist == best_dist and group_lo is not None and (best_lo is None or group_lo < best_lo))
                    ):
                        best_dist = group_dist
                        best_lo = group_lo
                        nearest_group = group_id
                if nearest_group is not None:
                    child = cur.children.get(nearest_group)
                if child is None:
                    fallback_counts = [0] * int(len(self.classes_))
                    fallback_prediction = int(cur.fallback_prediction)
                    if 0 <= fallback_prediction < len(fallback_counts):
                        fallback_counts[fallback_prediction] = 1
                    return MultiLeaf(
                        prediction=fallback_prediction,
                        loss=0.0,
                        n_samples=0,
                        class_counts=tuple(fallback_counts),
                    )
            cur = child
        return cur

    def _resolve_feature_index(self, feature_index: int, row_width: int) -> int:
        feature_index = int(feature_index)
        row_width = int(row_width)
        if row_width <= 0:
            raise ValueError("Cannot predict on an empty feature row")
        if 0 <= feature_index < row_width:
            return feature_index
        if feature_index == row_width:
            warnings.warn(
                f"MSPLIT tree feature index {feature_index} matched the row width; "
                f"clamping to the last available feature {row_width - 1}.",
                RuntimeWarning,
                stacklevel=3,
            )
            return row_width - 1
        raise IndexError(
            f"MSPLIT tree feature index {feature_index} is out of bounds for input width {row_width}"
        )

    def _solve_subproblem(self, indices: np.ndarray, depth_remaining: int, current_depth: int) -> BoundResult:
        self._check_timeout()
        if current_depth < self.effective_lookahead_depth_:
            self.exact_dp_subproblem_calls_above_lookahead_ += 1
        canonical_indices = np.sort(indices, kind="mergesort")
        key = (canonical_indices.tobytes(), depth_remaining)
        cached = self._dp_cache.get(key)
        if cached is not None:
            return cached

        leaf_objective, leaf_tree = self._leaf_solution(canonical_indices)
        pure = self._is_pure(canonical_indices)

        if depth_remaining <= 1 or pure:
            result = BoundResult(lb=leaf_objective, ub=leaf_objective, tree=leaf_tree)
            self._dp_cache[key] = result
            return result

        if current_depth == self.effective_lookahead_depth_:
            greedy_obj, greedy_tree = self._greedy_complete(indices, depth_remaining)
            result = BoundResult(lb=greedy_obj, ub=greedy_obj, tree=greedy_tree)
            self._dp_cache[key] = result
            return result

        best_tree: Union[MultiNode, MultiLeaf] = leaf_tree
        best_lb = leaf_objective
        best_ub = leaf_objective

        for feature in range(self._n_features):
            partition = self._partition_indices(canonical_indices, feature)
            if partition is None:
                continue

            split_lb = 0.0
            split_ub = 0.0
            children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
            child_spans: Dict[int, Tuple[Tuple[int, int], ...]] = {}
            largest_bin = -1
            largest_size = -1
            group_id = 0

            for bin_id in sorted(partition.keys()):
                child_indices = partition[bin_id]
                child_result = self._solve_subproblem(child_indices, depth_remaining - 1, current_depth + 1)
                split_lb += child_result.lb
                split_ub += child_result.ub
                children[group_id] = child_result.tree
                child_spans[group_id] = ((int(bin_id), int(bin_id)),)
                group_id += 1
                if child_indices.size > largest_size:
                    largest_size = int(child_indices.size)
                    largest_bin = int(bin_id)

            best_lb = min(best_lb, split_lb)
            if split_ub < best_ub:
                best_ub = split_ub
                best_tree = MultiNode(
                    feature=feature,
                    children=children,
                    child_spans=child_spans,
                    fallback_bin=largest_bin,
                    fallback_prediction=leaf_tree.prediction,
                    group_count=len(children),
                    n_samples=int(canonical_indices.size),
                )

        best_lb = min(best_lb, best_ub)
        result = BoundResult(lb=best_lb, ub=best_ub, tree=best_tree)
        if isinstance(best_tree, MultiNode):
            self._exact_internal_nodes_count += 1
        self._dp_cache[key] = result
        return result

    def _greedy_complete(self, indices: np.ndarray, depth_remaining: int) -> Tuple[float, Union[MultiNode, MultiLeaf]]:
        self._check_timeout()
        canonical_indices = np.sort(indices, kind="mergesort")
        key = (canonical_indices.tobytes(), depth_remaining)
        cached = self._greedy_cache.get(key)
        if cached is not None:
            return cached

        leaf_objective, leaf_tree = self._leaf_solution(canonical_indices)
        pure = self._is_pure(canonical_indices)
        if depth_remaining <= 1 or pure:
            result = (leaf_objective, leaf_tree)
            self._greedy_cache[key] = result
            return result

        best_objective = leaf_objective
        best_tree: Union[MultiNode, MultiLeaf] = leaf_tree

        for feature in range(self._n_features):
            partition = self._partition_indices(canonical_indices, feature)
            if partition is None:
                continue

            split_objective = 0.0
            children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
            child_spans: Dict[int, Tuple[Tuple[int, int], ...]] = {}
            largest_bin = -1
            largest_size = -1
            group_id = 0

            for bin_id in sorted(partition.keys()):
                child_indices = partition[bin_id]
                child_obj, child_tree = self._greedy_complete(child_indices, depth_remaining - 1)
                split_objective += child_obj
                children[group_id] = child_tree
                child_spans[group_id] = ((int(bin_id), int(bin_id)),)
                group_id += 1
                if child_indices.size > largest_size:
                    largest_size = int(child_indices.size)
                    largest_bin = int(bin_id)

            if split_objective < best_objective:
                best_objective = split_objective
                best_tree = MultiNode(
                    feature=feature,
                    children=children,
                    child_spans=child_spans,
                    fallback_bin=largest_bin,
                    fallback_prediction=leaf_tree.prediction,
                    group_count=len(children),
                    n_samples=int(canonical_indices.size),
                )

        result = (best_objective, best_tree)
        if isinstance(best_tree, MultiNode):
            self._greedy_internal_nodes_count += 1
        self._greedy_cache[key] = result
        return result

    def _partition_indices(self, indices: np.ndarray, feature: int) -> Optional[Dict[int, np.ndarray]]:
        feature_values = self._Z_train[indices, feature]
        unique_bins = np.unique(feature_values)
        if unique_bins.size <= 1:
            return None

        children: Dict[int, np.ndarray] = {}
        for bin_id in unique_bins:
            mask = feature_values == bin_id
            child_indices = indices[mask]
            if child_indices.size < self.min_child_size:
                return None
            children[int(bin_id)] = child_indices
        return children

    def _leaf_solution(self, indices: np.ndarray) -> Tuple[float, MultiLeaf]:
        y_subset = self._y_train[indices]
        w_subset = self._w_train[indices]
        n_classes = int(len(self.classes_))
        class_counts = np.bincount(y_subset, minlength=n_classes).astype(np.int32, copy=False)
        class_weight = np.bincount(y_subset, weights=w_subset, minlength=n_classes).astype(np.float64, copy=False)
        prediction = int(np.argmax(class_weight))
        mistakes_w = float(np.sum(class_weight) - class_weight[prediction])

        loss = mistakes_w + self.reg
        return loss, MultiLeaf(
            prediction=prediction,
            loss=loss,
            n_samples=int(indices.size),
            class_counts=tuple(int(v) for v in class_counts.tolist()),
        )

    def _is_pure(self, indices: np.ndarray) -> bool:
        y_subset = self._y_train[indices]
        return bool(np.all(y_subset == y_subset[0]))

    def _check_timeout(self):
        if self.time_limit and self.time_limit > 0:
            elapsed = time.perf_counter() - self._start_time
            if elapsed > self.time_limit:
                raise TimeoutError(f"MSPLIT exceeded time_limit={self.time_limit} seconds")

    def _dict_to_tree(self, tree_obj: dict) -> Union[MultiNode, MultiLeaf]:
        node_type = tree_obj.get("type")
        if node_type == "leaf":
            default_class_count = max(1, int(len(getattr(self, "class_labels_", []))))
            class_counts_raw = tree_obj.get("class_counts", [0] * default_class_count)
            class_counts = tuple(int(v) for v in class_counts_raw)
            return MultiLeaf(
                prediction=int(tree_obj["prediction"]),
                loss=float(tree_obj["loss"]),
                n_samples=int(tree_obj.get("n_samples", 0)),
                class_counts=class_counts,
            )

        children: Dict[int, Union[MultiNode, MultiLeaf]] = {}
        child_spans: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        groups_raw = tree_obj.get("groups")
        if isinstance(groups_raw, list):
            for group_id, entry in enumerate(groups_raw):
                if not isinstance(entry, dict):
                    continue
                child_obj = entry.get("child")
                if not isinstance(child_obj, dict):
                    continue
                spans_raw = entry.get("spans", [])
                spans: list[Tuple[int, int]] = []
                if isinstance(spans_raw, list):
                    for span in spans_raw:
                        if not isinstance(span, (list, tuple)) or len(span) != 2:
                            continue
                        lo = int(span[0])
                        hi = int(span[1])
                        if hi < lo:
                            lo, hi = hi, lo
                        spans.append((lo, hi))
                if not spans:
                    continue
                children[group_id] = self._dict_to_tree(child_obj)
                child_spans[group_id] = tuple(spans)
        else:
            # Backward compatibility: legacy per-bin child map.
            children_raw = tree_obj.get("children", {})
            if isinstance(children_raw, dict):
                group_id = 0
                for key in sorted(children_raw.keys(), key=lambda x: int(x)):
                    bin_id = int(key)
                    child_obj = children_raw[key]
                    if not isinstance(child_obj, dict):
                        continue
                    children[group_id] = self._dict_to_tree(child_obj)
                    child_spans[group_id] = ((bin_id, bin_id),)
                    group_id += 1

        return MultiNode(
            feature=int(tree_obj["feature"]),
            children=children,
            child_spans=child_spans,
            fallback_bin=int(tree_obj["fallback_bin"]),
            fallback_prediction=int(tree_obj["fallback_prediction"]),
            group_count=int(tree_obj.get("group_count", len(children))),
            n_samples=int(tree_obj.get("n_samples", 0)),
        )

    def _encode_target(self, y) -> Tuple[np.ndarray, np.ndarray]:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            y_arr = y_arr.ravel()

        classes = np.unique(y_arr)
        if classes.size == 0:
            raise ValueError("Target y must not be empty")

        ordered = sorted(classes.tolist(), key=lambda v: str(v))
        mapping = {label: idx for idx, label in enumerate(ordered)}
        y_bin = np.array([mapping[val] for val in y_arr], dtype=np.int32)
        labels = np.array([_to_python_scalar(label) for label in ordered], dtype=object)
        return y_bin, labels


    def _format_tree(self, node: Union[MultiNode, MultiLeaf], depth: int = 0) -> str:
        indent = "  " * depth
        if isinstance(node, MultiLeaf):
            pred_label = self.class_labels_[node.prediction]
            return (
                f"{indent}Leaf(pred={pred_label!r}, n={node.n_samples}, "
                f"class_counts={node.class_counts}, loss={node.loss:.6f})"
            )

        lines = [
            f"{indent}Node(feature={node.feature}, groups={node.group_count}, fallback_bin={node.fallback_bin}, "
            f"fallback_pred={self.class_labels_[node.fallback_prediction]!r}, n={node.n_samples})"
        ]
        for group_id in sorted(node.children.keys()):
            spans = node.child_spans.get(group_id, ())
            span_text_parts = []
            for lo, hi in spans:
                if int(lo) == int(hi):
                    span_text_parts.append(str(int(lo)))
                else:
                    span_text_parts.append(f"{int(lo)}-{int(hi)}")
            span_text = ",".join(span_text_parts) if span_text_parts else "?"
            lines.append(f"{indent}  group {group_id} bins {span_text} ->")
            lines.append(self._format_tree(node.children[group_id], depth + 2))
        return "\n".join(lines)
