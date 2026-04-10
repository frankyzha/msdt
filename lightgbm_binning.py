"""LightGBM-guided per-feature binning for multiway decision trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier


@dataclass
class LightGBMBinner:
    """Feature-wise binner built from LightGBM split thresholds."""

    bin_edges_per_feature: List[np.ndarray]
    fill_values_per_feature: np.ndarray
    max_bins: int
    min_samples_leaf: int
    random_state: Optional[int]
    device_type: str = "cpu"
    teacher_train_logit: Optional[np.ndarray] = None
    teacher_available: bool = False
    teacher_models: Optional[List[LGBMClassifier]] = None
    boundary_gain_per_feature: Optional[np.ndarray] = None
    boundary_cover_per_feature: Optional[np.ndarray] = None
    boundary_value_jump_per_feature: Optional[np.ndarray] = None
    bin_representatives_per_feature: Optional[List[np.ndarray]] = None

    @property
    def n_bins_per_feature(self) -> np.ndarray:
        return np.array([int(edges.size + 1) if edges.size else 1 for edges in self.bin_edges_per_feature], dtype=np.int32)

    def _prepare_input(self, X) -> np.ndarray:
        X_arr = _check_array_allow_nan(X)
        if X_arr.shape[1] != len(self.bin_edges_per_feature):
            raise ValueError(
                f"X has {X_arr.shape[1]} features but binner was fit with {len(self.bin_edges_per_feature)} features"
            )

        X_arr = X_arr.copy()
        nan_mask = np.isnan(X_arr)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            X_arr[rows, cols] = self.fill_values_per_feature[cols]
        return X_arr

    def transform(self, X) -> np.ndarray:
        X_arr = self._prepare_input(X)

        Z = np.zeros(X_arr.shape, dtype=np.int32)
        for j, edges in enumerate(self.bin_edges_per_feature):
            if edges.size == 0:
                Z[:, j] = 0
            else:
                Z[:, j] = np.digitize(X_arr[:, j], edges, right=False).astype(np.int32)
        return Z

    def predict_teacher_logit(self, X) -> np.ndarray:
        if not self.teacher_models:
            raise RuntimeError("teacher logits unavailable; fit with collect_teacher_logit=True")
        X_arr = self._prepare_input(X)
        margin_sum: Optional[np.ndarray] = None
        for model in self.teacher_models:
            raw_margin = _extract_raw_margin_predictions(model, X_arr)
            if margin_sum is None:
                margin_sum = np.zeros_like(raw_margin, dtype=np.float64)
            margin_sum += raw_margin
        if margin_sum is None:
            raise RuntimeError("teacher logits unavailable; no fitted teacher models found")
        return margin_sum / float(len(self.teacher_models))

    def compute_local_boundary_teacher_tensors(
        self,
        X,
        Z: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.teacher_models:
            raise RuntimeError("teacher tensors unavailable; fit with collect_teacher_logit=True")
        X_arr = self._prepare_input(X)
        if Z is None:
            Z_arr = self.transform(X_arr)
        else:
            Z_arr = np.asarray(Z, dtype=np.int32)
            if Z_arr.shape != X_arr.shape:
                raise ValueError("Z must have the same shape as X")
        n_rows, n_features = X_arr.shape
        left_delta = np.zeros((n_rows, n_features), dtype=np.float64)
        right_delta = np.zeros((n_rows, n_features), dtype=np.float64)
        left_conf = np.zeros((n_rows, n_features), dtype=np.float64)
        right_conf = np.zeros((n_rows, n_features), dtype=np.float64)

        representatives = self.bin_representatives_per_feature
        if representatives is None or len(representatives) != n_features:
            raise RuntimeError("bin representatives unavailable on binner")

        for feature in range(n_features):
            reps = np.asarray(representatives[feature], dtype=np.float64)
            if reps.size <= 1:
                continue
            feature_bins = Z_arr[:, feature]
            valid_mask = (feature_bins >= 0) & (feature_bins < reps.size)
            if not np.any(valid_mask):
                continue

            clipped_bins = np.clip(feature_bins, 0, reps.size - 1)
            X_curr = X_arr.copy()
            X_curr[:, feature] = reps[clipped_bins]
            current_margin = self.predict_teacher_logit(X_curr)
            if np.ndim(current_margin) != 1:
                raise RuntimeError(
                    "compute_local_boundary_teacher_tensors currently supports binary LightGBM teachers only."
                )

            prev_mask = valid_mask & (feature_bins > 0)
            if np.any(prev_mask):
                X_prev = X_curr.copy()
                X_prev[prev_mask, feature] = reps[feature_bins[prev_mask] - 1]
                prev_margin = self.predict_teacher_logit(X_prev)
                left_delta[prev_mask, feature] = current_margin[prev_mask] - prev_margin[prev_mask]
                left_conf[prev_mask, feature] = 0.5 * (
                    np.abs(current_margin[prev_mask]) + np.abs(prev_margin[prev_mask])
                )

            next_mask = valid_mask & (feature_bins + 1 < reps.size)
            if np.any(next_mask):
                X_next = X_curr.copy()
                X_next[next_mask, feature] = reps[feature_bins[next_mask] + 1]
                next_margin = self.predict_teacher_logit(X_next)
                right_delta[next_mask, feature] = next_margin[next_mask] - current_margin[next_mask]
                right_conf[next_mask, feature] = 0.5 * (
                    np.abs(current_margin[next_mask]) + np.abs(next_margin[next_mask])
                )

        return left_delta, right_delta, left_conf, right_conf


def _binary_edge(unique_values: np.ndarray) -> np.ndarray:
    low = float(np.min(unique_values))
    high = float(np.max(unique_values))
    if low == 0.0 and high == 1.0:
        return np.array([0.5], dtype=float)
    return np.array([(low + high) / 2.0], dtype=float)


def _check_array_allow_nan(X) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        X_arr = np.asarray(X.to_numpy(), dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {X_arr.shape}")
    return X_arr


def _check_X_y_allow_nan(X, y) -> tuple[np.ndarray, np.ndarray]:
    X_arr = _check_array_allow_nan(X)
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        y_arr = np.asarray(y_arr).reshape(-1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X and y have inconsistent lengths: {X_arr.shape[0]} != {y_arr.shape[0]}")
    return X_arr, np.asarray(y_arr)


def _quantile_edges(values: np.ndarray, max_edges: int) -> np.ndarray:
    if max_edges <= 0:
        return np.array([], dtype=float)

    unique_values = np.unique(values)
    if unique_values.size <= 1:
        return np.array([], dtype=float)
    if unique_values.size <= 2:
        return _binary_edge(unique_values)

    quantiles = np.linspace(0.0, 1.0, max_edges + 2, dtype=float)[1:-1]
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges.astype(float))

    lo = float(np.min(values))
    hi = float(np.max(values))
    edges = edges[(edges > lo) & (edges < hi)]
    if edges.size > max_edges:
        edges = edges[:max_edges]
    return np.sort(edges)


def _collect_threshold_scores(tree_node: Dict, feature_scores: List[Dict[float, float]]) -> None:
    if "split_feature" not in tree_node:
        return

    feature_idx = int(tree_node["split_feature"])
    threshold = float(tree_node["threshold"])
    split_gain = float(tree_node.get("split_gain", 1.0))

    if np.isfinite(threshold):
        score = max(split_gain, 1e-12)
        feature_scores[feature_idx][threshold] = feature_scores[feature_idx].get(threshold, 0.0) + score

    left = tree_node.get("left_child")
    right = tree_node.get("right_child")
    if isinstance(left, dict):
        _collect_threshold_scores(left, feature_scores)
    if isinstance(right, dict):
        _collect_threshold_scores(right, feature_scores)


def _subtree_leaf_summary(tree_node: Dict) -> tuple[float, float]:
    if "split_feature" not in tree_node:
        leaf_value = float(tree_node.get("leaf_value", 0.0))
        leaf_count = float(tree_node.get("leaf_count", 0.0))
        if leaf_count <= 0.0:
            leaf_count = 1.0
        return leaf_value * leaf_count, leaf_count

    left = tree_node.get("left_child")
    right = tree_node.get("right_child")
    left_sum = left_count = 0.0
    right_sum = right_count = 0.0
    if isinstance(left, dict):
        left_sum, left_count = _subtree_leaf_summary(left)
    if isinstance(right, dict):
        right_sum, right_count = _subtree_leaf_summary(right)
    return left_sum + right_sum, left_count + right_count


def _nearest_boundary_index(edges: np.ndarray, threshold: float) -> Optional[int]:
    if edges.size == 0 or not np.isfinite(threshold):
        return None
    pos = int(np.searchsorted(edges, threshold, side="left"))
    if pos <= 0:
        return 0
    if pos >= edges.size:
        return int(edges.size - 1)
    prev_edge = float(edges[pos - 1])
    next_edge = float(edges[pos])
    if abs(threshold - prev_edge) <= abs(next_edge - threshold):
        return pos - 1
    return pos


def _accumulate_boundary_priors(
    tree_node: Dict,
    bin_edges_per_feature: List[np.ndarray],
    boundary_gain: np.ndarray,
    boundary_cover: np.ndarray,
    boundary_value_jump: np.ndarray,
) -> tuple[float, float]:
    if "split_feature" not in tree_node:
        leaf_value = float(tree_node.get("leaf_value", 0.0))
        leaf_count = float(tree_node.get("leaf_count", 0.0))
        if leaf_count <= 0.0:
            leaf_count = 1.0
        return leaf_value * leaf_count, leaf_count

    left = tree_node.get("left_child")
    right = tree_node.get("right_child")
    left_sum = left_count = 0.0
    right_sum = right_count = 0.0
    if isinstance(left, dict):
        left_sum, left_count = _accumulate_boundary_priors(
            left,
            bin_edges_per_feature,
            boundary_gain,
            boundary_cover,
            boundary_value_jump,
        )
    if isinstance(right, dict):
        right_sum, right_count = _accumulate_boundary_priors(
            right,
            bin_edges_per_feature,
            boundary_gain,
            boundary_cover,
            boundary_value_jump,
        )

    feature_idx = int(tree_node["split_feature"])
    threshold = float(tree_node.get("threshold", np.nan))
    boundary_idx = _nearest_boundary_index(bin_edges_per_feature[feature_idx], threshold)
    if boundary_idx is not None:
        split_gain = max(float(tree_node.get("split_gain", 0.0)), 0.0)
        split_cover = float(tree_node.get("internal_count", left_count + right_count))
        if split_cover <= 0.0:
            split_cover = left_count + right_count
        left_mean = left_sum / left_count if left_count > 0.0 else 0.0
        right_mean = right_sum / right_count if right_count > 0.0 else 0.0
        boundary_gain[feature_idx, boundary_idx] += split_gain
        boundary_cover[feature_idx, boundary_idx] += split_cover
        boundary_value_jump[feature_idx, boundary_idx] += split_cover * abs(left_mean - right_mean)

    return left_sum + right_sum, left_count + right_count


def _merge_threshold_score_maps(
    dst: List[Dict[float, float]],
    src: List[Dict[float, float]],
) -> None:
    for j, src_map in enumerate(src):
        dst_map = dst[j]
        for threshold, score in src_map.items():
            dst_map[threshold] = dst_map.get(threshold, 0.0) + float(score)


def _encode_target(y_arr: np.ndarray) -> tuple[np.ndarray, list]:
    y = np.asarray(y_arr)
    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError(
            f"fit_lightgbm_binner expects at least 2 classes; received {unique.size}: {unique.tolist()}"
        )
    ordered = sorted(unique.tolist(), key=lambda v: str(v))
    mapping = {value: idx for idx, value in enumerate(ordered)}
    encoded = np.array([mapping[value] for value in y], dtype=np.int32)
    return encoded, ordered


def _encode_target_with_order(y_arr: np.ndarray, ordered_classes: list) -> np.ndarray:
    mapping = {value: idx for idx, value in enumerate(ordered_classes)}
    try:
        return np.array([mapping[value] for value in np.asarray(y_arr)], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(
            "Validation labels must be a subset of the training labels used to fit the LightGBM binner."
        ) from exc


def _extract_raw_margin_predictions(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        raw = np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)
    else:
        raw = np.asarray(model.predict(X, raw_score=True), dtype=np.float64)
    if raw.ndim == 2:
        if raw.shape[1] == 1:
            return np.ascontiguousarray(raw[:, 0], dtype=np.float64)
        return np.ascontiguousarray(raw, dtype=np.float64)
    return np.ascontiguousarray(raw.reshape(-1), dtype=np.float64)


def _compute_bin_representatives(
    X_work: np.ndarray,
    bin_edges_per_feature: List[np.ndarray],
) -> List[np.ndarray]:
    representatives: List[np.ndarray] = []
    n_features = X_work.shape[1]
    for j in range(n_features):
        edges = np.asarray(bin_edges_per_feature[j], dtype=np.float64)
        feature_values = X_work[:, j]
        n_bins = int(edges.size + 1)
        reps = np.zeros(n_bins, dtype=np.float64)
        bins = np.digitize(feature_values, edges, right=False).astype(np.int32) if edges.size else np.zeros(
            feature_values.shape[0], dtype=np.int32
        )
        feature_min = float(np.min(feature_values))
        feature_max = float(np.max(feature_values))
        for b in range(n_bins):
            mask = bins == b
            if np.any(mask):
                reps[b] = float(np.mean(feature_values[mask]))
                continue
            if n_bins == 1:
                reps[b] = feature_min
            elif b == 0:
                reps[b] = 0.5 * (feature_min + float(edges[0]))
            elif b == n_bins - 1:
                reps[b] = 0.5 * (float(edges[-1]) + feature_max)
            else:
                reps[b] = 0.5 * (float(edges[b - 1]) + float(edges[b]))
        representatives.append(reps)
    return representatives


def fit_lightgbm_binner(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    max_bins: int = 1024,
    min_samples_leaf: int = 10,
    random_state: Optional[int] = None,
    n_estimators: int = 10000,
    num_leaves: int = 31,
    learning_rate: float = 0.05,
    feature_fraction: float = 1.0,
    bagging_fraction: float = 1.0,
    bagging_freq: int = 0,
    max_depth: int = -1,
    min_data_in_bin: int = 1,
    min_data_in_leaf: Optional[int] = None,
    lambda_l2: float = 0.0,
    early_stopping_rounds: int = 100,
    num_threads: int = 1,
    device_type: str = "cpu",
    gpu_platform_id: int = 0,
    gpu_device_id: int = 0,
    gpu_fallback_to_cpu: bool = True,
    ensemble_runs: int = 1,
    ensemble_feature_fraction: float = 0.8,
    ensemble_bagging_fraction: float = 0.8,
    ensemble_bagging_freq: int = 1,
    threshold_dedup_eps: float = 1e-9,
    collect_teacher_logit: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> LightGBMBinner:
    """Fit LightGBM and extract per-feature split thresholds as discretization edges.

    When ``ensemble_runs > 1``, this fits multiple stochastic LightGBM models and
    unions split thresholds by aggregating split-gain scores per feature.
    """
    if max_bins < 2:
        raise ValueError("max_bins must be at least 2")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    if ensemble_runs < 1:
        raise ValueError("ensemble_runs must be at least 1")
    if not (0.0 < ensemble_feature_fraction <= 1.0):
        raise ValueError("ensemble_feature_fraction must be in (0, 1]")
    if not (0.0 < ensemble_bagging_fraction <= 1.0):
        raise ValueError("ensemble_bagging_fraction must be in (0, 1]")
    if ensemble_bagging_freq < 0:
        raise ValueError("ensemble_bagging_freq must be >= 0")
    if threshold_dedup_eps < 0:
        raise ValueError("threshold_dedup_eps must be >= 0")
    if min_data_in_leaf is None:
        min_data_in_leaf = int(min_samples_leaf)
    else:
        min_data_in_leaf = int(min_data_in_leaf)
    if min_data_in_leaf < 1:
        raise ValueError("min_data_in_leaf must be at least 1")
    if lambda_l2 < 0:
        raise ValueError("lambda_l2 must be >= 0")
    if early_stopping_rounds < 0:
        raise ValueError("early_stopping_rounds must be >= 0")

    X_arr, y_arr = _check_X_y_allow_nan(X_train, y_train)
    y_bin, ordered_classes = _encode_target(y_arr)
    n_classes = int(len(ordered_classes))
    n_features = X_arr.shape[1]
    X_val_work: Optional[np.ndarray] = None
    y_val_bin: Optional[np.ndarray] = None
    if X_val is not None or y_val is not None:
        if X_val is None or y_val is None:
            raise ValueError("X_val and y_val must both be provided for validation early stopping.")
        X_val_arr, y_val_arr = _check_X_y_allow_nan(X_val, y_val)
        if X_val_arr.shape[1] != n_features:
            raise ValueError(
                f"validation feature count mismatch: train has {n_features}, val has {X_val_arr.shape[1]}"
            )
        X_val_work = X_val_arr.copy()
        y_val_bin = _encode_target_with_order(y_val_arr, ordered_classes)

    X_work = X_arr.copy()
    fill_values = np.zeros(n_features, dtype=float)
    for j in range(n_features):
        col = X_work[:, j]
        if np.isnan(col).any():
            finite = col[~np.isnan(col)]
            fill = float(np.median(finite)) if finite.size > 0 else 0.0
            fill_values[j] = fill
            col[np.isnan(col)] = fill
            X_work[:, j] = col
        else:
            fill_values[j] = 0.0
    if X_val_work is not None:
        for j in range(n_features):
            col = X_val_work[:, j]
            if np.isnan(col).any():
                col[np.isnan(col)] = fill_values[j]
                X_val_work[:, j] = col

    requested_device = str(device_type).strip().lower()
    if requested_device not in {"cpu", "gpu", "cuda"}:
        raise ValueError(f"device_type must be one of ['cpu', 'gpu', 'cuda'], got {device_type!r}")

    def _progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(str(message))

    def _fit_for_device(
        device: str,
        run_seed: Optional[int],
        run_feature_fraction: float,
        run_bagging_fraction: float,
        run_bagging_freq: int,
    ) -> LGBMClassifier:
        subsample_freq = run_bagging_freq if run_bagging_fraction < 0.999 else 0
        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_bin": int(max_bins),
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_data_in_leaf,
            "min_data_in_bin": min_data_in_bin,
            "reg_lambda": lambda_l2,
            "colsample_bytree": run_feature_fraction,
            "subsample": run_bagging_fraction,
            "subsample_freq": subsample_freq,
            "random_state": run_seed,
            "n_jobs": num_threads,
            "deterministic": True,
            "force_col_wise": True,
            "verbose": -1,
            "device_type": device,
        }
        if n_classes == 2:
            params["objective"] = "binary"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = int(n_classes)
        if device != "cpu":
            params["gpu_platform_id"] = int(gpu_platform_id)
            params["gpu_device_id"] = int(gpu_device_id)
        model = LGBMClassifier(**params)
        fit_kwargs = {}
        if X_val_work is not None and y_val_bin is not None and int(early_stopping_rounds) > 0:
            fit_kwargs = {
                "eval_set": [(X_val_work, y_val_bin)],
                "eval_metric": "binary_logloss" if n_classes == 2 else "multi_logloss",
                "callbacks": [lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)],
            }
        model.fit(X_work, y_bin, **fit_kwargs)
        return model

    _progress(
        "lightgbm_binner:start "
        f"rows={X_work.shape[0]} features={n_features} max_bins={int(max_bins)} "
        f"ensemble_runs={int(ensemble_runs)} device={requested_device}"
    )

    feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
    teacher_margin_sum: Optional[np.ndarray]
    if collect_teacher_logit:
        if n_classes == 2:
            teacher_margin_sum = np.zeros(X_work.shape[0], dtype=np.float64)
        else:
            teacher_margin_sum = np.zeros((X_work.shape[0], n_classes), dtype=np.float64)
    else:
        teacher_margin_sum = None
    teacher_margin_runs = 0
    teacher_models: Optional[List[LGBMClassifier]] = [] if collect_teacher_logit else None
    actual_device = requested_device
    run_device = requested_device
    for run_idx in range(int(ensemble_runs)):
        _progress(
            f"lightgbm_binner:fit_run_start run={run_idx + 1}/{int(ensemble_runs)} device={run_device}"
        )
        if run_idx == 0:
            run_feature_fraction = float(feature_fraction)
            run_bagging_fraction = float(bagging_fraction)
            run_bagging_freq = int(bagging_freq)
        else:
            run_feature_fraction = min(float(feature_fraction), float(ensemble_feature_fraction))
            run_bagging_fraction = min(float(bagging_fraction), float(ensemble_bagging_fraction))
            run_bagging_freq = max(int(bagging_freq), int(ensemble_bagging_freq))

        run_seed = None if random_state is None else int(random_state + 1009 * run_idx)
        try:
            lgbm = _fit_for_device(
                run_device,
                run_seed=run_seed,
                run_feature_fraction=run_feature_fraction,
                run_bagging_fraction=run_bagging_fraction,
                run_bagging_freq=run_bagging_freq,
            )
        except Exception:
            if run_device == "cpu" or not gpu_fallback_to_cpu:
                raise
            run_device = "cpu"
            actual_device = "cpu"
            lgbm = _fit_for_device(
                run_device,
                run_seed=run_seed,
                run_feature_fraction=run_feature_fraction,
                run_bagging_fraction=run_bagging_fraction,
                run_bagging_freq=run_bagging_freq,
            )
        best_iter = getattr(lgbm, "best_iteration_", None)
        _progress(
            "lightgbm_binner:fit_run_done "
            f"run={run_idx + 1}/{int(ensemble_runs)} device={actual_device} "
            f"best_iteration={best_iter if best_iter is not None else 'na'}"
        )

        run_feature_scores: List[Dict[float, float]] = [dict() for _ in range(n_features)]
        dump = lgbm.booster_.dump_model()
        for tree_info in dump.get("tree_info", []):
            tree_structure = tree_info.get("tree_structure")
            if isinstance(tree_structure, dict):
                _collect_threshold_scores(tree_structure, run_feature_scores)
        _merge_threshold_score_maps(feature_scores, run_feature_scores)
        if collect_teacher_logit and teacher_margin_sum is not None:
            teacher_margin_sum += _extract_raw_margin_predictions(lgbm, X_work)
            teacher_margin_runs += 1
            if teacher_models is not None:
                teacher_models.append(lgbm)

    max_edges = max_bins - 1
    bin_edges_per_feature: List[np.ndarray] = []

    for j in range(n_features):
        feature_values = X_work[:, j]
        unique_values = np.unique(feature_values)

        if unique_values.size <= 1:
            edges = np.array([], dtype=float)
        elif unique_values.size <= 2:
            edges = _binary_edge(unique_values)
        else:
            lo = float(np.min(feature_values))
            hi = float(np.max(feature_values))

            candidates = [
                (thr, score)
                for thr, score in feature_scores[j].items()
                if np.isfinite(thr) and lo < float(thr) < hi
            ]

            if candidates:
                candidates.sort(key=lambda item: (-item[1], item[0]))
                chosen = []
                for threshold, _ in candidates:
                    t = float(threshold)
                    if threshold_dedup_eps > 0 and any(abs(prev - t) <= threshold_dedup_eps for prev in chosen):
                        continue
                    if threshold_dedup_eps == 0 and chosen and abs(chosen[-1] - t) < 1e-12:
                        continue
                    chosen.append(t)
                    if len(chosen) >= max_edges:
                        break
                edges = np.array(sorted(chosen), dtype=float)
            else:
                edges = _quantile_edges(feature_values, max_edges)

            if edges.size == 0:
                edges = _quantile_edges(feature_values, max_edges)

            if edges.size > max_edges:
                edges = edges[:max_edges]

        bin_edges_per_feature.append(np.sort(edges))

    n_bins_per_feature = np.array(
        [int(edges.size + 1) if edges.size else 1 for edges in bin_edges_per_feature],
        dtype=np.int32,
    )
    _progress(
        "lightgbm_binner:thresholds_ready "
        f"mean_bins={float(np.mean(n_bins_per_feature)):.2f} "
        f"max_bins_used={int(np.max(n_bins_per_feature)) if n_bins_per_feature.size else 0}"
    )

    boundary_gain = np.zeros((n_features, max_edges), dtype=np.float64)
    boundary_cover = np.zeros((n_features, max_edges), dtype=np.float64)
    boundary_value_jump = np.zeros((n_features, max_edges), dtype=np.float64)
    for model in teacher_models if teacher_models is not None else []:
        dump = model.booster_.dump_model()
        for tree_info in dump.get("tree_info", []):
            tree_structure = tree_info.get("tree_structure")
            if isinstance(tree_structure, dict):
                _accumulate_boundary_priors(
                    tree_structure,
                    bin_edges_per_feature,
                    boundary_gain,
                    boundary_cover,
                    boundary_value_jump,
                )
    bin_representatives_per_feature = _compute_bin_representatives(X_work, bin_edges_per_feature)
    _progress("lightgbm_binner:done")

    return LightGBMBinner(
        bin_edges_per_feature=bin_edges_per_feature,
        fill_values_per_feature=fill_values,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        device_type=actual_device,
        teacher_train_logit=(
            teacher_margin_sum / float(teacher_margin_runs)
            if teacher_margin_sum is not None and teacher_margin_runs > 0
            else None
        ),
        teacher_available=bool(collect_teacher_logit and teacher_margin_runs > 0),
        teacher_models=teacher_models if teacher_models else None,
        boundary_gain_per_feature=boundary_gain,
        boundary_cover_per_feature=boundary_cover,
        boundary_value_jump_per_feature=boundary_value_jump,
        bin_representatives_per_feature=bin_representatives_per_feature,
    )
