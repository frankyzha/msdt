#!/usr/bin/env python3
"""Analyze teacher signals on electricity/nswprice using the strict protocol split.

This script fits the same LightGBM teacher/binning setup used by the nswprice-only
experiments, extracts several signal families on the teacher bin grid, and scores
how well they recover the ShapeCART-critical nswprice bands.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import load_electricity
from experiment_utils import encode_binary_target, make_preprocessor
from lightgbm_binning import fit_lightgbm_binner


# Match the strict 70/10/20 protocol used by the experiment runner.
FIXED_TEST_FRACTION = 0.20
FIXED_VAL_WITHIN_TRAIN = 0.125
SEED = 0

# Match the nswprice-only run script.
MAX_BINS = 255
MIN_SAMPLES_LEAF = 8
LGB_NUM_THREADS = 16
LGB_NUM_LEAVES = 255
LGB_LEARNING_RATE = 0.05
LGB_MIN_DATA_IN_BIN = 1
LGB_LAMBDA_L2 = 0.0

# These boundaries came from the strict-protocol univariate ShapeCART rerun on electricity d4.
SHAPECART_ROOT_BOUNDARIES = np.array(
    [
        0.02820600010454655,
        0.041266001760959625,
        0.04174700006842613,
        0.06393400207161903,
        0.06444400176405907,
        0.06756599992513657,
        0.07792399823665619,
        0.0779539979994297,
        0.07927500084042549,
        0.07933499664068222,
        0.0811070017516613,
        0.0811369975656271,
        0.10134199913591146,
        0.10140199959278107,
    ],
    dtype=np.float64,
)


def cluster_reference_boundaries(values: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    if values.size == 0:
        return values
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    clusters: list[list[float]] = [[float(ordered[0])]]
    for value in ordered[1:]:
        if abs(float(value) - clusters[-1][-1]) <= tol:
            clusters[-1].append(float(value))
        else:
            clusters.append([float(value)])
    return np.asarray([float(np.mean(cluster)) for cluster in clusters], dtype=np.float64)


REFERENCE_CENTERS = cluster_reference_boundaries(SHAPECART_ROOT_BOUNDARIES, tol=1e-3)


@dataclass
class SignalScore:
    name: str
    strict_hits: int
    broad_hits: int
    mean_ref_distance: float
    max_ref_distance: float
    tail_strict_hits: int
    top_values: list[float]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cdf = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(max(idx, 0), sorted_values.size - 1)
    return float(sorted_values[idx])


def weighted_sse_cost(prefix_w, prefix_x, prefix_x2, u: int, v: int) -> float:
    total_w = prefix_w[v + 1] - prefix_w[u]
    if total_w <= 0.0:
        return 0.0
    total_x = prefix_x[v + 1] - prefix_x[u]
    total_x2 = prefix_x2[v + 1] - prefix_x2[u]
    mean_sq = float(np.dot(total_x, total_x)) / float(total_w)
    cost = float(np.sum(total_x2)) - mean_sq
    return max(cost, 0.0)


def exact_segmentation_boundaries(
    features: np.ndarray,
    weights: np.ndarray,
    n_segments: int,
    min_weight_per_segment: float = 1.0,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n_segments <= 1 or n <= 1:
        return np.array([], dtype=np.int32)
    n_segments = min(n_segments, n)
    prefix_w = np.zeros(n + 1, dtype=np.float64)
    prefix_w[1:] = np.cumsum(w)
    prefix_x = np.zeros((n + 1, x.shape[1]), dtype=np.float64)
    prefix_x[1:] = np.cumsum(w[:, None] * x, axis=0)
    prefix_x2 = np.zeros((n + 1, x.shape[1]), dtype=np.float64)
    prefix_x2[1:] = np.cumsum(w[:, None] * (x * x), axis=0)
    inf = float("inf")
    dp = np.full((n_segments + 1, n), inf, dtype=np.float64)
    prev = np.full((n_segments + 1, n), -1, dtype=np.int32)
    for t in range(n):
        if prefix_w[t + 1] >= min_weight_per_segment:
            dp[1, t] = weighted_sse_cost(prefix_w, prefix_x, prefix_x2, 0, t)
    for s in range(2, n_segments + 1):
        for t in range(n):
            best_cost = inf
            best_u = -1
            for u in range(s - 2, t):
                left_cost = dp[s - 1, u]
                if not np.isfinite(left_cost):
                    continue
                seg_w = prefix_w[t + 1] - prefix_w[u + 1]
                if seg_w < min_weight_per_segment:
                    continue
                cand = left_cost + weighted_sse_cost(prefix_w, prefix_x, prefix_x2, u + 1, t)
                if cand < best_cost:
                    best_cost = cand
                    best_u = u
            dp[s, t] = best_cost
            prev[s, t] = best_u
    end = n - 1
    if not np.isfinite(dp[n_segments, end]):
        return np.array([], dtype=np.int32)
    bounds = []
    s = n_segments
    t = end
    while s > 1:
        u = int(prev[s, t])
        if u < 0:
            return np.array([], dtype=np.int32)
        bounds.append(u)
        t = u
        s -= 1
    bounds.reverse()
    return np.asarray(bounds, dtype=np.int32)


def score_selected_boundaries(
    name: str,
    selected: np.ndarray,
    boundary_values: np.ndarray,
    ref_centers: np.ndarray,
    strict_tol: float = 0.002,
    broad_tol: float = 0.005,
) -> SignalScore:
    values = np.sort(np.asarray(boundary_values, dtype=np.float64)[np.asarray(selected, dtype=np.int32)])
    if values.size == 0:
        return SignalScore(name, 0, 0, float("inf"), float("inf"), 0, [])
    distances = np.min(np.abs(ref_centers[:, None] - values[None, :]), axis=1)
    strict_hits = int(np.sum(distances <= strict_tol))
    broad_hits = int(np.sum(distances <= broad_tol))
    tail_mask = ref_centers >= 0.06
    tail_strict_hits = int(np.sum(distances[tail_mask] <= strict_tol))
    return SignalScore(
        name=name,
        strict_hits=strict_hits,
        broad_hits=broad_hits,
        mean_ref_distance=float(np.mean(distances)),
        max_ref_distance=float(np.max(distances)),
        tail_strict_hits=tail_strict_hits,
        top_values=[float(v) for v in values.tolist()],
    )


def topk_indices(signal: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.array([], dtype=np.int32)
    idx = np.flatnonzero(valid)
    scores = np.abs(x[idx])
    order = np.argsort(-scores, kind="stable")
    return idx[order[: min(k, order.size)]]


def main() -> None:
    X, y = load_electricity()
    X = X.loc[:, ["nswprice"]].copy()
    y_bin = encode_binary_target(y, "electricity")
    all_idx = np.arange(y_bin.shape[0], dtype=np.int32)
    idx_train_all, idx_test = train_test_split(
        all_idx, test_size=FIXED_TEST_FRACTION, random_state=SEED, stratify=y_bin
    )
    y_train_all = y_bin[idx_train_all]
    idx_fit, idx_val = train_test_split(
        idx_train_all, test_size=FIXED_VAL_WITHIN_TRAIN, random_state=SEED, stratify=y_train_all
    )

    X_fit = X.iloc[idx_fit].reset_index(drop=True)
    X_val = X.iloc[idx_val].reset_index(drop=True)
    y_fit = np.asarray(y_bin[idx_fit], dtype=np.int32)
    y_val = np.asarray(y_bin[idx_val], dtype=np.int32)

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.ascontiguousarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.ascontiguousarray(pre.transform(X_val), dtype=np.float32)
    feature_names = pre.get_feature_names_out().tolist()
    feature_idx = int(feature_names.index("num__nswprice"))

    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=MAX_BINS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_estimators=10000,
        num_leaves=LGB_NUM_LEAVES,
        learning_rate=LGB_LEARNING_RATE,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        max_depth=-1,
        min_data_in_bin=LGB_MIN_DATA_IN_BIN,
        min_data_in_leaf=MIN_SAMPLES_LEAF,
        lambda_l2=LGB_LAMBDA_L2,
        early_stopping_rounds=100,
        num_threads=LGB_NUM_THREADS,
        device_type="cpu",
        collect_teacher_logit=True,
    )

    Z_fit = binner.transform(X_fit_proc)
    teacher_logit = np.asarray(binner.teacher_train_logit, dtype=np.float64).reshape(-1)
    teacher_prob = 1.0 / (1.0 + np.exp(-teacher_logit))
    left_delta, right_delta, left_conf, right_conf = binner.compute_local_boundary_teacher_tensors(X_fit_proc, Z_fit)

    bins = Z_fit[:, feature_idx]
    edges = np.asarray(binner.bin_edges_per_feature[feature_idx], dtype=np.float64)
    n_bins = int(edges.size + 1)
    reps = np.asarray(binner.bin_representatives_per_feature[feature_idx], dtype=np.float64)
    w = np.ones_like(teacher_logit, dtype=np.float64)

    bin_weight = np.zeros(n_bins, dtype=np.float64)
    bin_pos = np.zeros(n_bins, dtype=np.float64)
    bin_logit_w = np.zeros(n_bins, dtype=np.float64)
    bin_logit_sq_w = np.zeros(n_bins, dtype=np.float64)
    bin_abs_logit_w = np.zeros(n_bins, dtype=np.float64)
    bin_prob_w = np.zeros(n_bins, dtype=np.float64)
    bin_prob_sq_w = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            continue
        wb = w[mask]
        yb = y_fit[mask]
        sb = teacher_logit[mask]
        qb = teacher_prob[mask]
        bin_weight[b] = float(np.sum(wb))
        bin_pos[b] = float(np.sum(wb * yb))
        bin_logit_w[b] = float(np.sum(wb * sb))
        bin_logit_sq_w[b] = float(np.sum(wb * sb * sb))
        bin_abs_logit_w[b] = float(np.sum(wb * np.abs(sb)))
        bin_prob_w[b] = float(np.sum(wb * qb))
        bin_prob_sq_w[b] = float(np.sum(wb * qb * qb))

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_logit = np.divide(bin_logit_w, bin_weight, out=np.zeros_like(bin_logit_w), where=bin_weight > 0)
        mean_prob = np.divide(bin_prob_w, bin_weight, out=np.zeros_like(bin_prob_w), where=bin_weight > 0)
        mean_abs_logit = np.divide(
            bin_abs_logit_w, bin_weight, out=np.zeros_like(bin_abs_logit_w), where=bin_weight > 0
        )
        var_logit = np.divide(bin_logit_sq_w, bin_weight, out=np.zeros_like(bin_logit_sq_w), where=bin_weight > 0) - (
            mean_logit * mean_logit
        )
        var_prob = np.divide(bin_prob_sq_w, bin_weight, out=np.zeros_like(bin_prob_sq_w), where=bin_weight > 0) - (
            mean_prob * mean_prob
        )
    var_logit = np.maximum(var_logit, 0.0)
    var_prob = np.maximum(var_prob, 0.0)
    std_logit = np.sqrt(var_logit)
    std_prob = np.sqrt(var_prob)
    prob_entropy = np.zeros_like(mean_prob)
    valid_prob = (mean_prob > 0.0) & (mean_prob < 1.0)
    prob_entropy[valid_prob] = -(
        mean_prob[valid_prob] * np.log(mean_prob[valid_prob]) +
        (1.0 - mean_prob[valid_prob]) * np.log(1.0 - mean_prob[valid_prob])
    )

    boundary_values = edges.copy()
    n_boundaries = n_bins - 1
    boundary_support = np.zeros(n_boundaries, dtype=np.float64)
    delta_mean = np.zeros(n_boundaries, dtype=np.float64)
    delta_hetero = np.zeros(n_boundaries, dtype=np.float64)
    abs_delta_mean = np.zeros(n_boundaries, dtype=np.float64)
    for m in range(n_boundaries):
        left_mask = bins == m
        right_mask = bins == (m + 1)
        contrib_delta = []
        contrib_conf = []
        contrib_w = []
        if np.any(left_mask):
            contrib_delta.append(right_delta[left_mask, feature_idx])
            contrib_conf.append(right_conf[left_mask, feature_idx])
            contrib_w.append(w[left_mask])
        if np.any(right_mask):
            contrib_delta.append(left_delta[right_mask, feature_idx])
            contrib_conf.append(left_conf[right_mask, feature_idx])
            contrib_w.append(w[right_mask])
        if not contrib_delta:
            continue
        d = np.concatenate(contrib_delta).astype(np.float64, copy=False)
        c = np.concatenate(contrib_conf).astype(np.float64, copy=False)
        ww = np.concatenate(contrib_w).astype(np.float64, copy=False)
        total_w = float(np.sum(ww))
        boundary_support[m] = total_w
        if total_w <= 0.0:
            continue
        delta_mean[m] = float(np.sum(ww * d) / total_w)
        abs_delta_mean[m] = float(np.sum(ww * np.abs(d)) / total_w)
        med_c = weighted_median(c, ww)
        lo = c <= med_c
        hi = c > med_c
        lo_w = float(np.sum(ww[lo]))
        hi_w = float(np.sum(ww[hi]))
        lo_mean = float(np.sum(ww[lo] * d[lo]) / lo_w) if lo_w > 0.0 else delta_mean[m]
        hi_mean = float(np.sum(ww[hi] * d[hi]) / hi_w) if hi_w > 0.0 else delta_mean[m]
        delta_hetero[m] = hi_mean - lo_mean

    value_jump = np.asarray(binner.boundary_value_jump_per_feature[feature_idx, :n_boundaries], dtype=np.float64)
    gain = np.asarray(binner.boundary_gain_per_feature[feature_idx, :n_boundaries], dtype=np.float64)
    cover = np.asarray(binner.boundary_cover_per_feature[feature_idx, :n_boundaries], dtype=np.float64)

    scalar_boundary_signals = {
        "mean_logit_jump": np.abs(np.diff(mean_logit)),
        "mean_prob_jump": np.abs(np.diff(mean_prob)),
        "std_prob_jump": np.abs(np.diff(std_prob)),
        "mean_abs_margin_jump": np.abs(np.diff(mean_abs_logit)),
        "entropy_jump": np.abs(np.diff(prob_entropy)),
        "teacher_value_jump_prior": value_jump,
        "teacher_gain_prior": gain,
        "teacher_cover_prior": cover,
        "cover_x_prob_jump": cover * np.abs(np.diff(mean_prob)),
        "cover_x_std_prob_jump": cover * np.abs(np.diff(std_prob)),
        "cover_x_margin_jump": cover * np.abs(np.diff(mean_abs_logit)),
        "value_x_prob_jump": value_jump * np.abs(np.diff(mean_prob)),
        "local_delta_mean": np.abs(delta_mean),
        "local_delta_absmean": abs_delta_mean,
        "local_delta_hetero": np.abs(delta_hetero),
        "local_signature_l2": np.sqrt(delta_mean * delta_mean + delta_hetero * delta_hetero),
    }

    ranking_scores = []
    topk = REFERENCE_CENTERS.size
    for name, signal in scalar_boundary_signals.items():
        selected = topk_indices(signal, topk)
        ranking_scores.append(score_selected_boundaries(name, selected, boundary_values, REFERENCE_CENTERS))

    segmentation_scores = []
    target_runs = int(REFERENCE_CENTERS.size + 1)
    seg_specs = {
        "seg_mean_logit": (mean_logit.reshape(-1, 1), bin_weight),
        "seg_mean_prob": (mean_prob.reshape(-1, 1), bin_weight),
        "seg_std_prob": (std_prob.reshape(-1, 1), bin_weight),
        "seg_mean_abs_margin": (mean_abs_logit.reshape(-1, 1), bin_weight),
        "seg_prob_entropy": (prob_entropy.reshape(-1, 1), bin_weight),
        "seg_logit_mean_std": (np.column_stack([mean_logit, std_logit]), bin_weight),
        "seg_prob_mean_std": (np.column_stack([mean_prob, std_prob]), bin_weight),
        "seg_prob_mean_absmargin": (np.column_stack([mean_prob, mean_abs_logit]), bin_weight),
        "seg_prob_mean_entropy": (np.column_stack([mean_prob, prob_entropy]), bin_weight),
        "seg_prob_mean_std_absmargin": (np.column_stack([mean_prob, std_prob, mean_abs_logit]), bin_weight),
        "seg_boundary_delta_hetero": (np.column_stack([delta_mean, delta_hetero]), boundary_support),
        "seg_boundary_prior_value_gain": (np.column_stack([value_jump, gain]), np.maximum(cover, 1.0)),
        "seg_boundary_cover_jump": (
            np.column_stack([cover, np.abs(np.diff(mean_prob))]),
            np.maximum(boundary_support, 1.0),
        ),
        "seg_boundary_cover_value_jump": (
            np.column_stack([cover, value_jump, np.abs(np.diff(mean_prob))]),
            np.maximum(boundary_support, 1.0),
        ),
    }
    for name, (features, weights) in seg_specs.items():
        boundary_idx = exact_segmentation_boundaries(features, weights, target_runs, min_weight_per_segment=1.0)
        if name.startswith("seg_boundary_"):
            selected = boundary_idx
        else:
            selected = boundary_idx
        segmentation_scores.append(score_selected_boundaries(name, selected, boundary_values, REFERENCE_CENTERS))

    ranking_df = pd.DataFrame([s.__dict__ for s in ranking_scores]).sort_values(
        ["strict_hits", "tail_strict_hits", "broad_hits", "mean_ref_distance"],
        ascending=[False, False, False, True],
    )
    segmentation_df = pd.DataFrame([s.__dict__ for s in segmentation_scores]).sort_values(
        ["strict_hits", "tail_strict_hits", "broad_hits", "mean_ref_distance"],
        ascending=[False, False, False, True],
    )

    boundary_df = pd.DataFrame(
        {
            "boundary": boundary_values,
            "teacher_value_jump_prior": value_jump,
            "teacher_gain_prior": gain,
            "teacher_cover_prior": cover,
            "mean_logit_jump": np.abs(np.diff(mean_logit)),
            "mean_prob_jump": np.abs(np.diff(mean_prob)),
            "local_delta_mean": delta_mean,
            "local_delta_absmean": abs_delta_mean,
            "local_delta_hetero": delta_hetero,
            "local_signature_l2": np.sqrt(delta_mean * delta_mean + delta_hetero * delta_hetero),
        }
    ).sort_values("boundary")

    bin_df = pd.DataFrame(
        {
            "bin": np.arange(n_bins, dtype=np.int32),
            "left_edge": np.r_[-np.inf, edges],
            "right_edge": np.r_[edges, np.inf],
            "representative": reps,
            "weight": bin_weight,
            "mean_logit": mean_logit,
            "mean_prob": mean_prob,
            "std_prob": std_prob,
            "std_logit": std_logit,
            "mean_abs_logit": mean_abs_logit,
            "prob_entropy": prob_entropy,
            "label_rate": np.divide(bin_pos, bin_weight, out=np.zeros_like(bin_pos), where=bin_weight > 0),
        }
    )

    out_dir = Path("results/analysis/nswprice_teacher_signals_20260308")
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(out_dir / "boundary_signal_ranking.csv", index=False)
    segmentation_df.to_csv(out_dir / "segmentation_signal_ranking.csv", index=False)
    boundary_df.to_csv(out_dir / "boundary_signals.csv", index=False)
    bin_df.to_csv(out_dir / "bin_signals.csv", index=False)

    summary = {
        "feature_name": feature_names[feature_idx],
        "n_fit": int(X_fit_proc.shape[0]),
        "n_bins": int(n_bins),
        "n_boundaries": int(n_boundaries),
        "reference_boundaries_raw": SHAPECART_ROOT_BOUNDARIES.tolist(),
        "reference_centers": REFERENCE_CENTERS.tolist(),
        "best_boundary_signal": ranking_df.iloc[0].to_dict(),
        "best_segmentation_signal": segmentation_df.iloc[0].to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Feature: {feature_names[feature_idx]}")
    print(f"n_fit={X_fit_proc.shape[0]}, n_bins={n_bins}, n_boundaries={n_boundaries}")
    print("Reference centers:")
    print(", ".join(f"{v:.6f}" for v in REFERENCE_CENTERS))
    print("\nTop boundary signals:")
    print(ranking_df[["name", "strict_hits", "tail_strict_hits", "broad_hits", "mean_ref_distance", "top_values"]].to_string(index=False))
    print("\nTop segmentation signals:")
    print(segmentation_df[["name", "strict_hits", "tail_strict_hits", "broad_hits", "mean_ref_distance", "top_values"]].to_string(index=False))
    print(f"\nWrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
