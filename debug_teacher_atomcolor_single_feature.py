from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import load_electricity
from experiment_utils import encode_binary_target, make_preprocessor
from lightgbm_binning import fit_lightgbm_binner
from split._libgosdt import msplit_debug_teacher_guided_atomcolor_root_feature


def _predict_tree(tree: dict[str, Any], z_row: np.ndarray) -> int:
    node_type = tree.get("type", "")
    if node_type == "leaf":
        return int(tree["prediction"])

    if node_type == "node":
        feature = int(tree["feature"])
        bin_value = int(z_row[feature])
        for group in tree.get("groups", []):
            for lo, hi in group.get("spans", []):
                if int(lo) <= bin_value <= int(hi):
                    return _predict_tree(group["child"], z_row)
        return int(tree.get("fallback_prediction", 0))

    if node_type == "pair_node":
        feature_a = int(tree["feature_a"])
        feature_b = int(tree["feature_b"])
        bin_a = int(z_row[feature_a])
        bin_b = int(z_row[feature_b])
        primary_hit = any(int(lo) <= bin_a <= int(hi) for lo, hi in tree.get("primary_spans", []))
        secondary_hit = any(int(lo) <= bin_b <= int(hi) for lo, hi in tree.get("secondary_spans", []))
        child_idx = 2
        if primary_hit:
            child_idx = 0
        elif secondary_hit:
            child_idx = 1
        return _predict_tree(tree["children"][child_idx]["child"], z_row)

    raise ValueError(f"Unsupported tree node type: {node_type!r}")


def _predict_dataset(tree: dict[str, Any], z: np.ndarray) -> np.ndarray:
    out = np.empty(z.shape[0], dtype=np.int32)
    for i in range(z.shape[0]):
        out[i] = _predict_tree(tree, z[i])
    return out


def _accuracy(tree: dict[str, Any], z: np.ndarray, y: np.ndarray) -> float:
    pred = _predict_dataset(tree, z)
    return float(np.mean(pred == y))


def _root_partition_signature(group_spans: list[list[list[int]]], z_col: np.ndarray) -> tuple[tuple[int, ...], ...]:
    labels = np.full(z_col.shape[0], -1, dtype=np.int16)
    for group_idx, spans in enumerate(group_spans):
        for lo, hi in spans:
            mask = (z_col >= int(lo)) & (z_col <= int(hi))
            labels[mask] = group_idx
    if np.any(labels < 0):
        raise ValueError("Invalid group_spans: some rows were not assigned to a root group.")
    groups: list[tuple[int, ...]] = []
    for group_idx in sorted(set(int(v) for v in labels.tolist())):
        rows = np.flatnonzero(labels == group_idx)
        groups.append(tuple(int(i) for i in rows.tolist()))
    groups.sort()
    return tuple(groups)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug teacher_atomcolor on a single feature and inspect the root partition."
    )
    parser.add_argument("--feature", default="nswprice")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument("--min-child-size", type=int, default=8)
    parser.add_argument("--proposal-atom-cap", type=int, default=0)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--binner-estimators", type=int, default=2000)
    parser.add_argument("--binner-early-stop", type=int, default=50)
    parser.add_argument("--binner-threads", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    X, y = load_electricity()
    if not hasattr(X, "__getitem__"):
        raise TypeError("Expected tabular electricity data with named columns")
    X = pd.DataFrame(X).copy()
    if args.feature not in X.columns:
        raise ValueError(f"Feature {args.feature!r} not found. Available sample: {list(X.columns[:10])}")
    y_arr = encode_binary_target(y, "electricity")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_arr,
        test_size=float(args.test_size),
        random_state=int(args.seed),
        stratify=y_arr,
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=float(args.val_size),
        random_state=int(args.seed),
        stratify=y_train,
    )

    preprocessor = make_preprocessor(X_fit)
    X_fit_proc = np.ascontiguousarray(preprocessor.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.ascontiguousarray(preprocessor.transform(X_val), dtype=np.float32)
    X_test_proc = np.ascontiguousarray(preprocessor.transform(X_test), dtype=np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()

    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=int(args.max_bins),
        min_samples_leaf=int(args.min_child_size),
        random_state=int(args.seed),
        n_estimators=int(args.binner_estimators),
        num_leaves=int(args.max_bins),
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        max_depth=-1,
        min_data_in_bin=1,
        min_data_in_leaf=int(args.min_child_size),
        lambda_l2=0.0,
        early_stopping_rounds=int(args.binner_early_stop),
        num_threads=int(args.binner_threads),
        device_type="cpu",
        ensemble_runs=1,
        ensemble_feature_fraction=1.0,
        ensemble_bagging_fraction=1.0,
        ensemble_bagging_freq=0,
        threshold_dedup_eps=1e-9,
        collect_teacher_logit=True,
    )

    Z_fit = np.ascontiguousarray(binner.transform(X_fit_proc), dtype=np.int32)
    Z_val = np.ascontiguousarray(binner.transform(X_val_proc), dtype=np.int32)
    Z_test = np.ascontiguousarray(binner.transform(X_test_proc), dtype=np.int32)
    feature_name = f"num__{args.feature}"
    if feature_name not in feature_names:
        raise ValueError(
            f"Preprocessed feature {feature_name!r} not found. Sample names: {feature_names[:10]}"
        )
    feature_index = int(feature_names.index(feature_name))

    debug_json = msplit_debug_teacher_guided_atomcolor_root_feature(
        z=Z_fit,
        y=y_fit,
        sample_weight=None,
        feature=feature_index,
        depth_remaining=int(args.depth),
        full_depth_budget=int(args.depth),
        lookahead_depth_budget=int(args.lookahead),
        regularization=float(args.reg),
        min_child_size=int(args.min_child_size),
        min_atom_size=int(args.proposal_atom_cap),
        time_limit_seconds=300.0,
        max_branching=int(args.max_branching),
        partition_strategy=1,
        approx_feature_scan_limit=1,
        approx_ref_shortlist_enabled=False,
        approx_ref_widen_max=0,
        approx_challenger_sweep_enabled=False,
        approx_challenger_sweep_max_features=0,
        approx_challenger_sweep_max_patch_calls_per_node=0,
        approx_distilled_mode=True,
        approx_distilled_alpha=0.0,
        approx_distilled_max_depth=max(1, int(args.lookahead)),
        approx_distilled_geometry_mode=7,
        approx_score_order_enabled=False,
        teacher_logit=getattr(binner, "teacher_train_logit", None),
        teacher_boundary_gain=getattr(binner, "boundary_gain_per_feature", None),
        teacher_boundary_cover=getattr(binner, "boundary_cover_per_feature", None),
        teacher_boundary_value_jump=getattr(binner, "boundary_value_jump_per_feature", None),
        teacher_boundary_left_delta=None,
        teacher_boundary_right_delta=None,
        teacher_boundary_left_conf=None,
        teacher_boundary_right_conf=None,
    )
    result = json.loads(debug_json)
    result["feature_names"] = feature_names
    result["root_feature_name"] = feature_names[feature_index] if feature_names else args.feature

    proposer_candidates: list[dict[str, Any]] = []
    contiguous = result.get("contiguous", {})
    if contiguous.get("exact_ready") and contiguous.get("tree"):
        proposer_candidates.append(
            {
                "source": "contiguous",
                "index": -1,
                "group_spans": contiguous["group_spans"],
                "exact_ub": float(contiguous["exact_ub"]),
                "tree": contiguous["tree"],
            }
        )
    for candidate in result.get("candidates", []):
        if candidate.get("exact_ready") and candidate.get("tree"):
            proposer_candidates.append(
                {
                    "source": "teacher_atomcolor",
                    "index": int(candidate["index"]),
                    "group_spans": candidate["group_spans"],
                    "exact_ub": float(candidate["exact_ub"]),
                    "tree": candidate["tree"],
                }
            )

    epsilon = 1.0 / float(max(1, y_fit.shape[0]))
    z_fit_col = Z_fit[:, feature_index]
    dedup: dict[tuple[tuple[int, ...], ...], dict[str, Any]] = {}
    for candidate in proposer_candidates:
        signature = _root_partition_signature(candidate["group_spans"], z_fit_col)
        existing = dedup.get(signature)
        if existing is None or float(candidate["exact_ub"]) < float(existing["exact_ub"]) - 1e-12:
            dedup[signature] = candidate
    deduped_candidates = list(dedup.values())
    deduped_candidates.sort(key=lambda c: (float(c["exact_ub"]), c["source"], int(c["index"])))

    if deduped_candidates:
        best_exact = float(deduped_candidates[0]["exact_ub"])
        frontier = [
            c for c in deduped_candidates
            if float(c["exact_ub"]) <= best_exact + epsilon + 1e-12
        ]
    else:
        best_exact = float("inf")
        frontier = []

    for candidate in deduped_candidates:
        tree = candidate["tree"]
        candidate["val_accuracy"] = _accuracy(tree, Z_val, y_val)
        candidate["test_accuracy"] = _accuracy(tree, Z_test, y_test)
    frontier.sort(key=lambda c: (-float(c["val_accuracy"]), -float(c["test_accuracy"]), float(c["exact_ub"])))

    strict_best = deduped_candidates[0] if deduped_candidates else None
    frontier_best = frontier[0] if frontier else None
    result["rashomon_frontier"] = {
        "epsilon": epsilon,
        "candidate_count_before_dedup": len(proposer_candidates),
        "candidate_count_after_dedup": len(deduped_candidates),
        "best_exact_ub": best_exact,
        "strict_best": strict_best,
        "frontier_size": len(frontier),
        "frontier_best_by_val": frontier_best,
        "frontier_candidates": frontier,
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
