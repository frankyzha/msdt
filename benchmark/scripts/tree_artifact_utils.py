"""Shared helpers for serializing fitted trees into JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def to_label_text(value: object) -> str:
    return str(value)


def write_artifact_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def feature_display_name(name: str, max_len: int = 24) -> str:
    if name.startswith("num__"):
        out = name[5:]
    elif name.startswith("cat__"):
        out = name[5:]
    else:
        out = name
    out = out.replace("__", "_")
    if len(out) <= max_len:
        return out
    return out[: max_len - 2] + ".."


def _format_float(value: float) -> str:
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    text = f"{float(value):.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _bins_to_spans(bins: Iterable[int]) -> List[Tuple[int, int]]:
    sorted_bins = sorted(set(int(b) for b in bins))
    if not sorted_bins:
        return []

    spans: List[Tuple[int, int]] = []
    start = sorted_bins[0]
    end = sorted_bins[0]
    for b in sorted_bins[1:]:
        if b == end + 1:
            end = b
            continue
        spans.append((start, end))
        start, end = b, b
    spans.append((start, end))
    return spans


def _span_to_bounds(start_bin: int, end_bin: int, edges: np.ndarray) -> Tuple[float, float]:
    lo = -np.inf if start_bin == 0 else float(edges[start_bin - 1])
    hi = np.inf if end_bin >= len(edges) else float(edges[end_bin])
    return lo, hi


def _format_numeric_union(spans: List[Tuple[int, int]], edges: np.ndarray) -> str:
    parts: List[str] = []
    for start_bin, end_bin in spans:
        lo, hi = _span_to_bounds(start_bin, end_bin, edges)
        if np.isneginf(lo) and np.isfinite(hi):
            parts.append(f"<= {_format_float(hi)}")
        elif np.isfinite(lo) and np.isposinf(hi):
            parts.append(f"> {_format_float(lo)}")
        elif np.isfinite(lo) and np.isfinite(hi):
            parts.append(f"[{_format_float(lo)}, {_format_float(hi)}]")
        else:
            parts.append("all")
    return " U ".join(parts)


def _format_discrete_union(spans: List[Tuple[int, int]]) -> str:
    parts: List[str] = []
    for start_bin, end_bin in spans:
        if start_bin == end_bin:
            parts.append(f"= {int(start_bin)}")
        else:
            parts.append(f"[{int(start_bin)}, {int(end_bin)}]")
    return " U ".join(parts) if parts else "all"


def _is_binary_onehot_feature(feature_name: str, edges: np.ndarray) -> bool:
    if not feature_name.startswith("cat__"):
        return False
    if edges is None or len(edges) != 1:
        return False
    return abs(float(edges[0]) - 0.5) < 1e-6


def _resolve_matrix_feature_index(feature_idx: int, matrix_width: int) -> int:
    feature_idx = int(feature_idx)
    matrix_width = int(matrix_width)
    if matrix_width <= 0:
        raise ValueError("Cannot index an empty matrix")
    if 0 <= feature_idx < matrix_width:
        return feature_idx
    if feature_idx == matrix_width:
        return matrix_width - 1
    raise IndexError(f"feature index {feature_idx} is out of bounds for axis 1 with size {matrix_width}")


def _get_binner_feature_edges(binner) -> List[Optional[np.ndarray]]:
    edges = getattr(binner, "bin_edges_per_feature", None)
    if edges is None:
        return []
    return list(edges)


def _parse_onehot_name(feature_name: str) -> Tuple[str, str]:
    raw = feature_name[5:] if feature_name.startswith("cat__") else feature_name
    base, sep, category = raw.rpartition("_")
    if not sep:
        return feature_display_name(raw), "1"
    return feature_display_name(base), category


def format_msplit_condition(feature_idx: int, bins: Iterable[int], binner, feature_names: List[str]) -> str:
    feature_name = feature_names[feature_idx] if 0 <= feature_idx < len(feature_names) else f"x{feature_idx}"
    bins_list = sorted(set(int(b) for b in bins))
    bins_for_display = bins_list
    if len(bins_list) >= 2:
        # Merge gaps from unseen bins so labels stay readable.
        bins_for_display = list(range(bins_list[0], bins_list[-1] + 1))
    bin_edges = _get_binner_feature_edges(binner)
    if feature_idx < 0 or feature_idx >= len(bin_edges):
        return _format_discrete_union(_bins_to_spans(bins_for_display))
    edges = bin_edges[feature_idx]

    if edges is None or len(edges) == 0:
        cond = "all"
    elif _is_binary_onehot_feature(feature_name, edges):
        _, category = _parse_onehot_name(feature_name)
        bins_set = set(bins_list)
        if bins_set == {0}:
            cond = f"!= {category}"
        elif bins_set == {1}:
            cond = f"= {category}"
        else:
            cond = f"in {{{category}, others}}"
        return cond
    else:
        spans = _bins_to_spans(bins_for_display)
        cond = _format_numeric_union(spans, edges)

    return cond


def is_leaf_node(node: object) -> bool:
    return not hasattr(node, "children") and not hasattr(node, "primary_child")


def _expand_spans(spans: Iterable[Tuple[int, int]]) -> List[int]:
    bins: List[int] = []
    for lo, hi in spans:
        lo_i = int(lo)
        hi_i = int(hi)
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        bins.extend(range(lo_i, hi_i + 1))
    return bins


def _class_dist_from_counts(counts: Iterable[int], class_labels: np.ndarray) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, cnt in enumerate(counts):
        label = class_labels[i] if i < len(class_labels) else i
        out.append(
            {
                "class_index": int(i),
                "class_label": to_label_text(label),
                "count": int(cnt),
            }
        )
    return out


def _subtree_signature(node: object) -> Tuple:
    if is_leaf_node(node):
        prediction = int(getattr(node, "prediction", 0))
        class_counts = tuple(int(v) for v in getattr(node, "class_counts", ()))
        return ("leaf", prediction, class_counts)
    if hasattr(node, "primary_child"):
        return (
            "pair",
            int(getattr(node, "feature_a", -1)),
            int(getattr(node, "feature_b", -1)),
            _subtree_signature(getattr(node, "primary_child")),
            _subtree_signature(getattr(node, "secondary_child")),
            _subtree_signature(getattr(node, "else_child")),
        )
    child_sigs = tuple(
        (
            tuple(int(b) for b in bins),
            _subtree_signature(child),
        )
        for bins, child in group_children(node)
    )
    return ("node", int(getattr(node, "feature", -1)), child_sigs)


def group_children(node: object) -> List[Tuple[List[int], object]]:
    if is_leaf_node(node):
        return []
    if hasattr(node, "primary_child"):
        return []

    child_spans = getattr(node, "child_spans", None)
    if isinstance(child_spans, dict) and child_spans:
        grouped_children: List[Tuple[List[int], object]] = []
        for group_id in sorted(node.children.keys()):
            spans = child_spans.get(group_id, ())
            bins = _expand_spans(spans)
            if not bins:
                continue
            grouped_children.append((bins, node.children[group_id]))
        return grouped_children

    # C++ MSPLIT serializes one child per observed bin; merge adjacent observed-bin
    # entries when they share the same subtree.
    items = [(int(bin_id), child, _subtree_signature(child)) for bin_id, child in node.children.items()]
    items.sort(key=lambda item: item[0])
    if not items:
        return []

    grouped: List[Tuple[List[int], object, Tuple]] = []
    cur_bins = [items[0][0]]
    cur_child = items[0][1]
    cur_sig = items[0][2]
    for bin_id, child, sig in items[1:]:
        if sig == cur_sig:
            cur_bins.append(bin_id)
            continue
        grouped.append((cur_bins, cur_child, cur_sig))
        cur_bins = [bin_id]
        cur_child = child
        cur_sig = sig
    grouped.append((cur_bins, cur_child, cur_sig))
    return [(bins, child) for bins, child, _ in grouped]


def serialize_msplit_node(
    node: object,
    binner,
    feature_names: List[str],
    class_labels: np.ndarray,
    z_train: Optional[np.ndarray],
    idxs: Optional[np.ndarray],
    path_conditions: List[str],
) -> Dict[str, object]:
    n_samples = int(idxs.size) if idxs is not None else int(getattr(node, "n_samples", 0))
    if is_leaf_node(node):
        pred = int(node.prediction)
        pred_label = class_labels[pred] if pred < len(class_labels) else pred
        counts = [int(v) for v in getattr(node, "class_counts", ())]
        return {
            "node_type": "leaf",
            "n_samples": n_samples,
            "predicted_class_index": pred,
            "predicted_class_label": to_label_text(pred_label),
            "true_class_dist": _class_dist_from_counts(counts, class_labels),
            "path_conditions": path_conditions,
        }

    if hasattr(node, "primary_child"):
        feat_a = int(node.feature_a)
        feat_b = int(node.feature_b)
        feat_a_name = feature_names[feat_a] if feat_a < len(feature_names) else f"x{feat_a}"
        feat_b_name = feature_names[feat_b] if feat_b < len(feature_names) else f"x{feat_b}"
        feat_a_display = feature_display_name(feat_a_name)
        feat_b_display = feature_display_name(feat_b_name)
        primary_bins = _expand_spans(getattr(node, "primary_spans", ()))
        secondary_bins = _expand_spans(getattr(node, "secondary_spans", ()))
        primary_values = z_train[idxs, feat_a] if (z_train is not None and idxs is not None and idxs.size > 0) else None
        secondary_values = z_train[idxs, feat_b] if (z_train is not None and idxs is not None and idxs.size > 0) else None
        primary_idxs = secondary_idxs = else_idxs = None
        if primary_values is not None and secondary_values is not None:
            primary_mask = np.isin(primary_values, np.asarray(primary_bins, dtype=np.int32))
            residual_mask = ~primary_mask
            secondary_mask = residual_mask & np.isin(
                secondary_values,
                np.asarray(secondary_bins, dtype=np.int32),
            )
            else_mask = residual_mask & ~secondary_mask
            primary_idxs = idxs[primary_mask]
            secondary_idxs = idxs[secondary_mask]
            else_idxs = idxs[else_mask]
        cond_primary = format_msplit_condition(feat_a, primary_bins, binner, feature_names)
        cond_secondary = format_msplit_condition(feat_b, secondary_bins, binner, feature_names)
        return {
            "node_type": "pair_internal",
            "feature_index": feat_a,
            "feature_name": to_label_text(feat_a_name),
            "feature_display_name": f"{feat_a_display} / {feat_b_display}",
            "n_samples": n_samples,
            "n_way": 3,
            "children": [
                {
                    "branch": {
                        "condition": f"{feat_a_display} {cond_primary}",
                        "bins": [int(b) for b in primary_bins],
                    },
                    "child": serialize_msplit_node(
                        node.primary_child,
                        binner,
                        feature_names,
                        class_labels,
                        z_train,
                        primary_idxs,
                        path_conditions + [f"{feat_a_display} {cond_primary}"],
                    ),
                },
                {
                    "branch": {
                        "condition": f"else if {feat_b_display} {cond_secondary}",
                        "bins": [int(b) for b in secondary_bins],
                    },
                    "child": serialize_msplit_node(
                        node.secondary_child,
                        binner,
                        feature_names,
                        class_labels,
                        z_train,
                        secondary_idxs,
                        path_conditions + [f"else if {feat_b_display} {cond_secondary}"],
                    ),
                },
                {
                    "branch": {
                        "condition": "else",
                        "bins": [],
                    },
                    "child": serialize_msplit_node(
                        node.else_child,
                        binner,
                        feature_names,
                        class_labels,
                        z_train,
                        else_idxs,
                        path_conditions + ["else"],
                    ),
                },
            ],
        }

    feature_idx = int(node.feature)
    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"x{feature_idx}"
    feature_display = feature_display_name(feature_name)
    groups = group_children(node)
    feature_values = None
    if z_train is not None and idxs is not None and idxs.size > 0:
        matrix_feature_idx = _resolve_matrix_feature_index(feature_idx, z_train.shape[1])
        feature_values = z_train[idxs, matrix_feature_idx]

    children: List[Dict[str, object]] = []
    for bins, child in groups:
        bins_arr = np.asarray(bins, dtype=np.int32)
        child_idxs = None
        if feature_values is not None:
            child_idxs = idxs[np.isin(feature_values, bins_arr)]
        cond = format_msplit_condition(feature_idx, bins, binner, feature_names)
        children.append(
            {
                "branch": {
                    "condition": cond,
                    "bins": [int(b) for b in bins],
                },
                "child": serialize_msplit_node(
                    child,
                    binner,
                    feature_names,
                    class_labels,
                    z_train,
                    child_idxs,
                    path_conditions + [f"{feature_display} {cond}"],
                ),
            }
        )

    return {
        "node_type": "internal",
        "feature_index": feature_idx,
        "feature_name": to_label_text(feature_name),
        "feature_display_name": feature_display,
        "n_samples": n_samples,
        "n_way": int(len(groups)),
        "children": children,
    }


def build_msplit_artifact(
    *,
    dataset: str,
    pipeline: str,
    target_name: str,
    class_labels: np.ndarray,
    feature_names: List[str],
    accuracy: float,
    seed: int,
    test_size: float,
    depth_budget: int,
    lookahead: int,
    time_limit: float,
    max_bins: int,
    min_samples_leaf: int,
    min_child_size: int,
    max_branching: int,
    reg: float,
    branch_penalty: float | None = None,
    msplit_variant: Optional[str],
    tree_root: object,
    binner,
    z_train: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    root_idxs = np.arange(z_train.shape[0], dtype=np.int32)
    bin_edges: List[Optional[List[float]]] = []
    bin_edges_per_feature = _get_binner_feature_edges(binner)
    if not bin_edges_per_feature:
        bin_edges = [None for _ in feature_names]
    else:
        for edges in bin_edges_per_feature:
            if edges is None:
                bin_edges.append(None)
            else:
                bin_edges.append([float(v) for v in np.asarray(edges).tolist()])

    split_payload: Dict[str, object] = {
        "seed": int(seed),
        "test_size": float(test_size),
    }
    if train_indices is not None:
        split_payload["train_indices"] = np.asarray(train_indices, dtype=int).tolist()
    if test_indices is not None:
        split_payload["test_indices"] = np.asarray(test_indices, dtype=int).tolist()

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "pipeline": str(pipeline),
        "target_name": to_label_text(target_name),
        "class_labels": [to_label_text(v) for v in np.asarray(class_labels, dtype=object).tolist()],
        "feature_names": [to_label_text(v) for v in feature_names],
        "accuracy": float(accuracy),
        "split": split_payload,
        "model_config": {
            "depth_budget": int(depth_budget),
            "lookahead": int(lookahead),
            "time_limit": float(time_limit),
            "max_bins": int(max_bins),
            "min_samples_leaf": int(min_samples_leaf),
            "min_child_size": int(min_child_size),
            "max_branching": int(max_branching),
            "reg": float(reg),
            "branch_penalty": None if branch_penalty is None else float(branch_penalty),
            "msplit_variant": str(msplit_variant) if msplit_variant is not None else None,
        },
        "binner": {
            "n_features": int(len(feature_names)),
            "bin_edges_per_feature": bin_edges,
        },
        "tree_artifact": {
            "tree": serialize_msplit_node(
                tree_root,
                binner,
                feature_names,
                np.asarray(class_labels, dtype=object),
                z_train=z_train,
                idxs=root_idxs,
                path_conditions=[],
            )
        },
    }


def _expand_serialized_spans(spans: Iterable[Tuple[int, int]]) -> List[int]:
    bins: List[int] = []
    for lo, hi in spans:
        lo_i = int(lo)
        hi_i = int(hi)
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        bins.extend(range(lo_i, hi_i + 1))
    return bins


def serialize_msplit_json_node(
    node: Dict[str, object],
    binner,
    feature_names: List[str],
    class_labels: np.ndarray,
    z_train: Optional[np.ndarray],
    idxs: Optional[np.ndarray],
    path_conditions: List[str],
) -> Dict[str, object]:
    node_type = str(node.get("type", "leaf"))
    if node_type == "leaf":
        counts = [int(v) for v in node.get("class_counts", [])]
        pred = int(node.get("prediction", 0))
        pred_label = class_labels[pred] if pred < len(class_labels) else pred
        n_samples = int(node.get("n_samples", idxs.size if idxs is not None else sum(counts)))
        return {
            "node_type": "leaf",
            "n_samples": n_samples,
            "predicted_class_index": int(pred),
            "predicted_class_label": to_label_text(pred_label),
            "true_class_dist": _class_dist_from_counts(counts, class_labels),
            "path_conditions": path_conditions,
        }

    feature_idx = int(node.get("feature", -1))
    feature_name = feature_names[feature_idx] if 0 <= feature_idx < len(feature_names) else f"x{feature_idx}"
    feature_display = feature_display_name(feature_name)
    n_samples = int(node.get("n_samples", idxs.size if idxs is not None else 0))

    feature_values = None
    if z_train is not None and idxs is not None and idxs.size > 0:
        matrix_feature_idx = _resolve_matrix_feature_index(feature_idx, z_train.shape[1])
        feature_values = z_train[idxs, matrix_feature_idx]

    children: List[Dict[str, object]] = []
    for raw_group in node.get("groups", []):
        group = dict(raw_group)
        spans_raw = group.get("spans", [])
        spans = [(int(lo), int(hi)) for lo, hi in spans_raw]
        bins = _expand_serialized_spans(spans)
        bins_arr = np.asarray(bins, dtype=np.int32)
        child_idxs = None
        if feature_values is not None:
            child_idxs = idxs[np.isin(feature_values, bins_arr)]
        cond = format_msplit_condition(feature_idx, bins, binner, feature_names)
        children.append(
            {
                "branch": {
                    "condition": cond,
                    "bins": [int(v) for v in bins],
                    "spans": [[int(lo), int(hi)] for lo, hi in spans],
                },
                "child": serialize_msplit_json_node(
                    dict(group["child"]),
                    binner,
                    feature_names,
                    class_labels,
                    z_train,
                    child_idxs,
                    path_conditions + [f"{feature_display} {cond}"],
                ),
            }
        )

    return {
        "node_type": "internal",
        "feature_index": int(feature_idx),
        "feature_name": to_label_text(feature_name),
        "feature_display_name": feature_display,
        "n_samples": n_samples,
        "n_way": int(len(children)),
        "children": children,
    }


def build_msplit_artifact_from_serialized_tree(
    *,
    dataset: str,
    pipeline: str,
    target_name: str,
    class_labels: np.ndarray,
    feature_names: List[str],
    accuracy: float,
    seed: int,
    test_size: float,
    depth_budget: int,
    lookahead: int,
    time_limit: float,
    max_bins: int,
    min_samples_leaf: int,
    min_child_size: int,
    max_branching: int,
    reg: float,
    branch_penalty: float | None = None,
    msplit_variant: Optional[str],
    tree_root: Dict[str, object],
    binner,
    z_train: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    root_idxs = np.arange(z_train.shape[0], dtype=np.int32)
    bin_edges: List[Optional[List[float]]] = []
    bin_edges_per_feature = _get_binner_feature_edges(binner)
    if not bin_edges_per_feature:
        bin_edges = [None for _ in feature_names]
    else:
        for edges in bin_edges_per_feature:
            if edges is None:
                bin_edges.append(None)
            else:
                bin_edges.append([float(v) for v in np.asarray(edges).tolist()])

    split_payload: Dict[str, object] = {
        "seed": int(seed),
        "test_size": float(test_size),
    }
    if train_indices is not None:
        split_payload["train_indices"] = np.asarray(train_indices, dtype=int).tolist()
    if test_indices is not None:
        split_payload["test_indices"] = np.asarray(test_indices, dtype=int).tolist()

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "pipeline": str(pipeline),
        "target_name": to_label_text(target_name),
        "class_labels": [to_label_text(v) for v in np.asarray(class_labels, dtype=object).tolist()],
        "feature_names": [to_label_text(v) for v in feature_names],
        "accuracy": float(accuracy),
        "split": split_payload,
        "model_config": {
            "depth_budget": int(depth_budget),
            "lookahead": int(lookahead),
            "time_limit": float(time_limit),
            "max_bins": int(max_bins),
            "min_samples_leaf": int(min_samples_leaf),
            "min_child_size": int(min_child_size),
            "max_branching": int(max_branching),
            "reg": float(reg),
            "branch_penalty": None if branch_penalty is None else float(branch_penalty),
            "msplit_variant": str(msplit_variant) if msplit_variant is not None else None,
        },
        "binner": {
            "n_features": int(len(feature_names)),
            "bin_edges_per_feature": bin_edges,
        },
        "tree_artifact": {
            "tree": serialize_msplit_json_node(
                tree_root,
                binner,
                feature_names,
                np.asarray(class_labels, dtype=object),
                z_train=z_train,
                idxs=root_idxs,
                path_conditions=[],
            )
        },
    }


def xgb_parse_feature_idx(token: str) -> int:
    text = str(token)
    if text.startswith("f"):
        return int(text[1:])
    return int(text)


def serialize_xgb_tree(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    class_labels: np.ndarray,
    tree_index: int = 0,
) -> Dict[str, object]:
    tree_df = model.get_booster().trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == int(tree_index)].copy()
    rows = {str(row["ID"]): row for _, row in tree_df.iterrows()}
    root_id = f"{int(tree_index)}-0"

    children: Dict[str, Tuple[str, str]] = {}
    for node_id, row in rows.items():
        if str(row["Feature"]) == "Leaf":
            continue
        children[node_id] = (str(row["Yes"]), str(row["No"]))

    node_sample_counts: Dict[str, int] = {}
    leaf_class_counts: Dict[str, np.ndarray] = {}

    def _route_and_count(node_id: str, idxs: np.ndarray) -> None:
        node_sample_counts[node_id] = int(idxs.size)
        if node_id not in children:
            counts = np.bincount(y_train[idxs], minlength=len(class_labels)).astype(np.int64)
            leaf_class_counts[node_id] = counts
            return

        row = rows[node_id]
        yes_id, no_id = children[node_id]
        missing_id = str(row["Missing"])
        feat_idx = xgb_parse_feature_idx(str(row["Feature"]))
        thr = float(row["Split"])

        col = x_train[idxs, feat_idx]
        nan_mask = np.isnan(col)
        left_mask = (col < thr) & (~nan_mask)
        right_mask = (col >= thr) & (~nan_mask)

        left_idxs = idxs[left_mask]
        right_idxs = idxs[right_mask]
        if nan_mask.any():
            miss_idxs = idxs[nan_mask]
            if missing_id == yes_id:
                left_idxs = np.concatenate([left_idxs, miss_idxs])
            elif missing_id == no_id:
                right_idxs = np.concatenate([right_idxs, miss_idxs])

        _route_and_count(yes_id, left_idxs)
        _route_and_count(no_id, right_idxs)

    _route_and_count(root_id, np.arange(x_train.shape[0], dtype=np.int32))

    def _build_node(node_id: str, path_conditions: List[str]) -> Dict[str, object]:
        row = rows[node_id]
        if node_id not in children:
            leaf_score = float(row["Gain"])
            counts = leaf_class_counts.get(node_id, np.zeros(len(class_labels), dtype=np.int64))
            if counts.sum() > 0:
                pred_idx = int(np.argmax(counts))
            else:
                pred_idx = 1 if len(class_labels) == 2 and leaf_score >= 0.0 else 0
            pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else pred_idx
            return {
                "node_type": "leaf",
                "id": node_id,
                "n_samples": int(node_sample_counts.get(node_id, 0)),
                "cover": float(row["Cover"]),
                "leaf_score": leaf_score,
                "predicted_class_index": int(pred_idx),
                "predicted_class_label": to_label_text(pred_label),
                "true_class_dist": _class_dist_from_counts(counts.tolist(), class_labels),
                "path_conditions": path_conditions,
            }

        yes_id, no_id = children[node_id]
        feat_idx = xgb_parse_feature_idx(str(row["Feature"]))
        feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else str(row["Feature"])
        feature_disp = feature_display_name(feature_name)
        thr = float(row["Split"])
        yes_cond = f"<= {_format_float(thr)}"
        no_cond = f"> {_format_float(thr)}"
        return {
            "node_type": "internal",
            "id": node_id,
            "feature_index": int(feat_idx),
            "feature_name": to_label_text(feature_name),
            "feature_display_name": feature_disp,
            "n_samples": int(node_sample_counts.get(node_id, int(round(float(row["Cover"]))))),
            "cover": float(row["Cover"]),
            "threshold": thr,
            "n_way": 2,
            "missing_goes_to": str(row["Missing"]),
            "children": [
                {
                    "branch": {"condition": yes_cond},
                    "child": _build_node(yes_id, path_conditions + [f"{feature_disp} {yes_cond}"]),
                },
                {
                    "branch": {"condition": no_cond},
                    "child": _build_node(no_id, path_conditions + [f"{feature_disp} {no_cond}"]),
                },
            ],
        }

    return {
        "tree_index": int(tree_index),
        "root_id": root_id,
        "tree": _build_node(root_id, []),
    }


def build_xgb_artifact(
    *,
    dataset: str,
    target_name: str,
    class_labels: np.ndarray,
    feature_names: List[str],
    accuracy: float,
    seed: int,
    test_size: float,
    depth_budget: int,
    n_estimators: int,
    learning_rate: float,
    num_threads: int,
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    tree_index: int = 0,
) -> Dict[str, object]:
    split_payload: Dict[str, object] = {
        "seed": int(seed),
        "test_size": float(test_size),
    }
    if train_indices is not None:
        split_payload["train_indices"] = np.asarray(train_indices, dtype=int).tolist()
    if test_indices is not None:
        split_payload["test_indices"] = np.asarray(test_indices, dtype=int).tolist()

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "pipeline": "xgboost",
        "target_name": to_label_text(target_name),
        "class_labels": [to_label_text(v) for v in np.asarray(class_labels, dtype=object).tolist()],
        "feature_names": [to_label_text(v) for v in feature_names],
        "accuracy": float(accuracy),
        "split": split_payload,
        "model_config": {
            "depth_budget": int(depth_budget),
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "num_threads": int(num_threads),
            "tree_index": int(tree_index),
        },
        "tree_artifact": serialize_xgb_tree(
            model,
            x_train,
            y_train,
            feature_names,
            class_labels=np.asarray(class_labels, dtype=object),
            tree_index=int(tree_index),
        ),
    }


def _class_dist_from_probs(probs: np.ndarray, n_samples: int, class_labels: np.ndarray) -> List[Dict[str, object]]:
    counts = np.rint(np.asarray(probs, dtype=float) * float(max(n_samples, 0))).astype(int)
    return _class_dist_from_counts(counts.tolist(), class_labels)


def _flatten_feature_key(key: object, feature_dict: Dict[object, List[int]]) -> List[int]:
    if key in feature_dict:
        return [int(v) for v in feature_dict[key]]
    if isinstance(key, tuple):
        out: List[int] = []
        for part in key:
            if part in feature_dict:
                out.extend(int(v) for v in feature_dict[part])
            else:
                try:
                    out.append(int(part))
                except Exception:
                    continue
        return out
    try:
        return [int(key)]
    except Exception:
        return []


def serialize_shapecart_node(
    model,
    node_idx: int,
    feature_names: List[str],
    class_labels: np.ndarray,
    path_conditions: List[str],
) -> Dict[str, object]:
    probs = np.asarray(model.values[node_idx], dtype=float)
    n_samples = int(model.n_samples[node_idx])
    pred_idx = int(np.argmax(probs)) if probs.size else 0
    pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else pred_idx

    if bool(model.is_leaf[node_idx]) or model.children[node_idx] is None:
        return {
            "node_type": "leaf",
            "node_index": int(node_idx),
            "n_samples": n_samples,
            "predicted_class_index": pred_idx,
            "predicted_class_label": to_label_text(pred_label),
            "true_class_dist": _class_dist_from_probs(probs, n_samples, class_labels),
            "path_conditions": path_conditions,
        }

    node = model.nodes[node_idx]
    feature_key = getattr(node, "final_key", None)
    raw_feature_dict = getattr(node, "feature_dict", {}) or {}
    selected_indices = _flatten_feature_key(feature_key, raw_feature_dict)
    selected_names = [
        feature_names[i] if 0 <= int(i) < len(feature_names) else f"x{int(i)}"
        for i in selected_indices
    ]
    display_parts = [feature_display_name(str(name), max_len=18) for name in selected_names]
    feature_display = " + ".join(display_parts) if display_parts else str(feature_key)

    inner_tree = getattr(node, "final_tree", None)
    inner_internal = inner_leaves = None
    if inner_tree is not None and hasattr(inner_tree, "tree_") and hasattr(inner_tree.tree_, "children_left"):
        children_left = np.asarray(inner_tree.tree_.children_left)
        leaf_mask = children_left == -1
        inner_leaves = int(np.sum(leaf_mask))
        inner_internal = int(children_left.size - np.sum(leaf_mask))

    children = []
    child_indices = list(model.children[node_idx] or [])
    for branch_idx, child_idx in enumerate(child_indices):
        cond = f"group = {int(branch_idx)}"
        children.append(
            {
                "branch": {"condition": cond, "branch_index": int(branch_idx)},
                "child": serialize_shapecart_node(
                    model,
                    int(child_idx),
                    feature_names,
                    class_labels,
                    path_conditions + [cond],
                ),
            }
        )

    return {
        "node_type": "internal",
        "node_index": int(node_idx),
        "feature_index": int(selected_indices[0]) if selected_indices else -1,
        "feature_name": to_label_text(selected_names[0] if selected_names else str(feature_key)),
        "feature_display_name": feature_display,
        "n_samples": n_samples,
        "n_way": int(len(child_indices)),
        "split_metadata": {
            "selected_feature_key": to_label_text(feature_key),
            "selected_feature_indices": [int(v) for v in selected_indices],
            "selected_feature_names": [to_label_text(v) for v in selected_names],
            "inner_tree_internal_nodes": inner_internal,
            "inner_tree_leaves": inner_leaves,
        },
        "children": children,
    }


def build_shapecart_artifact(
    *,
    dataset: str,
    target_name: str,
    class_labels: np.ndarray,
    feature_names: List[str],
    accuracy: float,
    seed: int,
    test_size: float,
    depth_budget: int,
    k: int,
    min_samples_leaf: int,
    min_samples_split: int,
    inner_min_samples_leaf: int | float,
    inner_min_samples_split: int,
    inner_max_depth: int,
    inner_max_leaf_nodes: int,
    max_iter: int,
    model,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    split_payload: Dict[str, object] = {
        "seed": int(seed),
        "test_size": float(test_size),
    }
    if train_indices is not None:
        split_payload["train_indices"] = np.asarray(train_indices, dtype=int).tolist()
    if test_indices is not None:
        split_payload["test_indices"] = np.asarray(test_indices, dtype=int).tolist()

    return {
        "schema_version": 1,
        "dataset": str(dataset),
        "pipeline": "shapecart",
        "target_name": to_label_text(target_name),
        "class_labels": [to_label_text(v) for v in np.asarray(class_labels, dtype=object).tolist()],
        "feature_names": [to_label_text(v) for v in feature_names],
        "accuracy": float(accuracy),
        "split": split_payload,
        "model_config": {
            "depth_budget": int(depth_budget),
            "k": int(k),
            "min_samples_leaf": int(min_samples_leaf),
            "min_samples_split": int(min_samples_split),
            "inner_min_samples_leaf": (
                int(inner_min_samples_leaf)
                if float(inner_min_samples_leaf).is_integer()
                else float(inner_min_samples_leaf)
            ),
            "inner_min_samples_split": int(inner_min_samples_split),
            "inner_max_depth": int(inner_max_depth),
            "inner_max_leaf_nodes": int(inner_max_leaf_nodes),
            "max_iter": int(max_iter),
        },
        "tree_artifact": {
            "tree": serialize_shapecart_node(
                model,
                node_idx=0,
                feature_names=feature_names,
                class_labels=np.asarray(class_labels, dtype=object),
                path_conditions=[],
            )
        },
    }
