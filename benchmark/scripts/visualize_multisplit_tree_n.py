"""Visualize one fitted tree for XGBoost or LightGBM-binned MSPLIT."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msdt_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.benchmark_paths import BENCHMARK_ARTIFACTS_ROOT, ensure_repo_import_paths

ensure_repo_import_paths(include_msplit_src=True)

from benchmark.scripts.experiment_utils import DATASET_LOADERS, encode_binary_target, make_preprocessor
from benchmark.scripts.lightgbm_binning import fit_lightgbm_binner
from benchmark.scripts.tree_artifact_utils import write_artifact_json
from split import MSPLIT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one model and save a tree visualization PNG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_LOADERS.keys()), required=True)
    parser.add_argument("--pipeline", choices=["xgboost", "lightgbm"], required=True)
    parser.add_argument("--depth-budget", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--time-limit", type=float, default=3000.0)
    parser.add_argument("--lookahead-cap", type=int, default=3)
    parser.add_argument("--max-bins", type=int, default=255)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=8)
    parser.add_argument("--max-branching", type=int, default=0)
    parser.add_argument("--reg", type=float, default=0.01)
    parser.add_argument("--xgb-n-estimators", type=int, default=300)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-num-threads", type=int, default=4)
    parser.add_argument(
        "--lgb-device-type",
        choices=["cpu", "gpu", "cuda"],
        default="gpu",
        help="LightGBM backend for preprocessing binning in lightgbm pipeline.",
    )
    parser.add_argument("--lgb-ensemble-runs", type=int, default=1)
    parser.add_argument("--lgb-ensemble-feature-fraction", type=float, default=0.8)
    parser.add_argument("--lgb-ensemble-bagging-fraction", type=float, default=0.8)
    parser.add_argument("--lgb-ensemble-bagging-freq", type=int, default=1)
    parser.add_argument("--lgb-threshold-dedup-eps", type=float, default=1e-9)
    parser.add_argument("--lgb-gpu-platform-id", type=int, default=0)
    parser.add_argument("--lgb-gpu-device-id", type=int, default=0)
    parser.add_argument(
        "--lgb-gpu-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If GPU binning fails, retry LightGBM binning on CPU.",
    )

    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--artifact-in", type=str, default=None, help="Optional input artifact path. If set, skip training and render directly.")
    parser.add_argument("--artifact-out", type=str, default=None, help="Optional JSON artifact path for precise tree metadata.")
    return parser.parse_args()


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f6f8fb",
            "axes.facecolor": "#f6f8fb",
            "font.family": "monospace",
            "font.monospace": [
                "JetBrains Mono",
                "Fira Code",
                "Source Code Pro",
                "DejaVu Sans Mono",
                "Liberation Mono",
                "Consolas",
                "Monaco",
            ],
            "axes.titleweight": "bold",
            "axes.titlepad": 12.0,
        }
    )


def _tree_figure_size(leaf_count: int, depth: int) -> Tuple[float, float]:
    leaf_count = max(1, int(leaf_count))
    depth = max(1, int(depth))
    fig_w = max(13.0, min(60.0, 2.9 * leaf_count))
    fig_h = max(7.5, min(24.0, 2.3 * (depth + 2)))
    return fig_w, fig_h


def _tree_figure_dpi(leaf_count: int) -> int:
    leaf_count = max(1, int(leaf_count))
    if leaf_count >= 250:
        return 90
    if leaf_count >= 150:
        return 110
    if leaf_count >= 80:
        return 150
    return 230


def _add_config_box(ax, entries: List[Tuple[str, str]]) -> None:
    if not entries:
        return

    n_cols = 2 if len(entries) >= 8 else 1
    rows: List[List[Tuple[str, str]]] = [entries[i : i + n_cols] for i in range(0, len(entries), n_cols)]
    n_rows = len(rows)

    col_entries: List[List[Tuple[str, str]]] = []
    for c in range(n_cols):
        col_entries.append([row[c] for row in rows if c < len(row)])

    col_widths: List[float] = []
    for col in col_entries:
        key_w = max(len(k) for k, _ in col)
        val_w = max(len(str(v)) for _, v in col)
        col_w = min(0.195, max(0.110, 0.0042 * (key_w + val_w + 5)))
        col_widths.append(col_w)

    line_h = 0.026
    header_h = 0.026
    pad_x = 0.007
    pad_top = 0.006
    pad_bottom = 0.005
    col_gap = 0.010
    width = sum(col_widths) + 2 * pad_x + (n_cols - 1) * col_gap
    width = min(0.40, max(0.22, width))
    height = pad_top + header_h + n_rows * line_h + pad_bottom
    height = min(0.58, max(0.10, height))

    x_right = 0.985
    y_top = 0.985
    x_left = x_right - width
    y_bottom = y_top - height

    box = FancyBboxPatch(
        (x_left, y_bottom),
        width,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=0.9,
        edgecolor="#94a3b8",
        facecolor="#ffffff",
        alpha=0.96,
        zorder=8,
    )
    ax.add_patch(box)

    ax.text(
        x_left + pad_x,
        y_top - pad_top,
        "Config:",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.1,
        fontweight="bold",
        color="#0f172a",
        zorder=9,
    )

    body_top = y_top - pad_top - header_h
    col_left = x_left + pad_x
    for col_i, col in enumerate(col_entries):
        this_left = col_left
        this_right = this_left + col_widths[col_i]
        for row_i, (key, value) in enumerate(col):
            y = body_top - row_i * line_h - 0.5 * line_h
            ax.text(
                this_left,
                y,
                str(key),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8.3,
                color="#0f172a",
                zorder=9,
            )
            ax.text(
                this_right,
                y,
                str(value),
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8.3,
                fontweight="bold",
                color="#0f172a",
                zorder=9,
            )
        col_left = this_right + col_gap


def _is_leaf(node: object) -> bool:
    return not hasattr(node, "children")


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    text = f"{float(value):.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _feature_display_name(name: str, max_len: int = 24) -> str:
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


def _to_label_text(value: object) -> str:
    return str(value)


def _escape_mathtext(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("_", r"\_")


def _bold_key_text(key: str) -> str:
    return f"$\\mathbf{{{_escape_mathtext(key)}}}$"


def _kv_line(key: str, value: object, bold_value: bool = True) -> str:
    value_text = str(value)
    if bold_value and re.fullmatch(r"[A-Za-z0-9_.+\-]+", value_text):
        return f"{key}=$\\mathbf{{{_escape_mathtext(value_text)}}}$"
    return f"{key}={value_text}"


def _wrap_line(text: str, width: int = 34) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False))


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


def _expand_spans(spans: Iterable[Tuple[int, int]]) -> List[int]:
    bins: List[int] = []
    for lo, hi in spans:
        lo_i = int(lo)
        hi_i = int(hi)
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i
        bins.extend(range(lo_i, hi_i + 1))
    return bins


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


def _is_binary_onehot_feature(feature_name: str, edges: np.ndarray) -> bool:
    if not feature_name.startswith("cat__"):
        return False
    if edges is None or len(edges) != 1:
        return False
    return abs(float(edges[0]) - 0.5) < 1e-6


def _parse_onehot_name(feature_name: str) -> Tuple[str, str]:
    raw = feature_name[5:] if feature_name.startswith("cat__") else feature_name
    base, sep, category = raw.rpartition("_")
    if not sep:
        return _feature_display_name(raw), "1"
    return _feature_display_name(base), category


def _format_msplit_condition(feature_idx: int, bins: Iterable[int], binner, feature_names: List[str]) -> str:
    feature_name = feature_names[feature_idx] if 0 <= feature_idx < len(feature_names) else f"x{feature_idx}"
    bins_list = sorted(set(int(b) for b in bins))
    bins_for_display = bins_list
    if len(bins_list) >= 2:
        # Merge gaps from unseen bins so labels stay readable.
        bins_for_display = list(range(bins_list[0], bins_list[-1] + 1))
    edges = binner.bin_edges_per_feature[feature_idx]

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


def _subtree_signature(node: object) -> Tuple:
    if _is_leaf(node):
        prediction = int(getattr(node, "prediction", 0))
        class_counts = tuple(int(v) for v in getattr(node, "class_counts", ()))
        return ("leaf", prediction, class_counts)
    child_sigs = tuple(
        (
            tuple(int(b) for b in bins),
            _subtree_signature(child),
        )
        for bins, child in _group_children(node)
    )
    return ("node", int(getattr(node, "feature", -1)), child_sigs)


def _group_children(node: object) -> List[Tuple[List[int], object]]:
    if _is_leaf(node):
        return []

    child_spans = getattr(node, "child_spans", None)
    if isinstance(child_spans, dict) and child_spans:
        groups: List[Tuple[List[int], object]] = []
        for group_id in sorted(node.children.keys()):
            spans = child_spans.get(group_id, ())
            bins = _expand_spans(spans)
            if not bins:
                continue
            groups.append((bins, node.children[group_id]))
        return groups

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


def _count_leaves(node: object) -> int:
    if _is_leaf(node):
        return 1
    return sum(_count_leaves(child) for _, child in _group_children(node))


def _tree_depth(node: object) -> int:
    if _is_leaf(node):
        return 0
    return 1 + max(_tree_depth(child) for _, child in _group_children(node))


def _assign_positions(node: object, depth: int, x_cursor: List[float], positions: Dict[int, Tuple[float, float]]) -> None:
    if _is_leaf(node):
        x = x_cursor[0]
        x_cursor[0] += 1.0
        positions[id(node)] = (x, float(depth))
        return

    groups = _group_children(node)
    for _, child in groups:
        _assign_positions(child, depth + 1, x_cursor, positions)
    child_x = [positions[id(child)][0] for _, child in groups]
    positions[id(node)] = (float(np.mean(child_x)), float(depth))


def _box_dims(text: str, is_leaf: bool) -> Tuple[float, float]:
    lines = text.split("\n")
    max_chars = max(len(line) for line in lines) if lines else 10
    n_lines = max(1, len(lines))

    base_w = 0.13 if is_leaf else 0.15
    char_scale = 0.0027
    max_w = 0.22 if is_leaf else 0.27
    width = min(max_w, base_w + char_scale * max_chars)

    base_h = 0.05 if is_leaf else 0.06
    line_scale = 0.017
    height = min(0.23, base_h + line_scale * max(0, n_lines - 1))
    return width, height


def _interval_label(feature: int, bins: List[int], binner, feature_names: List[str]) -> str:
    return _format_msplit_condition(feature, bins, binner, feature_names)


def _node_label(
    node: object,
    classes: np.ndarray,
    feature_names: List[str],
    n_override: Optional[int] = None,
    group_count_override: Optional[int] = None,
) -> str:
    n_samples = int(n_override) if n_override is not None else int(getattr(node, "n_samples", 0))
    if _is_leaf(node):
        pred = int(node.prediction)
        pred_label = classes[pred] if pred < len(classes) else pred
        class_counts = tuple(int(v) for v in getattr(node, "class_counts", ()))
        if class_counts:
            pairs = []
            for i, cnt in enumerate(class_counts):
                label = classes[i] if i < len(classes) else i
                pairs.append(f"({label}, {cnt})")
            true_class_line = _bold_key_text("true_class_dist") + "=[" + ", ".join(pairs) + "]"
        else:
            true_class_line = _bold_key_text("true_class_dist") + "=[]"
        return f"{_kv_line('predicted_class', pred_label)}\n{true_class_line}"

    feat_idx = int(node.feature)
    feat = feature_names[feat_idx] if feat_idx < len(feature_names) else f"x{feat_idx}"
    feat = _feature_display_name(feat)
    group_count = int(group_count_override) if group_count_override is not None else int(getattr(node, "group_count", 0))
    return f"{_kv_line('feature', feat)}\n({group_count}-way)\n{_kv_line('n', n_samples)}"


def _draw_msplit_tree(
    ax,
    root: object,
    binner,
    feature_names: List[str],
    classes: np.ndarray,
    z_train: Optional[np.ndarray] = None,
):
    positions: Dict[int, Tuple[float, float]] = {}
    _assign_positions(root, depth=0, x_cursor=[0.0], positions=positions)

    n_leaves = max(1.0, float(_count_leaves(root)))
    depth = max(1, _tree_depth(root))
    leaf_slot = 0.92 / n_leaves

    def _to_plot_coords(x: float, d: float) -> Tuple[float, float]:
        nx = (x + 1.0) / (n_leaves + 1.0)
        ny = 1.0 - ((d + 0.55) / (depth + 1.3))
        return nx, ny

    def _draw_node(node: object, idxs: Optional[np.ndarray]):
        x, d = positions[id(node)]
        nx, ny = _to_plot_coords(x, d)

        is_leaf = _is_leaf(node)
        groups = _group_children(node)

        if is_leaf:
            pred = int(node.prediction)
            color = "#8fc2e8" if pred == 0 else "#f3c6bc"
            style = "round,pad=0.008,rounding_size=0.028"
        else:
            color = "#d6e4f1"
            style = "round,pad=0.008,rounding_size=0.006"

        n_override = int(idxs.size) if idxs is not None else None
        label = _node_label(
            node,
            classes,
            feature_names,
            n_override=n_override,
            group_count_override=(len(groups) if not is_leaf else None),
        )
        label = "\n".join(_wrap_line(line, width=30) for line in label.split("\n"))
        w, h = _box_dims(label, is_leaf=is_leaf)
        subtree_leaves = max(1, _count_leaves(node))
        w = min(w, leaf_slot * 0.78 * subtree_leaves)

        box = FancyBboxPatch(
            (nx - w / 2.0, ny - h / 2.0),
            w,
            h,
            boxstyle=style,
            linewidth=1.2,
            edgecolor="#2f3640",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        node_font = 7.9 if (is_leaf and n_leaves > 6) else (8.6 if is_leaf else 9.1)
        ax.text(nx, ny, label, ha="center", va="center", fontsize=node_font, zorder=3)

        if is_leaf:
            return

        feature_idx = int(node.feature)
        feature_values = z_train[idxs, feature_idx] if (z_train is not None and idxs is not None and idxs.size > 0) else None

        branch_count = len(groups)
        for branch_i, (bins, child) in enumerate(groups):
            cx, cd = positions[id(child)]
            cnx, cny = _to_plot_coords(cx, cd)

            arr = FancyArrowPatch(
                (nx, ny - h / 2.0),
                (cnx, cny + 0.045),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#2f3640",
                alpha=0.40,
                zorder=1,
            )
            ax.add_patch(arr)

            child_idxs = None
            if feature_values is not None:
                mask = np.isin(feature_values, np.asarray(bins, dtype=np.int32))
                child_idxs = idxs[mask]

            edge_cond = _interval_label(feature_idx, bins, binner, feature_names)
            edge_text = _wrap_line(edge_cond, width=22)

            mx = (nx + cnx) / 2.0
            my = (ny + cny) / 2.0 + 0.015 + 0.010 * (branch_i - (branch_count - 1) / 2.0)
            ax.text(
                mx,
                my,
                edge_text,
                fontsize=8.8,
                ha="center",
                va="center",
                color="#111827",
                fontweight="semibold",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.92},
                zorder=4,
            )

            _draw_node(child, child_idxs)

    root_idxs = None if z_train is None else np.arange(z_train.shape[0], dtype=np.int32)
    _draw_node(root, root_idxs)


def _xgb_assign_positions(
    node_id: str,
    children: Dict[str, Tuple[str, str]],
    x_cursor: List[float],
    pos: Dict[str, Tuple[float, float]],
    depth: int,
) -> None:
    if node_id not in children:
        x = x_cursor[0]
        x_cursor[0] += 1.0
        pos[node_id] = (x, float(depth))
        return
    yes_id, no_id = children[node_id]
    _xgb_assign_positions(yes_id, children, x_cursor, pos, depth + 1)
    _xgb_assign_positions(no_id, children, x_cursor, pos, depth + 1)
    cx = (pos[yes_id][0] + pos[no_id][0]) / 2.0
    pos[node_id] = (cx, float(depth))


def _xgb_parse_feature_idx(token: str) -> int:
    text = str(token)
    if text.startswith("f"):
        return int(text[1:])
    return int(text)


def _draw_xgb_tree(
    ax,
    model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    class_labels: np.ndarray,
) -> None:
    tree_df = model.get_booster().trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == 0].copy()
    rows = {str(row["ID"]): row for _, row in tree_df.iterrows()}
    root_id = "0-0"

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
        feat_idx = _xgb_parse_feature_idx(str(row["Feature"]))
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

    positions: Dict[str, Tuple[float, float]] = {}
    _xgb_assign_positions(root_id, children, [0.0], positions, 0)
    leaf_count = max(1.0, float(sum(1 for node_id in rows if node_id not in children)))
    max_depth = max(int(d) for _, d in positions.values()) if positions else 1
    leaf_slot = 0.92 / leaf_count

    desc_leaf_count: Dict[str, int] = {}

    def _count_desc(node_id: str) -> int:
        if node_id in desc_leaf_count:
            return desc_leaf_count[node_id]
        if node_id not in children:
            desc_leaf_count[node_id] = 1
            return 1
        yes_id, no_id = children[node_id]
        count = _count_desc(yes_id) + _count_desc(no_id)
        desc_leaf_count[node_id] = count
        return count

    _count_desc(root_id)

    def _to_plot_coords(x: float, d: float) -> Tuple[float, float]:
        nx = (x + 1.0) / (leaf_count + 1.0)
        ny = 1.0 - ((d + 0.55) / (max_depth + 1.3))
        return nx, ny

    for node_id, row in rows.items():
        x, d = positions[node_id]
        nx, ny = _to_plot_coords(x, d)
        is_leaf = node_id not in children

        if is_leaf:
            color = "#f7d8a8"
            leaf_score = float(row["Gain"])
            vote = 1 if leaf_score >= 0.0 else 0
            pred_label = class_labels[vote] if vote < len(class_labels) else vote
            counts = leaf_class_counts.get(node_id, np.zeros(len(class_labels), dtype=np.int64))
            pairs = []
            for i, cnt in enumerate(counts.tolist()):
                label = class_labels[i] if i < len(class_labels) else i
                pairs.append(f"({label}, {int(cnt)})")
            text = (
                _kv_line("predicted_class", pred_label)
                + "\n"
                + _bold_key_text("true_class_dist")
                + "=["
                + ", ".join(pairs)
                + "]"
            )
            w, h = _box_dims(text, is_leaf=True)
            style = "round,pad=0.008,rounding_size=0.02"
        else:
            feat_idx = _xgb_parse_feature_idx(str(row["Feature"]))
            feat = feature_names[feat_idx] if feat_idx < len(feature_names) else str(row["Feature"])
            feat = _feature_display_name(feat)
            text = (
                _kv_line("feature", feat)
                + "\n(2-way)\n"
                + _kv_line("n", node_sample_counts.get(node_id, int(round(float(row["Cover"])))))
            )
            w, h = _box_dims(text, is_leaf=False)
            color = "#d6e4f1"
            style = "round,pad=0.008,rounding_size=0.006"

        subtree_leaves = max(1, desc_leaf_count.get(node_id, 1))
        w = min(w, leaf_slot * 0.78 * subtree_leaves)

        box = FancyBboxPatch(
            (nx - w / 2.0, ny - h / 2.0),
            w,
            h,
            boxstyle=style,
            linewidth=1.1,
            edgecolor="#2f3640",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        node_font = 7.0 if (is_leaf and leaf_count > 6) else 8.2
        ax.text(nx, ny, text, ha="center", va="center", fontsize=node_font, zorder=3)

    for node_id, (yes_id, no_id) in children.items():
        x, d = positions[node_id]
        nx, ny = _to_plot_coords(x, d)
        row = rows[node_id]
        feat_idx = _xgb_parse_feature_idx(str(row["Feature"]))
        feat = feature_names[feat_idx] if feat_idx < len(feature_names) else str(row["Feature"])
        feat = _feature_display_name(feat)
        thr = float(row["Split"])

        edge_labels = [(yes_id, f"<= {_format_float(thr)}"), (no_id, f"> {_format_float(thr)}")]
        for k, (child_id, edge_label) in enumerate(edge_labels):
            cx, cd = positions[child_id]
            cnx, cny = _to_plot_coords(cx, cd)
            arr = FancyArrowPatch(
                (nx, ny - 0.038),
                (cnx, cny + 0.040),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#2f3640",
                alpha=0.38,
                zorder=1,
            )
            ax.add_patch(arr)

            mx = (nx + cnx) / 2.0
            my = (ny + cny) / 2.0 + 0.012 + (k - 0.5) * 0.010
            txt = _wrap_line(edge_label, width=18)
            ax.text(
                mx,
                my,
                txt,
                fontsize=8.7,
                ha="center",
                va="center",
                color="#111827",
                fontweight="semibold",
                bbox={"boxstyle": "round,pad=0.14", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.92},
                zorder=4,
            )


def _artifact_is_leaf(node: Dict[str, Any]) -> bool:
    if str(node.get("node_type", "")).lower() == "leaf":
        return True
    children = node.get("children")
    return not isinstance(children, list) or len(children) == 0


def _artifact_children(
    node: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    def _artifact_children_raw(raw_node: Dict[str, Any]) -> List[Tuple[str, List[int], Dict[str, Any]]]:
        out_raw: List[Tuple[str, List[int], Dict[str, Any]]] = []
        children = raw_node.get("children")
        if not isinstance(children, list):
            return out_raw
        for entry in children:
            if not isinstance(entry, dict):
                continue
            child = entry.get("child")
            if not isinstance(child, dict):
                continue
            branch = entry.get("branch") if isinstance(entry.get("branch"), dict) else {}
            cond = str(branch.get("condition", ""))
            bins_raw = branch.get("bins", [])
            bins: List[int] = []
            if isinstance(bins_raw, list):
                bins = [int(v) for v in bins_raw]
            out_raw.append((cond, bins, child))
        return out_raw

    def _artifact_subtree_signature(sig_node: Dict[str, Any]) -> Tuple:
        if _artifact_is_leaf(sig_node):
            pred_idx = int(sig_node.get("predicted_class_index", -1))
            true_dist = sig_node.get("true_class_dist", [])
            counts: List[Tuple[int, int]] = []
            if isinstance(true_dist, list):
                for entry in true_dist:
                    if not isinstance(entry, dict):
                        continue
                    counts.append((int(entry.get("class_index", len(counts))), int(entry.get("count", 0))))
            return ("leaf", pred_idx, tuple(counts))

        kids = _artifact_children_raw(sig_node)
        child_sigs = tuple(_artifact_subtree_signature(child) for _, _, child in kids)
        return (
            "internal",
            int(sig_node.get("feature_index", -1)),
            int(sig_node.get("n_samples", 0)),
            child_sigs,
        )

    def _merged_condition(
        feature_idx: int,
        bins: List[int],
        fallback_cond: str,
        cond_count: int,
    ) -> str:
        if cond_count <= 1:
            return fallback_cond
        if (
            binner is not None
            and hasattr(binner, "bin_edges_per_feature")
            and feature_idx >= 0
            and feature_idx < len(getattr(binner, "bin_edges_per_feature", []))
        ):
            return _format_msplit_condition(feature_idx, bins, binner, feature_names or [])
        if not bins:
            return fallback_cond
        lo = min(bins)
        hi = max(bins)
        return f"bin {lo}" if lo == hi else f"bins {lo}-{hi}"

    raw = _artifact_children_raw(node)
    if not raw:
        return []

    out: List[Tuple[str, Dict[str, Any]]] = []
    feature_idx = int(node.get("feature_index", -1))
    cur_cond = raw[0][0]
    cur_bins = list(raw[0][1])
    cur_cond_count = 1
    cur_child = raw[0][2]
    cur_sig = _artifact_subtree_signature(cur_child)
    for cond, bins, child in raw[1:]:
        sig = _artifact_subtree_signature(child)
        if sig == cur_sig:
            cur_bins.extend(bins)
            cur_cond_count += 1
            continue
        merged_cond = _merged_condition(feature_idx, cur_bins, cur_cond, cur_cond_count)
        out.append((merged_cond, cur_child))
        cur_cond = cond
        cur_bins = list(bins)
        cur_cond_count = 1
        cur_child = child
        cur_sig = sig
    merged_cond = _merged_condition(feature_idx, cur_bins, cur_cond, cur_cond_count)
    out.append((merged_cond, cur_child))
    return out


def _artifact_count_leaves(
    node: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> int:
    kids = _artifact_children(node, binner=binner, feature_names=feature_names)
    if not kids:
        return 1
    return sum(_artifact_count_leaves(child, binner=binner, feature_names=feature_names) for _, child in kids)


def _artifact_depth(
    node: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> int:
    kids = _artifact_children(node, binner=binner, feature_names=feature_names)
    if not kids:
        return 0
    return 1 + max(_artifact_depth(child, binner=binner, feature_names=feature_names) for _, child in kids)


def _artifact_assign_positions(
    node: Dict[str, Any],
    depth: int,
    x_cursor: List[float],
    positions: Dict[int, Tuple[float, float]],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> None:
    node_key = id(node)
    kids = _artifact_children(node, binner=binner, feature_names=feature_names)
    if not kids:
        x = x_cursor[0]
        x_cursor[0] += 1.0
        positions[node_key] = (x, float(depth))
        return
    child_x: List[float] = []
    for _, child in kids:
        _artifact_assign_positions(child, depth + 1, x_cursor, positions, binner=binner, feature_names=feature_names)
        child_x.append(positions[id(child)][0])
    positions[node_key] = (float(np.mean(child_x)), float(depth))


def _artifact_leaf_label(node: Dict[str, Any]) -> str:
    pred = node.get("predicted_class_label", node.get("predicted_class_index", ""))
    true_dist = node.get("true_class_dist", [])
    pairs: List[str] = []
    if isinstance(true_dist, list):
        for entry in true_dist:
            if not isinstance(entry, dict):
                continue
            label = entry.get("class_label", entry.get("class_index", ""))
            count = entry.get("count", 0)
            pairs.append(f"({label}, {count})")
    if not pairs:
        pairs = ["(unknown, 0)"]
    return f"{_kv_line('predicted_class', pred)}\n{_bold_key_text('true_class_dist')}=[{', '.join(pairs)}]"


def _artifact_internal_label(node: Dict[str, Any], n_way: int) -> str:
    feat = node.get("feature_display_name", node.get("feature_name", "x"))
    n_samples = node.get("n_samples", 0)
    return f"{_kv_line('feature', feat)}\n({int(n_way)}-way)\n{_kv_line('n', n_samples)}"


def _draw_serialized_tree(
    ax,
    root: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> tuple[int, int]:
    positions: Dict[int, Tuple[float, float]] = {}
    _artifact_assign_positions(root, depth=0, x_cursor=[0.0], positions=positions, binner=binner, feature_names=feature_names)

    n_leaves = max(1.0, float(_artifact_count_leaves(root, binner=binner, feature_names=feature_names)))
    depth = max(1, _artifact_depth(root, binner=binner, feature_names=feature_names))
    leaf_slot = 0.92 / n_leaves

    def _to_plot_coords(x: float, d: float) -> Tuple[float, float]:
        nx = (x + 1.0) / (n_leaves + 1.0)
        ny = 1.0 - ((d + 0.55) / (depth + 1.3))
        return nx, ny

    def _draw_node(node: Dict[str, Any]) -> None:
        node_key = id(node)
        x, d = positions[node_key]
        nx, ny = _to_plot_coords(x, d)
        kids = _artifact_children(node, binner=binner, feature_names=feature_names)
        is_leaf = len(kids) == 0

        if is_leaf:
            pred_idx = int(node.get("predicted_class_index", 0))
            color = "#8fc2e8" if pred_idx == 0 else "#f3c6bc"
            style = "round,pad=0.008,rounding_size=0.028"
            label = _artifact_leaf_label(node)
        else:
            color = "#d6e4f1"
            style = "round,pad=0.008,rounding_size=0.006"
            label = _artifact_internal_label(node, n_way=max(2, len(kids)))

        label = "\n".join(_wrap_line(line, width=30) for line in label.split("\n"))
        w, h = _box_dims(label, is_leaf=is_leaf)
        subtree_leaves = max(1, _artifact_count_leaves(node, binner=binner, feature_names=feature_names))
        w = min(w, leaf_slot * 0.78 * subtree_leaves)

        box = FancyBboxPatch(
            (nx - w / 2.0, ny - h / 2.0),
            w,
            h,
            boxstyle=style,
            linewidth=1.2,
            edgecolor="#2f3640",
            facecolor=color,
            zorder=2,
        )
        ax.add_patch(box)
        node_font = 7.9 if (is_leaf and n_leaves > 6) else (8.6 if is_leaf else 9.1)
        ax.text(nx, ny, label, ha="center", va="center", fontsize=node_font, zorder=3)

        if is_leaf:
            return

        branch_count = len(kids)
        for branch_i, (edge_cond, child) in enumerate(kids):
            cx, cd = positions[id(child)]
            cnx, cny = _to_plot_coords(cx, cd)

            arr = FancyArrowPatch(
                (nx, ny - h / 2.0),
                (cnx, cny + 0.045),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.0,
                color="#2f3640",
                alpha=0.30,
                zorder=1,
            )
            ax.add_patch(arr)

            edge_text = _wrap_line(str(edge_cond), width=22)
            mx = (nx + cnx) / 2.0
            my = (ny + cny) / 2.0 + 0.015 + 0.010 * (branch_i - (branch_count - 1) / 2.0)
            ax.text(
                mx,
                my,
                edge_text,
                fontsize=9.4,
                ha="center",
                va="center",
                color="#111827",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.94},
                zorder=4,
            )

            _draw_node(child)

    _draw_node(root)
    return int(depth), int(n_leaves)


def _class_dist_from_counts(counts: Iterable[int], class_labels: np.ndarray) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for i, cnt in enumerate(counts):
        label = class_labels[i] if i < len(class_labels) else i
        out.append(
            {
                "class_index": int(i),
                "class_label": _to_label_text(label),
                "count": int(cnt),
            }
        )
    return out


def _serialize_msplit_node(
    node: object,
    binner,
    feature_names: List[str],
    class_labels: np.ndarray,
    z_train: Optional[np.ndarray],
    idxs: Optional[np.ndarray],
    path_conditions: List[str],
) -> Dict[str, object]:
    n_samples = int(idxs.size) if idxs is not None else int(getattr(node, "n_samples", 0))
    if _is_leaf(node):
        pred = int(node.prediction)
        pred_label = class_labels[pred] if pred < len(class_labels) else pred
        counts = [int(v) for v in getattr(node, "class_counts", ())]
        return {
            "node_type": "leaf",
            "n_samples": n_samples,
            "predicted_class_index": pred,
            "predicted_class_label": _to_label_text(pred_label),
            "true_class_dist": _class_dist_from_counts(counts, class_labels),
            "path_conditions": path_conditions,
        }

    feature_idx = int(node.feature)
    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"x{feature_idx}"
    feature_display = _feature_display_name(feature_name)
    groups = _group_children(node)
    feature_values = z_train[idxs, feature_idx] if (z_train is not None and idxs is not None and idxs.size > 0) else None
    children: List[Dict[str, object]] = []
    for bins, child in groups:
        bins_arr = np.asarray(bins, dtype=np.int32)
        child_idxs = None
        if feature_values is not None:
            child_idxs = idxs[np.isin(feature_values, bins_arr)]
        cond = _interval_label(feature_idx, bins, binner, feature_names)
        children.append(
            {
                "branch": {
                    "condition": cond,
                    "bins": [int(b) for b in bins],
                },
                "child": _serialize_msplit_node(
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
        "feature_name": _to_label_text(feature_name),
        "feature_display_name": feature_display,
        "n_samples": n_samples,
        "n_way": int(len(groups)),
        "children": children,
    }


def _serialize_xgb_tree(
    model: XGBClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    class_labels: np.ndarray,
) -> Dict[str, object]:
    tree_df = model.get_booster().trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == 0].copy()
    rows = {str(row["ID"]): row for _, row in tree_df.iterrows()}
    root_id = "0-0"

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
        feat_idx = _xgb_parse_feature_idx(str(row["Feature"]))
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
            pred_idx = 1 if leaf_score >= 0.0 else 0
            pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else pred_idx
            counts = leaf_class_counts.get(node_id, np.zeros(len(class_labels), dtype=np.int64))
            return {
                "node_type": "leaf",
                "id": node_id,
                "n_samples": int(node_sample_counts.get(node_id, 0)),
                "cover": float(row["Cover"]),
                "leaf_score": leaf_score,
                "predicted_class_index": int(pred_idx),
                "predicted_class_label": _to_label_text(pred_label),
                "true_class_dist": _class_dist_from_counts(counts.tolist(), class_labels),
                "path_conditions": path_conditions,
            }

        yes_id, no_id = children[node_id]
        feat_idx = _xgb_parse_feature_idx(str(row["Feature"]))
        feature_name = feature_names[feat_idx] if feat_idx < len(feature_names) else str(row["Feature"])
        feature_disp = _feature_display_name(feature_name)
        thr = float(row["Split"])
        yes_cond = f"<= {_format_float(thr)}"
        no_cond = f"> {_format_float(thr)}"
        return {
            "node_type": "internal",
            "id": node_id,
            "feature_index": int(feat_idx),
            "feature_name": _to_label_text(feature_name),
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
        "tree_index": 0,
        "root_id": root_id,
        "tree": _build_node(root_id, []),
    }


def _write_artifact(path: Path, payload: Dict[str, object]) -> None:
    write_artifact_json(path, payload)


def _slice_rows(x, idx: np.ndarray):
    if hasattr(x, "iloc"):
        return x.iloc[idx]
    return x[idx]


def _fit_msplit_tree(
    args: argparse.Namespace,
    x_train_proc: np.ndarray,
    x_test_proc: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
):
    if args.pipeline == "lightgbm":
        binner = fit_lightgbm_binner(
            x_train_proc,
            y_train,
            max_bins=int(args.max_bins),
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
            num_threads=1,
            device_type=args.lgb_device_type,
            gpu_platform_id=args.lgb_gpu_platform_id,
            gpu_device_id=args.lgb_gpu_device_id,
            gpu_fallback_to_cpu=args.lgb_gpu_fallback,
            ensemble_runs=max(1, int(args.lgb_ensemble_runs)),
            ensemble_feature_fraction=min(1.0, max(1e-6, float(args.lgb_ensemble_feature_fraction))),
            ensemble_bagging_fraction=min(1.0, max(1e-6, float(args.lgb_ensemble_bagging_fraction))),
            ensemble_bagging_freq=max(0, int(args.lgb_ensemble_bagging_freq)),
            threshold_dedup_eps=max(0.0, float(args.lgb_threshold_dedup_eps)),
        )
    z_train = binner.transform(x_train_proc)
    z_test = binner.transform(x_test_proc)
    lookahead = min(args.lookahead_cap, args.depth_budget - 1)

    model = MSPLIT(
        lookahead_depth_budget=lookahead,
        full_depth_budget=args.depth_budget,
        reg=args.reg,
        max_bins=args.max_bins,
        min_samples_leaf=args.min_samples_leaf,
        min_child_size=args.min_child_size,
        max_branching=args.max_branching,
        time_limit=args.time_limit,
        verbose=False,
        random_state=args.seed,
        use_cpp_solver=True,
    )
    model.fit(z_train, y_train)

    y_pred = model.predict(z_test).astype(np.int32)
    acc = float(np.mean(y_pred == y_test))
    return model, binner, acc, lookahead, z_train


def _pipeline_title(pipeline: str) -> str:
    if pipeline == "lightgbm":
        return "LightGBM + MultiwaySPLIT"
    if pipeline == "shapecart":
        return "ShapeCART"
    return "XGBoost"


def main() -> None:
    args = _parse_args()
    _apply_theme()

    if args.artifact_in:
        artifact_path = Path(args.artifact_in)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        pipeline = str(payload.get("pipeline", args.pipeline))
        dataset_name = str(payload.get("dataset", args.dataset))
        tree_artifact = payload.get("tree_artifact")
        if isinstance(tree_artifact, dict) and isinstance(tree_artifact.get("tree"), dict):
            root = tree_artifact["tree"]
        elif isinstance(tree_artifact, dict):
            root = tree_artifact
        else:
            raise ValueError(f"Invalid artifact format: missing tree_artifact in {artifact_path}")

        artifact_feature_names = payload.get("feature_names", [])
        if not isinstance(artifact_feature_names, list):
            artifact_feature_names = []

        artifact_binner = None
        binner_payload = payload.get("binner")
        if isinstance(binner_payload, dict):
            raw_edges = binner_payload.get("bin_edges_per_feature", [])
            if isinstance(raw_edges, list):
                parsed_edges: List[np.ndarray] = []
                for edges in raw_edges:
                    if isinstance(edges, list):
                        parsed_edges.append(np.asarray(edges, dtype=float))
                    else:
                        parsed_edges.append(np.asarray([], dtype=float))
                artifact_binner = SimpleNamespace(bin_edges_per_feature=parsed_edges, n_features=len(parsed_edges))

        if args.out:
            out_path = Path(args.out)
        else:
            out_dir = BENCHMARK_ARTIFACTS_ROOT / "tree_viz"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{dataset_name}_{pipeline}_artifact.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        depth = max(1, _artifact_depth(root, binner=artifact_binner, feature_names=artifact_feature_names))
        leaves = max(1, _artifact_count_leaves(root, binner=artifact_binner, feature_names=artifact_feature_names))
        fig_w, fig_h = _tree_figure_size(leaves, depth)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_axis_off()
        ax.set_facecolor("#f6f8fb")
        realized_depth, realized_leaves = _draw_serialized_tree(
            ax,
            root,
            binner=artifact_binner,
            feature_names=artifact_feature_names,
        )
        ax.set_title(f"{_pipeline_title(pipeline)} on {dataset_name}", fontsize=17, y=0.962)

        model_cfg = payload.get("model_config", {}) if isinstance(payload.get("model_config"), dict) else {}
        split_cfg = payload.get("split", {}) if isinstance(payload.get("split"), dict) else {}
        acc = payload.get("accuracy", np.nan)

        if pipeline == "xgboost":
            config_entries = [
                ("depth", str(model_cfg.get("depth_budget", args.depth_budget))),
                ("leaves", str(realized_leaves)),
                ("acc", f"{float(acc):.4f}" if not pd.isna(acc) else "nan"),
                ("n_estim", str(model_cfg.get("n_estimators", args.xgb_n_estimators))),
                ("lr", str(model_cfg.get("learning_rate", args.xgb_learning_rate))),
                ("seed", str(split_cfg.get("seed", args.seed))),
                ("test", str(split_cfg.get("test_size", args.test_size))),
            ]
        elif pipeline == "shapecart":
            config_entries = [
                ("depth", str(model_cfg.get("depth_budget", args.depth_budget))),
                ("k", str(model_cfg.get("k", "n/a"))),
                ("leaves", str(realized_leaves)),
                ("acc", f"{float(acc):.4f}" if not pd.isna(acc) else "nan"),
                ("min_leaf", str(model_cfg.get("min_samples_leaf", "n/a"))),
                ("min_split", str(model_cfg.get("min_samples_split", "n/a"))),
                ("inner_depth", str(model_cfg.get("inner_max_depth", "n/a"))),
                ("inner_leaves", str(model_cfg.get("inner_max_leaf_nodes", "n/a"))),
                ("max_iter", str(model_cfg.get("max_iter", "n/a"))),
                ("seed", str(split_cfg.get("seed", args.seed))),
                ("test", str(split_cfg.get("test_size", args.test_size))),
            ]
        else:
            config_entries = [
                ("depth", str(model_cfg.get("depth_budget", args.depth_budget))),
                ("la", str(model_cfg.get("lookahead", args.lookahead_cap))),
                ("leaves", str(realized_leaves)),
                ("acc", f"{float(acc):.4f}" if not pd.isna(acc) else "nan"),
                ("t_limit", str(model_cfg.get("time_limit", args.time_limit))),
                ("max_bins", str(model_cfg.get("max_bins", args.max_bins))),
                ("max_branch", str(model_cfg.get("max_branching", args.max_branching))),
                ("min_leaf", str(model_cfg.get("min_samples_leaf", args.min_samples_leaf))),
                ("min_child", str(model_cfg.get("min_child_size", args.min_child_size))),
                ("reg", str(model_cfg.get("reg", args.reg))),
                ("msplit_var", str(model_cfg.get("msplit_variant", "msplit"))),
                ("seed", str(split_cfg.get("seed", args.seed))),
                ("test", str(split_cfg.get("test_size", args.test_size))),
            ]
        _add_config_box(ax, config_entries)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.962])
        fig.savefig(out_path, dpi=_tree_figure_dpi(realized_leaves), bbox_inches="tight")
        plt.close(fig)

        if args.artifact_out:
            _write_artifact(Path(args.artifact_out), payload)
        print(f"saved {out_path}")
        if not pd.isna(acc):
            print(f"accuracy {float(acc):.10f}")
        print(f"realized_depth {int(realized_depth)}")
        return

    x, y = DATASET_LOADERS[args.dataset]()
    class_labels = np.array(sorted(np.unique(np.asarray(y)).tolist(), key=lambda v: str(v)), dtype=object)
    y_bin = encode_binary_target(y, args.dataset)
    y_bin = np.asarray(y_bin, dtype=np.int32)
    target_name = getattr(y, "name", None) or "target"

    all_idx = np.arange(len(y_bin), dtype=np.int32)
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_bin,
    )

    x_train = _slice_rows(x, idx_train)
    x_test = _slice_rows(x, idx_test)
    y_train = y_bin[idx_train]
    y_test = y_bin[idx_test]

    preprocessor = make_preprocessor(x_train)
    x_train_proc = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float32)
    x_test_proc = np.asarray(preprocessor.transform(x_test), dtype=np.float32)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"x{i}" for i in range(x_train_proc.shape[1])]

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = BENCHMARK_ARTIFACTS_ROOT / "tree_viz"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.dataset}_{args.pipeline}_d{args.depth_budget}_s{args.seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.pipeline == "xgboost":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=int(args.depth_budget),
            random_state=int(args.seed),
            n_jobs=int(args.xgb_num_threads),
        )
        model.fit(x_train_proc, y_train)
        y_pred = model.predict(x_test_proc).astype(np.int32)
        acc = float(np.mean(y_pred == y_test))

        depth = int(args.depth_budget)
        fig_w = max(13.0, 3.2 * (2 ** min(depth, 4)))
        fig_h = max(7.5, 2.2 * (depth + 2))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor("#f6f8fb")
        ax.set_axis_off()
        _draw_xgb_tree(ax, model, x_train_proc, y_train, feature_names, class_labels=class_labels)
        ax.set_title(f"XGBoost on {args.dataset}", fontsize=17, y=0.965)
        _add_config_box(
            ax,
            [
                ("depth", str(args.depth_budget)),
                ("acc", f"{acc:.4f}"),
                ("tree", "first"),
                ("n_estim", str(args.xgb_n_estimators)),
                ("lr", str(args.xgb_learning_rate)),
                ("seed", str(args.seed)),
                ("test", str(args.test_size)),
            ],
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        fig.savefig(out_path, dpi=230, bbox_inches="tight")
        plt.close(fig)
        if args.artifact_out:
            artifact = {
                "schema_version": 1,
                "dataset": args.dataset,
                "pipeline": "xgboost",
                "target_name": _to_label_text(target_name),
                "class_labels": [_to_label_text(v) for v in class_labels.tolist()],
                "feature_names": [_to_label_text(v) for v in feature_names],
                "accuracy": float(acc),
                "split": {
                    "seed": int(args.seed),
                    "test_size": float(args.test_size),
                    "train_indices": idx_train.astype(int).tolist(),
                    "test_indices": idx_test.astype(int).tolist(),
                },
                "model_config": {
                    "depth_budget": int(args.depth_budget),
                    "n_estimators": int(args.xgb_n_estimators),
                    "learning_rate": float(args.xgb_learning_rate),
                    "num_threads": int(args.xgb_num_threads),
                },
                "tree_artifact": _serialize_xgb_tree(
                    model,
                    x_train_proc,
                    y_train,
                    feature_names,
                    class_labels=class_labels,
                ),
            }
            _write_artifact(Path(args.artifact_out), artifact)
        print(f"saved {out_path}")
        print(f"accuracy {acc:.10f}")
        return

    model, binner, acc, lookahead, z_train = _fit_msplit_tree(args, x_train_proc, x_test_proc, y_train, y_test)
    leaves = _count_leaves(model.tree_)
    depth = _tree_depth(model.tree_)

    fig_w, fig_h = _tree_figure_size(leaves, depth)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_facecolor("#f6f8fb")
    pipeline_label = "LightGBM + MultiwaySPLIT"
    ax.set_title(f"{pipeline_label} on {args.dataset}", fontsize=17, y=0.965)
    _draw_msplit_tree(
        ax,
        model.tree_,
        binner,
        feature_names,
        class_labels,
        z_train=z_train,
    )
    _add_config_box(
        ax,
        [
            ("depth", str(args.depth_budget)),
            ("la", str(lookahead)),
            ("leaves", str(leaves)),
            ("acc", f"{acc:.4f}"),
            ("t_limit", str(args.time_limit)),
            ("max_bins", str(args.max_bins)),
            ("max_branch", str(args.max_branching)),
            ("min_leaf", str(args.min_samples_leaf)),
            ("min_child", str(args.min_child_size)),
            ("reg", str(args.reg)),
            ("msplit_var", "msplit"),
            ("seed", str(args.seed)),
            ("test", str(args.test_size)),
        ],
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_path, dpi=_tree_figure_dpi(leaves), bbox_inches="tight")
    plt.close(fig)
    if args.artifact_out:
        root_idxs = np.arange(z_train.shape[0], dtype=np.int32)
        bin_edges = []
        for edges in binner.bin_edges_per_feature:
            if edges is None:
                bin_edges.append(None)
            else:
                bin_edges.append([float(v) for v in np.asarray(edges).tolist()])
        artifact = {
            "schema_version": 1,
            "dataset": args.dataset,
            "pipeline": args.pipeline,
            "target_name": _to_label_text(target_name),
            "class_labels": [_to_label_text(v) for v in class_labels.tolist()],
            "feature_names": [_to_label_text(v) for v in feature_names],
            "accuracy": float(acc),
            "split": {
                "seed": int(args.seed),
                "test_size": float(args.test_size),
                "train_indices": idx_train.astype(int).tolist(),
                "test_indices": idx_test.astype(int).tolist(),
            },
            "model_config": {
                "depth_budget": int(args.depth_budget),
                "lookahead": int(lookahead),
                "time_limit": float(args.time_limit),
                "max_bins": int(args.max_bins),
                "lgb_ensemble_runs": int(max(1, int(args.lgb_ensemble_runs))),
                "min_samples_leaf": int(args.min_samples_leaf),
                "min_child_size": int(args.min_child_size),
                "max_branching": int(args.max_branching),
                "reg": float(args.reg),
                "msplit_variant": "msplit",
            },
            "binner": {
                "n_features": int(len(feature_names)),
                "bin_edges_per_feature": bin_edges,
            },
            "tree_artifact": _serialize_msplit_node(
                model.tree_,
                binner,
                feature_names,
                class_labels,
                z_train=z_train,
                idxs=root_idxs,
                path_conditions=[],
            ),
        }
        _write_artifact(Path(args.artifact_out), artifact)
    print(f"saved {out_path}")
    print(f"accuracy {acc:.10f}")


if __name__ == "__main__":
    main()
