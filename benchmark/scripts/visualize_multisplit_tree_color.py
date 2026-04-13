"""Visualize one fitted tree with color-coded bin-to-branch assignments."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts import visualize_multisplit_tree_n as base

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def _raw_artifact_children(node: Dict[str, Any]) -> List[Tuple[str, List[int], Dict[str, Any]]]:
    out: List[Tuple[str, List[int], Dict[str, Any]]] = []
    children = node.get("children")
    if not isinstance(children, list):
        return out

    for entry in children:
        if not isinstance(entry, dict):
            continue
        child = entry.get("child")
        if not isinstance(child, dict):
            continue
        branch = entry.get("branch") if isinstance(entry.get("branch"), dict) else {}
        cond = str(branch.get("condition", ""))
        bins_raw = branch.get("bins", [])
        bins = [int(v) for v in bins_raw] if isinstance(bins_raw, list) else []
        out.append((cond, bins, child))
    return out


def _artifact_subtree_signature(node: Dict[str, Any]) -> Tuple[Any, ...]:
    if base._artifact_is_leaf(node):
        pred_idx = int(node.get("predicted_class_index", -1))
        true_dist = node.get("true_class_dist", [])
        counts: List[Tuple[int, int]] = []
        if isinstance(true_dist, list):
            for entry in true_dist:
                if not isinstance(entry, dict):
                    continue
                counts.append((int(entry.get("class_index", len(counts))), int(entry.get("count", 0))))
        return ("leaf", pred_idx, tuple(counts))

    kids = _raw_artifact_children(node)
    child_sigs = tuple(_artifact_subtree_signature(child) for _, _, child in kids)
    return ("internal", int(node.get("feature_index", -1)), int(node.get("n_samples", 0)), child_sigs)


def _merged_condition(
    feature_idx: int,
    bins: List[int],
    fallback_cond: str,
    cond_count: int,
    binner: Optional[object],
    feature_names: Optional[List[str]],
) -> str:
    if cond_count <= 1:
        return fallback_cond
    if (
        binner is not None
        and hasattr(binner, "bin_edges_per_feature")
        and feature_idx >= 0
        and feature_idx < len(getattr(binner, "bin_edges_per_feature", []))
    ):
        return base._format_msplit_condition(feature_idx, bins, binner, feature_names or [])
    if not bins:
        return fallback_cond
    lo = min(bins)
    hi = max(bins)
    return f"bin {lo}" if lo == hi else f"bins {lo}-{hi}"


def _artifact_color_children(
    node: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    raw = _raw_artifact_children(node)
    if not raw:
        return []

    feature_idx = int(node.get("feature_index", -1))
    out: List[Dict[str, Any]] = []
    cur_cond, cur_bins, cur_child = raw[0]
    cur_sig = _artifact_subtree_signature(cur_child)
    cur_cond_count = 1
    for cond, bins, child in raw[1:]:
        sig = _artifact_subtree_signature(child)
        if sig == cur_sig:
            cur_bins.extend(bins)
            cur_cond_count += 1
            continue
        out.append(
            {
                "condition": _merged_condition(
                    feature_idx,
                    cur_bins,
                    cur_cond,
                    cur_cond_count,
                    binner=binner,
                    feature_names=feature_names,
                ),
                "bins": sorted(set(int(v) for v in cur_bins)),
                "child": cur_child,
            }
        )
        cur_cond = cond
        cur_bins = list(bins)
        cur_child = child
        cur_sig = sig
        cur_cond_count = 1
    out.append(
        {
            "condition": _merged_condition(
                feature_idx,
                cur_bins,
                cur_cond,
                cur_cond_count,
                binner=binner,
                feature_names=feature_names,
            ),
            "bins": sorted(set(int(v) for v in cur_bins)),
            "child": cur_child,
        }
    )
    return out


def _artifact_count_leaves(
    node: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> int:
    kids = _artifact_color_children(node, binner=binner, feature_names=feature_names)
    if not kids:
        return 1
    return sum(_artifact_count_leaves(entry["child"], binner=binner, feature_names=feature_names) for entry in kids)


def _artifact_depth(
    node: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> int:
    kids = _artifact_color_children(node, binner=binner, feature_names=feature_names)
    if not kids:
        return 0
    return 1 + max(_artifact_depth(entry["child"], binner=binner, feature_names=feature_names) for entry in kids)


def _artifact_assign_positions(
    node: Dict[str, Any],
    depth: int,
    x_cursor: List[float],
    positions: Dict[int, Tuple[float, float]],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
) -> None:
    node_key = id(node)
    kids = _artifact_color_children(node, binner=binner, feature_names=feature_names)
    if not kids:
        x = x_cursor[0]
        x_cursor[0] += 1.0
        positions[node_key] = (x, float(depth))
        return

    child_x: List[float] = []
    for entry in kids:
        child = entry["child"]
        _artifact_assign_positions(child, depth + 1, x_cursor, positions, binner=binner, feature_names=feature_names)
        child_x.append(positions[id(child)][0])
    positions[node_key] = (float(np.mean(child_x)), float(depth))


def _branch_palette(branch_count: int) -> List[str]:
    base_colors = [
        "#2563eb",
        "#ea580c",
        "#059669",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
    ]
    if branch_count <= len(base_colors):
        return base_colors[:branch_count]
    return [base_colors[i % len(base_colors)] for i in range(branch_count)]


def _paper_tree_dpi(leaf_count: int) -> int:
    leaf_count = max(1, int(leaf_count))
    if leaf_count >= 250:
        return 220
    if leaf_count >= 150:
        return 240
    if leaf_count >= 80:
        return 280
    return 320


def _bin_assignment_segments(children: List[Dict[str, Any]]) -> Optional[Tuple[List[int], List[Tuple[int, int, int]]]]:
    observed_bins: List[Tuple[int, int]] = []
    for branch_i, entry in enumerate(children):
        for bin_id in entry.get("bins", []):
            observed_bins.append((int(bin_id), branch_i))

    if not observed_bins:
        return None

    observed_bins.sort(key=lambda item: item[0])
    ordered_bins = [bin_id for bin_id, _ in observed_bins]

    segments: List[Tuple[int, int, int]] = []
    cur_start_idx = 0
    cur_branch = observed_bins[0][1]
    for idx, (_, branch_i) in enumerate(observed_bins[1:], start=1):
        if branch_i == cur_branch:
            continue
        segments.append((cur_start_idx, idx - 1, cur_branch))
        cur_start_idx = idx
        cur_branch = branch_i
    segments.append((cur_start_idx, len(observed_bins) - 1, cur_branch))
    return ordered_bins, segments


def _draw_node_fill(
    ax,
    center_x: float,
    center_y: float,
    box_w: float,
    box_h: float,
    children: List[Dict[str, Any]],
    palette: List[str],
    boxstyle: str,
) -> Optional[Tuple[int, int]]:
    spec = _bin_assignment_segments(children)
    if spec is None:
        return None

    ordered_bins, segments = spec
    total_bins = max(1, len(ordered_bins))
    box_left = center_x - box_w / 2.0
    box_bottom = center_y - box_h / 2.0

    clip_box = FancyBboxPatch(
        (box_left, box_bottom),
        box_w,
        box_h,
        boxstyle=boxstyle,
        linewidth=0.0,
        edgecolor="none",
        facecolor="none",
        zorder=1.5,
    )
    ax.add_patch(clip_box)

    for start_idx, end_idx, branch_idx in segments:
        frac_left = start_idx / total_bins
        frac_right = (end_idx + 1) / total_bins
        face = palette[int(branch_idx) % len(palette)]
        rect = Rectangle(
            (box_left + box_w * frac_left, box_bottom),
            box_w * max(0.002, frac_right - frac_left),
            box_h,
            linewidth=0.0,
            edgecolor="none",
            facecolor=face,
            alpha=0.95,
            zorder=1.6,
        )
        rect.set_clip_path(clip_box)
        ax.add_patch(rect)

    return ordered_bins[0], ordered_bins[-1]


def _draw_serialized_tree_color(
    ax,
    root: Dict[str, Any],
    binner: Optional[object] = None,
    feature_names: Optional[List[str]] = None,
    n_classes: Optional[int] = None,
    target_name: Optional[str] = None,
) -> Tuple[int, int]:
    positions: Dict[int, Tuple[float, float]] = {}
    _artifact_assign_positions(root, depth=0, x_cursor=[0.0], positions=positions, binner=binner, feature_names=feature_names)

    n_leaves = max(1.0, float(_artifact_count_leaves(root, binner=binner, feature_names=feature_names)))
    depth = max(1, _artifact_depth(root, binner=binner, feature_names=feature_names))
    leaf_slot = 0.92 / n_leaves

    def _to_plot_coords(x: float, d: float) -> Tuple[float, float]:
        nx = (x + 1.0) / (n_leaves + 1.0)
        ny = 1.0 - ((d + 0.55) / (depth + 1.3))
        return nx, ny

    def _compact_internal_label(node: Dict[str, Any]) -> str:
        feat = node.get("feature_display_name", node.get("feature_name", "x"))
        n_samples = node.get("n_samples", 0)
        return f"{feat} (n = {int(n_samples):,})"

    def _compact_leaf_label(node: Dict[str, Any]) -> str:
        n_samples = int(node.get("n_samples", 0))
        label = str(
            node.get(
                "predicted_class_label",
                node.get("predicted_class_index", target_name or "target"),
            )
        )
        return f"{label} (n = {n_samples:,})"

    def _leaf_face_and_text(node: Dict[str, Any]) -> Tuple[str, str]:
        pred_idx = int(node.get("predicted_class_index", 0))
        class_count = max(1, int(n_classes or 2))
        if class_count <= 2:
            if pred_idx > 0:
                return "#ffffff", "#0f172a"
            return "#1f2937", "#f8fafc"

        frac = 0.0 if class_count <= 1 else min(1.0, max(0.0, pred_idx / max(1, class_count - 1)))
        level = int(round(248 - frac * 208))
        face = f"#{level:02x}{level:02x}{level:02x}"
        text = "#0f172a" if level >= 150 else "#ffffff"
        return face, text

    def _draw_node(node: Dict[str, Any]) -> None:
        node_key = id(node)
        x, d = positions[node_key]
        nx, ny = _to_plot_coords(x, d)
        kids = _artifact_color_children(node, binner=binner, feature_names=feature_names)
        is_leaf = len(kids) == 0

        if is_leaf:
            color, text_color = _leaf_face_and_text(node)
            style = "square,pad=0.0"
            label = _compact_leaf_label(node)
        else:
            color = "#d6e4f1"
            text_color = "#0f172a"
            style = "square,pad=0.0"
            label = _compact_internal_label(node)

        wrap_width = 26 if not is_leaf else 30
        label = "\n".join(base._wrap_line(line, width=wrap_width) for line in label.split("\n"))
        w, h = base._box_dims(label, is_leaf=is_leaf)
        if not is_leaf:
            h = base._box_dims(label, is_leaf=True)[1]
        subtree_leaves = max(1, _artifact_count_leaves(node, binner=binner, feature_names=feature_names))
        w = min(w, leaf_slot * 0.78 * subtree_leaves)

        if is_leaf:
            box = FancyBboxPatch(
                (nx - w / 2.0, ny - h / 2.0),
                w,
                h,
                boxstyle=style,
                linewidth=0.55,
                edgecolor="#64748b" if color == "#ffffff" else "#475569",
                facecolor=color,
                zorder=2,
            )
            ax.add_patch(box)
            node_font = 12.0
            ax.text(
                nx,
                ny,
                label,
                ha="center",
                va="center",
                fontsize=node_font,
                fontweight="bold",
                color=text_color,
                zorder=3,
                path_effects=[pe.withStroke(linewidth=1.1, foreground=("#ffffff" if text_color == "#0f172a" else "#0f172a"))],
            )
            return

        palette = _branch_palette(len(kids))
        bin_bounds = _draw_node_fill(ax, nx, ny, w, h, kids, palette, style)
        box = FancyBboxPatch(
            (nx - w / 2.0, ny - h / 2.0),
            w,
            h,
            boxstyle=style,
            linewidth=0.7,
            edgecolor="#2f3640",
            facecolor="none",
            zorder=2.4,
        )
        ax.add_patch(box)
        node_font = 12.0
        ax.text(
            nx,
            ny,
            label,
            ha="center",
            va="center",
            fontsize=node_font,
            fontweight="bold",
            color=text_color,
            zorder=3,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="#ffffff")],
        )

        branch_count = len(kids)
        for branch_i, entry in enumerate(kids):
            child = entry["child"]
            edge_cond = str(entry["condition"])
            edge_color = palette[branch_i % len(palette)] if bin_bounds is not None else "#2f3640"
            child_kids = _artifact_color_children(child, binner=binner, feature_names=feature_names)
            child_is_leaf = len(child_kids) == 0
            child_label = _compact_leaf_label(child) if child_is_leaf else _compact_internal_label(child)
            child_wrap_width = 26 if not child_is_leaf else 30
            child_label = "\n".join(base._wrap_line(line, width=child_wrap_width) for line in child_label.split("\n"))
            _, child_h = base._box_dims(child_label, is_leaf=child_is_leaf)
            if not child_is_leaf:
                child_h = base._box_dims(child_label, is_leaf=True)[1]
            cx, cd = positions[id(child)]
            cnx, cny = _to_plot_coords(cx, cd)
            edge_rad = 0.045 * (branch_i - (branch_count - 1) / 2.0) if branch_count > 1 else 0.0

            arr = FancyArrowPatch(
                (nx, ny - h / 2.0),
                (cnx, cny + child_h / 2.0),
                arrowstyle="Simple,tail_width=1.12,head_width=9.4,head_length=10.4",
                connectionstyle=f"arc3,rad={edge_rad}",
                mutation_scale=1.0,
                linewidth=0.0,
                shrinkA=1.5,
                shrinkB=7.0,
                color=edge_color,
                alpha=0.98 if bin_bounds is not None else 0.58,
                zorder=1,
                capstyle="round",
                joinstyle="round",
            )
            ax.add_patch(arr)

            if bin_bounds is None:
                edge_text = base._wrap_line(edge_cond, width=22)
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


def _save_color_figure(fig, out_path: Path, leaf_count: int) -> Optional[Path]:
    suffix = out_path.suffix.lower()
    raster_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    facecolor = fig.get_facecolor()
    if suffix in raster_suffixes:
        fig.savefig(out_path, dpi=_paper_tree_dpi(leaf_count), bbox_inches="tight", facecolor=facecolor)
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=facecolor)
        return pdf_path
    fig.savefig(out_path, bbox_inches="tight", facecolor=facecolor)
    return None


def _paper_title(ax, title: str, subtitle: Optional[str] = None) -> None:
    ax.text(
        0.5,
        0.992,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=20.5,
        fontweight="bold",
        color="#0f172a",
        zorder=10,
    )
    if subtitle:
        ax.text(
            0.5,
            0.972,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.2,
            color="#475569",
            zorder=10,
        )


def _msplit_title(dataset_name: str) -> str:
    dataset_text = base._escape_mathtext(str(dataset_name))
    return rf"MSPLIT on $\mathtt{{{dataset_text}}}$"


def _format_fit_seconds(value: object) -> str:
    try:
        seconds = float(value)
    except Exception:
        return str(value)
    return f"{seconds:.3f}s"


def _msplit_config_entries(
    *,
    depth: object,
    lookahead: object,
    leaves: object,
    test_accuracy: object,
    max_bins: object,
    max_branching: object,
    min_leaf_size: object,
    min_child_size: object,
    regularization: object,
    seed: object,
    test_size: object,
    fit_seconds: object | None = None,
    time_limit: object | None = None,
) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = [
        ("depth", str(depth)),
        ("lookahead depth", str(lookahead)),
        ("max bins", str(max_bins)),
        ("max branching", str(max_branching)),
        ("min leaf size", str(min_leaf_size)),
        ("min child size", str(min_child_size)),
        ("regularization", str(regularization)),
        ("seed", str(seed)),
    ]
    if fit_seconds is not None:
        entries.extend(
            [
                ("fit time", _format_fit_seconds(fit_seconds)),
                ("test size", str(test_size)),
            ]
        )
    elif time_limit is not None:
        entries.extend(
            [
                ("time limit", str(time_limit)),
                ("test size", str(test_size)),
            ]
        )
    else:
        entries.append(("test size", str(test_size)))

    entries.extend(
        [
            ("leaves", str(leaves)),
            ("test accuracy", str(test_accuracy)),
        ]
    )
    return entries


def _add_compact_config_card(ax, entries: List[Tuple[str, str]]) -> None:
    items = [(str(key), str(value)) for key, value in entries]
    if not items:
        return

    rows: List[List[Tuple[str, str]]] = [items[i : i + 2] for i in range(0, len(items), 2)]
    n_rows = len(rows)
    row_gap = 0.0017
    pad_x = 0.004
    pad_y = 0.003
    pair_gap = 0.003
    col_gap = 0.010

    pair_widths_by_col: List[List[float]] = [[], []]
    key_widths_by_col: List[List[float]] = [[], []]
    value_widths_by_col: List[List[float]] = [[], []]
    for row in rows:
        for col in range(2):
            if col >= len(row):
                continue
            key, value = row[col]
            key_w = min(0.105, max(0.040, 0.0026 * len(key) + 0.007))
            val_w = min(0.034, max(0.015, 0.0024 * len(value) + 0.002))
            pair_widths_by_col[col].append(key_w + pair_gap + val_w)
            key_widths_by_col[col].append(key_w)
            value_widths_by_col[col].append(val_w)

    col_widths = [max(widths, default=0.0) for widths in pair_widths_by_col]
    col_key_widths = [max(widths, default=0.0) for widths in key_widths_by_col]
    col_value_widths = [max(widths, default=0.0) for widths in value_widths_by_col]
    row_h = 0.0122
    width = 2 * pad_x + sum(col_widths) + (col_gap if col_widths[1] > 0 else 0.0)
    height = 2 * pad_y + n_rows * row_h + max(0, n_rows - 1) * row_gap

    x_right = 0.992
    y_top = 0.992
    x_left = x_right - width
    y_bottom = y_top - height

    card = FancyBboxPatch(
        (x_left, y_bottom),
        width,
        height,
        transform=ax.transAxes,
        boxstyle="square,pad=0.0",
        linewidth=0.65,
        edgecolor="#cbd5e1",
        facecolor="#ffffff",
        alpha=0.94,
        zorder=9,
    )
    ax.add_patch(card)

    second_col_left = x_left + pad_x + col_widths[0] + (col_gap if col_widths[1] > 0 else 0.0)
    for row_idx, row in enumerate(rows):
        cell_top = y_top - pad_y - row_idx * (row_h + row_gap)
        for col_idx, (key, value) in enumerate(row):
            cell_left = x_left + pad_x if col_idx == 0 else second_col_left
            key_w = col_key_widths[col_idx]
            pair_w = col_widths[col_idx]
            value_w = col_value_widths[col_idx]
            ax.text(
                cell_left,
                cell_top,
                key.upper(),
                transform=ax.transAxes,
                ha="left",
                va="top",
            fontsize=9.4,
            fontweight="bold",
            color="#64748b",
            zorder=10,
        )
            ax.text(
                cell_left + pair_w,
                cell_top,
                value,
                transform=ax.transAxes,
                ha="right",
                va="top",
            fontsize=11.6,
            fontweight="bold",
            color="#0f172a",
            zorder=10,
        )


def _default_out_path(dataset_name: str, pipeline: str, *, artifact_mode: bool, depth_budget: int, seed: int) -> Path:
    out_dir = base.BENCHMARK_ARTIFACTS_ROOT / "tree_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    if artifact_mode:
        return out_dir / f"{dataset_name}_{pipeline}_artifact_color.png"
    return out_dir / f"{dataset_name}_{pipeline}_d{depth_budget}_s{seed}_color.png"


def _artifact_binner_from_payload(payload: Dict[str, Any]) -> Optional[object]:
    binner_payload = payload.get("binner")
    if not isinstance(binner_payload, dict):
        return None
    raw_edges = binner_payload.get("bin_edges_per_feature", [])
    if not isinstance(raw_edges, list):
        return None

    parsed_edges: List[np.ndarray] = []
    for edges in raw_edges:
        if isinstance(edges, list):
            parsed_edges.append(np.asarray(edges, dtype=float))
        else:
            parsed_edges.append(np.asarray([], dtype=float))
    return SimpleNamespace(bin_edges_per_feature=parsed_edges, n_features=len(parsed_edges))


def main() -> None:
    args = base._parse_args()
    base._apply_theme()

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
        artifact_binner = _artifact_binner_from_payload(payload)

        out_path = Path(args.out) if args.out else _default_out_path(
            dataset_name,
            pipeline,
            artifact_mode=True,
            depth_budget=args.depth_budget,
            seed=args.seed,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        depth = max(1, _artifact_depth(root, binner=artifact_binner, feature_names=artifact_feature_names))
        leaves = max(1, _artifact_count_leaves(root, binner=artifact_binner, feature_names=artifact_feature_names))
        fig_w, fig_h = base._tree_figure_size(leaves, depth)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#ffffff")
        ax.set_axis_off()
        ax.set_facecolor("#ffffff")
        realized_depth, realized_leaves = _draw_serialized_tree_color(
            ax,
            root,
            binner=artifact_binner,
            feature_names=artifact_feature_names,
            n_classes=len(payload.get("class_labels", [])) if isinstance(payload.get("class_labels"), list) else None,
            target_name=str(payload.get("target_name", "target")),
        )
        _paper_title(ax, _msplit_title(dataset_name))

        model_cfg = payload.get("model_config", {}) if isinstance(payload.get("model_config"), dict) else {}
        split_cfg = payload.get("split", {}) if isinstance(payload.get("split"), dict) else {}
        acc = payload.get("accuracy", np.nan)

        if pipeline == "xgboost":
            config_entries = [
                ("depth", str(model_cfg.get("depth_budget", args.depth_budget))),
                ("leaves", str(realized_leaves)),
                ("test accuracy", f"{float(acc):.4f}" if not pd.isna(acc) else "nan"),
                ("trees", str(model_cfg.get("n_estimators", args.xgb_n_estimators))),
                ("learning rate", str(model_cfg.get("learning_rate", args.xgb_learning_rate))),
                ("seed", str(split_cfg.get("seed", args.seed))),
                ("test size", str(split_cfg.get("test_size", args.test_size))),
            ]
        else:
            fit_seconds = model_cfg.get("fit_seconds")
            config_entries = _msplit_config_entries(
                depth=model_cfg.get("depth_budget", args.depth_budget),
                lookahead=model_cfg.get("lookahead", args.lookahead_cap),
                leaves=realized_leaves,
                test_accuracy=(f"{float(acc):.4f}" if not pd.isna(acc) else "nan"),
                fit_seconds=fit_seconds,
                time_limit=model_cfg.get("time_limit", args.time_limit),
                max_bins=model_cfg.get("max_bins", args.max_bins),
                max_branching=model_cfg.get("max_branching", args.max_branching),
                min_leaf_size=model_cfg.get("min_samples_leaf", args.min_samples_leaf),
                min_child_size=model_cfg.get("min_child_size", args.min_child_size),
                regularization=model_cfg.get("reg", args.reg),
                seed=split_cfg.get("seed", args.seed),
                test_size=split_cfg.get("test_size", args.test_size),
            )
        _add_compact_config_card(ax, config_entries)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.988])
        pdf_path = _save_color_figure(fig, out_path, realized_leaves)
        plt.close(fig)

        if args.artifact_out:
            base._write_artifact(Path(args.artifact_out), payload)
        print(f"saved {out_path}")
        if pdf_path is not None:
            print(f"saved {pdf_path}")
        if not pd.isna(acc):
            print(f"accuracy {float(acc):.10f}")
        print(f"realized_depth {int(realized_depth)}")
        return

    x, y = base.DATASET_LOADERS[args.dataset]()
    class_labels = np.array(sorted(np.unique(np.asarray(y)).tolist(), key=lambda v: str(v)), dtype=object)
    y_bin = base.encode_binary_target(y, args.dataset)
    y_bin = np.asarray(y_bin, dtype=np.int32)
    target_name = getattr(y, "name", None) or "target"

    all_idx = np.arange(len(y_bin), dtype=np.int32)
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_bin,
    )

    x_train = base._slice_rows(x, idx_train)
    x_test = base._slice_rows(x, idx_test)
    y_train = y_bin[idx_train]
    y_test = y_bin[idx_test]

    preprocessor = base.make_preprocessor(x_train)
    x_train_proc = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float32)
    x_test_proc = np.asarray(preprocessor.transform(x_test), dtype=np.float32)
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"x{i}" for i in range(x_train_proc.shape[1])]

    out_path = Path(args.out) if args.out else _default_out_path(
        args.dataset,
        args.pipeline,
        artifact_mode=False,
        depth_budget=args.depth_budget,
        seed=args.seed,
    )
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
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_axis_off()
        base._draw_xgb_tree(ax, model, x_train_proc, y_train, feature_names, class_labels=class_labels)
        _paper_title(ax, _msplit_title(args.dataset))
        _add_compact_config_card(
            ax,
            [
                ("depth", str(args.depth_budget)),
                ("test accuracy", f"{acc:.4f}"),
                ("tree", "first"),
                ("trees", str(args.xgb_n_estimators)),
                ("learning rate", str(args.xgb_learning_rate)),
                ("seed", str(args.seed)),
                ("test size", str(args.test_size)),
            ],
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.988])
        pdf_path = _save_color_figure(fig, out_path, 2 ** max(1, depth))
        plt.close(fig)
        if args.artifact_out:
            artifact = {
                "schema_version": 1,
                "dataset": args.dataset,
                "pipeline": "xgboost",
                "target_name": base._to_label_text(target_name),
                "class_labels": [base._to_label_text(v) for v in class_labels.tolist()],
                "feature_names": [base._to_label_text(v) for v in feature_names],
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
                "tree_artifact": base._serialize_xgb_tree(
                    model,
                    x_train_proc,
                    y_train,
                    feature_names,
                    class_labels=class_labels,
                ),
            }
            base._write_artifact(Path(args.artifact_out), artifact)
        print(f"saved {out_path}")
        if pdf_path is not None:
            print(f"saved {pdf_path}")
        print(f"accuracy {acc:.10f}")
        return

    model, binner, acc, lookahead, z_train = base._fit_msplit_tree(args, x_train_proc, x_test_proc, y_train, y_test)
    root_idxs = np.arange(z_train.shape[0], dtype=np.int32)
    root = base._serialize_msplit_node(
        model.tree_,
        binner,
        feature_names,
        class_labels,
        z_train=z_train,
        idxs=root_idxs,
        path_conditions=[],
    )

    leaves = _artifact_count_leaves(root, binner=binner, feature_names=feature_names)
    depth = _artifact_depth(root, binner=binner, feature_names=feature_names)
    fig_w, fig_h = base._tree_figure_size(leaves, depth)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_axis_off()
    ax.set_facecolor("#ffffff")
    _paper_title(ax, _msplit_title(args.dataset))
    _draw_serialized_tree_color(
        ax,
        root,
        binner=binner,
        feature_names=feature_names,
        n_classes=len(class_labels),
        target_name=str(target_name),
    )
    _add_compact_config_card(
        ax,
        _msplit_config_entries(
            depth=args.depth_budget,
            lookahead=lookahead,
            leaves=leaves,
            test_accuracy=f"{acc:.4f}",
            time_limit=args.time_limit,
            max_bins=args.max_bins,
            max_branching=args.max_branching,
            min_leaf_size=args.min_samples_leaf,
            min_child_size=args.min_child_size,
            regularization=args.reg,
            seed=args.seed,
            test_size=args.test_size,
        ),
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.988])
    pdf_path = _save_color_figure(fig, out_path, leaves)
    plt.close(fig)

    if args.artifact_out:
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
            "target_name": base._to_label_text(target_name),
            "class_labels": [base._to_label_text(v) for v in class_labels.tolist()],
            "feature_names": [base._to_label_text(v) for v in feature_names],
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
            "tree_artifact": root,
        }
        base._write_artifact(Path(args.artifact_out), artifact)
    print(f"saved {out_path}")
    if pdf_path is not None:
        print(f"saved {pdf_path}")
    print(f"accuracy {acc:.10f}")


if __name__ == "__main__":
    main()
