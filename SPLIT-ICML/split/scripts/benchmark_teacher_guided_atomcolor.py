#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from split import MSPLIT


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_rows: int
    n_features: int
    n_bins: int
    positive_bins: tuple[int, ...]


@dataclass(frozen=True)
class ShapeLeaf:
    prediction: int


@dataclass(frozen=True)
class ShapeNode:
    feature: int
    left_bins: frozenset[int]
    left_child: object
    right_child: object


def make_teacher_logits(signal: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    logits = 2.75 * signal + rng.normal(scale=noise, size=signal.shape[0])
    return logits.astype(np.float64)


def build_islands_dataset(spec: DatasetSpec, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.integers(0, spec.n_bins, size=(spec.n_rows, spec.n_features), dtype=np.int32)
    positive = np.isin(X[:, 0], np.asarray(spec.positive_bins, dtype=np.int32))
    y = positive.astype(np.int32)
    teacher_signal = 2.0 * positive.astype(np.float64) - 1.0
    teacher_logit = make_teacher_logits(teacher_signal, rng, noise=0.10)
    return X, y, teacher_logit


def count_tree_nodes(tree) -> tuple[int, int]:
    if tree.__class__.__name__ == "MultiLeaf":
        return 1, 1
    children = getattr(tree, "children", None)
    if isinstance(children, dict):
        children = list(children.values())
    if children is None:
        children = getattr(tree, "group_nodes", None) or []
    internal = 1
    leaves = 0
    for child in children:
        child_internal, child_leaves = count_tree_nodes(child)
        internal += child_internal
        leaves += child_leaves
    return internal, leaves


def is_noncontiguous_group_spans(child_spans: object) -> bool:
    if not isinstance(child_spans, dict):
        return False
    for spans in child_spans.values():
        if len(spans) > 1:
            return True
    return False


def majority_loss(y: np.ndarray) -> tuple[float, int]:
    if y.size == 0:
        return 0.0, 0
    prediction = int(np.mean(y) >= 0.5)
    loss = float(np.sum(y != prediction)) / float(y.size)
    return loss, prediction


def best_shapecart_split(
    X: np.ndarray,
    y: np.ndarray,
    min_child_size: int,
) -> tuple[float, tuple[float, int, frozenset[int], np.ndarray] | None]:
    n_rows, n_features = X.shape
    leaf_loss, _ = majority_loss(y)
    best: tuple[float, int, frozenset[int], np.ndarray] | None = None
    for feature in range(n_features):
        bins = sorted(int(v) for v in np.unique(X[:, feature]))
        if len(bins) < 2:
            continue
        for subset_size in range(1, len(bins)):
            for subset_tuple in combinations(bins, subset_size):
                # Canonicalize complements so we do not evaluate both A and ~A.
                if bins[0] not in subset_tuple:
                    continue
                subset = frozenset(subset_tuple)
                mask = np.isin(X[:, feature], tuple(subset))
                left_count = int(np.sum(mask))
                right_count = n_rows - left_count
                if left_count < min_child_size or right_count < min_child_size:
                    continue
                left_loss, _ = majority_loss(y[mask])
                right_loss, _ = majority_loss(y[~mask])
                split_loss = (left_count / n_rows) * left_loss + (right_count / n_rows) * right_loss
                if best is None or split_loss < best[0] - 1e-12:
                    best = (split_loss, feature, subset, mask)
    return leaf_loss, best


def train_shapecart_greedy(
    X: np.ndarray,
    y: np.ndarray,
    depth_budget: int,
    min_child_size: int,
) -> tuple[object, float]:
    leaf_loss, prediction = majority_loss(y)
    if depth_budget <= 1 or y.size == 0:
        return ShapeLeaf(prediction), leaf_loss
    _, best = best_shapecart_split(X, y, min_child_size)
    if best is None or best[0] >= leaf_loss - 1e-12:
        return ShapeLeaf(prediction), leaf_loss
    _, feature, subset, mask = best
    left_child, _ = train_shapecart_greedy(X[mask], y[mask], depth_budget - 1, min_child_size)
    right_child, _ = train_shapecart_greedy(X[~mask], y[~mask], depth_budget - 1, min_child_size)
    return ShapeNode(feature, subset, left_child, right_child), best[0]


def predict_shapecart_one(x_row: np.ndarray, tree: object) -> int:
    node = tree
    while isinstance(node, ShapeNode):
        node = node.left_child if int(x_row[node.feature]) in node.left_bins else node.right_child
    return node.prediction


def count_shapecart_nodes(tree: object) -> tuple[int, int]:
    if isinstance(tree, ShapeLeaf):
        return 1, 1
    left_internal, left_leaves = count_shapecart_nodes(tree.left_child)
    right_internal, right_leaves = count_shapecart_nodes(tree.right_child)
    return 1 + left_internal + right_internal, left_leaves + right_leaves


def make_msplit_model(method: str) -> MSPLIT:
    common = dict(
        full_depth_budget=2,
        lookahead_depth_budget=1,
        reg=0.0,
        min_child_size=5,
        max_branching=3,
        use_cpp_solver=True,
        time_limit=60,
    )
    if method == "teacher_guided_atomcolor":
        return MSPLIT(
            **common,
            approx_distilled_mode=True,
            approx_distilled_alpha=0.8,
            approx_distilled_max_depth=1,
            approx_distilled_geometry_mode="teacher_guided_atomcolor",
        )
    if method == "hard_label_approx":
        return MSPLIT(**common)
    raise ValueError(f"unknown MSPLIT method: {method}")


def fit_once(method: str, X: np.ndarray, y: np.ndarray, teacher_logit: np.ndarray) -> dict[str, float | int]:
    started = time.perf_counter()
    if method == "shapecart_greedy":
        tree, objective = train_shapecart_greedy(X, y, depth_budget=2, min_child_size=5)
        pred = np.fromiter((predict_shapecart_one(row, tree) for row in X), dtype=np.int32, count=X.shape[0])
        fit_seconds = time.perf_counter() - started
        internal_nodes, leaves = count_shapecart_nodes(tree)
        root_noncontiguous = 0
        if isinstance(tree, ShapeNode):
            root_bins = sorted(tree.left_bins)
            root_noncontiguous = int(
                any(root_bins[idx] + 1 < root_bins[idx + 1] for idx in range(len(root_bins) - 1))
            )
        return {
            "fit_seconds": fit_seconds,
            "objective": float(objective),
            "train_accuracy": float(np.mean(pred == y)),
            "internal_nodes": int(internal_nodes),
            "leaves": int(leaves),
            "root_noncontiguous": int(root_noncontiguous),
            "shape_nodes": int(root_noncontiguous),
        }

    model = make_msplit_model(method)
    fit_kwargs = {}
    if method == "teacher_guided_atomcolor":
        fit_kwargs["teacher_logit"] = teacher_logit
    model.fit(X, y, **fit_kwargs)
    fit_seconds = time.perf_counter() - started
    pred = model.predict(X)
    internal_nodes, leaves = count_tree_nodes(model.tree_)
    child_spans = getattr(model.tree_, "child_spans", None)
    return {
        "fit_seconds": fit_seconds,
        "objective": float(model.objective_),
        "train_accuracy": float(np.mean(pred == y)),
        "internal_nodes": int(internal_nodes),
        "leaves": int(leaves),
        "root_noncontiguous": int(is_noncontiguous_group_spans(child_spans)),
        "shape_nodes": int(getattr(model, "approx_distilled_shape_nodes_", 0)),
    }


def summarize_records(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    fit_seconds = [float(r["fit_seconds"]) for r in records]
    objective = [float(r["objective"]) for r in records]
    train_accuracy = [float(r["train_accuracy"]) for r in records]
    internal_nodes = [int(r["internal_nodes"]) for r in records]
    leaves = [int(r["leaves"]) for r in records]
    root_noncontiguous = [int(r["root_noncontiguous"]) for r in records]
    shape_nodes = [int(r["shape_nodes"]) for r in records]
    return {
        "fit_seconds_median": statistics.median(fit_seconds),
        "objective_median": statistics.median(objective),
        "train_accuracy_mean": statistics.fmean(train_accuracy),
        "internal_nodes_median": int(round(statistics.median(internal_nodes))),
        "leaves_median": int(round(statistics.median(leaves))),
        "root_noncontiguous_rate": statistics.fmean(root_noncontiguous),
        "shape_nodes_mean": statistics.fmean(shape_nodes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark teacher-guided atomcolor against contiguous hard-label MSPLIT "
            "and a local ShapeCART-style greedy baseline on root-local island tasks."
        )
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write per-run results as JSON.",
    )
    args = parser.parse_args()

    dataset_specs = [
        DatasetSpec("three_islands_a", n_rows=1200, n_features=6, n_bins=12, positive_bins=(1, 2, 5, 8, 9)),
        DatasetSpec("three_islands_b", n_rows=1200, n_features=6, n_bins=12, positive_bins=(0, 3, 4, 7, 10)),
        DatasetSpec("four_islands", n_rows=1200, n_features=6, n_bins=12, positive_bins=(1, 4, 7, 10)),
    ]
    methods = ["teacher_guided_atomcolor", "hard_label_approx", "shapecart_greedy"]

    full_results: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in dataset_specs:
        for method in methods:
            records: list[dict[str, float | int]] = []
            for repeat in range(args.repeats):
                X, y, teacher_logit = build_islands_dataset(spec, seed=1000 + repeat)
                record = fit_once(method, X, y, teacher_logit)
                records.append(record)
                full_results.append(
                    {
                        "dataset": spec.name,
                        "method": method,
                        "repeat": repeat,
                        **record,
                    }
                )
            summary_rows.append(
                {
                    "dataset": spec.name,
                    "method": method,
                    **summarize_records(records),
                }
            )

    headers = [
        "dataset",
        "method",
        "fit_seconds_median",
        "objective_median",
        "train_accuracy_mean",
        "root_noncontiguous_rate",
        "shape_nodes_mean",
        "internal_nodes_median",
        "leaves_median",
    ]
    print("\t".join(headers))
    for row in summary_rows:
        print(
            "\t".join(
                [
                    str(row["dataset"]),
                    str(row["method"]),
                    f'{float(row["fit_seconds_median"]):.4f}',
                    f'{float(row["objective_median"]):.6f}',
                    f'{float(row["train_accuracy_mean"]):.4f}',
                    f'{float(row["root_noncontiguous_rate"]):.2f}',
                    f'{float(row["shape_nodes_mean"]):.2f}',
                    str(row["internal_nodes_median"]),
                    str(row["leaves_median"]),
                ]
            )
        )

    if args.json is not None:
        args.json.write_text(json.dumps(full_results, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
