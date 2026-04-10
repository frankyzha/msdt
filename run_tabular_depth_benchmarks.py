"""Run XGBoost, ShapeCART, and atomized MSPLIT benchmarks over tabular datasets."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msdt_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_DATASETS,
    DEFAULT_OPENML_CACHE,
    get_dataset_spec,
    load_dataset,
    materialize_all_datasets,
)
from experiment_utils import (
    canonical_dataset_list,
    encode_target,
    feature_sparsity,
    make_preprocessor,
    resolve_feature_names,
    stratified_train_test_indices,
)
from lightgbm_binning import fit_lightgbm_binner
from tree_artifact_utils import (
    build_msplit_artifact,
    build_shapecart_artifact,
    build_xgb_artifact,
    write_artifact_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SHAPECART_ROOT = PROJECT_ROOT / "Empowering-DTs-via-Shape-Functions"
SPLIT_SRC = PROJECT_ROOT / "SPLIT-ICML" / "split" / "src"
if str(SHAPECART_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPECART_ROOT))
if str(SPLIT_SRC) not in sys.path:
    sys.path.insert(0, str(SPLIT_SRC))

from src.ShapeCARTClassifier import ShapeCARTClassifier  # type: ignore
from split import MSPLIT

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - runtime dependency
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


MODEL_ORDER = ("xgboost", "shapecart", "msplit_atomized")
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "shapecart": "ShapeCART",
    "msplit_atomized": "MSPLIT atomized",
}
MODEL_COLORS = {
    "xgboost": "#1f7a8c",
    "shapecart": "#e07a1f",
    "msplit_atomized": "#c1121f",
}
MODEL_MARKERS = {
    "xgboost": "o",
    "shapecart": "s",
    "msplit_atomized": "^",
}


@dataclass
class PreparedSplit:
    seed: int
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    X_train_raw: Any
    X_val_raw: Any
    X_test_raw: Any
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    X_train_proc: np.ndarray
    X_val_proc: np.ndarray
    X_test_proc: np.ndarray
    feature_names: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate accuracy-vs-depth artifacts under datasets/<name>/ for 8 tabular datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=sorted(DEFAULT_DATASETS))
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--openml-cache-dir", type=str, default=str(DEFAULT_OPENML_CACHE))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--no-render-trees", action="store_true")
    parser.add_argument(
        "--graphs-only",
        action="store_true",
        help="Skip artifact/split/tree outputs and only save the accuracy-vs-depth test graph plus summary CSVs.",
    )

    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--xgb-num-threads", type=int, default=4)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=30)

    parser.add_argument("--shapecart-k", type=int, default=3)
    parser.add_argument("--shapecart-min-samples-leaf", type=int, default=4)
    parser.add_argument("--shapecart-min-samples-split", type=int, default=8)
    parser.add_argument("--shapecart-inner-max-depth", type=int, default=6)
    parser.add_argument("--shapecart-inner-max-leaf-nodes", type=int, default=24)
    parser.add_argument("--shapecart-max-iter", type=int, default=10)

    parser.add_argument("--msplit-reg", type=float, default=0.0)
    parser.add_argument("--msplit-leaf-frac", type=float, default=0.001)
    parser.add_argument("--msplit-max-branching", type=int, default=0)
    parser.add_argument("--msplit-time-limit", type=int, default=10)
    return parser.parse_args()


def _slice_rows(x, idx: np.ndarray):
    if hasattr(x, "iloc"):
        return x.iloc[idx]
    return x[idx]


def _json_default(value: object):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _maybe_write_json(enabled: bool, path: Path, payload: dict[str, object]) -> None:
    if enabled:
        _write_json(path, payload)


def _append_log(path: Path, message: str) -> None:
    stamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(stamped, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(stamped + "\n")


def _count_serialized_tree(node: dict[str, object]) -> tuple[int, int]:
    if str(node.get("node_type", "")) == "leaf":
        return 0, 1
    children = node.get("children", [])
    internal = 1
    leaves = 0
    if not isinstance(children, list):
        return 0, 1
    for entry in children:
        if not isinstance(entry, dict) or not isinstance(entry.get("child"), dict):
            continue
        child_internal, child_leaves = _count_serialized_tree(entry["child"])
        internal += child_internal
        leaves += child_leaves
    return internal, leaves


def _artifact_tree_stats(artifact: dict[str, object]) -> tuple[int, int]:
    tree_artifact = artifact.get("tree_artifact", {})
    if isinstance(tree_artifact, dict) and isinstance(tree_artifact.get("tree"), dict):
        return _count_serialized_tree(tree_artifact["tree"])
    if isinstance(tree_artifact, dict):
        return _count_serialized_tree(tree_artifact)
    return 0, 0


def _optional_int_attr(obj: object, name: str) -> int | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _optional_float_attr(obj: object, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _default_msplit_lookahead(depth: int) -> int:
    return max(1, (int(depth) + 1) // 2)


def _xgb_tree_stats(model, total_feature_count: int) -> dict[str, object]:
    tree_df = model.get_booster().trees_to_dataframe()
    internal_mask = tree_df["Feature"] != "Leaf"
    leaves_mask = ~internal_mask
    used_feature_ids = set()
    for token in tree_df.loc[internal_mask, "Feature"].tolist():
        text = str(token)
        if text.startswith("f"):
            text = text[1:]
        try:
            used_feature_ids.add(int(text))
        except Exception:
            continue
    used_count = len(used_feature_ids)
    return {
        "n_internal_nodes": int(internal_mask.sum()),
        "n_leaves": int(leaves_mask.sum()),
        "used_feature_count": int(used_count),
        "used_feature_indices": sorted(int(v) for v in used_feature_ids),
        "sparsity": float(feature_sparsity(used_count, total_feature_count)),
    }


def _flatten_feature_key(key: object, feature_dict: dict[object, list[int]]) -> list[int]:
    if key in feature_dict:
        return [int(v) for v in feature_dict[key]]
    if isinstance(key, tuple):
        out: list[int] = []
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


def _shapecart_tree_stats(model, total_feature_count: int) -> dict[str, object]:
    internal = 0
    leaves = 0
    inner_internal = 0
    inner_leaves = 0
    used_features: set[int] = set()
    stack = [0]
    while stack:
        node_idx = int(stack.pop())
        if bool(model.is_leaf[node_idx]) or model.children[node_idx] is None:
            leaves += 1
            continue
        internal += 1
        node = model.nodes[node_idx]
        feature_key = getattr(node, "final_key", None)
        used_features.update(_flatten_feature_key(feature_key, getattr(node, "feature_dict", {}) or {}))
        inner_tree = getattr(node, "final_tree", None)
        if inner_tree is not None and hasattr(inner_tree, "tree_") and hasattr(inner_tree.tree_, "children_left"):
            child_left = np.asarray(inner_tree.tree_.children_left)
            leaf_mask = child_left == -1
            inner_leaves += int(np.sum(leaf_mask))
            inner_internal += int(child_left.size - np.sum(leaf_mask))
        stack.extend(int(v) for v in (model.children[node_idx] or []))

    used_count = len(used_features)
    return {
        "n_internal_nodes": int(internal),
        "n_leaves": int(leaves),
        "inner_tree_internal_nodes": int(inner_internal),
        "inner_tree_leaves": int(inner_leaves),
        "used_feature_count": int(used_count),
        "used_feature_indices": sorted(int(v) for v in used_features),
        "sparsity": float(feature_sparsity(used_count, total_feature_count)),
    }


def _msplit_tree_stats(model, total_feature_count: int) -> dict[str, object]:
    internal = 0
    leaves = 0
    used_features: set[int] = set()
    stack = [model.tree_]
    while stack:
        node = stack.pop()
        if hasattr(node, "children"):
            internal += 1
            used_features.add(int(getattr(node, "feature", -1)))
            for child in node.children.values():
                stack.append(child)
        else:
            leaves += 1
    used_features = {v for v in used_features if v >= 0}
    used_count = len(used_features)
    return {
        "n_internal_nodes": int(internal),
        "n_leaves": int(leaves),
        "used_feature_count": int(used_count),
        "used_feature_indices": sorted(int(v) for v in used_features),
        "sparsity": float(feature_sparsity(used_count, total_feature_count)),
    }


def _prepare_split_cache(
    X,
    y_encoded: np.ndarray,
    seeds: list[int],
    train_size: float,
    val_size: float,
    test_size: float,
    max_train_rows: int,
) -> dict[int, PreparedSplit]:
    cache: dict[int, PreparedSplit] = {}
    total = float(train_size) + float(val_size) + float(test_size)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"train/val/test sizes must sum to 1.0; received {total:.6f}")
    for seed in seeds:
        all_idx = np.arange(y_encoded.shape[0], dtype=np.int32)
        holdout_size = float(val_size) + float(test_size)
        idx_train, idx_holdout = train_test_split(
            all_idx,
            train_size=float(train_size),
            random_state=int(seed),
            stratify=y_encoded,
        )
        idx_train = np.asarray(idx_train, dtype=np.int32)
        idx_holdout = np.asarray(idx_holdout, dtype=np.int32)
        y_holdout = y_encoded[idx_holdout]
        val_fraction_within_holdout = float(val_size) / holdout_size
        idx_val, idx_test = train_test_split(
            idx_holdout,
            train_size=val_fraction_within_holdout,
            random_state=int(seed) + 1,
            stratify=y_holdout,
        )
        idx_val = np.asarray(idx_val, dtype=np.int32)
        idx_test = np.asarray(idx_test, dtype=np.int32)
        if int(max_train_rows) > 0 and idx_train.size > int(max_train_rows):
            y_train_full = y_encoded[idx_train]
            idx_train, _ = train_test_split(
                idx_train,
                train_size=int(max_train_rows),
                random_state=int(seed),
                stratify=y_train_full,
            )
            idx_train = np.asarray(idx_train, dtype=np.int32)
        X_val_raw = _slice_rows(X, idx_val)
        X_test_raw = _slice_rows(X, idx_test)
        X_train_raw = _slice_rows(X, idx_train)
        y_train = y_encoded[idx_train]
        y_val = y_encoded[idx_val]
        y_test = y_encoded[idx_test]
        preprocessor = make_preprocessor(X_train_raw)
        X_train_proc = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=np.float32)
        X_val_proc = np.asarray(preprocessor.transform(X_val_raw), dtype=np.float32)
        X_test_proc = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float32)
        feature_names = resolve_feature_names(preprocessor, X_train_proc.shape[1])
        cache[int(seed)] = PreparedSplit(
            seed=int(seed),
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            X_train_raw=X_train_raw,
            X_val_raw=X_val_raw,
            X_test_raw=X_test_raw,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            X_train_proc=X_train_proc,
            X_val_proc=X_val_proc,
            X_test_proc=X_test_proc,
            feature_names=feature_names,
        )
    return cache


def _fit_xgboost(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    depth: int,
    split: PreparedSplit,
    class_labels: np.ndarray,
    target_name: str,
    artifact_dir: Path,
) -> dict[str, object]:
    if XGBClassifier is None:
        raise RuntimeError(f"XGBoost is unavailable: {XGBOOST_IMPORT_ERROR}")

    train_classes = np.unique(split.y_train).astype(np.int32, copy=False)
    if train_classes.size == 1:
        majority_class = int(train_classes[0])
        pred_train = np.full(split.y_train.shape[0], majority_class, dtype=np.int32)
        pred_val = np.full(split.y_val.shape[0], majority_class, dtype=np.int32)
        pred_test = np.full(split.y_test.shape[0], majority_class, dtype=np.int32)
        train_accuracy = float(np.mean(pred_train == split.y_train))
        val_accuracy = float(np.mean(pred_val == split.y_val))
        test_accuracy = float(np.mean(pred_test == split.y_test))
        balanced_accuracy = float(balanced_accuracy_score(split.y_test, pred_test))
        return {
            "dataset": dataset_name,
            "model": "xgboost",
            "depth_budget": int(depth),
            "seed": int(split.seed),
            "fit_time_sec": 0.0,
            "train_accuracy": float(train_accuracy),
            "val_accuracy": float(val_accuracy),
            "test_accuracy": float(test_accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "artifact_path": "",
            "n_internal_nodes": 0,
            "n_leaves": 1,
            "used_feature_count": 0,
            "used_feature_indices": "[]",
            "sparsity": 1.0,
            "artifact_internal_nodes": 0,
            "artifact_leaves": 1,
            "status": "single_class_fallback",
        }

    train_class_map = {int(label): idx for idx, label in enumerate(train_classes.tolist())}
    y_train_fit = np.asarray([train_class_map[int(v)] for v in split.y_train], dtype=np.int32)
    n_classes = int(train_classes.size)
    params: dict[str, object] = {
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "tree_method": "hist",
        "n_estimators": int(args.xgb_n_estimators),
        "learning_rate": float(args.xgb_learning_rate),
        "max_depth": int(depth),
        "random_state": int(split.seed),
        "n_jobs": int(args.xgb_num_threads),
    }
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = int(n_classes)
    else:
        params["objective"] = "binary:logistic"

    fit_kwargs: dict[str, object] = {}
    if split.X_val_proc.shape[0] > 0:
        y_val_fit = np.asarray([train_class_map.get(int(v), -1) for v in split.y_val], dtype=np.int32)
        valid_mask = y_val_fit >= 0
        if np.any(valid_mask):
            params["early_stopping_rounds"] = int(args.xgb_early_stopping_rounds)
            fit_kwargs["eval_set"] = [(split.X_val_proc[valid_mask], y_val_fit[valid_mask])]
            fit_kwargs["verbose"] = False
    model = XGBClassifier(**params)
    t0 = time.perf_counter()
    model.fit(split.X_train_proc, y_train_fit, **fit_kwargs)
    fit_time_sec = time.perf_counter() - t0

    pred_train_fit = np.asarray(model.predict(split.X_train_proc), dtype=np.int32)
    pred_val_fit = np.asarray(model.predict(split.X_val_proc), dtype=np.int32)
    pred_test_fit = np.asarray(model.predict(split.X_test_proc), dtype=np.int32)
    pred_train = train_classes[pred_train_fit]
    pred_val = train_classes[pred_val_fit]
    pred_test = train_classes[pred_test_fit]
    train_accuracy = float(np.mean(pred_train == split.y_train))
    val_accuracy = float(np.mean(pred_val == split.y_val))
    test_accuracy = float(np.mean(pred_test == split.y_test))
    balanced_accuracy = float(balanced_accuracy_score(split.y_test, pred_test))

    stats = _xgb_tree_stats(model, total_feature_count=len(split.feature_names))
    artifact_path = ""
    stats["artifact_internal_nodes"] = int(stats["n_internal_nodes"])
    stats["artifact_leaves"] = int(stats["n_leaves"])
    if not args.graphs_only:
        artifact = build_xgb_artifact(
            dataset=dataset_name,
            target_name=target_name,
            class_labels=class_labels,
            feature_names=split.feature_names,
            accuracy=test_accuracy,
            seed=int(split.seed),
            test_size=float(args.test_size),
            depth_budget=int(depth),
            n_estimators=int(args.xgb_n_estimators),
            learning_rate=float(args.xgb_learning_rate),
            num_threads=int(args.xgb_num_threads),
            model=model,
            x_train=split.X_train_proc,
            y_train=split.y_train,
            train_indices=split.idx_train,
            test_indices=split.idx_test,
            tree_index=0,
        )
        artifact_path = str(artifact_dir / f"depth_{int(depth)}" / f"seed_{int(split.seed)}.json")
        write_artifact_json(Path(artifact_path), artifact)
        stats["artifact_internal_nodes"], stats["artifact_leaves"] = _artifact_tree_stats(artifact)

    return {
        "dataset": dataset_name,
        "model": "xgboost",
        "depth_budget": int(depth),
        "seed": int(split.seed),
        "fit_time_sec": float(fit_time_sec),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "artifact_path": str(artifact_path),
        **stats,
    }


def _fit_shapecart(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    depth: int,
    split: PreparedSplit,
    class_labels: np.ndarray,
    target_name: str,
    artifact_dir: Path,
) -> dict[str, object]:
    model = ShapeCARTClassifier(
        max_depth=int(depth),
        min_samples_leaf=int(args.shapecart_min_samples_leaf),
        min_samples_split=int(args.shapecart_min_samples_split),
        inner_min_samples_leaf=int(args.shapecart_min_samples_leaf),
        inner_min_samples_split=int(args.shapecart_min_samples_split),
        inner_max_depth=int(args.shapecart_inner_max_depth),
        inner_max_leaf_nodes=int(args.shapecart_inner_max_leaf_nodes),
        max_iter=int(args.shapecart_max_iter),
        k=int(args.shapecart_k),
        branching_penalty=0.0,
        random_state=int(split.seed),
        verbose=False,
    )
    t0 = time.perf_counter()
    model.fit(split.X_train_proc, split.y_train)
    fit_time_sec = time.perf_counter() - t0

    pred_train = np.asarray(model.predict(split.X_train_proc), dtype=np.int32)
    pred_val = np.asarray(model.predict(split.X_val_proc), dtype=np.int32)
    pred_test = np.asarray(model.predict(split.X_test_proc), dtype=np.int32)
    train_accuracy = float(np.mean(pred_train == split.y_train))
    val_accuracy = float(np.mean(pred_val == split.y_val))
    test_accuracy = float(np.mean(pred_test == split.y_test))
    balanced_accuracy = float(balanced_accuracy_score(split.y_test, pred_test))

    stats = _shapecart_tree_stats(model, total_feature_count=len(split.feature_names))
    artifact_path = ""
    stats["artifact_internal_nodes"] = int(stats["n_internal_nodes"])
    stats["artifact_leaves"] = int(stats["n_leaves"])
    if not args.graphs_only:
        artifact = build_shapecart_artifact(
            dataset=dataset_name,
            target_name=target_name,
            class_labels=class_labels,
            feature_names=split.feature_names,
            accuracy=test_accuracy,
            seed=int(split.seed),
            test_size=float(args.test_size),
            depth_budget=int(depth),
            k=int(args.shapecart_k),
            min_samples_leaf=int(args.shapecart_min_samples_leaf),
            min_samples_split=int(args.shapecart_min_samples_split),
            inner_min_samples_leaf=int(args.shapecart_min_samples_leaf),
            inner_min_samples_split=int(args.shapecart_min_samples_split),
            inner_max_depth=int(args.shapecart_inner_max_depth),
            inner_max_leaf_nodes=int(args.shapecart_inner_max_leaf_nodes),
            max_iter=int(args.shapecart_max_iter),
            model=model,
            train_indices=split.idx_train,
            test_indices=split.idx_test,
        )
        artifact_path = str(artifact_dir / f"depth_{int(depth)}" / f"seed_{int(split.seed)}.json")
        write_artifact_json(Path(artifact_path), artifact)
        stats["artifact_internal_nodes"], stats["artifact_leaves"] = _artifact_tree_stats(artifact)

    return {
        "dataset": dataset_name,
        "model": "shapecart",
        "depth_budget": int(depth),
        "seed": int(split.seed),
        "fit_time_sec": float(fit_time_sec),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "artifact_path": str(artifact_path),
        **stats,
    }


def _fit_msplit_model(
    *,
    X_train_proc: np.ndarray,
    X_val_proc: np.ndarray,
    X_test_proc: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    depth: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[MSPLIT, Any, np.ndarray, np.ndarray, np.ndarray, int]:
    derived_min_child = max(2, int(args.shapecart_min_samples_leaf))
    derived_min_split = max(2, int(args.shapecart_min_samples_split))
    binner = fit_lightgbm_binner(
        X_train_proc,
        y_train,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=1024,
        min_samples_leaf=int(derived_min_child),
        min_data_in_leaf=int(derived_min_child),
        lambda_l2=0.0,
        random_state=int(seed),
        collect_teacher_logit=True,
    )
    Z_train = np.asarray(binner.transform(X_train_proc), dtype=np.int32)
    Z_val = np.asarray(binner.transform(X_val_proc), dtype=np.int32)
    Z_test = np.asarray(binner.transform(X_test_proc), dtype=np.int32)
    resolved_lookahead_depth = _default_msplit_lookahead(int(depth))
    model = MSPLIT(
        full_depth_budget=int(depth),
        lookahead_depth=resolved_lookahead_depth,
        reg=float(args.msplit_reg),
        min_split_size=int(derived_min_split),
        min_child_size=int(derived_min_child),
        max_branching=int(args.shapecart_k),
        time_limit=float(args.msplit_time_limit),
        verbose=False,
        random_state=int(seed),
        use_cpp_solver=True,
    )
    model.fit(
        Z_train,
        y_train,
        teacher_logit=getattr(binner, "teacher_train_logit", None),
        teacher_boundary_gain=getattr(binner, "boundary_gain_per_feature", None),
        teacher_boundary_cover=getattr(binner, "boundary_cover_per_feature", None),
        teacher_boundary_value_jump=getattr(binner, "boundary_value_jump_per_feature", None),
    )
    return model, binner, Z_train, Z_val, Z_test, int(derived_min_child)


def _fit_msplit(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    depth: int,
    split: PreparedSplit,
    class_labels: np.ndarray,
    target_name: str,
    artifact_dir: Path,
) -> dict[str, object]:
    n_classes = len(class_labels)
    total_feature_count = len(split.feature_names)
    X_train_subset = split.X_train_proc
    y_train_subset = split.y_train
    train_indices_subset = split.idx_train
    t0 = time.perf_counter()
    per_class_artifacts: list[str] = []
    model, binner, Z_train, Z_val, Z_test, derived_min_child = _fit_msplit_model(
        X_train_proc=X_train_subset,
        X_val_proc=split.X_val_proc,
        X_test_proc=split.X_test_proc,
        y_train=y_train_subset,
        y_val=split.y_val,
        depth=depth,
        seed=split.seed,
        args=args,
    )
    pred_train = np.asarray(model.predict(Z_train), dtype=np.int32)
    pred_val = np.asarray(model.predict(Z_val), dtype=np.int32)
    pred_test = np.asarray(model.predict(Z_test), dtype=np.int32)

    stats = _msplit_tree_stats(model, total_feature_count=total_feature_count)
    stats["artifact_internal_nodes"] = int(stats["n_internal_nodes"])
    stats["artifact_leaves"] = int(stats["n_leaves"])
    stats["debr_source_group_row_size_histogram"] = list(
        getattr(model, "debr_source_group_row_size_histogram_", [])
    )
    stats["debr_source_component_atom_size_histogram"] = list(
        getattr(model, "debr_source_component_atom_size_histogram_", [])
    )
    stats["debr_source_component_row_size_histogram"] = list(
        getattr(model, "debr_source_component_row_size_histogram_", [])
    )
    stats["profiling_greedy_complete_calls_by_depth"] = list(
        getattr(model, "profiling_greedy_complete_calls_by_depth_", [])
    )
    stats["exact_dp_subproblem_calls_above_lookahead"] = _optional_int_attr(
        model, "exact_dp_subproblem_calls_above_lookahead_"
    )
    if not args.graphs_only:
        artifact = build_msplit_artifact(
            dataset=dataset_name,
            pipeline="msplit_atomized",
            target_name=target_name,
            class_labels=class_labels,
            feature_names=split.feature_names,
            accuracy=float(np.mean(pred_test == split.y_test)),
            seed=int(split.seed),
            test_size=float(args.test_size),
            depth_budget=int(depth),
            lookahead=int(getattr(model, "effective_lookahead_depth_", _default_msplit_lookahead(depth))),
            time_limit=float(args.msplit_time_limit),
            max_bins=int(getattr(binner, "max_bins", 1024)),
            min_samples_leaf=int(derived_min_child),
            min_child_size=int(derived_min_child),
            max_branching=int(args.shapecart_k),
            reg=float(args.msplit_reg),
            msplit_variant="atomized_cpp_native" if n_classes > 2 else "atomized_cpp",
            tree_root=model.tree_,
            binner=binner,
            z_train=np.asarray(Z_train, dtype=np.int32),
            train_indices=train_indices_subset,
            test_indices=split.idx_test,
        )
        artifact_path = artifact_dir / f"depth_{int(depth)}" / f"seed_{int(split.seed)}.json"
        write_artifact_json(artifact_path, artifact)
        per_class_artifacts.append(str(artifact_path))
        stats["artifact_internal_nodes"], stats["artifact_leaves"] = _artifact_tree_stats(artifact)

    fit_time_sec = time.perf_counter() - t0
    train_accuracy = float(np.mean(pred_train == split.y_train))
    val_accuracy = float(np.mean(pred_val == split.y_val))
    test_accuracy = float(np.mean(pred_test == split.y_test))
    balanced_accuracy = float(balanced_accuracy_score(split.y_test, pred_test))

    ensemble_path = ""
    if not args.graphs_only:
        ensemble_path = str(artifact_dir / f"depth_{int(depth)}" / f"seed_{int(split.seed)}_ensemble.json")
        _write_json(
            Path(ensemble_path),
            {
                "schema_version": 1,
                "dataset": dataset_name,
                "pipeline": "msplit_atomized",
                "mode": "native_multiclass" if n_classes > 2 else "binary",
                "seed": int(split.seed),
                "depth_budget": int(depth),
                "val_accuracy": float(val_accuracy),
                "test_accuracy": float(test_accuracy),
                "train_accuracy": float(train_accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "fit_time_sec": float(fit_time_sec),
                "artifact_paths": per_class_artifacts,
                "n_classes": int(n_classes),
                "class_labels": [str(v) for v in class_labels.tolist()],
                "stats": stats,
                "train_rows_used": int(split.X_train_proc.shape[0]),
            },
        )

    return {
        "dataset": dataset_name,
        "model": "msplit_atomized",
        "depth_budget": int(depth),
        "seed": int(split.seed),
        "fit_time_sec": float(fit_time_sec),
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "artifact_path": str(ensemble_path),
        "primary_tree_artifact_path": str(per_class_artifacts[0]) if per_class_artifacts else "",
        "submodel_artifact_paths": json.dumps(per_class_artifacts),
        "n_internal_nodes": int(stats["n_internal_nodes"]),
        "n_leaves": int(stats["n_leaves"]),
        "artifact_internal_nodes": int(stats["artifact_internal_nodes"]),
        "artifact_leaves": int(stats["artifact_leaves"]),
        "used_feature_count": int(stats["used_feature_count"]),
        "used_feature_indices": json.dumps(stats["used_feature_indices"]),
        "sparsity": float(stats["sparsity"]),
        "n_submodels": 1,
        "greedy_subproblem_calls": _optional_int_attr(model, "greedy_subproblem_calls_"),
        "exact_dp_subproblem_calls_above_lookahead": _optional_int_attr(
            model, "exact_dp_subproblem_calls_above_lookahead_"
        ),
        "greedy_unique_states": _optional_int_attr(model, "greedy_unique_states_"),
        "greedy_cache_hits": _optional_int_attr(model, "greedy_cache_hits_"),
        "greedy_cache_entries_peak": _optional_int_attr(model, "greedy_cache_entries_peak_"),
        "greedy_cache_clears": _optional_int_attr(model, "greedy_cache_clears_"),
        "family_compare_total": _optional_int_attr(model, "family_compare_total_"),
        "family_compare_equivalent": _optional_int_attr(model, "family_compare_equivalent_"),
        "family1_both_wins": _optional_int_attr(model, "family1_both_wins_"),
        "family2_hard_loss_wins": _optional_int_attr(model, "family2_hard_loss_wins_"),
        "family2_hard_impurity_wins": _optional_int_attr(model, "family2_hard_impurity_wins_"),
        "family2_both_wins": _optional_int_attr(model, "family2_both_wins_"),
        "family_metric_disagreement": _optional_int_attr(model, "family_metric_disagreement_"),
        "family_hard_loss_ties": _optional_int_attr(model, "family_hard_loss_ties_"),
        "family_hard_impurity_ties": _optional_int_attr(model, "family_hard_impurity_ties_"),
        "family_joint_impurity_ties": _optional_int_attr(model, "family_joint_impurity_ties_"),
        "family_neither_both_wins": _optional_int_attr(model, "family_neither_both_wins_"),
        "family1_selected_by_equivalence": _optional_int_attr(model, "family1_selected_by_equivalence_"),
        "family1_selected_by_dominance": _optional_int_attr(model, "family1_selected_by_dominance_"),
        "family2_selected_by_dominance": _optional_int_attr(model, "family2_selected_by_dominance_"),
        "family_sent_both": _optional_int_attr(model, "family_sent_both_"),
        "family1_hard_loss_sum": _optional_float_attr(model, "family1_hard_loss_sum_"),
        "family2_hard_loss_sum": _optional_float_attr(model, "family2_hard_loss_sum_"),
        "family_hard_loss_delta_sum": _optional_float_attr(model, "family_hard_loss_delta_sum_"),
        "family1_hard_impurity_sum": _optional_float_attr(model, "family1_hard_impurity_sum_"),
        "family2_hard_impurity_sum": _optional_float_attr(model, "family2_hard_impurity_sum_"),
        "family_hard_impurity_delta_sum": _optional_float_attr(model, "family_hard_impurity_delta_sum_"),
        "family1_soft_impurity_sum": _optional_float_attr(model, "family1_soft_impurity_sum_"),
        "family2_soft_impurity_sum": _optional_float_attr(model, "family2_soft_impurity_sum_"),
        "family_soft_impurity_delta_sum": _optional_float_attr(model, "family_soft_impurity_delta_sum_"),
        "family1_joint_impurity_sum": _optional_float_attr(model, "family1_joint_impurity_sum_"),
        "family2_joint_impurity_sum": _optional_float_attr(model, "family2_joint_impurity_sum_"),
        "family_joint_impurity_delta_sum": _optional_float_attr(model, "family_joint_impurity_delta_sum_"),
        "debr_refine_calls": _optional_int_attr(model, "debr_refine_calls_"),
        "debr_refine_improved": _optional_int_attr(model, "debr_refine_improved_"),
        "debr_total_moves": _optional_int_attr(model, "debr_total_moves_"),
        "debr_bridge_policy_calls": _optional_int_attr(model, "debr_bridge_policy_calls_"),
        "debr_refine_windowed_calls": _optional_int_attr(model, "debr_refine_windowed_calls_"),
        "debr_refine_unwindowed_calls": _optional_int_attr(model, "debr_refine_unwindowed_calls_"),
        "debr_refine_overlap_segments": _optional_int_attr(model, "debr_refine_overlap_segments_"),
        "debr_refine_calls_with_overlap": _optional_int_attr(model, "debr_refine_calls_with_overlap_"),
        "debr_refine_calls_without_overlap": _optional_int_attr(model, "debr_refine_calls_without_overlap_"),
        "debr_candidate_total": _optional_int_attr(model, "debr_candidate_total_"),
        "debr_candidate_legal": _optional_int_attr(model, "debr_candidate_legal_"),
        "debr_candidate_source_size_rejects": _optional_int_attr(model, "debr_candidate_source_size_rejects_"),
        "debr_candidate_target_size_rejects": _optional_int_attr(model, "debr_candidate_target_size_rejects_"),
        "debr_candidate_descent_eligible": _optional_int_attr(model, "debr_candidate_descent_eligible_"),
        "debr_candidate_descent_rejected": _optional_int_attr(model, "debr_candidate_descent_rejected_"),
        "debr_candidate_bridge_eligible": _optional_int_attr(model, "debr_candidate_bridge_eligible_"),
        "debr_candidate_bridge_window_blocked": _optional_int_attr(model, "debr_candidate_bridge_window_blocked_"),
        "debr_candidate_bridge_used_blocked": _optional_int_attr(model, "debr_candidate_bridge_used_blocked_"),
        "debr_candidate_bridge_guide_rejected": _optional_int_attr(model, "debr_candidate_bridge_guide_rejected_"),
        "debr_candidate_cleanup_eligible": _optional_int_attr(model, "debr_candidate_cleanup_eligible_"),
        "debr_candidate_cleanup_primary_rejected": _optional_int_attr(model, "debr_candidate_cleanup_primary_rejected_"),
        "debr_candidate_cleanup_complexity_rejected": _optional_int_attr(model, "debr_candidate_cleanup_complexity_rejected_"),
        "debr_candidate_score_rejected": _optional_int_attr(model, "debr_candidate_score_rejected_"),
        "debr_descent_moves": _optional_int_attr(model, "debr_descent_moves_"),
        "debr_bridge_moves": _optional_int_attr(model, "debr_bridge_moves_"),
        "debr_simplify_moves": _optional_int_attr(model, "debr_simplify_moves_"),
        "debr_source_group_row_size_histogram": json.dumps(
            getattr(model, "debr_source_group_row_size_histogram_", [])
        ),
        "debr_source_component_atom_size_histogram": json.dumps(
            getattr(model, "debr_source_component_atom_size_histogram_", [])
        ),
        "debr_source_component_row_size_histogram": json.dumps(
            getattr(model, "debr_source_component_row_size_histogram_", [])
        ),
        "debr_total_hard_gain": _optional_float_attr(model, "debr_total_hard_gain_"),
        "debr_total_soft_gain": _optional_float_attr(model, "debr_total_soft_gain_"),
        "debr_total_delta_j": _optional_float_attr(model, "debr_total_delta_j_"),
        "debr_total_component_delta": _optional_int_attr(model, "debr_total_component_delta_"),
        "debr_final_geo_wins": _optional_int_attr(model, "debr_final_geo_wins_"),
        "debr_final_block_wins": _optional_int_attr(model, "debr_final_block_wins_"),
        "profiling_signature_bound_calls": _optional_int_attr(model, "profiling_signature_bound_calls_"),
        "profiling_signature_bound_sec": _optional_float_attr(model, "profiling_signature_bound_sec_"),
        "profiling_path_bound_calls": _optional_int_attr(model, "profiling_path_bound_calls_"),
        "profiling_path_bound_sec": _optional_float_attr(model, "profiling_path_bound_sec_"),
        "profiling_path_bound_cache_hits": _optional_int_attr(model, "profiling_path_bound_cache_hits_"),
        "profiling_path_bound_skip_trivial": _optional_int_attr(model, "profiling_path_bound_skip_trivial_"),
        "profiling_path_bound_skip_disabled": _optional_int_attr(model, "profiling_path_bound_skip_disabled_"),
        "profiling_path_bound_skip_small_state": _optional_int_attr(model, "profiling_path_bound_skip_small_state_"),
        "profiling_path_bound_skip_too_many_blocks": _optional_int_attr(model, "profiling_path_bound_skip_too_many_blocks_"),
        "profiling_path_bound_skip_large_child": _optional_int_attr(model, "profiling_path_bound_skip_large_child_"),
        "profiling_path_bound_tighten_attempts": _optional_int_attr(model, "profiling_path_bound_tighten_attempts_"),
        "profiling_path_bound_tighten_effective": _optional_int_attr(model, "profiling_path_bound_tighten_effective_"),
        "profiling_lp_solve_calls": _optional_int_attr(model, "profiling_lp_solve_calls_"),
        "profiling_lp_solve_sec": _optional_float_attr(model, "profiling_lp_solve_sec_"),
        "profiling_pricing_calls": _optional_int_attr(model, "profiling_pricing_calls_"),
        "profiling_pricing_sec": _optional_float_attr(model, "profiling_pricing_sec_"),
        "profiling_greedy_complete_calls": _optional_int_attr(model, "profiling_greedy_complete_calls_"),
        "profiling_greedy_complete_calls_by_depth": json.dumps(
            getattr(model, "profiling_greedy_complete_calls_by_depth_", [])
        ),
        "profiling_greedy_complete_sec": _optional_float_attr(model, "profiling_greedy_complete_sec_"),
        "profiling_candidate_generation_sec": _optional_float_attr(
            model, "profiling_candidate_generation_sec_"
        ),
        "profiling_recursive_child_eval_sec": _optional_float_attr(
            model, "profiling_recursive_child_eval_sec_"
        ),
        "profiling_refine_calls": _optional_int_attr(model, "profiling_refine_calls_"),
        "profiling_refine_sec": _optional_float_attr(model, "profiling_refine_sec_"),
        "native_n_classes": int(getattr(model, "native_n_classes_", n_classes)),
        "native_teacher_class_count": int(getattr(model, "native_teacher_class_count_", 0)),
        "native_binary_mode": int(bool(getattr(model, "native_binary_mode_", n_classes == 2))),
        "atomized_features_prepared": int(getattr(model, "atomized_features_prepared_", 0)),
        "atomized_coarse_candidates": int(getattr(model, "atomized_coarse_candidates_", 0)),
        "atomized_final_candidates": int(getattr(model, "atomized_final_candidates_", 0)),
        "atomized_coarse_pruned_candidates": int(getattr(model, "atomized_coarse_pruned_candidates_", 0)),
        "greedy_feature_survivor_histogram": json.dumps(
            getattr(model, "greedy_feature_survivor_histogram_", [])
        ),
        "atomized_feature_atom_count_histogram": json.dumps(
            getattr(model, "atomized_feature_atom_count_histogram_", [])
        ),
        "atomized_feature_block_atom_count_histogram": json.dumps(
            getattr(model, "atomized_feature_block_atom_count_histogram_", [])
        ),
        "atomized_feature_q_effective_histogram": json.dumps(
            getattr(model, "atomized_feature_q_effective_histogram_", [])
        ),
        "greedy_state_block_count_histogram": json.dumps(
            getattr(model, "greedy_state_block_count_histogram_", [])
        ),
        "greedy_feature_preserved_histogram": json.dumps(
            getattr(model, "greedy_feature_preserved_histogram_", [])
        ),
        "greedy_candidate_count_histogram": json.dumps(
            getattr(model, "greedy_candidate_count_histogram_", [])
        ),
        "msplit_train_rows_used": int(split.X_train_proc.shape[0]),
    }


def _plot_dataset_accuracy(dataset_name: str, summary_df: pd.DataFrame, depths: list[int], out_path: Path, metric: str) -> None:
    metric = str(metric)
    plt.rcParams.update({"figure.facecolor": "#fbfaf7", "axes.facecolor": "#fbfaf7", "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for model_name in MODEL_ORDER:
        subset = summary_df[summary_df["model"] == model_name].sort_values("depth_budget")
        if subset.empty:
            continue
        col = f"mean_{metric}"
        std_col = f"std_{metric}"
        x = subset["depth_budget"].to_numpy(dtype=int)
        y = subset[col].to_numpy(dtype=float)
        yerr = subset[std_col].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            y * 100.0,
            color=MODEL_COLORS[model_name],
            marker=MODEL_MARKERS[model_name],
            linewidth=2.2,
            label=MODEL_LABELS[model_name],
        )
        if np.any(yerr > 0):
            ax.fill_between(x, (y - yerr) * 100.0, (y + yerr) * 100.0, color=MODEL_COLORS[model_name], alpha=0.15)
    ax.set_title(f"{dataset_name}: {metric.replace('_', ' ').title()} vs Depth")
    ax.set_xlabel("Depth budget")
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_xticks(depths)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summarize_dataset(seed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_df.groupby(["dataset", "model", "depth_budget"], as_index=False)
        .agg(
            mean_train_accuracy=("train_accuracy", "mean"),
            std_train_accuracy=("train_accuracy", "std"),
            mean_val_accuracy=("val_accuracy", "mean"),
            std_val_accuracy=("val_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_fit_time_sec=("fit_time_sec", "mean"),
            std_fit_time_sec=("fit_time_sec", "std"),
            mean_internal_nodes=("n_internal_nodes", "mean"),
            mean_leaves=("n_leaves", "mean"),
            mean_sparsity=("sparsity", "mean"),
            n_runs=("seed", "count"),
        )
        .sort_values(["dataset", "model", "depth_budget"])
        .reset_index(drop=True)
    )
    for col in summary.columns:
        if col.startswith("std_"):
            summary[col] = summary[col].fillna(0.0)
    return summary


def _select_best_runs(seed_df: pd.DataFrame) -> pd.DataFrame:
    ranked = seed_df.sort_values(
        ["dataset", "model", "test_accuracy", "train_accuracy", "depth_budget", "seed"],
        ascending=[True, True, False, False, True, True],
    )
    return ranked.groupby(["dataset", "model"], as_index=False).head(1).reset_index(drop=True)


def _render_artifact_tree(dataset_name: str, artifact_path: Path, output_path: Path) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "visualize_multisplit_tree.py"),
        "--dataset",
        dataset_name,
        "--pipeline",
        "lightgbm",
        "--artifact-in",
        str(artifact_path),
        "--out",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _coalesce_artifact_path(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        return text
    return ""


def _build_dataset_metadata(dataset_name: str, X, y, class_labels: np.ndarray, split_cache: dict[int, PreparedSplit]) -> dict[str, object]:
    numeric_cols = list(X.select_dtypes(include=[np.number]).columns) if hasattr(X, "select_dtypes") else []
    categorical_cols = [str(c) for c in getattr(X, "columns", []) if c not in numeric_cols]
    sample_seed = sorted(split_cache)[0]
    sample_split = split_cache[sample_seed]
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "spec": get_dataset_spec(dataset_name).__dict__,
        "n_rows": int(len(y)),
        "n_features_raw": int(X.shape[1]),
        "n_features_numeric_raw": int(len(numeric_cols)),
        "n_features_categorical_raw": int(len(categorical_cols)),
        "n_classes": int(len(class_labels)),
        "class_labels": [str(v) for v in class_labels.tolist()],
        "class_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "sample_preprocessed_feature_count": int(sample_split.X_train_proc.shape[1]),
        "sample_train_rows_used": int(sample_split.idx_train.size),
        "sample_val_rows": int(sample_split.idx_val.size),
        "sample_test_rows": int(sample_split.idx_test.size),
        "seeds": sorted(int(s) for s in split_cache),
        "train_size": float(len(sample_split.idx_train) / len(y)),
        "val_size": float(len(sample_split.idx_val) / len(y)),
        "test_size": float(len(sample_split.idx_test) / len(y)),
    }


def _run_dataset(args: argparse.Namespace, dataset_name: str) -> None:
    dataset_dir = Path(args.output_root) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = dataset_dir / "artifacts"
    split_root = dataset_dir / "splits"
    plot_root = dataset_dir / "plots"
    tree_root = dataset_dir / "tree_visualizations"
    required_paths = [plot_root]
    if not args.graphs_only:
        required_paths.extend([artifact_root, split_root, tree_root])
    for path in required_paths:
        path.mkdir(parents=True, exist_ok=True)
    log_path = dataset_dir / "run.log"
    if log_path.exists():
        log_path.unlink()

    X, y = load_dataset(dataset_name, data_root=args.output_root)
    y_encoded, class_labels, _ = encode_target(y)
    target_name = getattr(y, "name", None) or get_dataset_spec(dataset_name).target_name or "target"
    split_cache = _prepare_split_cache(
        X,
        y_encoded,
        seeds=args.seeds,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        max_train_rows=int(args.max_train_rows),
    )

    if not args.graphs_only:
        for seed, split in split_cache.items():
            _write_json(
                split_root / f"seed_{int(seed)}.json",
                {
                    "seed": int(seed),
                    "idx_train": split.idx_train,
                    "idx_test": split.idx_test,
                    "idx_val": split.idx_val,
                    "n_train": int(split.idx_train.size),
                    "n_val": int(split.idx_val.size),
                    "n_test": int(split.idx_test.size),
                    "max_train_rows": int(args.max_train_rows),
                    "n_features_preprocessed": int(split.X_train_proc.shape[1]),
                    "feature_names": split.feature_names,
                },
            )

        metadata = _build_dataset_metadata(dataset_name, X, y, class_labels, split_cache)
        _write_json(dataset_dir / "dataset_metadata.json", metadata)
    else:
        metadata = {
            "n_classes": int(len(class_labels)),
        }

    rows: list[dict[str, object]] = []
    for depth in args.depths:
        for seed in args.seeds:
            split = split_cache[int(seed)]
            trial_specs = [
                ("xgboost", _fit_xgboost, artifact_root / "xgboost"),
                ("shapecart", _fit_shapecart, artifact_root / "shapecart"),
                ("msplit_atomized", _fit_msplit, artifact_root / "msplit_atomized"),
            ]
            for model_name, fit_fn, model_artifact_dir in trial_specs:
                _append_log(log_path, f"start dataset={dataset_name} model={model_name} depth={depth} seed={seed}")
                try:
                    row = fit_fn(
                        args=args,
                        dataset_name=dataset_name,
                        depth=int(depth),
                        split=split,
                        class_labels=class_labels,
                        target_name=str(target_name),
                        artifact_dir=model_artifact_dir,
                    )
                except Exception as exc:
                    error_payload = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "depth_budget": int(depth),
                        "seed": int(seed),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    _write_json(dataset_dir / "last_error.json", error_payload)
                    _append_log(
                        log_path,
                        f"fail dataset={dataset_name} model={model_name} depth={depth} seed={seed}: {exc}",
                    )
                    raise
                rows.append(row)
                _append_log(
                    log_path,
                    (
                        f"done dataset={dataset_name} model={model_name} depth={depth} seed={seed} "
                        f"test_acc={row['test_accuracy']:.4f} fit_sec={row['fit_time_sec']:.2f}"
                    ),
                )

    seed_df = pd.DataFrame(rows).sort_values(["model", "depth_budget", "seed"]).reset_index(drop=True)
    summary_df = _summarize_dataset(seed_df)

    summary_df.to_csv(dataset_dir / "summary_by_depth.csv", index=False)
    if not args.graphs_only:
        seed_df.to_csv(dataset_dir / "seed_results.csv", index=False)
        best_df = _select_best_runs(seed_df)
        best_df.to_csv(dataset_dir / "best_runs.csv", index=False)
    else:
        best_df = pd.DataFrame()

    _plot_dataset_accuracy(dataset_name, summary_df, args.depths, plot_root / "accuracy_vs_depth_test.png", "test_accuracy")
    _plot_dataset_accuracy(dataset_name, summary_df, args.depths, plot_root / "accuracy_vs_depth_val.png", "val_accuracy")
    if not args.graphs_only:
        _plot_dataset_accuracy(
            dataset_name,
            summary_df,
            args.depths,
            plot_root / "accuracy_vs_depth_train.png",
            "train_accuracy",
        )

    tree_manifest: list[dict[str, object]] = []
    if not args.graphs_only and not args.no_render_trees:
        for _, row in best_df.iterrows():
            model_name = str(row["model"])
            artifact_text = _coalesce_artifact_path(
                row.get("primary_tree_artifact_path"),
                row.get("artifact_path"),
            )
            if not artifact_text:
                continue
            artifact_path = Path(artifact_text)
            output_path = tree_root / f"{model_name}_best_tree.png"
            _render_artifact_tree(dataset_name, artifact_path, output_path)
            tree_manifest.append(
                {
                    "model": model_name,
                    "artifact_path": str(artifact_path),
                    "image_path": str(output_path),
                    "note": "best run tree",
                }
            )

    if not args.graphs_only:
        _write_json(dataset_dir / "tree_visualization_manifest.json", {"items": tree_manifest})


def main() -> None:
    args = _parse_args()
    args.datasets = canonical_dataset_list(args.datasets)
    args.depths = sorted(set(int(v) for v in args.depths))
    args.seeds = sorted(set(int(v) for v in args.seeds))
    if args.graphs_only:
        args.no_render_trees = True

    materialize_all_datasets(
        dataset_names=args.datasets,
        data_root=args.output_root,
        force=bool(args.force_download),
        openml_cache_dir=args.openml_cache_dir,
    )
    if args.download_only:
        return

    for dataset_name in args.datasets:
        _run_dataset(args, dataset_name)


if __name__ == "__main__":
    main()
