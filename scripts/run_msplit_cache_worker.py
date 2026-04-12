#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_teacher_guided_atomcolor_cached import predict_tree, root_has_noncontiguous_group, tree_stats


def _load_libgosdt(build_dir: str) -> object:
    build_path = Path(build_dir)
    if not build_path.is_absolute():
        build_path = (REPO_ROOT / "SPLIT-ICML" / "split" / build_path).resolve()
    candidates = sorted(build_path.glob("_libgosdt*.so"))
    if not candidates:
        raise FileNotFoundError(f"Could not find _libgosdt extension under {build_path}")
    so_path = candidates[0]
    module_name = "split._libgosdt"
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as npz:
        return {name: np.asarray(npz[name]) for name in npz.files}


def _tree_depth(tree: dict[str, object]) -> int:
    if tree.get("type") == "leaf":
        return 0
    groups = tree.get("groups", [])
    if not groups:
        return 0
    return 1 + max(_tree_depth(group["child"]) for group in groups)


def _group_span_counts(tree: dict[str, object]) -> dict[str, int]:
    if tree.get("type") == "leaf":
        return {"noncontiguous_groups": 0, "groups_with_multiple_spans": 0}
    noncontiguous = 0
    multi_span = 0
    for group in tree.get("groups", []):
        spans = group.get("spans", [])
        if len(spans) > 1:
            noncontiguous += 1
            multi_span += 1
        child_counts = _group_span_counts(group["child"])
        noncontiguous += int(child_counts["noncontiguous_groups"])
        multi_span += int(child_counts["groups_with_multiple_spans"])
    return {
        "noncontiguous_groups": noncontiguous,
        "groups_with_multiple_spans": multi_span,
    }


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single MSPLIT native build against a cached dataset artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--build-dir", type=str, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--lookahead-depth", type=int, required=True)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--min-split-size", type=int, required=True)
    parser.add_argument("--min-child-size", type=int, required=True)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--exactify-top-k", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cache = _load_cache(args.cache_path)
    libgosdt = _load_libgosdt(args.build_dir)

    z_fit = np.asarray(cache["Z_fit"], dtype=np.int32)
    z_test = np.asarray(cache["Z_test"], dtype=np.int32)
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
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
        int(args.depth),
        int(args.lookahead_depth),
        float(args.reg),
        int(args.min_split_size),
        int(args.min_child_size),
        28800.0,
        int(args.max_branching),
        int(args.exactify_top_k),
    )
    fit_seconds = time.perf_counter() - started

    tree = json.loads(str(cpp_result["tree"]))
    pred_train = predict_tree(tree, z_fit)
    pred_test = predict_tree(tree, z_test)
    stats = tree_stats(tree)
    span_counts = _group_span_counts(tree)

    payload = {
        "fit_seconds": float(fit_seconds),
        "train_accuracy": float(np.mean(pred_train == y_fit)),
        "test_accuracy": float(np.mean(pred_test == y_test)),
        "objective": float(cpp_result.get("objective", 0.0)),
        "root_feature": int(tree.get("feature", -1)),
        "root_group_count": int(tree.get("group_count", 0)),
        "root_has_noncontiguous_group": bool(root_has_noncontiguous_group(tree)),
        "tree_depth": int(_tree_depth(tree)),
        "n_leaves": int(stats["n_leaves"]),
        "n_internal": int(stats["n_internal"]),
        "max_arity": int(stats["max_arity"]),
        "noncontiguous_group_count": int(span_counts["noncontiguous_groups"]),
        "groups_with_multiple_spans": int(span_counts["groups_with_multiple_spans"]),
        "tree": tree,
    }

    for key, value in cpp_result.items():
        if key in {"tree", "objective"}:
            continue
        payload[key] = _jsonify(value)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
