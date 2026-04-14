#!/usr/bin/env python3
"""Shared helpers for cached MSPLIT vs ShapeCART benchmark runners."""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.scripts.benchmark_paths import ensure_repo_import_paths
from benchmark.scripts.cache_utils import (
    default_cache_path,
    resolve_compatible_cache,
)
from benchmark.scripts.lightgbm_binning import deserialize_lightgbm_binner
from benchmark.scripts.runtime_guard import guarded_fit
from benchmark.scripts.tree_artifact_utils import (
    build_msplit_artifact_from_serialized_tree,
    build_shapecart_artifact,
    write_artifact_json,
)

ensure_repo_import_paths(include_msplit_src=True, include_shapecart=True)

from src.ShapeCARTClassifier import ShapeCARTClassifier  # type: ignore


DEFAULT_DEPTHS = [2, 3, 4, 5, 6]
DEFAULT_SEEDS = [0]
DEFAULT_CV_FOLDS = 3
DEFAULT_SHARED_MIN_LEAF_VALUES = [1, 2, 4, 8, 16]
DEFAULT_SHARED_SPLIT_MULTIPLIERS = [2, 4]
DEFAULT_MSPLIT_LOOKAHEAD_DEPTH_VALUES = [2, 3, 4]
DEFAULT_MSPLIT_EXACTIFY_TOP_K_VALUES = [1, 2, 3, 4, 5, 6]
DEFAULT_MSPLIT_MAX_BRANCHING_VALUES = [3]
DEFAULT_MSPLIT_REG_VALUES = ["0.0", "1e-6", "3e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]
DEFAULT_SHAPE_CRITERION_VALUES = ["gini", "entropy"]
DEFAULT_SHAPE_INNER_MAX_LEAF_VALUES = [4, 8, 12, 16, 24, 32, 48, 64]
DEFAULT_SHAPE_INNER_MIN_LEAF_VALUES = ["1", "5e-4", "1e-3", "5e-3", "1e-2"]
DEFAULT_SHAPE_BRANCHING_PENALTY_VALUES = ["0.0", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
DEFAULT_SHAPE_TAO_REG_VALUES = ["0.0", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]


@dataclass
class CachedProtocol:
    dataset: str
    seed: int
    requested_test_size: float
    requested_val_size: float
    requested_max_bins: int
    requested_binner_min_samples_leaf: int
    requested_cache_min_child_size: int
    cache_path: Path
    cache: dict[str, np.ndarray]
    cache_meta: dict[str, object]
    cache_used_fallback: bool
    requested_cache_path: Path

    @property
    def n_fit(self) -> int:
        return int(np.asarray(self.cache["X_fit_proc"]).shape[0])

    @property
    def n_val(self) -> int:
        return int(np.asarray(self.cache["X_val_proc"]).shape[0])

    @property
    def n_test(self) -> int:
        return int(np.asarray(self.cache["X_test_proc"]).shape[0])

    @property
    def test_size(self) -> float:
        return float(self.cache_meta.get("test_size", self.requested_test_size))

    @property
    def val_size(self) -> float:
        return float(self.cache_meta.get("val_size", self.requested_val_size))

    @property
    def max_bins(self) -> int:
        return int(self.cache_meta.get("max_bins", self.requested_max_bins))

    @property
    def binner_min_samples_leaf(self) -> int:
        return int(self.cache_meta.get("min_samples_leaf", self.requested_binner_min_samples_leaf))

    @property
    def cache_min_child_size(self) -> int:
        return int(self.cache_meta.get("min_child_size", self.requested_cache_min_child_size))

    @property
    def cache_build_seconds(self) -> float:
        return float(self.cache_meta.get("build_seconds", 0.0))

    @property
    def feature_names(self) -> list[str]:
        return [str(v) for v in np.asarray(self.cache["feature_names"], dtype=object).tolist()]

    @property
    def class_labels(self) -> np.ndarray:
        return np.asarray(self.cache["class_labels"], dtype=object)


def coerce_numeric_token(token: str) -> int | float:
    token = str(token).strip()
    if any(ch in token.lower() for ch in (".", "e")):
        return float(token)
    return int(token)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def configure_timing_mode(mode: str) -> None:
    os.environ["MSDT_BENCHMARK_GUARD"] = "1" if mode == "fair" else "0"


@contextmanager
def timing_guard_scope(enabled: bool):
    previous = os.environ.get("MSDT_BENCHMARK_GUARD")
    os.environ["MSDT_BENCHMARK_GUARD"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MSDT_BENCHMARK_GUARD", None)
        else:
            os.environ["MSDT_BENCHMARK_GUARD"] = previous


def default_search_jobs() -> int:
    for env_name in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        raw = os.environ.get(env_name)
        if raw and raw.isdigit() and int(raw) > 0:
            return int(raw)
    return max(1, int(os.cpu_count() or 1))


def resolve_search_jobs(requested_jobs: int | None, timing_mode: str) -> int:
    if requested_jobs is not None:
        return max(1, int(requested_jobs))
    return default_search_jobs()


def relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def cached_protocol_manifest_row(protocol: CachedProtocol) -> dict[str, Any]:
    return {
        "dataset": protocol.dataset,
        "seed": int(protocol.seed),
        "cache_path": str(protocol.cache_path),
        "cache_requested_path": str(protocol.requested_cache_path),
        "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
        "test_size": float(protocol.test_size),
        "val_size": float(protocol.val_size),
        "max_bins": int(protocol.max_bins),
        "binner_min_samples_leaf": int(protocol.binner_min_samples_leaf),
        "cache_min_child_size": int(protocol.cache_min_child_size),
        "cache_build_seconds": float(protocol.cache_build_seconds),
        "n_fit": int(protocol.n_fit),
        "n_val": int(protocol.n_val),
        "n_test": int(protocol.n_test),
    }


def write_csv_tables(out_dir: Path, tables: dict[str, pd.DataFrame | list[dict[str, Any]]]) -> None:
    for filename, payload in tables.items():
        df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        df.to_csv(out_dir / filename, index=False)


def benchmark_timing_fields(
    *,
    algorithm: str,
    model_fit_time_sec: float,
    shared_cache_build_seconds: float,
) -> dict[str, float]:
    model_fit = float(model_fit_time_sec)
    shared_cache = float(shared_cache_build_seconds) if algorithm == "msplit" else float("nan")
    return {
        "model_fit_time_sec": model_fit,
        "pipeline_fit_time_sec": model_fit if algorithm != "msplit" else model_fit + shared_cache,
        "shared_cache_build_seconds": shared_cache,
    }


def load_cached_protocol(
    *,
    dataset: str,
    seed: int,
    test_size: float,
    val_size: float,
    max_bins: int,
    binner_min_samples_leaf: int,
    cache_min_child_size: int,
    cache_version: int,
) -> CachedProtocol:
    requested_cache_path = default_cache_path(
        dataset=dataset,
        seed=int(seed),
        test_size=float(test_size),
        val_size=float(val_size),
        max_bins=int(max_bins),
        min_samples_leaf=int(binner_min_samples_leaf),
        min_child_size=int(cache_min_child_size),
        cache_version=int(cache_version),
    )
    cache_path, cache, cache_meta, cache_hit, cache_used_fallback = resolve_compatible_cache(
        requested_cache_path,
        force_rebuild=False,
    )
    if not cache_hit:
        raise FileNotFoundError(
            "Required benchmark cache is missing. "
            f"expected={requested_cache_path}"
        )
    return CachedProtocol(
        dataset=dataset,
        seed=int(seed),
        requested_test_size=float(test_size),
        requested_val_size=float(val_size),
        requested_max_bins=int(max_bins),
        requested_binner_min_samples_leaf=int(binner_min_samples_leaf),
        requested_cache_min_child_size=int(cache_min_child_size),
        cache_path=cache_path,
        cache=cache,
        cache_meta=cache_meta,
        cache_used_fallback=bool(cache_used_fallback),
        requested_cache_path=requested_cache_path,
    )


def fit_shapecart_once(
    *,
    cache: dict[str, np.ndarray],
    depth: int,
    random_state: int,
    params: dict[str, Any],
    include_model: bool,
) -> dict[str, Any]:
    X_fit = np.asarray(cache["X_fit_proc"], dtype=np.float32)
    X_val = np.asarray(cache["X_val_proc"], dtype=np.float32)
    X_test = np.asarray(cache["X_test_proc"], dtype=np.float32)
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    y_val = np.asarray(cache["y_val"], dtype=np.int32)
    y_test = np.asarray(cache["y_test"], dtype=np.int32)

    def _run_once() -> ShapeCARTClassifier:
        model = ShapeCARTClassifier(
            max_depth=int(depth),
            min_samples_leaf=params["min_samples_leaf"],
            min_samples_split=params["min_samples_split"],
            inner_min_samples_leaf=params["inner_min_samples_leaf"],
            inner_min_samples_split=params["min_samples_split"],
            inner_max_depth=int(params["inner_max_depth"]),
            inner_max_leaf_nodes=int(params["inner_max_leaf_nodes"]),
            max_iter=int(params["max_iter"]),
            k=int(params["k"]),
            branching_penalty=float(params["branching_penalty"]),
            criterion=str(params["criterion"]),
            random_state=int(random_state),
            verbose=False,
            pairwise_candidates=float(params["pairwise_candidates"]),
            smart_init=bool(params["smart_init"]),
            random_pairs=bool(params["random_pairs"]),
            use_dpdt=bool(params["use_dpdt"]),
            use_tao=bool(params["use_tao"]),
            n_runs=int(params["tao_n_runs"]),
            tao_reg=float(params["tao_reg"]),
            tao_pair_scale=float(params["tao_pair_scale"]),
        )
        model.fit(X_fit, y_fit)
        return model

    model, fit_seconds, timing_guard = guarded_fit(_run_once, repo_root=PROJECT_ROOT)
    pred_train = np.asarray(model.predict(X_fit), dtype=np.int32)
    pred_val = np.asarray(model.predict(X_val), dtype=np.int32)
    pred_test = np.asarray(model.predict(X_test), dtype=np.int32)
    n_leaves = int(sum(bool(v) for v in model.is_leaf))
    n_internal = int(len(model.is_leaf) - n_leaves)
    result = {
        "train_accuracy": float(np.mean(pred_train == y_fit)),
        "val_accuracy": float(np.mean(pred_val == y_val)),
        "test_accuracy": float(np.mean(pred_test == y_test)),
        "fit_seconds": float(fit_seconds),
        "n_internal_nodes": int(n_internal),
        "n_leaves": int(n_leaves),
        "timing_guard": timing_guard,
    }
    if include_model:
        result["model"] = model
    return result


def write_msplit_artifacts(
    *,
    out_dir: Path,
    protocol: CachedProtocol,
    depth: int,
    best_params: dict[str, Any],
    final_result: dict[str, Any],
) -> tuple[str, str]:
    model_dir = (
        out_dir
        / "best_models"
        / protocol.dataset
        / f"seed{int(protocol.seed)}"
        / f"depth{int(depth)}"
        / "msplit"
    )
    tree_artifact_path = model_dir / "tree_artifact.json"
    metrics_path = model_dir / "metrics.json"

    artifact = build_msplit_artifact_from_serialized_tree(
        dataset=protocol.dataset,
        pipeline="cached_msplit",
        target_name="target",
        class_labels=protocol.class_labels,
        feature_names=protocol.feature_names,
        accuracy=float(final_result["test_accuracy"]),
        seed=int(protocol.seed),
        test_size=float(protocol.test_size),
        depth_budget=int(depth),
        lookahead=int(best_params["lookahead_depth"]),
        time_limit=28800.0,
        max_bins=int(protocol.max_bins),
        min_samples_leaf=int(protocol.binner_min_samples_leaf),
        min_child_size=int(best_params["min_child_size"]),
        max_branching=int(best_params["max_branching"]),
        reg=float(best_params["reg"]),
        branch_penalty=None,
        msplit_variant="build-nonlinear-py",
        tree_root=dict(final_result["tree"]),
        binner=deserialize_lightgbm_binner(protocol.cache),
        z_train=np.asarray(protocol.cache["Z_fit"], dtype=np.int32),
        train_indices=np.asarray(protocol.cache["idx_fit"], dtype=np.int32),
        test_indices=np.asarray(protocol.cache["idx_test"], dtype=np.int32),
    )
    write_artifact_json(tree_artifact_path, artifact)
    write_artifact_json(
        metrics_path,
        {
            "dataset": protocol.dataset,
            "seed": int(protocol.seed),
            "depth_budget": int(depth),
            "algorithm": "msplit",
            "cache_path": str(protocol.cache_path),
            "cache_requested_path": str(protocol.requested_cache_path),
            "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
            "selected_params": json_safe(best_params),
            "result": json_safe({k: v for k, v in final_result.items() if k != "tree"}),
        },
    )
    return relative_path(tree_artifact_path, out_dir), relative_path(metrics_path, out_dir)


def write_shapecart_artifacts(
    *,
    out_dir: Path,
    protocol: CachedProtocol,
    depth: int,
    final_params: dict[str, Any],
    final_result: dict[str, Any],
) -> tuple[str, str]:
    model_dir = (
        out_dir
        / "best_models"
        / protocol.dataset
        / f"seed{int(protocol.seed)}"
        / f"depth{int(depth)}"
        / "shapecart"
    )
    tree_artifact_path = model_dir / "tree_artifact.json"
    metrics_path = model_dir / "metrics.json"

    artifact = build_shapecart_artifact(
        dataset=protocol.dataset,
        target_name="target",
        class_labels=protocol.class_labels,
        feature_names=protocol.feature_names,
        accuracy=float(final_result["test_accuracy"]),
        seed=int(protocol.seed),
        test_size=float(protocol.test_size),
        depth_budget=int(depth),
        k=int(final_params["k"]),
        min_samples_leaf=int(final_params["min_samples_leaf"]),
        min_samples_split=int(final_params["min_samples_split"]),
        inner_min_samples_leaf=final_params["inner_min_samples_leaf"],
        inner_min_samples_split=int(final_params["min_samples_split"]),
        inner_max_depth=int(final_params["inner_max_depth"]),
        inner_max_leaf_nodes=int(final_params["inner_max_leaf_nodes"]),
        max_iter=int(final_params["max_iter"]),
        model=final_result["model"],
        train_indices=np.asarray(protocol.cache["idx_fit"], dtype=np.int32),
        test_indices=np.asarray(protocol.cache["idx_test"], dtype=np.int32),
    )
    write_artifact_json(tree_artifact_path, artifact)
    write_artifact_json(
        metrics_path,
        {
            "dataset": protocol.dataset,
            "seed": int(protocol.seed),
            "depth_budget": int(depth),
            "algorithm": "shapecart",
            "cache_path": str(protocol.cache_path),
            "cache_requested_path": str(protocol.requested_cache_path),
            "cache_used_compatible_fallback": bool(protocol.cache_used_fallback),
            "selected_params": json_safe(final_params),
            "result": json_safe({k: v for k, v in final_result.items() if k != "model"}),
        },
    )
    return relative_path(tree_artifact_path, out_dir), relative_path(metrics_path, out_dir)


def aggregate_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = [
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "model_fit_time_sec",
        "pipeline_fit_time_sec",
        "search_time_sec",
        "shared_cache_build_seconds",
        "n_internal_nodes",
        "n_leaves",
    ]
    summary = (
        df.groupby(["dataset", "algorithm", "depth_budget"])[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    flat_cols: list[str] = []
    for col in summary.columns:
        if isinstance(col, tuple):
            left, right = col
            if right == "":
                flat_cols.append(str(left))
            elif left in {"dataset", "algorithm", "depth_budget"}:
                flat_cols.append(str(left))
            else:
                flat_cols.append(f"{left}_{right}")
        else:
            flat_cols.append(str(col))
    summary.columns = flat_cols
    counts = (
        df.groupby(["dataset", "algorithm", "depth_budget"], as_index=False)
        .size()
        .rename(columns={"size": "n_seeds"})
    )
    return summary.merge(counts, on=["dataset", "algorithm", "depth_budget"], how="left")


def best_depth_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.sort_values(
        by=["dataset", "algorithm", "test_accuracy_mean", "pipeline_fit_time_sec_mean", "depth_budget"],
        ascending=[True, True, False, True, True],
    )
    return ranked.groupby(["dataset", "algorithm"], as_index=False).head(1).reset_index(drop=True)
