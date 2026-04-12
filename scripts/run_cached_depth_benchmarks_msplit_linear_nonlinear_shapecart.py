#!/usr/bin/env python3
"""Compare cached LightGBM-binned MSPLIT variants against ShapeCART.

This runner consumes the precomputed cached LightGBM bins under
`results/cache/lightgbm_binner` and compares:

* `msplit_linear` via `SPLIT-ICML/split/build-linear-py`
* `msplit_nonlinear` via `SPLIT-ICML/split/build-nonlinear-py`
* `shapecart`

The benchmark is depth-swept and reports train/test accuracy, fit time,
internal node count, and leaf count for each algorithm.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_utils import canonical_dataset_list

SHAPECART_ROOT = PROJECT_ROOT / "Empowering-DTs-via-Shape-Functions"
if str(SHAPECART_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPECART_ROOT))

from src.ShapeCARTClassifier import ShapeCARTClassifier  # type: ignore


ALGORITHMS = ("msplit_linear", "msplit_nonlinear", "shapecart")
ALGORITHM_LABELS = {
    "msplit_linear": "MSPLIT linear",
    "msplit_nonlinear": "MSPLIT nonlinear",
    "shapecart": "ShapeCART",
}
ALGORITHM_COLORS = {
    "msplit_linear": "#1f77b4",
    "msplit_nonlinear": "#2ca02c",
    "shapecart": "#ff8c00",
}
ALGORITHM_MARKERS = {
    "msplit_linear": "o",
    "msplit_nonlinear": "s",
    "shapecart": "^",
}

DEFAULT_CACHE_VERSION = 4
CACHE_REQUIRED_KEYS = {
    "X_fit_proc",
    "X_test_proc",
    "y_fit",
    "y_test",
    "Z_fit",
    "Z_test",
    "teacher_logit",
    "teacher_boundary_gain",
    "teacher_boundary_cover",
    "teacher_boundary_value_jump",
}


@dataclass(frozen=True)
class CacheSpec:
    dataset: str
    seed: int
    test_size: float
    val_size: float
    max_bins: int
    min_samples_leaf: int
    min_child_size: int
    cache_version: int


def default_cache_path(
    dataset: str,
    seed: int,
    test_size: float,
    val_size: float,
    max_bins: int,
    min_samples_leaf: int,
    min_child_size: int,
    cache_version: int,
) -> Path:
    stem = (
        f"{dataset}_seed{int(seed)}_"
        f"test{str(float(test_size)).replace('.', 'p')}_"
        f"val{str(float(val_size)).replace('.', 'p')}_"
        f"bins{int(max_bins)}_leaf{int(min_samples_leaf)}_child{int(min_child_size)}"
    )
    if dataset == "compas":
        stem += "_raw"
    if cache_version:
        stem += f"_v{int(cache_version)}"
    return PROJECT_ROOT / "results" / "cache" / "lightgbm_binner" / f"{stem}.npz"


def load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as npz:
        return {name: np.asarray(npz[name]) for name in npz.files}


def cache_is_complete(cache: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    missing = [key for key in sorted(CACHE_REQUIRED_KEYS) if key not in cache]
    return (len(missing) == 0, missing)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Depth-sweep cached LightGBM MSPLIT vs ShapeCART comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["electricity", "eye-movement", "eye-state", "compas", "heloc", "adult", "bike_sharing", "spambase"],
        help="Datasets to benchmark.",
    )
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.125)
    parser.add_argument("--max-bins", type=int, default=1024)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=8)
    parser.add_argument(
        "--cache-version",
        type=int,
        default=DEFAULT_CACHE_VERSION,
        help="Cache file version suffix used when locating cached LightGBM bins.",
    )
    parser.add_argument("--lookahead-depth", type=int, default=2)
    parser.add_argument("--reg", type=float, default=0.0005)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--lgb-num-threads", type=int, default=6)
    parser.add_argument(
        "--msplit-min-child-size",
        type=int,
        default=None,
        help="Override the MSPLIT child-size threshold passed to the cached benchmark.",
    )
    parser.add_argument(
        "--msplit-min-split-size",
        type=int,
        default=None,
        help="Override the MSPLIT split-size threshold passed to the cached benchmark.",
    )
    parser.add_argument("--linear-build-dir", type=str, default="build-linear-py")
    parser.add_argument("--nonlinear-build-dir", type=str, default="build-nonlinear-py")
    parser.add_argument("--shape-k", type=int, default=3)
    parser.add_argument("--shape-min-samples-leaf", type=int, default=8)
    parser.add_argument("--shape-min-samples-split", type=int, default=8)
    parser.add_argument("--shape-inner-max-depth", type=int, default=6)
    parser.add_argument("--shape-inner-max-leaf-nodes", type=int, default=32)
    parser.add_argument("--shape-max-iter", type=int, default=20)
    parser.add_argument("--shape-pairwise-candidates", type=float, default=0.0)
    parser.add_argument(
        "--shape-smart-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the weighted K-means initialization used by the paper's ShapeCART.",
    )
    parser.add_argument(
        "--shape-random-pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize pairwise candidate selection for the Shape2 variants.",
    )
    parser.add_argument(
        "--shape-use-dpdt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the DPDT-backed inner tree learner instead of the paper's greedy ShapeCART backend.",
    )
    parser.add_argument(
        "--shape-use-tao",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable TAO post-processing in ShapeCART.",
    )
    parser.add_argument("--results-root", type=str, default="results/comparison_cached_msplit_linear_nonlinear_shapecart")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing run directory and overwrite its tables.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating accuracy-vs-depth plots.")
    return parser.parse_args()


def _load_cache_for_dataset(spec: CacheSpec) -> dict[str, np.ndarray]:
    cache_path = default_cache_path(
        dataset=spec.dataset,
        seed=spec.seed,
        test_size=spec.test_size,
        val_size=spec.val_size,
        max_bins=spec.max_bins,
        min_samples_leaf=spec.min_samples_leaf,
        min_child_size=spec.min_child_size,
        cache_version=spec.cache_version,
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cache file: {cache_path}")
    cache = load_cache(cache_path)
    ok, missing = cache_is_complete(cache)
    if not ok:
        raise RuntimeError(f"Stale cache file {cache_path}: missing {missing}")
    return cache


def _build_cache_for_dataset(
    *,
    spec: CacheSpec,
    cache_path: Path,
    build_depth: int,
    lgb_num_threads: int,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_teacher_guided_atomcolor_cached.py"),
        "--dataset",
        spec.dataset,
        "--depth",
        str(int(build_depth)),
        "--seed",
        str(int(spec.seed)),
        "--test-size",
        str(float(spec.test_size)),
        "--val-size",
        str(float(spec.val_size)),
        "--max-bins",
        str(int(spec.max_bins)),
        "--min-samples-leaf",
        str(int(spec.min_samples_leaf)),
        "--min-child-size",
        str(int(spec.min_child_size)),
        "--lgb-num-threads",
        str(int(lgb_num_threads)),
        "--cache-path",
        str(cache_path),
        "--force-rebuild-cache",
        "--build-cache-only",
    ]
    print(f"[cache] building missing cache for {spec.dataset}: {cache_path}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Cache build failed for "
            f"dataset={spec.dataset}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if result.stdout.strip():
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.stderr.strip():
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)


def _ensure_cache_for_dataset(
    *,
    spec: CacheSpec,
    build_depth: int,
    lgb_num_threads: int,
) -> tuple[Path, dict[str, np.ndarray]]:
    cache_path = default_cache_path(
        dataset=spec.dataset,
        seed=spec.seed,
        test_size=spec.test_size,
        val_size=spec.val_size,
        max_bins=spec.max_bins,
        min_samples_leaf=spec.min_samples_leaf,
        min_child_size=spec.min_child_size,
        cache_version=spec.cache_version,
    )
    if cache_path.exists():
        cache = load_cache(cache_path)
        ok, missing = cache_is_complete(cache)
        if ok:
            return cache_path, cache
        print(f"[cache] stale cache file {cache_path}: missing {missing}", flush=True)
    else:
        print(f"[cache] missing cache file {cache_path}", flush=True)

    _build_cache_for_dataset(
        spec=spec,
        cache_path=cache_path,
        build_depth=build_depth,
        lgb_num_threads=lgb_num_threads,
    )
    cache = load_cache(cache_path)
    ok, missing = cache_is_complete(cache)
    if not ok:
        raise RuntimeError(f"Cache build completed but file is still incomplete: {cache_path} missing {missing}")
    return cache_path, cache


def _build_msplit_command(
    *,
    dataset: str,
    depth: int,
    cache_path: Path,
    lookahead_depth: int,
    reg: float,
    max_branching: int,
    min_samples_leaf: int,
    min_child_size: int,
    max_bins: int,
    lgb_num_threads: int,
    test_size: float,
    val_size: float,
    json_path: Path,
    msplit_min_child_size: int | None = None,
    msplit_min_split_size: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "benchmark_teacher_guided_atomcolor_cached.py"),
        "--dataset",
        dataset,
        "--cache-path",
        str(cache_path),
        "--depth",
        str(int(depth)),
        "--lookahead-depth",
        str(int(lookahead_depth)),
        "--seed",
        "0",
        "--test-size",
        str(float(test_size)),
        "--val-size",
        str(float(val_size)),
        "--max-bins",
        str(int(max_bins)),
        "--min-samples-leaf",
        str(int(min_samples_leaf)),
        "--min-child-size",
        str(int(min_child_size)),
        "--max-branching",
        str(int(max_branching)),
        "--reg",
        str(float(reg)),
        "--lgb-num-threads",
        str(int(lgb_num_threads)),
        "--json",
        str(json_path),
    ]
    if msplit_min_child_size is not None:
        cmd.extend(["--min-child-size", str(int(msplit_min_child_size))])
    if msplit_min_split_size is not None:
        cmd.extend(["--min-split-size", str(int(msplit_min_split_size))])
    return cmd


def _run_msplit_variant(
    *,
    dataset: str,
    depth: int,
    variant: str,
    build_dir: str,
    cache_path: Path,
    lookahead_depth: int,
    reg: float,
    max_branching: int,
    min_samples_leaf: int,
    min_child_size: int,
    max_bins: int,
    lgb_num_threads: int,
    test_size: float,
    val_size: float,
    out_dir: Path,
    msplit_min_child_size: int | None = None,
    msplit_min_split_size: int | None = None,
) -> dict[str, object]:
    tmp_dir = out_dir / "_tmp" / variant
    tmp_dir.mkdir(parents=True, exist_ok=True)
    json_path = tmp_dir / f"{dataset}_depth{int(depth)}.json"
    if json_path.exists():
        json_path.unlink()

    cmd = _build_msplit_command(
        dataset=dataset,
        depth=depth,
        cache_path=cache_path,
        lookahead_depth=lookahead_depth,
        reg=reg,
        max_branching=max_branching,
        min_samples_leaf=min_samples_leaf,
        min_child_size=min_child_size,
        max_bins=max_bins,
        lgb_num_threads=lgb_num_threads,
        test_size=float(test_size),
        val_size=float(val_size),
        json_path=json_path,
        msplit_min_child_size=msplit_min_child_size,
        msplit_min_split_size=msplit_min_split_size,
    )
    env = os.environ.copy()
    env["MSPLIT_BUILD_DIR"] = build_dir
    started = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "MSPLIT subprocess failed for "
            f"dataset={dataset} depth={depth} variant={variant}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    if not json_path.exists():
        raise FileNotFoundError(f"MSPLIT subprocess did not produce JSON output: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    fit_time_sec = float(payload.get("fit_seconds", payload.get("elapsed_time_sec", 0.0)))
    return {
        "dataset": dataset,
        "depth_budget": int(depth),
        "algorithm": variant,
        "train_accuracy": float(payload["train_accuracy"]),
        "test_accuracy": float(payload["test_accuracy"]),
        "fit_time_sec": float(fit_time_sec),
        "n_internal_nodes": int(payload["n_internal"]),
        "n_leaves": int(payload["n_leaves"]),
        "objective": float(payload.get("objective", np.nan)),
        "cache_hit": bool(payload.get("cache_hit", False)),
        "cache_path": str(payload.get("cache_path", "")),
        "build_dir": build_dir,
        "elapsed_wall_sec": float(time.perf_counter() - started),
    }


def _count_shapecart_nodes(model: ShapeCARTClassifier) -> tuple[int, int]:
    n_leaves = int(sum(bool(v) for v in model.is_leaf))
    n_internal = int(len(model.is_leaf) - n_leaves)
    return n_internal, n_leaves


def _fit_shapecart(
    *,
    cache: dict[str, np.ndarray],
    depth: int,
    seed: int,
    shape_k: int,
    shape_min_samples_leaf: int,
    shape_min_samples_split: int,
    shape_inner_max_depth: int,
    shape_inner_max_leaf_nodes: int,
    shape_max_iter: int,
    shape_pairwise_candidates: float,
    shape_smart_init: bool,
    shape_random_pairs: bool,
    shape_use_dpdt: bool,
    shape_use_tao: bool,
) -> dict[str, object]:
    X_fit = np.asarray(cache["X_fit_proc"], dtype=np.float32)
    X_test = np.asarray(cache["X_test_proc"], dtype=np.float32)
    y_fit = np.asarray(cache["y_fit"], dtype=np.int32)
    y_test = np.asarray(cache["y_test"], dtype=np.int32)

    model = ShapeCARTClassifier(
        max_depth=int(depth),
        min_samples_leaf=int(shape_min_samples_leaf),
        min_samples_split=int(shape_min_samples_split),
        inner_min_samples_leaf=int(shape_min_samples_leaf),
        inner_min_samples_split=int(shape_min_samples_split),
        inner_max_depth=int(shape_inner_max_depth),
        inner_max_leaf_nodes=int(shape_inner_max_leaf_nodes),
        max_iter=int(shape_max_iter),
        k=int(shape_k),
        branching_penalty=0.0,
        random_state=int(seed),
        verbose=False,
        pairwise_candidates=float(shape_pairwise_candidates),
        smart_init=bool(shape_smart_init),
        random_pairs=bool(shape_random_pairs),
        use_dpdt=bool(shape_use_dpdt),
        use_tao=bool(shape_use_tao),
    )
    started = time.perf_counter()
    model.fit(X_fit, y_fit)
    fit_time_sec = time.perf_counter() - started

    pred_train = np.asarray(model.predict(X_fit), dtype=np.int32)
    pred_test = np.asarray(model.predict(X_test), dtype=np.int32)
    n_internal, n_leaves = _count_shapecart_nodes(model)
    return {
        "dataset": "",
        "depth_budget": int(depth),
        "algorithm": "shapecart",
        "train_accuracy": float(np.mean(pred_train == y_fit)),
        "test_accuracy": float(np.mean(pred_test == y_test)),
        "fit_time_sec": float(fit_time_sec),
        "n_internal_nodes": int(n_internal),
        "n_leaves": int(n_leaves),
        "objective": np.nan,
        "cache_hit": True,
        "cache_path": "",
        "build_dir": "",
        "elapsed_wall_sec": float(fit_time_sec),
    }


def _plot_accuracy_vs_depth(
    summary_df: pd.DataFrame,
    *,
    datasets: list[str],
    depths: list[int],
    metric: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = str(metric).strip().lower()
    if metric not in {"train", "test"}:
        raise ValueError(f"Unknown metric {metric!r}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes_flat = list(axes.flatten())
    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        sub = summary_df[summary_df["dataset"] == dataset].sort_values("depth_budget")
        for algorithm in ALGORITHMS:
            col = f"{algorithm}_{metric}_accuracy"
            if col not in sub.columns:
                continue
            ax.plot(
                sub["depth_budget"],
                sub[col].to_numpy(dtype=float) * 100.0,
                marker=ALGORITHM_MARKERS[algorithm],
                color=ALGORITHM_COLORS[algorithm],
                linewidth=2.0,
                label=ALGORITHM_LABELS[algorithm],
            )
        ax.set_title(dataset)
        ax.set_xlabel("Depth budget")
        ax.set_ylabel(f"{metric.capitalize()} accuracy (%)")
        ax.set_xticks(depths)
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(datasets):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    datasets = canonical_dataset_list(args.datasets)
    depths = sorted(set(int(v) for v in args.depths))
    out_root = Path(args.results_root)
    run_name = args.run_name or datetime.now().strftime("cached_msplit_linear_nonlinear_shapecart_%Y%m%d_%H%M%S")
    out_dir = out_root / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "datasets": datasets,
        "depths": depths,
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "max_bins": int(args.max_bins),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_child_size": int(args.min_child_size),
        "cache_version": int(args.cache_version),
        "lookahead_depth": int(args.lookahead_depth),
        "reg": float(args.reg),
        "max_branching": int(args.max_branching),
        "lgb_num_threads": int(args.lgb_num_threads),
        "msplit_min_child_size": None if args.msplit_min_child_size is None else int(args.msplit_min_child_size),
        "msplit_min_split_size": None if args.msplit_min_split_size is None else int(args.msplit_min_split_size),
        "linear_build_dir": str(args.linear_build_dir),
        "nonlinear_build_dir": str(args.nonlinear_build_dir),
        "shape_k": int(args.shape_k),
        "shape_min_samples_leaf": int(args.shape_min_samples_leaf),
        "shape_min_samples_split": int(args.shape_min_samples_split),
        "shape_inner_max_depth": int(args.shape_inner_max_depth),
        "shape_inner_max_leaf_nodes": int(args.shape_inner_max_leaf_nodes),
        "shape_max_iter": int(args.shape_max_iter),
        "shape_pairwise_candidates": float(args.shape_pairwise_candidates),
        "shape_smart_init": bool(args.shape_smart_init),
        "shape_random_pairs": bool(args.shape_random_pairs),
        "shape_use_dpdt": bool(args.shape_use_dpdt),
        "shape_use_tao": bool(args.shape_use_tao),
    }

    _write_json(out_dir / "run_config.json", config)

    cache_spec_template = CacheSpec(
        dataset="",
        seed=int(args.seed),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
        max_bins=int(args.max_bins),
        min_samples_leaf=int(args.min_samples_leaf),
        min_child_size=int(args.min_child_size),
        cache_version=int(args.cache_version),
    )

    long_rows: list[dict[str, object]] = []
    cache_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    cache_path_by_dataset: dict[str, Path] = {}
    cache_build_depth = int(depths[0]) if depths else 2

    for dataset in datasets:
        spec = CacheSpec(
            dataset=dataset,
            seed=cache_spec_template.seed,
            test_size=cache_spec_template.test_size,
            val_size=cache_spec_template.val_size,
            max_bins=cache_spec_template.max_bins,
            min_samples_leaf=cache_spec_template.min_samples_leaf,
            min_child_size=cache_spec_template.min_child_size,
            cache_version=cache_spec_template.cache_version,
        )
        cache_path, cache = _ensure_cache_for_dataset(
            spec=spec,
            build_depth=cache_build_depth,
            lgb_num_threads=int(args.lgb_num_threads),
        )
        cache_by_dataset[dataset] = cache
        cache_path_by_dataset[dataset] = cache_path

    t_total = time.perf_counter()
    for dataset in datasets:
        cache = cache_by_dataset[dataset]
        cache_path = cache_path_by_dataset[dataset]
        for depth in depths:
            row_base = {
                "dataset": dataset,
                "depth_budget": int(depth),
                "seed": int(args.seed),
                "cache_path": str(cache_path),
            }

            print(f"[run] dataset={dataset} depth={depth} algorithm=msplit_linear", flush=True)
            linear_row = _run_msplit_variant(
                dataset=dataset,
                depth=depth,
                variant="msplit_linear",
                build_dir=str(args.linear_build_dir),
                cache_path=cache_path,
                lookahead_depth=int(args.lookahead_depth),
                reg=float(args.reg),
                max_branching=int(args.max_branching),
                min_samples_leaf=int(args.min_samples_leaf),
                min_child_size=int(args.min_child_size),
                max_bins=int(args.max_bins),
                lgb_num_threads=int(args.lgb_num_threads),
                test_size=float(args.test_size),
                val_size=float(args.val_size),
                out_dir=out_dir,
                msplit_min_child_size=(
                    int(args.msplit_min_child_size)
                    if args.msplit_min_child_size is not None
                    else int(args.min_child_size)
                ),
                msplit_min_split_size=(
                    int(args.msplit_min_split_size)
                    if args.msplit_min_split_size is not None
                    else None
                ),
            )
            long_rows.append({**row_base, **linear_row})

            print(f"[run] dataset={dataset} depth={depth} algorithm=msplit_nonlinear", flush=True)
            nonlinear_row = _run_msplit_variant(
                dataset=dataset,
                depth=depth,
                variant="msplit_nonlinear",
                build_dir=str(args.nonlinear_build_dir),
                cache_path=cache_path,
                lookahead_depth=int(args.lookahead_depth),
                reg=float(args.reg),
                max_branching=int(args.max_branching),
                min_samples_leaf=int(args.min_samples_leaf),
                min_child_size=int(args.min_child_size),
                max_bins=int(args.max_bins),
                lgb_num_threads=int(args.lgb_num_threads),
                test_size=float(args.test_size),
                val_size=float(args.val_size),
                out_dir=out_dir,
                msplit_min_child_size=(
                    int(args.msplit_min_child_size)
                    if args.msplit_min_child_size is not None
                    else int(args.min_child_size)
                ),
                msplit_min_split_size=(
                    int(args.msplit_min_split_size)
                    if args.msplit_min_split_size is not None
                    else None
                ),
            )
            long_rows.append({**row_base, **nonlinear_row})

            print(f"[run] dataset={dataset} depth={depth} algorithm=shapecart", flush=True)
            shapecart_row = _fit_shapecart(
                cache=cache,
                depth=int(depth),
                seed=int(args.seed),
                shape_k=int(args.shape_k),
                shape_min_samples_leaf=int(args.shape_min_samples_leaf),
                shape_min_samples_split=int(args.shape_min_samples_split),
                shape_inner_max_depth=int(args.shape_inner_max_depth),
                shape_inner_max_leaf_nodes=int(args.shape_inner_max_leaf_nodes),
                shape_max_iter=int(args.shape_max_iter),
                shape_pairwise_candidates=float(args.shape_pairwise_candidates),
                shape_smart_init=bool(args.shape_smart_init),
                shape_random_pairs=bool(args.shape_random_pairs),
                shape_use_dpdt=bool(args.shape_use_dpdt),
                shape_use_tao=bool(args.shape_use_tao),
            )
            long_rows.append({**shapecart_row, **row_base})

            print(
                (
                    f"[run] done dataset={dataset} depth={depth} "
                    f"linear_test={linear_row['test_accuracy']:.4f} "
                    f"nonlinear_test={nonlinear_row['test_accuracy']:.4f} "
                    f"shape_test={shapecart_row['test_accuracy']:.4f}"
                ),
                flush=True,
            )

    long_df = pd.DataFrame(long_rows).sort_values(["dataset", "depth_budget", "algorithm"]).reset_index(drop=True)
    long_csv = out_dir / "seed_results.csv"
    long_df.to_csv(long_csv, index=False)

    summary_df = pd.MultiIndex.from_product([datasets, depths], names=["dataset", "depth_budget"]).to_frame(index=False)
    for algorithm in ALGORITHMS:
        sub = long_df[long_df["algorithm"] == algorithm].copy()
        rename_map = {
            "train_accuracy": f"{algorithm}_train_accuracy",
            "test_accuracy": f"{algorithm}_test_accuracy",
            "fit_time_sec": f"{algorithm}_fit_time_sec",
            "n_internal_nodes": f"{algorithm}_n_internal_nodes",
            "n_leaves": f"{algorithm}_n_leaves",
        }
        sub = sub[["dataset", "depth_budget", *rename_map.keys()]].rename(columns=rename_map)
        summary_df = summary_df.merge(sub, on=["dataset", "depth_budget"], how="left")
    summary_df = summary_df.sort_values(["dataset", "depth_budget"]).reset_index(drop=True)
    summary_csv = out_dir / "summary_by_depth.csv"
    summary_df.to_csv(summary_csv, index=False)

    if not args.no_plots:
        _plot_accuracy_vs_depth(summary_df, datasets=datasets, depths=depths, metric="test", out_path=out_dir / "accuracy_vs_depth_test.png")
        _plot_accuracy_vs_depth(summary_df, datasets=datasets, depths=depths, metric="train", out_path=out_dir / "accuracy_vs_depth_train.png")

    timing_payload = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_name": run_name,
        "run_dir": str(out_dir),
        "config": config,
        "total_seconds": float(time.perf_counter() - t_total),
    }
    _write_json(out_dir / "timing_profile.json", timing_payload)

    print("[done] saved:")
    print(f"- {long_csv}")
    print(f"- {summary_csv}")
    if not args.no_plots:
        print(f"- {out_dir / 'accuracy_vs_depth_test.png'}")
        print(f"- {out_dir / 'accuracy_vs_depth_train.png'}")
    print(f"- {out_dir / 'run_config.json'}")
    print(f"- {out_dir / 'timing_profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
