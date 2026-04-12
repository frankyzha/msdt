#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import optuna
except Exception as exc:  # pragma: no cover - CLI guard
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_teacher_guided_atomcolor_cached import (
    _protocol_split_indices,
    build_cache,
    default_cache_path,
    derive_min_child_size,
    derive_min_split_size,
    resolve_compatible_cache,
    run_cached_msplit,
)
from experiment_utils import DATASET_LOADERS, encode_binary_target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune MSPLIT regularization with Optuna on the cached benchmark protocol."
    )
    parser.add_argument("--dataset", default="electricity")
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, required=True)
    parser.add_argument("--max-bins", type=int, default=1024)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=4)
    parser.add_argument("--min-split-size", type=int, default=8)
    parser.add_argument("--leaf-frac", type=float, default=0.001)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--reg-min", type=float, default=1e-6)
    parser.add_argument("--reg-max", type=float, default=5e-4)
    parser.add_argument("--family-mode", choices=("single", "dual"), default="single")
    parser.add_argument("--lgb-num-threads", type=int, default=3)
    parser.add_argument("--exactify-top-k", type=int, default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if optuna is None:
        raise RuntimeError("Optuna is not installed in this environment.") from _OPTUNA_IMPORT_ERROR
    if int(args.depth) < 1:
        raise ValueError("--depth must be at least 1")
    if float(args.reg_min) <= 0.0 or float(args.reg_max) <= 0.0:
        raise ValueError("regularization bounds must be positive")
    if float(args.reg_min) >= float(args.reg_max):
        raise ValueError("--reg-min must be smaller than --reg-max")
    if args.exactify_top_k is not None and int(args.exactify_top_k) < 1:
        raise ValueError("--exactify-top-k must be a positive integer when specified")
    return args


def _configure_family_mode(mode: str) -> None:
    if mode == "dual":
        os.environ["MSPLIT_ATOM_FAMILY_MODE"] = "dual"
    else:
        os.environ["MSPLIT_ATOM_FAMILY_MODE"] = "single"


def _load_or_build_cache(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], Path, dict[str, object], bool, bool]:
    preview_X, preview_y = DATASET_LOADERS[args.dataset]()
    preview_y_bin = encode_binary_target(preview_y, args.dataset)
    split_idx = _protocol_split_indices(
        y_bin=np.asarray(preview_y_bin, dtype=np.int32),
        seed=int(args.seed),
        test_size=float(args.test_size),
        val_size=float(args.val_size),
    )
    n_fit = int(split_idx["idx_fit"].shape[0])
    resolved_min_child_size = int(args.min_child_size)
    if resolved_min_child_size <= 0:
        resolved_min_child_size = derive_min_child_size(leaf_frac=float(args.leaf_frac), n_fit=n_fit)
    resolved_min_split_size = int(args.min_split_size)
    if resolved_min_split_size <= 0:
        resolved_min_split_size = derive_min_split_size(leaf_frac=float(args.leaf_frac), n_fit=n_fit)

    cache_path = default_cache_path(
        args.dataset,
        args.seed,
        args.test_size,
        args.val_size,
        args.max_bins,
        args.min_samples_leaf,
        resolved_min_child_size,
    )

    cache_resolved_path, cache, cache_meta, cache_hit, cache_used_fallback = resolve_compatible_cache(
        cache_path,
        force_rebuild=bool(args.force_rebuild_cache),
    )
    needs_val_bins = "Z_val" not in cache or "y_val" not in cache
    if cache_hit and not needs_val_bins:
        return cache, cache_resolved_path, cache_meta, cache_hit, cache_used_fallback

    if cache_hit and needs_val_bins:
        print(f"[cache] rebuilding because validation bins are missing in {cache_resolved_path}", flush=True)
    else:
        print(f"[cache] miss: {cache_path}", flush=True)

    cache_resolved_path = cache_path
    cache = build_cache(
        dataset=args.dataset,
        depth=args.depth,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        max_bins=args.max_bins,
        min_samples_leaf=args.min_samples_leaf,
        min_child_size=resolved_min_child_size,
        lgb_num_threads=args.lgb_num_threads,
        cache_path=cache_resolved_path,
    )
    cache_meta_path = cache_resolved_path.with_suffix(".json")
    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8")) if cache_meta_path.exists() else {}
    return cache, cache_resolved_path, cache_meta, False, False


def main() -> int:
    args = _parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _configure_family_mode(args.family_mode)

    cache, cache_path, cache_meta, cache_hit, cache_used_fallback = _load_or_build_cache(args)

    def _objective(trial) -> float:
        reg = trial.suggest_float("regularization", float(args.reg_min), float(args.reg_max), log=True)
        result = run_cached_msplit(
            cache=cache,
            depth=int(args.depth),
            lookahead_depth=int(args.lookahead_depth),
            reg=float(reg),
            exactify_top_k=args.exactify_top_k,
            min_split_size=int(args.min_split_size),
            min_child_size=int(args.min_child_size),
            max_branching=int(args.max_branching),
        )
        val_accuracy = result.get("val_accuracy")
        if val_accuracy is None:
            raise RuntimeError("Validation accuracy is unavailable; cache must include Z_val and y_val.")
        trial.set_user_attr("test_accuracy", float(result["test_accuracy"]))
        trial.set_user_attr("train_accuracy", float(result["train_accuracy"]))
        trial.set_user_attr("fit_seconds", float(result["fit_seconds"]))
        trial.set_user_attr("objective", float(result["objective"]))
        return float(val_accuracy)

    started = time.perf_counter()
    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(_objective, n_trials=int(args.n_trials))
    search_seconds = time.perf_counter() - started

    best_reg = float(study.best_params["regularization"])
    best_result = run_cached_msplit(
        cache=cache,
        depth=int(args.depth),
        lookahead_depth=int(args.lookahead_depth),
        reg=best_reg,
        exactify_top_k=args.exactify_top_k,
        min_split_size=int(args.min_split_size),
        min_child_size=int(args.min_child_size),
        max_branching=int(args.max_branching),
    )

    trials = [
        {
            "number": int(trial.number),
            "value": float(trial.value) if trial.value is not None else None,
            "regularization": (
                float(trial.params["regularization"]) if "regularization" in trial.params else None
            ),
            "train_accuracy": trial.user_attrs.get("train_accuracy"),
            "test_accuracy": trial.user_attrs.get("test_accuracy"),
            "fit_seconds": trial.user_attrs.get("fit_seconds"),
            "objective": trial.user_attrs.get("objective"),
        }
        for trial in study.trials
    ]

    summary = {
        "dataset": args.dataset,
        "depth": int(args.depth),
        "lookahead_depth": int(args.lookahead_depth),
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "max_bins": int(args.max_bins),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_child_size": int(args.min_child_size),
        "min_split_size": int(args.min_split_size),
        "max_branching": int(args.max_branching),
        "n_trials": int(args.n_trials),
        "reg_min": float(args.reg_min),
        "reg_max": float(args.reg_max),
        "family_mode": args.family_mode,
        "cache_path": str(cache_path),
        "cache_hit": bool(cache_hit),
        "cache_used_compatible_fallback": bool(cache_used_fallback),
        "cache_build_seconds": float(cache_meta.get("build_seconds", 0.0)) if cache_meta else 0.0,
        "search_seconds": float(search_seconds),
        "best_regularization": float(best_reg),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_val_accuracy": float(study.best_value),
        "best_result": best_result,
        "trials": trials,
    }

    print(json.dumps(summary, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
