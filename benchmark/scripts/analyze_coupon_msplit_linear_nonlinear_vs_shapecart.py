#!/usr/bin/env python3
"""Compare cached coupon variants across MSPLIT linear/nonlinear and ShapeCART."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.benchmark_paths import (
    BENCHMARK_ARTIFACTS_ROOT,
    BENCHMARK_CACHE_ROOT,
    BENCHMARK_DATA_ROOT,
    SCRIPT_ROOT,
    ensure_repo_import_paths,
)

ensure_repo_import_paths(include_shapecart=True)

from benchmark.scripts.cache_utils import _protocol_split_indices, _slice_rows
from benchmark.scripts.experiment_utils import encode_binary_target, make_preprocessor
from benchmark.scripts.lightgbm_binning import fit_lightgbm_binner
from benchmark.scripts.runtime_guard import guarded_fit
from src.ShapeCARTClassifier import ShapeCARTClassifier


CSV_PATH = BENCHMARK_DATA_ROOT / "coupon" / "coupon_source.csv"
VARIANTS = (
    ("coupon_overall", "Coupon overall", None),
    ("coupon_bar", "Coupon (Bar)", "Bar"),
    ("coupon_carryout", "Coupon (Carry Out & Take Away)", "Carry out & Take away"),
    ("coupon_coffeehouse", "Coupon (Coffee House)", "Coffee House"),
    ("coupon_rest20to50", "Coupon (Restaurant, 20-50)", "Restaurant(20-50)"),
    ("coupon_restlt20", "Coupon (Restaurant, <20)", "Restaurant(<20)"),
)


def _load_clean_coupon_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)
    drop_cols = [col for col in ("click", "car") if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.dropna().reset_index(drop=True)


def _build_variants(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, str | None]]:
    variants: list[tuple[str, str, pd.DataFrame, str | None]] = []
    for slug, label, coupon_value in VARIANTS:
        if coupon_value is None:
            variants.append((slug, label, df.copy(), None))
        else:
            sub = df[df["coupon"] == coupon_value].copy().reset_index(drop=True)
            variants.append((slug, label, sub, coupon_value))
    return variants


def _tree_depth_shapecart(model: ShapeCARTClassifier, node_idx: int = 0) -> int:
    if model.is_leaf[node_idx]:
        return 0
    children = model.children[node_idx] or []
    if not children:
        return 0
    return 1 + max(_tree_depth_shapecart(model, child) for child in children)


def _tree_stats_shapecart(model: ShapeCARTClassifier) -> dict[str, int]:
    n_leaves = int(sum(bool(v) for v in model.is_leaf))
    n_internal = int(len(model.is_leaf) - n_leaves)
    max_arity = 0
    for children in model.children:
        if children:
            max_arity = max(max_arity, len(children))
    return {
        "n_leaves": n_leaves,
        "n_internal": n_internal,
        "max_arity": int(max_arity),
    }


def _node_prediction(model: ShapeCARTClassifier, node_idx: int) -> int:
    values = np.asarray(model.values[node_idx], dtype=float)
    if values.size == 0:
        return 0
    return int(np.argmax(values))


def _serialize_shapecart_node(model: ShapeCARTClassifier, node_idx: int = 0) -> dict[str, object]:
    payload = {
        "node_idx": int(node_idx),
        "depth": int(model.depths[node_idx]),
        "n_samples": int(model.n_samples[node_idx]),
        "prediction": int(_node_prediction(model, node_idx)),
        "class_distribution": np.asarray(model.values[node_idx], dtype=float).tolist(),
        "is_leaf": bool(model.is_leaf[node_idx]),
    }
    if model.is_leaf[node_idx]:
        return payload

    node = model.nodes[node_idx]
    final_key = None if node is None else node.final_key
    payload["feature_key"] = (
        [int(v) if isinstance(v, (np.integer, int)) else str(v) for v in final_key]
        if isinstance(final_key, tuple)
        else (None if final_key is None else int(final_key) if isinstance(final_key, (np.integer, int)) else str(final_key))
    )
    children = model.children[node_idx] or []
    payload["children"] = [_serialize_shapecart_node(model, child) for child in children]
    return payload


def _fit_shapecart(
    *,
    x_fit_proc: np.ndarray,
    y_fit: np.ndarray,
    x_test_proc: np.ndarray,
    y_test: np.ndarray,
    depth: int,
    seed: int,
    min_samples_leaf: int,
    min_samples_split: int,
    k: int,
    use_tao: bool,
) -> dict[str, object]:
    def _run_once() -> ShapeCARTClassifier:
        model = ShapeCARTClassifier(
            max_depth=int(depth),
            min_samples_leaf=int(min_samples_leaf),
            min_samples_split=int(min_samples_split),
            inner_min_samples_leaf=int(min_samples_leaf),
            inner_min_samples_split=int(min_samples_split),
            inner_max_depth=6,
            inner_max_leaf_nodes=32,
            max_iter=20,
            k=int(k),
            branching_penalty=0.0,
            random_state=int(seed),
            verbose=False,
            pairwise_candidates=0.0,
            smart_init=True,
            random_pairs=False,
            use_dpdt=False,
            use_tao=bool(use_tao),
        )
        model.fit(x_fit_proc, y_fit)
        return model

    model, fit_seconds, timing_guard = guarded_fit(_run_once, repo_root=REPO_ROOT)
    pred_train = np.asarray(model.predict(x_fit_proc), dtype=np.int32)
    pred_test = np.asarray(model.predict(x_test_proc), dtype=np.int32)
    stats = _tree_stats_shapecart(model)
    return {
        "fit_seconds": float(fit_seconds),
        "train_accuracy": float(np.mean(pred_train == y_fit)),
        "test_accuracy": float(np.mean(pred_test == y_test)),
        "root_feature_key": (
            None
            if model.nodes[0] is None
            else model.nodes[0].final_key
        ),
        "tree_depth": int(_tree_depth_shapecart(model, 0)),
        "n_leaves": int(stats["n_leaves"]),
        "n_internal": int(stats["n_internal"]),
        "max_arity": int(stats["max_arity"]),
        "timing_guard": timing_guard,
        "tree": _serialize_shapecart_node(model, 0),
    }


def _prepare_variant_cache(
    *,
    variant_df: pd.DataFrame,
    coupon_value: str | None,
    seed: int,
    test_size: float,
    val_size: float,
    max_bins: int,
    min_samples_leaf: int,
    lgb_num_threads: int,
    cache_path: Path,
) -> dict[str, object]:
    y = encode_binary_target(variant_df["Y"].to_numpy(), "coupon").astype(np.int32)
    if coupon_value is None:
        X = variant_df.drop(columns=["Y"])
    else:
        X = variant_df.drop(columns=["coupon", "Y"])

    split_idx = _protocol_split_indices(
        y_encoded=y,
        seed=int(seed),
        test_size=float(test_size),
        val_size=float(val_size),
    )
    idx_fit = split_idx["idx_fit"]
    idx_val = split_idx["idx_val"]
    idx_test = split_idx["idx_test"]

    X_fit = _slice_rows(X, idx_fit)
    X_val = _slice_rows(X, idx_val)
    X_test = _slice_rows(X, idx_test)
    y_fit = np.asarray(y[idx_fit], dtype=np.int32)
    y_val = np.asarray(y[idx_val], dtype=np.int32)
    y_test = np.asarray(y[idx_test], dtype=np.int32)

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.asarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.asarray(pre.transform(X_val), dtype=np.float32)
    X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)

    started = time.perf_counter()
    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=int(max_bins),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        n_estimators=10000,
        num_leaves=255,
        learning_rate=0.05,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        min_data_in_bin=1,
        min_data_in_leaf=int(min_samples_leaf),
        lambda_l2=0.0,
        early_stopping_rounds=100,
        num_threads=int(lgb_num_threads),
        device_type="cpu",
        collect_teacher_logit=True,
    )
    binner_fit_seconds = time.perf_counter() - started

    arrays = {
        "idx_fit": np.asarray(idx_fit, dtype=np.int32),
        "idx_val": np.asarray(idx_val, dtype=np.int32),
        "idx_test": np.asarray(idx_test, dtype=np.int32),
        "X_fit_proc": np.asarray(X_fit_proc, dtype=np.float32),
        "X_val_proc": np.asarray(X_val_proc, dtype=np.float32),
        "X_test_proc": np.asarray(X_test_proc, dtype=np.float32),
        "y_fit": np.asarray(y_fit, dtype=np.int32),
        "y_val": np.asarray(y_val, dtype=np.int32),
        "y_test": np.asarray(y_test, dtype=np.int32),
        "Z_fit": np.asarray(binner.transform(X_fit_proc), dtype=np.int32),
        "Z_test": np.asarray(binner.transform(X_test_proc), dtype=np.int32),
        "teacher_logit": np.asarray(getattr(binner, "teacher_train_logit"), dtype=np.float64),
        "teacher_boundary_gain": np.asarray(getattr(binner, "boundary_gain_per_feature"), dtype=np.float64),
        "teacher_boundary_cover": np.asarray(getattr(binner, "boundary_cover_per_feature"), dtype=np.float64),
        "teacher_boundary_value_jump": np.asarray(
            getattr(binner, "boundary_value_jump_per_feature"), dtype=np.float64
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return {
        "cache_path": str(cache_path),
        "binner_fit_seconds": float(binner_fit_seconds),
        "n_total": int(len(variant_df)),
        "n_fit": int(len(idx_fit)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "positive_rate": float(np.mean(y)),
        "fit_positive_rate": float(np.mean(y_fit)),
        "test_positive_rate": float(np.mean(y_test)),
        "n_features_preprocessed": int(X_fit_proc.shape[1]),
        "x_fit_proc": X_fit_proc,
        "x_test_proc": X_test_proc,
        "y_fit": y_fit,
        "y_test": y_test,
    }


def _run_msplit_worker(
    *,
    cache_path: Path,
    build_dir: str,
    depth: int,
    lookahead_depth: int,
    reg: float,
    min_split_size: int,
    min_child_size: int,
    max_branching: int,
    exactify_top_k: int,
    json_path: Path,
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(SCRIPT_ROOT / "run_cached_msplit_worker.py"),
        "--cache-path",
        str(cache_path),
        "--build-dir",
        str(build_dir),
        "--depth",
        str(int(depth)),
        "--lookahead-depth",
        str(int(lookahead_depth)),
        "--reg",
        str(float(reg)),
        "--min-split-size",
        str(int(min_split_size)),
        "--min-child-size",
        str(int(min_child_size)),
        "--max-branching",
        str(int(max_branching)),
        "--exactify-top-k",
        str(int(exactify_top_k)),
        "--json",
        str(json_path),
    ]
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "MSPLIT worker failed.\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ShapeCART, msplit_linear, and msplit_nonlinear on the 6 coupon variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--max-bins", type=int, default=1024)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--min-child-size", type=int, default=4)
    parser.add_argument("--min-split-size", type=int, default=8)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--reg", type=float, default=0.0)
    parser.add_argument("--max-branching", type=int, default=3)
    parser.add_argument("--shape-k", type=int, default=3)
    parser.add_argument("--shape-min-samples-leaf", type=int, default=4)
    parser.add_argument("--shape-min-samples-split", type=int, default=8)
    parser.add_argument("--shape-use-tao", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--linear-build-dir", type=str, default="build-linear-py")
    parser.add_argument("--nonlinear-build-dir", type=str, default="build-nonlinear-py")
    parser.add_argument("--compression-rule", type=str, default=None)
    parser.add_argument("--family-mode", type=str, default=None)
    parser.add_argument("--results-root", type=Path, default=BENCHMARK_ARTIFACTS_ROOT / "coupon_gap_analysis")
    parser.add_argument("--run-name", type=str, default="current_coupon_gap")
    parser.add_argument("--lgb-num-threads", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = args.results_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = BENCHMARK_CACHE_ROOT / "coupon_gap_analysis" / args.run_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": int(args.seed),
        "test_size": float(args.test_size),
        "val_size": float(args.val_size),
        "depths": [int(v) for v in args.depths],
        "max_bins": int(args.max_bins),
        "min_samples_leaf": int(args.min_samples_leaf),
        "min_child_size": int(args.min_child_size),
        "min_split_size": int(args.min_split_size),
        "lookahead_depth": int(args.lookahead_depth),
        "reg": float(args.reg),
        "max_branching": int(args.max_branching),
        "shape_k": int(args.shape_k),
        "shape_min_samples_leaf": int(args.shape_min_samples_leaf),
        "shape_min_samples_split": int(args.shape_min_samples_split),
        "shape_use_tao": bool(args.shape_use_tao),
        "linear_build_dir": str(args.linear_build_dir),
        "nonlinear_build_dir": str(args.nonlinear_build_dir),
        "compression_rule": args.compression_rule,
        "family_mode": args.family_mode,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    df = _load_clean_coupon_df()
    variants = _build_variants(df)
    prepared_variants: dict[str, dict[str, object]] = {}

    for slug, label, variant_df, coupon_value in variants:
        print(f"[prep] {label}", flush=True)
        cache_path = cache_dir / f"{slug}.npz"
        prepared = _prepare_variant_cache(
            variant_df=variant_df,
            coupon_value=coupon_value,
            seed=int(args.seed),
            test_size=float(args.test_size),
            val_size=float(args.val_size),
            max_bins=int(args.max_bins),
            min_samples_leaf=int(args.min_samples_leaf),
            lgb_num_threads=int(args.lgb_num_threads),
            cache_path=cache_path,
        )
        prepared_variants[slug] = {
            "slug": slug,
            "label": label,
            "coupon_value": coupon_value,
            **prepared,
        }

    summary_rows: list[dict[str, object]] = []
    results_payload: dict[str, object] = {
        "config": config,
        "datasets": {},
    }

    extra_env: dict[str, str] = {}
    if args.compression_rule is not None:
        extra_env["MSPLIT_ATOM_COMPRESSION_RULE"] = str(args.compression_rule)
    if args.family_mode is not None:
        extra_env["MSPLIT_ATOM_FAMILY_MODE"] = str(args.family_mode)

    for slug, label, _, _ in variants:
        prepared = prepared_variants[slug]
        dataset_payload = {
            "label": label,
            "meta": {
                key: value
                for key, value in prepared.items()
                if key
                in {
                    "coupon_value",
                    "cache_path",
                    "binner_fit_seconds",
                    "n_total",
                    "n_fit",
                    "n_val",
                    "n_test",
                    "positive_rate",
                    "fit_positive_rate",
                    "test_positive_rate",
                    "n_features_preprocessed",
                }
            },
            "depths": {},
        }

        x_fit_proc = np.asarray(prepared["x_fit_proc"], dtype=np.float32)
        x_test_proc = np.asarray(prepared["x_test_proc"], dtype=np.float32)
        y_fit = np.asarray(prepared["y_fit"], dtype=np.int32)
        y_test = np.asarray(prepared["y_test"], dtype=np.int32)
        cache_path = Path(str(prepared["cache_path"]))

        for depth in args.depths:
            lookahead = min(int(depth), int(args.lookahead_depth))
            print(f"[run] {label} depth={depth}", flush=True)
            linear_json = raw_dir / f"{slug}_depth{depth}_linear.json"
            nonlinear_json = raw_dir / f"{slug}_depth{depth}_nonlinear.json"

            linear = _run_msplit_worker(
                cache_path=cache_path,
                build_dir=str(args.linear_build_dir),
                depth=int(depth),
                lookahead_depth=lookahead,
                reg=float(args.reg),
                min_split_size=int(args.min_split_size),
                min_child_size=int(args.min_child_size),
                max_branching=int(args.max_branching),
                exactify_top_k=0,
                json_path=linear_json,
                extra_env=extra_env or None,
            )
            nonlinear = _run_msplit_worker(
                cache_path=cache_path,
                build_dir=str(args.nonlinear_build_dir),
                depth=int(depth),
                lookahead_depth=lookahead,
                reg=float(args.reg),
                min_split_size=int(args.min_split_size),
                min_child_size=int(args.min_child_size),
                max_branching=int(args.max_branching),
                exactify_top_k=0,
                json_path=nonlinear_json,
                extra_env=extra_env or None,
            )
            shapecart = _fit_shapecart(
                x_fit_proc=x_fit_proc,
                y_fit=y_fit,
                x_test_proc=x_test_proc,
                y_test=y_test,
                depth=int(depth),
                seed=int(args.seed),
                min_samples_leaf=int(args.shape_min_samples_leaf),
                min_samples_split=int(args.shape_min_samples_split),
                k=int(args.shape_k),
                use_tao=bool(args.shape_use_tao),
            )

            dataset_payload["depths"][str(depth)] = {
                "msplit_linear": linear,
                "msplit_nonlinear": nonlinear,
                "shapecart": shapecart,
            }

            for algorithm, payload in (
                ("msplit_linear", linear),
                ("msplit_nonlinear", nonlinear),
                ("shapecart", shapecart),
            ):
                summary_rows.append(
                    {
                        "dataset": label,
                        "slug": slug,
                        "depth": int(depth),
                        "algorithm": algorithm,
                        "train_accuracy": float(payload["train_accuracy"]),
                        "test_accuracy": float(payload["test_accuracy"]),
                        "generalization_gap": float(payload["train_accuracy"]) - float(payload["test_accuracy"]),
                        "fit_seconds": float(payload["fit_seconds"]),
                        "n_internal": int(payload["n_internal"]),
                        "n_leaves": int(payload["n_leaves"]),
                        "max_arity": int(payload["max_arity"]),
                        "tree_depth": int(payload["tree_depth"]),
                        "root_feature": payload.get("root_feature", payload.get("root_feature_key")),
                        "root_group_count": payload.get("root_group_count"),
                        "root_has_noncontiguous_group": payload.get("root_has_noncontiguous_group"),
                        "atomized_coarse_candidates": payload.get("atomized_coarse_candidates"),
                        "atomized_coarse_pruned_candidates": payload.get("atomized_coarse_pruned_candidates"),
                        "atomized_final_candidates": payload.get("atomized_final_candidates"),
                        "debr_refine_calls": payload.get("debr_refine_calls"),
                        "family_compare_total": payload.get("family_compare_total"),
                        "family_sent_both": payload.get("family_sent_both"),
                        "heuristic_selector_candidate_total": payload.get("heuristic_selector_candidate_total"),
                        "heuristic_selector_candidate_pruned_total": payload.get(
                            "heuristic_selector_candidate_pruned_total"
                        ),
                    }
                )

            print(
                (
                    f"  linear train/test={linear['train_accuracy']:.4f}/{linear['test_accuracy']:.4f} "
                    f"nonlinear train/test={nonlinear['train_accuracy']:.4f}/{nonlinear['test_accuracy']:.4f} "
                    f"shape train/test={shapecart['train_accuracy']:.4f}/{shapecart['test_accuracy']:.4f}"
                ),
                flush=True,
            )

        results_payload["datasets"][slug] = dataset_payload

    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "depth", "algorithm"]).reset_index(drop=True)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[done] wrote {out_dir / 'summary.csv'}", flush=True)
    print(f"[done] wrote {out_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
