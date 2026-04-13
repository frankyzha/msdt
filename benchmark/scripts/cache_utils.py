"""Shared LightGBM cache helpers for benchmark datasets and scripts."""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np

from benchmark.scripts.benchmark_paths import BENCHMARK_CACHE_ROOT
from benchmark.scripts.experiment_utils import DATASET_LOADERS, encode_target, make_preprocessor
from benchmark.scripts.lightgbm_binning import fit_lightgbm_binner, serialize_lightgbm_binner


DEFAULT_CACHE_VERSION = 6
DEFAULT_CACHE_SEED = 0
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1
DEFAULT_MAX_BINS = 1024
DEFAULT_MIN_SAMPLES_LEAF = 4
DEFAULT_MIN_CHILD_SIZE = 4
DEFAULT_LGB_NUM_THREADS = 6

DEFAULT_LIGHTGBM_BINNING_KWARGS = {
    "n_estimators": 10000,
    "num_leaves": 255,
    "learning_rate": 0.05,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_data_in_bin": 1,
    "lambda_l2": 0.0,
    "early_stopping_rounds": 100,
    "device_type": "cpu",
    "collect_teacher_logit": True,
}

CACHE_REQUIRED_KEYS = {
    "idx_fit",
    "idx_val",
    "idx_test",
    "X_fit_proc",
    "X_val_proc",
    "X_test_proc",
    "y_fit",
    "y_val",
    "y_test",
    "Z_fit",
    "Z_val",
    "Z_test",
    "teacher_logit",
    "teacher_boundary_gain",
    "teacher_boundary_cover",
    "teacher_boundary_value_jump",
    "feature_names",
    "class_labels",
    "binner_bin_edges_lengths",
    "binner_bin_edges_flat",
    "binner_fill_values",
}


def _slice_rows(x, idx: np.ndarray):
    if hasattr(x, "iloc"):
        return x.iloc[idx]
    return x[idx]


def _protocol_split_indices(
    y_encoded: np.ndarray,
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import train_test_split

    total = float(test_size) + float(val_size)
    if total >= 1.0:
        raise ValueError(f"test_size + val_size must be < 1.0, got {total:.6f}")

    all_idx = np.arange(y_encoded.shape[0], dtype=np.int32)
    train_size = 1.0 - float(test_size) - float(val_size)
    idx_fit, idx_holdout = train_test_split(
        all_idx,
        train_size=float(train_size),
        random_state=int(seed),
        stratify=y_encoded,
    )
    y_holdout = y_encoded[idx_holdout]
    holdout_size = float(test_size) + float(val_size)
    val_fraction_within_holdout = float(val_size) / holdout_size
    idx_val, idx_test = train_test_split(
        idx_holdout,
        train_size=float(val_fraction_within_holdout),
        random_state=int(seed) + 1,
        stratify=y_holdout,
    )
    return {
        "idx_fit": np.asarray(idx_fit, dtype=np.int32),
        "idx_val": np.asarray(idx_val, dtype=np.int32),
        "idx_test": np.asarray(idx_test, dtype=np.int32),
    }


def default_cache_path(
    dataset: str,
    seed: int = DEFAULT_CACHE_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    max_bins: int = DEFAULT_MAX_BINS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    min_child_size: int = DEFAULT_MIN_CHILD_SIZE,
    cache_version: int = DEFAULT_CACHE_VERSION,
    cache_root: str | Path = BENCHMARK_CACHE_ROOT,
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
    return Path(cache_root) / f"{stem}.npz"


def _cache_stem_parts(stem: str) -> tuple[str, int | None]:
    match = re.match(r"^(?P<base>.+)_v(?P<version>\d+)$", stem)
    if match is None:
        return stem, None
    return match.group("base"), int(match.group("version"))


def compatible_cache_candidates(requested_path: Path) -> list[Path]:
    requested = requested_path.resolve()
    parent = requested.parent
    base_stem, _ = _cache_stem_parts(requested.stem)
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    add_candidate(requested)
    versioned = []
    for path in parent.glob(f"{base_stem}_v*.npz"):
        _, version = _cache_stem_parts(path.stem)
        versioned.append((version if version is not None else -1, path))
    for _, path in sorted(versioned, key=lambda item: item[0], reverse=True):
        add_candidate(path)
    add_candidate(parent / f"{base_stem}.npz")
    return candidates


def cache_is_complete(cache: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    missing = sorted(key for key in CACHE_REQUIRED_KEYS if key not in cache)
    return (len(missing) == 0), missing


def load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def resolve_compatible_cache(
    requested_path: Path,
    *,
    force_rebuild: bool,
) -> tuple[Path, dict[str, np.ndarray], dict[str, object], bool, bool]:
    if force_rebuild:
        return requested_path, {}, {}, False, False

    for candidate in compatible_cache_candidates(requested_path):
        if not candidate.exists():
            continue
        print(f"[cache] probing: {candidate}", flush=True)
        cache = load_cache(candidate)
        cache_ok, missing = cache_is_complete(cache)
        if not cache_ok:
            print(f"[cache] stale cache skipped (missing: {missing})", flush=True)
            continue
        meta_path = candidate.with_suffix(".json")
        cache_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        used_fallback = candidate.resolve() != requested_path.resolve()
        print(
            f"[cache] {'compatible fallback hit' if used_fallback else 'hit'}: {candidate}",
            flush=True,
        )
        return candidate, cache, cache_meta, True, used_fallback

    return requested_path, {}, {}, False, False


def derive_min_child_size(*, leaf_frac: float, n_fit: int) -> int:
    frac = float(leaf_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"leaf_frac must be positive and finite, got {leaf_frac!r}")
    return max(2, int(math.ceil(frac * max(1, int(n_fit)))))


def derive_min_split_size(*, leaf_frac: float, n_fit: int) -> int:
    frac = float(leaf_frac)
    if not np.isfinite(frac) or frac <= 0.0:
        raise ValueError(f"leaf_frac must be positive and finite, got {leaf_frac!r}")
    return max(2, int(math.ceil((2.0 * frac) * max(1, int(n_fit)))))


def resolve_protocol_support_sizes(
    *,
    dataset: str | None = None,
    y_encoded: np.ndarray | None = None,
    seed: int = DEFAULT_CACHE_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    leaf_frac: float = 0.001,
    min_child_size: int = 0,
    min_split_size: int = 0,
) -> tuple[int | None, int, int]:
    resolved_min_child_size = int(min_child_size)
    resolved_min_split_size = int(min_split_size)
    if resolved_min_child_size > 0 and resolved_min_split_size > 0:
        return None, resolved_min_child_size, resolved_min_split_size

    if y_encoded is None:
        if dataset is None:
            raise ValueError("dataset must be provided when y_encoded is not available")
        _, y = DATASET_LOADERS[dataset]()
        y_encoded, _, _ = encode_target(y)

    split_idx = _protocol_split_indices(
        y_encoded=np.asarray(y_encoded, dtype=np.int32),
        seed=int(seed),
        test_size=float(test_size),
        val_size=float(val_size),
    )
    n_fit = int(split_idx["idx_fit"].shape[0])
    if resolved_min_child_size <= 0:
        resolved_min_child_size = derive_min_child_size(leaf_frac=float(leaf_frac), n_fit=n_fit)
    if resolved_min_split_size <= 0:
        resolved_min_split_size = derive_min_split_size(leaf_frac=float(leaf_frac), n_fit=n_fit)
    return n_fit, resolved_min_child_size, resolved_min_split_size


def build_cache(
    *,
    dataset: str,
    seed: int = DEFAULT_CACHE_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    max_bins: int = DEFAULT_MAX_BINS,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    min_child_size: int = DEFAULT_MIN_CHILD_SIZE,
    lgb_num_threads: int = DEFAULT_LGB_NUM_THREADS,
    cache_path: Path | None = None,
) -> dict[str, np.ndarray]:
    print(f"[cache] loading dataset={dataset} seed={seed}", flush=True)
    X, y = DATASET_LOADERS[dataset]()
    y_encoded, class_labels, _ = encode_target(y)
    split_idx = _protocol_split_indices(
        y_encoded=np.asarray(y_encoded, dtype=np.int32),
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
    y_fit = y_encoded[idx_fit]
    y_val = y_encoded[idx_val]
    y_test = y_encoded[idx_test]

    pre = make_preprocessor(X_fit)
    X_fit_proc = np.asarray(pre.fit_transform(X_fit), dtype=np.float32)
    X_val_proc = np.asarray(pre.transform(X_val), dtype=np.float32)
    X_test_proc = np.asarray(pre.transform(X_test), dtype=np.float32)
    feature_names = np.asarray(pre.get_feature_names_out(), dtype=str)

    print("[cache] fitting LightGBM binner", flush=True)

    def _progress(message: str) -> None:
        print(f"[cache] {message}", flush=True)

    started = time.perf_counter()
    binner = fit_lightgbm_binner(
        X_fit_proc,
        y_fit,
        X_val=X_val_proc,
        y_val=y_val,
        max_bins=int(max_bins),
        min_samples_leaf=int(min_samples_leaf),
        random_state=int(seed),
        min_data_in_leaf=int(min_samples_leaf),
        num_threads=int(lgb_num_threads),
        progress_callback=_progress,
        **DEFAULT_LIGHTGBM_BINNING_KWARGS,
    )
    build_seconds = time.perf_counter() - started
    print(f"[cache] binner done in {build_seconds:.3f}s", flush=True)

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
        "Z_val": np.asarray(binner.transform(X_val_proc), dtype=np.int32),
        "Z_test": np.asarray(binner.transform(X_test_proc), dtype=np.int32),
        "teacher_logit": np.asarray(getattr(binner, "teacher_train_logit"), dtype=np.float64),
        "teacher_boundary_gain": np.asarray(getattr(binner, "boundary_gain_per_feature"), dtype=np.float64),
        "teacher_boundary_cover": np.asarray(getattr(binner, "boundary_cover_per_feature"), dtype=np.float64),
        "teacher_boundary_value_jump": np.asarray(
            getattr(binner, "boundary_value_jump_per_feature"), dtype=np.float64
        ),
        "feature_names": feature_names,
        "class_labels": np.asarray([str(label) for label in class_labels], dtype=str),
    }
    arrays.update(serialize_lightgbm_binner(binner))

    cache_path = cache_path or default_cache_path(
        dataset=dataset,
        seed=seed,
        test_size=test_size,
        val_size=val_size,
        max_bins=max_bins,
        min_samples_leaf=min_samples_leaf,
        min_child_size=min_child_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    meta = {
        "dataset": dataset,
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "max_bins": int(max_bins),
        "min_samples_leaf": int(min_samples_leaf),
        "min_child_size": int(min_child_size),
        "lgb_num_threads": int(lgb_num_threads),
        "build_seconds": float(build_seconds),
        "cache_file": cache_path.name,
        "cache_bytes": int(cache_path.stat().st_size),
        "n_rows": int(X_fit_proc.shape[0] + X_val_proc.shape[0] + X_test_proc.shape[0]),
        "n_fit": int(X_fit_proc.shape[0]),
        "n_val": int(X_val_proc.shape[0]),
        "n_test": int(X_test_proc.shape[0]),
        "n_features_preprocessed": int(X_fit_proc.shape[1]),
        "class_count": int(len(class_labels)),
        "class_labels": [str(label) for label in class_labels],
    }
    cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return arrays
