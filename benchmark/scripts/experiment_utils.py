"""Shared utilities for tabular depth-benchmark experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from benchmark.scripts.dataset import (
    DEFAULT_DATASETS,
    get_dataset_spec,
    load_adult,
    load_avila,
    load_bank,
    load_bean,
    load_bike_sharing,
    load_bidding,
    load_compas,
    load_coupon,
    load_electricity,
    load_eye_movements,
    load_eye_state,
    load_heloc,
    load_spambase,
)


@dataclass(frozen=True)
class SplitIndices:
    idx_train: np.ndarray
    idx_test: np.ndarray


class SimpleLabelEncoder:
    def __init__(self) -> None:
        self.classes_: np.ndarray | None = None
        self._mapping: dict[object, int] | None = None

    def fit(self, y) -> "SimpleLabelEncoder":
        y_series = pd.Series(y).reset_index(drop=True)
        classes = sorted(pd.unique(y_series).tolist(), key=lambda v: str(v))
        self.classes_ = np.asarray(classes, dtype=object)
        self._mapping = {value: idx for idx, value in enumerate(self.classes_.tolist())}
        return self

    def transform(self, y) -> np.ndarray:
        if self._mapping is None:
            raise RuntimeError("SimpleLabelEncoder must be fit before transform")
        y_values = pd.Series(y).reset_index(drop=True).tolist()
        try:
            return np.asarray([self._mapping[value] for value in y_values], dtype=np.int32)
        except KeyError as exc:
            raise ValueError("Input contains labels not seen during fit") from exc

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, y) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("SimpleLabelEncoder must be fit before inverse_transform")
        y_arr = np.asarray(y, dtype=np.int32)
        if np.any((y_arr < 0) | (y_arr >= self.classes_.shape[0])):
            raise ValueError("Encoded labels are out of range")
        return self.classes_[y_arr]


class SimpleTabularPreprocessor:
    def __init__(self) -> None:
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.numeric_medians_: dict[str, float] = {}
        self.categorical_categories_: dict[str, list[object]] = {}
        self.categorical_modes_: dict[str, object] = {}
        self.categorical_fill_codes_: dict[str, int] = {}
        self.feature_slices_: dict[str, slice] = {}
        self.feature_names_out_: list[str] = []
        self._fitted = False

    @staticmethod
    def _is_dataframe(X) -> bool:
        return hasattr(X, "select_dtypes") and hasattr(X, "columns")

    @staticmethod
    def _to_1d_float(col) -> np.ndarray:
        arr = pd.to_numeric(pd.Series(col), errors="coerce").to_numpy(dtype=float)
        return arr

    def fit(self, X) -> "SimpleTabularPreprocessor":
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.numeric_medians_ = {}
        self.categorical_categories_ = {}
        self.categorical_modes_ = {}
        self.categorical_fill_codes_ = {}
        self.feature_slices_ = {}
        self.feature_names_out_ = []

        if self._is_dataframe(X):
            df = X.reset_index(drop=True).copy()
            self.numeric_cols_ = list(df.select_dtypes(include=[np.number]).columns)
            self.categorical_cols_ = [c for c in df.columns if c not in self.numeric_cols_]
            offset = 0

            for col in self.numeric_cols_:
                values = self._to_1d_float(df[col])
                finite = values[np.isfinite(values)]
                med = float(np.median(finite)) if finite.size > 0 else 0.0
                self.numeric_medians_[str(col)] = med
                self.feature_slices_[str(col)] = slice(offset, offset + 1)
                self.feature_names_out_.append(str(col))
                offset += 1

            for col in self.categorical_cols_:
                series = pd.Series(df[col], copy=False).astype(object)
                non_missing = series[~pd.isna(series)]
                if non_missing.empty:
                    categories: list[object] = ["__missing__"]
                    mode_value: object = "__missing__"
                else:
                    categories = list(dict.fromkeys(non_missing.tolist()))
                    mode_value = non_missing.value_counts(dropna=True).index[0]
                self.categorical_categories_[str(col)] = categories
                self.categorical_modes_[str(col)] = mode_value
                self.categorical_fill_codes_[str(col)] = categories.index(mode_value)
                width = len(categories)
                self.feature_slices_[str(col)] = slice(offset, offset + width)
                self.feature_names_out_.extend(f"{col}__{cat}" for cat in categories)
                offset += width
        else:
            arr = np.asarray(X, dtype=float)
            if arr.ndim != 2:
                raise ValueError("Expected a 2D array or DataFrame")
            n_features = int(arr.shape[1])
            self.numeric_cols_ = [f"x{i}" for i in range(n_features)]
            self.categorical_cols_ = []
            self.feature_names_out_ = list(self.numeric_cols_)
            self.feature_slices_ = {col: slice(j, j + 1) for j, col in enumerate(self.numeric_cols_)}
            for j in range(n_features):
                values = arr[:, j]
                finite = values[np.isfinite(values)]
                self.numeric_medians_[self.numeric_cols_[j]] = float(np.median(finite)) if finite.size > 0 else 0.0

        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SimpleTabularPreprocessor must be fit before transform")

        if self._is_dataframe(X):
            df = X.reset_index(drop=True).copy()
            n_rows = int(df.shape[0])
            n_features = len(self.feature_names_out_)
            if n_features == 0:
                return np.zeros((n_rows, 0), dtype=np.float32)
            out = np.zeros((n_rows, n_features), dtype=np.float32)

            for col in self.numeric_cols_:
                values = self._to_1d_float(df[col])
                fill = self.numeric_medians_.get(col, 0.0)
                mask = ~np.isfinite(values)
                if np.any(mask):
                    values = values.copy()
                    values[mask] = fill
                feature_idx = self.feature_slices_[col].start
                if feature_idx is None:
                    raise RuntimeError(f"Missing numeric feature slice for column {col!r}")
                out[:, feature_idx] = values.astype(np.float32, copy=False)

            for col in self.categorical_cols_:
                categories = self.categorical_categories_[col]
                mode_value = self.categorical_modes_[col]
                values = pd.Series(df[col], copy=False).astype(object)
                if values.isna().any():
                    values = values.where(~values.isna(), mode_value)
                codes = pd.Categorical(values, categories=categories).codes
                unknown_mask = codes < 0
                if np.any(unknown_mask):
                    codes = codes.copy()
                    codes[unknown_mask] = self.categorical_fill_codes_[col]
                feature_slice = self.feature_slices_[col]
                start = feature_slice.start
                if start is None:
                    raise RuntimeError(f"Missing categorical feature slice for column {col!r}")
                out[np.arange(n_rows), start + codes] = 1.0
            return out

        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Expected a 2D array or DataFrame")
        out = arr.copy()
        for j, col in enumerate(self.numeric_cols_):
            fill = self.numeric_medians_.get(col, 0.0)
            mask = ~np.isfinite(out[:, j])
            if np.any(mask):
                out[mask, j] = fill
        return out.astype(np.float32, copy=False)

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.feature_names_out_, dtype=object)


DATASET_LOADERS = {
    "adult": load_adult,
    "bike_sharing": load_bike_sharing,
    "bike-sharing": load_bike_sharing,
    "bikeshare": load_bike_sharing,
    "spambase": load_spambase,
    "avila": load_avila,
    "bank": load_bank,
    "bean": load_bean,
    "bidding": load_bidding,
    "electricity": load_electricity,
    "compas": load_compas,
    "coupon": load_coupon,
    "coupon_source": load_coupon,
    "heloc": load_heloc,
    "eye-movement": load_eye_movements,
    "eye-movements": load_eye_movements,
    "eye-state": load_eye_state,
}

PAPER_SOTA: dict[str, dict[int, float]] = {
    "eye-movement": {2: 0.601, 3: 0.636, 4: 0.662, 5: 0.675, 6: 0.666},
    "electricity": {2: 0.849, 3: 0.826, 4: 0.866, 5: 0.882, 6: 0.888},
    "eye-state": {2: 0.759, 3: 0.763, 4: 0.793, 5: 0.811, 6: 0.817},
}


def canonical_dataset_list(dataset_names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    if dataset_names is None:
        return list(DEFAULT_DATASETS)
    return [get_dataset_spec(name).name for name in dataset_names]


def make_preprocessor(X_train):
    return SimpleTabularPreprocessor()


def encode_target(y) -> tuple[np.ndarray, np.ndarray, SimpleLabelEncoder]:
    encoder = SimpleLabelEncoder()
    encoded = encoder.fit_transform(y)
    class_labels = np.asarray(encoder.classes_, dtype=object) if encoder.classes_ is not None else np.array([], dtype=object)
    return encoded.astype(np.int32), class_labels, encoder


def encode_binary_target(y, dataset_name: str) -> np.ndarray:
    encoded, class_labels, _ = encode_target(y)
    if len(class_labels) != 2:
        counts = pd.Series(y).value_counts(dropna=False).to_dict()
        raise ValueError(
            f"{dataset_name}: expected binary target with exactly 2 classes, "
            f"got {len(class_labels)} classes. counts={counts}"
        )
    return encoded


def _allocate_stratified_counts(class_counts: np.ndarray, sample_total: int) -> np.ndarray:
    class_counts = np.asarray(class_counts, dtype=np.int64)
    if class_counts.ndim != 1:
        raise ValueError("class_counts must be one-dimensional")
    if sample_total <= 0:
        return np.zeros_like(class_counts, dtype=np.int64)

    total = int(class_counts.sum())
    if total <= 0:
        return np.zeros_like(class_counts, dtype=np.int64)
    if sample_total >= total:
        return class_counts.copy()

    expected = class_counts.astype(np.float64) * (float(sample_total) / float(total))
    counts = np.floor(expected).astype(np.int64)
    counts = np.minimum(counts, class_counts)
    remainder = int(sample_total - int(counts.sum()))
    if remainder <= 0:
        return counts

    frac = expected - np.floor(expected)
    order = np.argsort(-frac, kind="stable")
    while remainder > 0:
        progressed = False
        for idx in order:
            if counts[idx] < class_counts[idx]:
                counts[idx] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            break
    return counts


def _stratified_split_indices(y, *, seed: int, sample_size: float) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y)
    idx_all = np.arange(y_arr.shape[0], dtype=np.int32)
    if idx_all.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    rng = np.random.default_rng(int(seed))
    unique, inverse = np.unique(y_arr, return_inverse=True)
    class_counts = np.bincount(inverse, minlength=unique.size).astype(np.int64)
    sample_total = int(round(float(sample_size) * float(idx_all.size)))
    sample_total = max(0, min(sample_total, int(idx_all.size)))
    sample_counts = _allocate_stratified_counts(class_counts, sample_total)

    selected_parts: list[np.ndarray] = []
    remaining_parts: list[np.ndarray] = []
    for class_id in range(unique.size):
        class_idx = idx_all[inverse == class_id].copy()
        rng.shuffle(class_idx)
        n_sample = int(sample_counts[class_id])
        n_sample = max(0, min(n_sample, int(class_idx.size)))
        if n_sample > 0:
            selected_parts.append(class_idx[:n_sample])
        if n_sample < class_idx.size:
            remaining_parts.append(class_idx[n_sample:])

    selected = np.concatenate(selected_parts) if selected_parts else np.array([], dtype=np.int32)
    remaining = np.concatenate(remaining_parts) if remaining_parts else np.array([], dtype=np.int32)
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return np.asarray(remaining, dtype=np.int32), np.asarray(selected, dtype=np.int32)


def stratified_train_test_indices(y, *, seed: int, test_size: float) -> SplitIndices:
    idx_train, idx_test = _stratified_split_indices(y, seed=seed, sample_size=test_size)
    return SplitIndices(idx_train=np.asarray(idx_train, dtype=np.int32), idx_test=np.asarray(idx_test, dtype=np.int32))


def resolve_feature_names(preprocessor, transformed_shape: int) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return [f"x{i}" for i in range(int(transformed_shape))]


def feature_sparsity(used_feature_count: int, total_feature_count: int) -> float:
    if total_feature_count <= 0:
        return float("nan")
    clipped = min(max(int(used_feature_count), 0), int(total_feature_count))
    return 1.0 - (float(clipped) / float(total_feature_count))
