"""Dataset registry and local materialization helpers for tabular benchmarks."""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import zipfile

import pandas as pd

try:
    from ucimlrepo import fetch_ucirepo
except Exception:  # pragma: no cover - optional until installed
    fetch_ucirepo = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_OPENML_CACHE = PROJECT_ROOT / "results" / "openml_cache"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    task: str
    target_name: str
    openml_id: int | None = None
    uci_id: int | None = None
    local_path: str | None = None
    description: str = ""


DATASET_SPECS: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        source="uci",
        task="classification",
        target_name="income",
        uci_id=2,
        description="Adult income classification.",
    ),
    "bike_sharing": DatasetSpec(
        name="bike_sharing",
        source="uci",
        task="classification",
        target_name="cnt_binary",
        uci_id=275,
        description="Bike sharing demand classification (median split on cnt).",
    ),
    "spambase": DatasetSpec(
        name="spambase",
        source="uci",
        task="classification",
        target_name="Class",
        uci_id=94,
        description="UCI Spambase spam classification.",
    ),
    "avila": DatasetSpec(
        name="avila",
        source="uci",
        task="classification",
        target_name="class",
        uci_id=459,
        description="Avila authorship classification.",
    ),
    "bank": DatasetSpec(
        name="bank",
        source="uci",
        task="classification",
        target_name="class",
        uci_id=267,
        description="Banknote authentication.",
    ),
    "bean": DatasetSpec(
        name="bean",
        source="uci",
        task="classification",
        target_name="Class",
        uci_id=602,
        description="Dry bean classification.",
    ),
    "bidding": DatasetSpec(
        name="bidding",
        source="uci",
        task="classification",
        target_name="Class",
        uci_id=563,
        description="Shill bidding classification.",
    ),
    "electricity": DatasetSpec(
        name="electricity",
        source="openml",
        task="classification",
        target_name="class",
        openml_id=151,
        description="Electricity pricing classification.",
    ),
    "compas": DatasetSpec(
        name="compas",
        source="local_csv",
        task="classification",
        target_name="two_year_recid",
        local_path="datasets/compas/raw/compas_data_matrix.csv",
        description="COMPAS recidivism benchmark (raw non-binarized source).",
    ),
    "heloc": DatasetSpec(
        name="heloc",
        source="openml",
        task="classification",
        target_name="RiskPerformance",
        openml_id=45023,
        description="HELOC credit risk classification.",
    ),
    "eye-movement": DatasetSpec(
        name="eye-movement",
        source="openml",
        task="classification",
        target_name="target",
        openml_id=45073,
        description="Eye movement classification.",
    ),
    "eye-state": DatasetSpec(
        name="eye-state",
        source="openml",
        task="classification",
        target_name="Class",
        openml_id=1471,
        description="EEG eye state classification.",
    ),
    "coupon": DatasetSpec(
        name="coupon",
        source="local_csv",
        task="classification",
        target_name="Y",
        local_path="datasets/coupon_source.csv",
        description="Coupon recommendation classification from the in-repo coupon source CSV.",
    ),
}

DATASET_ALIASES = {
    "bike-sharing": "bike_sharing",
    "bikeshare": "bike_sharing",
    "spam-base": "spambase",
    "eye-movements": "eye-movement",
    "coupon_source": "coupon",
    "coupon-source": "coupon",
}

DEFAULT_DATASETS = tuple(DATASET_SPECS.keys())


def canonical_dataset_name(name: str) -> str:
    raw = str(name).strip().lower()
    return DATASET_ALIASES.get(raw, raw)


def get_dataset_spec(name: str) -> DatasetSpec:
    canonical = canonical_dataset_name(name)
    if canonical not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Expected one of {sorted(DATASET_SPECS)}.")
    return DATASET_SPECS[canonical]


def dataset_dir(name: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    spec = get_dataset_spec(name)
    return Path(data_root) / spec.name


def dataset_csv_path(name: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    spec = get_dataset_spec(name)
    return dataset_dir(spec.name, data_root) / "raw" / f"{spec.name}_data_matrix.csv"


def dataset_metadata_path(name: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    spec = get_dataset_spec(name)
    return dataset_dir(spec.name, data_root) / "metadata.json"


def _ensure_openml_cache(openml_cache_dir: str | Path | None) -> None:
    if openml_cache_dir is None:
        return
    import openml

    cache_dir = Path(openml_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    openml.config.cache_directory = str(cache_dir)


def _target_to_frame(y, fallback_name: str) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        out = y.copy()
    elif isinstance(y, pd.Series):
        out = y.to_frame(name=y.name or fallback_name)
    else:
        out = pd.DataFrame({fallback_name: pd.Series(y)})

    columns = [str(col) if col is not None else fallback_name for col in out.columns]
    out.columns = columns
    return out.reset_index(drop=True)


def _fetch_openml_dataset(spec: DatasetSpec, openml_cache_dir: str | Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.openml_id is None:
        raise ValueError(f"{spec.name} does not define an OpenML dataset id.")
    import openml

    _ensure_openml_cache(openml_cache_dir)
    dataset = openml.datasets.get_dataset(spec.openml_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    X_df = pd.DataFrame(X).reset_index(drop=True)
    y_df = _target_to_frame(y, dataset.default_target_attribute or spec.target_name)
    return X_df, y_df


def _fetch_uci_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if spec.uci_id is None:
        raise ValueError(f"{spec.name} does not define a UCI repository id.")
    if fetch_ucirepo is None:
        if spec.name == "avila":
            return _fetch_avila_from_zip()
        raise ImportError(
            "ucimlrepo is not installed. Install it with `pip install ucimlrepo pandas` before fetching UCI datasets."
        )

    try:
        if spec.name == "adult":
            return _fetch_adult_from_uci()
        if spec.name == "bike_sharing":
            return _fetch_bike_sharing_from_uci()
        if spec.name == "spambase":
            return _fetch_spambase_from_uci()
        dataset = fetch_ucirepo(id=spec.uci_id)
        X_df = pd.DataFrame(dataset.data.features).reset_index(drop=True)
        y_df = _target_to_frame(dataset.data.targets, spec.target_name)
        return X_df, y_df
    except Exception:
        if spec.name == "avila":
            return _fetch_avila_from_zip()
        raise


def _fetch_adult_from_uci() -> tuple[pd.DataFrame, pd.DataFrame]:
    if fetch_ucirepo is None:
        raise ImportError(
            "ucimlrepo is not installed. Install it with `pip install ucimlrepo pandas` before fetching UCI datasets."
        )

    dataset = fetch_ucirepo(id=2)
    X_df = pd.DataFrame(dataset.data.features).reset_index(drop=True)
    y_df = _target_to_frame(dataset.data.targets, "income")
    if y_df.empty:
        raise ValueError("adult: expected a non-empty target column from ucimlrepo.")

    income = pd.Series(y_df.iloc[:, 0], copy=False).astype(str).str.strip().str.rstrip(".")
    return X_df, income.to_frame(name="income")


def _fetch_bike_sharing_from_uci() -> tuple[pd.DataFrame, pd.DataFrame]:
    if fetch_ucirepo is None:
        raise ImportError(
            "ucimlrepo is not installed. Install it with `pip install ucimlrepo pandas` before fetching UCI datasets."
        )

    dataset = fetch_ucirepo(id=275)
    X_df = pd.DataFrame(dataset.data.features).reset_index(drop=True)
    y_df = _target_to_frame(dataset.data.targets, "cnt")

    drop_cols = [col for col in ["instant", "dteday", "casual", "registered"] if col in X_df.columns]
    if drop_cols:
        X_df = X_df.drop(columns=drop_cols)

    if "cnt" in y_df.columns:
        cnt = pd.Series(y_df["cnt"], copy=False).reset_index(drop=True)
        if "cnt" in X_df.columns:
            X_df = X_df.drop(columns=["cnt"])
    elif "cnt" in X_df.columns:
        cnt = pd.Series(X_df.pop("cnt"), copy=False).reset_index(drop=True)
    else:
        raise KeyError("bike_sharing: expected a cnt column in either features or targets.")

    threshold = float(pd.Series(cnt, copy=False).median())
    y_binary = (pd.Series(cnt, copy=False) >= threshold).astype(int).rename("cnt_binary")
    return X_df.reset_index(drop=True), y_binary.to_frame()


def _fetch_spambase_from_uci() -> tuple[pd.DataFrame, pd.DataFrame]:
    if fetch_ucirepo is None:
        raise ImportError(
            "ucimlrepo is not installed. Install it with `pip install ucimlrepo pandas` before fetching UCI datasets."
        )

    dataset = fetch_ucirepo(id=94)
    X_df = pd.DataFrame(dataset.data.features).reset_index(drop=True)
    y_df = _target_to_frame(dataset.data.targets, "Class")
    if "Class" not in y_df.columns:
        raise KeyError("spambase: expected a Class column in the UCI target frame.")
    y_binary = pd.Series(y_df["Class"], copy=False).astype(int).rename("Class")
    return X_df.reset_index(drop=True), y_binary.to_frame()


def _clean_coupon_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in ["click", "car"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.dropna().reset_index(drop=True)


def _fetch_local_csv_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not spec.local_path:
        raise ValueError(f"{spec.name} does not define a local CSV path.")
    csv_path = PROJECT_ROOT / spec.local_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Local dataset CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    if spec.name == "coupon":
        df = _clean_coupon_frame(df)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least one feature and one target column in {csv_path}, got shape {df.shape}.")
    target_col = spec.target_name if spec.target_name in df.columns else df.columns[-1]
    X_df = df.drop(columns=[target_col]).reset_index(drop=True)
    y_df = _target_to_frame(df[target_col], target_col)
    return X_df, y_df


def _fetch_avila_from_zip() -> tuple[pd.DataFrame, pd.DataFrame]:
    url = "https://archive.ics.uci.edu/static/public/459/avila.zip"
    import requests

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in ["avila/avila-tr.txt", "avila/avila-ts.txt"]:
            with archive.open(member) as handle:
                frames.append(pd.read_csv(handle, header=None))

    df = pd.concat(frames, axis=0, ignore_index=True)
    if df.shape[1] != 11:
        raise ValueError(f"Unexpected Avila shape {df.shape}; expected 10 features plus 1 target column.")
    feature_cols = [f"F{i}" for i in range(1, 11)]
    df.columns = feature_cols + ["Class"]
    X_df = df[feature_cols].reset_index(drop=True)
    y_df = df[["Class"]].reset_index(drop=True)
    return X_df, y_df


def fetch_dataset_frame(
    name: str,
    *,
    openml_cache_dir: str | Path | None = DEFAULT_OPENML_CACHE,
) -> tuple[pd.DataFrame, list[str], DatasetSpec]:
    spec = get_dataset_spec(name)
    if spec.source == "openml":
        X_df, y_df = _fetch_openml_dataset(spec, openml_cache_dir=openml_cache_dir)
    elif spec.source == "uci":
        X_df, y_df = _fetch_uci_dataset(spec)
    elif spec.source == "local_csv":
        X_df, y_df = _fetch_local_csv_dataset(spec)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported source '{spec.source}' for dataset '{spec.name}'.")

    df = pd.concat([X_df, y_df], axis=1)
    target_columns = [str(col) for col in y_df.columns]
    return df, target_columns, spec


def materialize_dataset(
    name: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    force: bool = False,
    openml_cache_dir: str | Path | None = DEFAULT_OPENML_CACHE,
) -> tuple[Path, Path]:
    spec = get_dataset_spec(name)
    csv_path = dataset_csv_path(spec.name, data_root=data_root)
    metadata_path = dataset_metadata_path(spec.name, data_root=data_root)
    if csv_path.exists() and metadata_path.exists() and not force:
        return csv_path, metadata_path

    df, target_columns, _ = fetch_dataset_frame(spec.name, openml_cache_dir=openml_cache_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    metadata = {
        "schema_version": 1,
        "spec": asdict(spec),
        "target_columns": target_columns,
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "feature_columns": [str(col) for col in df.columns if str(col) not in set(target_columns)],
        "csv_path": str(csv_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, metadata_path


def materialize_all_datasets(
    dataset_names: list[str] | tuple[str, ...] = DEFAULT_DATASETS,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    force: bool = False,
    openml_cache_dir: str | Path | None = DEFAULT_OPENML_CACHE,
) -> dict[str, dict[str, str]]:
    realized: dict[str, dict[str, str]] = {}
    for dataset_name in dataset_names:
        spec = get_dataset_spec(dataset_name)
        csv_path, metadata_path = materialize_dataset(
            spec.name,
            data_root=data_root,
            force=force,
            openml_cache_dir=openml_cache_dir,
        )
        realized[spec.name] = {
            "csv_path": str(csv_path),
            "metadata_path": str(metadata_path),
        }
    return realized


def load_dataset(
    name: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    ensure_local: bool = True,
    openml_cache_dir: str | Path | None = DEFAULT_OPENML_CACHE,
) -> tuple[pd.DataFrame, pd.Series]:
    spec = get_dataset_spec(name)
    csv_path = dataset_csv_path(spec.name, data_root=data_root)
    metadata_path = dataset_metadata_path(spec.name, data_root=data_root)
    if ensure_local and (not csv_path.exists() or not metadata_path.exists()):
        materialize_dataset(spec.name, data_root=data_root, openml_cache_dir=openml_cache_dir)

    if ensure_local and spec.name == "compas" and csv_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("target_columns") != ["two_year_recid"]:
            materialize_dataset(
                spec.name,
                data_root=data_root,
                force=True,
                openml_cache_dir=openml_cache_dir,
            )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_columns = [str(col) for col in metadata["target_columns"]]
    df = pd.read_csv(csv_path, low_memory=False)
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        raise ValueError(f"{spec.name}: missing target columns {missing_targets} in {csv_path}")

    if len(target_columns) != 1:
        raise ValueError(
            f"{spec.name}: expected exactly one target column for classification, got {target_columns}."
        )
    target_col = target_columns[0]
    X = df.drop(columns=target_columns)
    y = df[target_col].rename(target_col)
    return X, y


def _loader(name: str) -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _wrapped() -> tuple[pd.DataFrame, pd.Series]:
        return load_dataset(name)

    return _wrapped


def load_adult():
    return load_dataset("adult")


def load_bike_sharing():
    return load_dataset("bike_sharing")


def load_spambase():
    return load_dataset("spambase")


def load_avila():
    return load_dataset("avila")


def load_bank():
    return load_dataset("bank")


def load_bean():
    return load_dataset("bean")


def load_bidding():
    return load_dataset("bidding")


def load_electricity():
    return load_dataset("electricity")


def load_compas():
    return load_dataset("compas")


def load_heloc():
    return load_dataset("heloc")


def load_eye_movements():
    return load_dataset("eye-movement")


def load_eye_state():
    return load_dataset("eye-state")


def load_coupon():
    return load_dataset("coupon")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/cache benchmark datasets under datasets/<name>/raw.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=sorted(DATASET_SPECS))
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--openml-cache-dir", type=str, default=str(DEFAULT_OPENML_CACHE))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    realized = materialize_all_datasets(
        dataset_names=args.datasets,
        data_root=args.data_root,
        force=args.force,
        openml_cache_dir=args.openml_cache_dir,
    )
    print(json.dumps(realized, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
