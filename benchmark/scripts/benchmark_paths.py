"""Shared repository paths for benchmark scripts."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = PROJECT_ROOT / "benchmark"
SCRIPT_ROOT = BENCHMARK_ROOT / "scripts"
BENCHMARK_DATA_ROOT = BENCHMARK_ROOT / "datasets"
BENCHMARK_CACHE_ROOT = BENCHMARK_ROOT / "cache"
BENCHMARK_ARTIFACTS_ROOT = BENCHMARK_ROOT / "artifacts"
ALGORITHM_ROOT = PROJECT_ROOT / "algorithm"
MSPLIT_ROOT = ALGORITHM_ROOT / "msplit"
MSPLIT_SRC_ROOT = MSPLIT_ROOT / "src"
SHAPECART_ROOT = ALGORITHM_ROOT / "shapecart"
DEFAULT_OPENML_CACHE = BENCHMARK_ARTIFACTS_ROOT / "openml_cache"


def ensure_repo_import_paths(*, include_msplit_src: bool = False, include_shapecart: bool = False) -> None:
    """Add the moved repository roots to ``sys.path`` for direct script execution."""

    candidates = [PROJECT_ROOT]
    if include_msplit_src:
        candidates.append(MSPLIT_SRC_ROOT)
    if include_shapecart:
        candidates.append(SHAPECART_ROOT)

    for path in candidates:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def resolve_msplit_build_dir(build_dir: str | Path) -> Path:
    """Resolve an MSPLIT build directory relative to ``algorithm/msplit``."""

    build_path = Path(build_dir)
    if build_path.is_absolute():
        return build_path
    project_relative = (PROJECT_ROOT / build_path).resolve()
    if project_relative.exists():
        return project_relative
    return (MSPLIT_ROOT / build_path).resolve()
