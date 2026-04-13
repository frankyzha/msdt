"""Repository-local import shim for the moved MSPLIT package.

This keeps plain ``import split`` pinned to ``algorithm/msplit/src/split`` when
running from the repository root, instead of accidentally picking up an
unrelated site-packages install.
"""

from __future__ import annotations

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_PACKAGE_DIR = _REPO_ROOT / "algorithm" / "msplit" / "src" / "split"
_REAL_INIT = _REAL_PACKAGE_DIR / "__init__.py"

if not _REAL_INIT.exists():
    raise ImportError(f"Unable to locate repository split package at {_REAL_INIT}")

__file__ = str(_REAL_INIT)
__path__ = [str(_REAL_PACKAGE_DIR)]

if __spec__ is not None:
    __spec__.origin = __file__
    if __spec__.submodule_search_locations is not None:
        __spec__.submodule_search_locations[:] = __path__

exec(compile(_REAL_INIT.read_text(encoding="utf-8"), __file__, "exec"), globals(), globals())
