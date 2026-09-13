from __future__ import annotations

import os
from pathlib import Path

_ROOT_ENV = "LEGAL_RETRIEVAL_PROJECT_ROOT"


def _looks_like_project_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "configs").is_dir()
        and (path / "src" / "legal_retrieval").is_dir()
    )


def _search_upwards(start: Path) -> Path | None:
    start = start.resolve()
    for candidate in (start, *start.parents):
        if _looks_like_project_root(candidate):
            return candidate
    return None


def project_root() -> Path:
    """Resolve the runtime project/data root without assuming package depth.

    Resolution order:
    1. LEGAL_RETRIEVAL_PROJECT_ROOT (recommended for Docker/deployment)
    2. Search upward from the current working directory
    3. Search upward from this source file (developer/editable installs)
    """
    configured = os.environ.get(_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(f"{_ROOT_ENV} points to a missing directory: {root}")
        return root

    root = _search_upwards(Path.cwd())
    if root is not None:
        return root

    root = _search_upwards(Path(__file__).resolve().parent)
    if root is not None:
        return root

    raise RuntimeError(
        "Could not locate the legal-retrieval project root. "
        f"Set {_ROOT_ENV} to the directory containing pyproject.toml and configs/."
    )


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value).expanduser()
    return p if p.is_absolute() else project_root() / p
