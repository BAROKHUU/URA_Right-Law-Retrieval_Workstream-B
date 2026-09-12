from __future__ import annotations
from pathlib import Path


def project_root() -> Path:
    # src/legal_retrieval/paths.py -> project root
    return Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else project_root() / p
