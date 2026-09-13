from pathlib import Path

import pytest

from legal_retrieval.paths import project_root


def test_project_root_uses_environment_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("LEGAL_RETRIEVAL_PROJECT_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_project_root_rejects_missing_environment_override(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("LEGAL_RETRIEVAL_PROJECT_ROOT", str(missing))
    with pytest.raises(RuntimeError, match="missing directory"):
        project_root()
