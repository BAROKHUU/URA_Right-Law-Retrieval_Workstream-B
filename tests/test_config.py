from pathlib import Path

import pytest
import yaml

from legal_retrieval.config import deep_merge, load_config


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_load_base_config():
    cfg = load_config("configs/base.yaml")
    assert cfg["retrieval"]["method"] == "sparse"
    assert cfg["representation"]["mode"] == "hierarchical_units"
    assert cfg["_hypothesis_diff"]["actual_changes"] == []


def test_example_configs_have_no_undeclared_changes():
    for path in sorted(Path("configs/examples").glob("*.yaml")):
        cfg = load_config(path)
        assert cfg["_hypothesis_diff"]["undeclared_changes"] == []


def test_undeclared_hypothesis_change_is_rejected(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        "\n".join([
            f"extends: {Path('configs/base.yaml').resolve()}",
            "hypothesis:",
            "  id: H_BAD",
            "  variables: [reranking.enabled]",
            "retrieval:",
            "  candidate_top_k: 99",
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Declared changes =") as exc:
        load_config(config)
    assert "retrieval.candidate_top_k" in str(exc.value)


def test_final_mode_requires_dense_commit_hash(tmp_path: Path):
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["runtime"]["reproducibility_mode"] = "final"
    base["indexing"]["sparse"]["enabled"] = False
    base["indexing"]["dense"]["enabled"] = True
    base["retrieval"]["method"] = "dense"
    base["indexing"]["dense"]["revision"] = None
    path = tmp_path / "final.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="indexing.dense.revision"):
        load_config(path)


def test_final_mode_accepts_dense_commit_hash(tmp_path: Path):
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["runtime"]["reproducibility_mode"] = "final"
    base["indexing"]["sparse"]["enabled"] = False
    base["indexing"]["dense"]["enabled"] = True
    base["retrieval"]["method"] = "dense"
    base["indexing"]["dense"]["revision"] = "a" * 40
    path = tmp_path / "final_pinned.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    cfg = load_config(path)
    assert cfg["indexing"]["dense"]["revision"] == "a" * 40
