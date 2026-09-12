from legal_retrieval.config import deep_merge, load_config


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_load_base_config():
    cfg = load_config("configs/base.yaml")
    assert cfg["retrieval"]["method"] == "sparse"
    assert cfg["representation"]["mode"] == "hierarchical_units"
