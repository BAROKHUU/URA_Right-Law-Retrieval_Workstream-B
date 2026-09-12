from legal_retrieval.config import load_config
from legal_retrieval.corpus import build_records


def test_enriched_corpus_loads():
    cfg = load_config("configs/base.yaml")
    records = build_records(cfg)
    assert len(records) > 1000
    first = records[0]
    assert first.record_id
    assert first.source_unit_ids
    assert "[TEXT]" in first.text


def test_fixed_length_produces_source_mapping():
    cfg = load_config("configs/base.yaml")
    cfg["representation"]["mode"] = "fixed_length"
    cfg["representation"]["fixed_length"]["chunk_size"] = 100
    cfg["representation"]["fixed_length"]["overlap"] = 20
    records = build_records(cfg)
    assert records
    assert records[0].metadata["unit_type"] == "fixed_chunk"
    assert len(records[0].source_unit_ids) >= 1
