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


def test_contextual_child_adds_parent_heading_without_changing_source_unit():
    baseline = build_records(load_config("configs/examples/h7_child_temporal.yaml"))
    contextual = build_records(load_config("configs/examples/h8_contextual_child_temporal.yaml"))
    baseline_by_id = {record.record_id: record for record in baseline}
    contextual_by_id = {record.record_id: record for record in contextual}
    record_id = next(iter(baseline_by_id))

    assert "[PARENT]" not in baseline_by_id[record_id].text
    assert "[PARENT]" in contextual_by_id[record_id].text
    assert contextual_by_id[record_id].source_unit_ids == [record_id]
    assert contextual_by_id[record_id].metadata["parent_id"]


def test_parent_to_child_representation_contains_both_roles():
    records = build_records(load_config("configs/examples/h9_parent_to_child_temporal.yaml"))
    roles = {record.metadata["retrieval_role"] for record in records}
    assert roles == {"parent", "child"}
    record_ids = {record.record_id for record in records}
    assert all(
        record.metadata["parent_id"] in record_ids
        for record in records
        if record.metadata["retrieval_role"] == "child"
    )
