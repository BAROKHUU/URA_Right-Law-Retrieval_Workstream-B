from datetime import date

from legal_retrieval.temporal import parse_temporal_query, record_matches_temporal


CFG = {
    "enabled": True,
    "reference_date": "2026-09-19",
    "strip_time_from_query": True,
}


def test_query_without_time_uses_pinned_current_date():
    parsed = parse_temporal_query("hỗ trợ ngư dân", CFG)
    assert parsed.mode == "current"
    assert parsed.start == date(2026, 9, 19)
    assert not parsed.explicit


def test_as_of_year_is_extracted_from_vietnamese_query():
    parsed = parse_temporal_query("năm 2022 ngư dân được hỗ trợ thế nào", CFG)
    assert parsed.mode == "as_of"
    assert parsed.start == date(2022, 1, 1)
    assert parsed.end == date(2022, 12, 31)
    assert "2022" not in parsed.semantic_query


def test_law_name_year_without_temporal_cue_is_not_treated_as_time():
    parsed = parse_temporal_query("Luật Đất đai 2024 quy định gì", CFG)
    assert parsed.mode == "current"
    assert parsed.semantic_query == "Luật Đất đai 2024 quy định gì"


def test_before_and_validity_interval_matching():
    parsed = parse_temporal_query("trước năm 2020 có quy định nào", CFG)
    old = {"unit_id": "old", "effective_from": "01/01/2018", "effective_to": "31/12/2019"}
    new = {"unit_id": "new", "effective_from": "01/01/2021", "effective_to": None}
    assert record_matches_temporal(old, parsed, "error")
    assert not record_matches_temporal(new, parsed, "error")


def test_as_of_excludes_law_not_yet_effective():
    parsed = parse_temporal_query("tại thời điểm 01/06/2022 áp dụng thế nào", CFG)
    future = {"unit_id": "future", "effective_from": "01/01/2023", "effective_to": None}
    assert not record_matches_temporal(future, parsed, "error")
