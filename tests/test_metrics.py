from legal_retrieval.evaluation import evaluate_query, metrics_at_k
from legal_retrieval.models import RetrievalRecord, SearchHit


def test_metrics():
    m = metrics_at_k([0, 1, 0, 1], relevant_count=2, k=4)
    assert m["hit_rate"] == 1.0
    assert m["precision"] == 0.5
    assert m["recall"] == 1.0
    assert m["mrr"] == 0.5
    assert 0.0 < m["ndcg"] <= 1.0


def test_overlapping_chunks_do_not_double_count_relevance():
    record_a = RetrievalRecord("chunk-a", "", "", ["unit-1"])
    record_b = RetrievalRecord("chunk-b", "", "", ["unit-1"])
    hits = [SearchHit("chunk-a", 1.0, record=record_a), SearchHit("chunk-b", 0.9, record=record_b)]
    cfg = {"evaluation": {"match_by": "source_unit", "metrics": ["recall", "ndcg", "map"], "k_values": [2]}}
    result = evaluate_query(hits, ["unit-1"], cfg)
    assert result == {"recall@2": 1.0, "ndcg@2": 1.0, "map@2": 1.0}


def test_one_chunk_can_cover_multiple_relevant_units():
    record = RetrievalRecord("chunk", "", "", ["unit-1", "unit-2"])
    hits = [SearchHit("chunk", 1.0, record=record)]
    cfg = {"evaluation": {"match_by": "source_unit", "metrics": ["recall"], "k_values": [1]}}
    assert evaluate_query(hits, ["unit-1", "unit-2"], cfg)["recall@1"] == 1.0
