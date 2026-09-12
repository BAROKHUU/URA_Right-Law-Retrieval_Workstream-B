from legal_retrieval.indexing.sparse import BM25Index
from legal_retrieval.models import RetrievalRecord


def test_bm25_prefers_matching_document():
    records = [
        RetrievalRecord("a", "alpha beta", "alpha beta", ["a"]),
        RetrievalRecord("b", "land registration certificate", "land registration certificate", ["b"]),
    ]
    index = BM25Index()
    index.build(records)
    hits = index.search("registration certificate", 2)
    assert hits[0].record_id == "b"
