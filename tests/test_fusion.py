from legal_retrieval.fusion import rrf_fusion, weighted_fusion
from legal_retrieval.models import SearchHit


def test_rrf_fusion():
    sparse = [SearchHit("a", 10, 1), SearchHit("b", 8, 2)]
    dense = [SearchHit("b", .9, 1), SearchHit("c", .8, 2)]
    hits = rrf_fusion({"sparse": sparse, "dense": dense})
    assert hits[0].record_id == "b"


def test_weighted_fusion_returns_union():
    sparse = [SearchHit("a", 10, 1), SearchHit("b", 5, 2)]
    dense = [SearchHit("c", .9, 1), SearchHit("b", .1, 2)]
    hits = weighted_fusion({"sparse": sparse, "dense": dense}, {"sparse": .5, "dense": .5})
    assert {h.record_id for h in hits} == {"a", "b", "c"}
