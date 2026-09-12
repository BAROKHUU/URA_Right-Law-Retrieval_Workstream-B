from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .models import SearchHit


def _minmax(hits: list[SearchHit]) -> dict[str, float]:
    if not hits:
        return {}
    values = [h.score for h in hits]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {h.record_id: 1.0 for h in hits}
    return {h.record_id: (h.score - lo) / (hi - lo) for h in hits}


def rrf_fusion(lists: dict[str, list[SearchHit]], k: int = 60, weights: dict[str, float] | None = None) -> list[SearchHit]:
    scores = defaultdict(float)
    components: dict[str, dict[str, float]] = defaultdict(dict)
    weights = weights or {}
    for name, hits in lists.items():
        weight = float(weights.get(name, 1.0))
        for rank, hit in enumerate(hits, start=1):
            contribution = weight / (k + rank)
            scores[hit.record_id] += contribution
            components[hit.record_id][name] = hit.score
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchHit(record_id=rid, score=score, rank=i + 1, source="fusion", component_scores=components[rid])
            for i, (rid, score) in enumerate(ordered)]


def weighted_fusion(lists: dict[str, list[SearchHit]], weights: dict[str, float]) -> list[SearchHit]:
    scores = defaultdict(float)
    components: dict[str, dict[str, float]] = defaultdict(dict)
    for name, hits in lists.items():
        norm = _minmax(hits)
        weight = float(weights.get(name, 1.0))
        for hit in hits:
            scores[hit.record_id] += weight * norm[hit.record_id]
            components[hit.record_id][name] = hit.score
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [SearchHit(record_id=rid, score=score, rank=i + 1, source="fusion", component_scores=components[rid])
            for i, (rid, score) in enumerate(ordered)]
