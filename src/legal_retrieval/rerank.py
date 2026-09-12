from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .embedding import resolve_device
from .fusion import rrf_fusion
from .models import SearchHit, RetrievalRecord


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(x - lo) / (hi - lo) for x in values]


class MultiCrossEncoderReranker:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._models = []

    def _load(self):
        if self._models:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Reranking requires sentence-transformers. Install `pip install -e \".[dense]\"`.") from exc
        for spec in self.cfg.get("models", []):
            model = CrossEncoder(
                spec["model_name"],
                device=resolve_device(spec.get("device", "auto")),
                trust_remote_code=spec.get("trust_remote_code", False),
                revision=spec.get("revision"),
            )
            self._models.append((spec, model))

    def rerank(self, query: str, hits: list[SearchHit], records: dict[str, RetrievalRecord]) -> list[SearchHit]:
        self._load()
        top_n = min(int(self.cfg.get("top_n", 30)), len(hits))
        head, tail = hits[:top_n], hits[top_n:]
        if not head:
            return hits
        pairs = [(query, records[h.record_id].text) for h in head]
        aggregation = self.cfg.get("aggregation", "weighted")

        per_model_hits: dict[str, list[SearchHit]] = {}
        weighted_scores = [0.0] * len(head)
        for model_index, (spec, model) in enumerate(self._models):
            raw = model.predict(pairs, batch_size=int(self.cfg.get("batch_size", 8)), show_progress_bar=False)
            scores = [float(x) for x in raw]
            name = spec.get("name") or spec["model_name"]
            if aggregation == "rrf":
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                per_model_hits[name] = [SearchHit(record_id=head[i].record_id, score=scores[i], rank=r + 1, source=name)
                                        for r, i in enumerate(order)]
            else:
                norm = _minmax(scores)
                weight = float(spec.get("weight", 1.0))
                for i, value in enumerate(norm):
                    weighted_scores[i] += weight * value
                    head[i].component_scores[f"reranker:{name}"] = scores[i]

        if aggregation == "rrf":
            fused = rrf_fusion(per_model_hits, k=int(self.cfg.get("rrf_k", 60)))
            id_to_original = {h.record_id: h for h in head}
            for h in fused:
                h.record = id_to_original[h.record_id].record
                h.component_scores.update(id_to_original[h.record_id].component_scores)
                h.source = "reranker_rrf"
            reranked = fused
        else:
            for h, score in zip(head, weighted_scores):
                h.score = score
                h.source = "reranker"
            reranked = sorted(head, key=lambda h: h.score, reverse=True)

        output = reranked + tail
        for rank, hit in enumerate(output, start=1):
            hit.rank = rank
        return output
