from __future__ import annotations

import logging
from typing import Any

from .embedding import SentenceTransformerEncoder
from .fusion import rrf_fusion, weighted_fusion
from .indexing.manager import IndexManager
from .models import SearchHit, RetrievalRecord
from .query import QueryProcessor
from .rerank import MultiCrossEncoderReranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.manager = IndexManager(cfg)
        self.manager.ensure()
        records = self.manager.load_records()
        self.records: dict[str, RetrievalRecord] = {r.record_id: r for r in records}
        self.query_processor = QueryProcessor(cfg)
        self.sparse = None
        self.dense = None
        self.encoder = None
        method = cfg["retrieval"]["method"]
        if method in {"sparse", "hybrid"}:
            self.sparse = self.manager.load_sparse()
        if method in {"dense", "hybrid"}:
            self.dense = self.manager.load_dense()
            self.encoder = SentenceTransformerEncoder(cfg["indexing"]["dense"])
        self.reranker = MultiCrossEncoderReranker(cfg["reranking"]) if cfg.get("reranking", {}).get("enabled") else None

    @staticmethod
    def _matches_filters(record: RetrievalRecord, filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = record.metadata.get(key)
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _filter_hits(self, hits: list[SearchHit], filters: dict[str, Any], limit: int) -> list[SearchHit]:
        if not filters:
            return hits[:limit]
        out = [h for h in hits if self._matches_filters(self.records[h.record_id], filters)]
        return out[:limit]

    def _attach_records(self, hits: list[SearchHit]) -> list[SearchHit]:
        for hit in hits:
            hit.record = self.records[hit.record_id]
        return hits

    def _deduplicate(self, hits: list[SearchHit]) -> list[SearchHit]:
        mode = self.cfg.get("ranking", {}).get("deduplicate_by", "source_unit")
        if mode == "none":
            return hits
        seen = set()
        out = []
        for hit in hits:
            if mode == "record_id":
                key = hit.record_id
            elif mode == "source_unit":
                # For fixed chunks, use the first source legal unit as the primary dedupe key.
                key = hit.record.source_unit_ids[0] if hit.record and hit.record.source_unit_ids else hit.record_id
            else:
                raise ValueError(f"Unknown ranking.deduplicate_by={mode}")
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
        return out

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> list[SearchHit]:
        processed = self.query_processor.process(query, filters)
        rcfg = self.cfg["retrieval"]
        method = rcfg["method"]
        filter_cfg = self.cfg.get("query_processing", {}).get("metadata_filter", {})
        factor = int(filter_cfg.get("oversample_factor", 4)) if processed.filters and filter_cfg.get("enabled", True) else 1

        lists: dict[str, list[SearchHit]] = {}
        if method in {"sparse", "hybrid"}:
            k = int(rcfg.get("sparse_top_k", 50))
            lists["sparse"] = self.sparse.search(processed.sparse, k * factor)
        if method in {"dense", "hybrid"}:
            k = int(rcfg.get("dense_top_k", 50))
            qvec = self.encoder.encode_query(processed.dense)
            lists["dense"] = self.dense.search(qvec, k * factor)

        candidate_k = int(rcfg.get("candidate_top_k", 50))
        if method == "sparse":
            hits = lists["sparse"]
        elif method == "dense":
            hits = lists["dense"]
        else:
            fcfg = self.cfg["fusion"]
            if fcfg["method"] == "rrf":
                hits = rrf_fusion(
                    lists,
                    k=int(fcfg.get("rrf_k", 60)),
                    weights={"sparse": fcfg.get("sparse_weight", 1.0), "dense": fcfg.get("dense_weight", 1.0)},
                )
            elif fcfg["method"] == "weighted":
                hits = weighted_fusion(
                    lists,
                    weights={"sparse": fcfg.get("sparse_weight", 0.5), "dense": fcfg.get("dense_weight", 0.5)},
                )
            else:
                raise ValueError(fcfg["method"])

        hits = self._filter_hits(hits, processed.filters, candidate_k)
        hits = self._attach_records(hits)

        if self.reranker:
            hits = self.reranker.rerank(processed.dense, hits, self.records)

        hits = self._deduplicate(hits)
        final_k = int(self.cfg.get("ranking", {}).get("final_top_k", 10) if top_k is None else top_k)
        if final_k <= 0:
            raise ValueError("top_k must be > 0")
        hits = hits[:final_k]
        for rank, hit in enumerate(hits, start=1):
            hit.rank = rank
        return hits
