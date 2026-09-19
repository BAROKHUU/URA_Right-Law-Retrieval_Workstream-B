from __future__ import annotations

import logging
from typing import Any

from .embedding import SentenceTransformerEncoder
from .fusion import rrf_fusion, weighted_fusion
from .indexing.manager import IndexManager
from .models import SearchHit, RetrievalRecord
from .query import QueryProcessor
from .rerank import MultiCrossEncoderReranker
from .temporal import record_matches_temporal

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
        self.last_query_context: dict[str, Any] = {}

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

    def _eligible_record_ids(self, processed) -> set[str]:
        temporal_cfg = self.cfg.get("temporal", {})
        missing_policy = temporal_cfg.get("missing_date_policy", "exclude")
        eligible: set[str] = set()
        for record_id, record in self.records.items():
            if not self._matches_filters(record, processed.filters):
                continue
            if processed.temporal and not record_matches_temporal(record.metadata, processed.temporal, missing_policy):
                continue
            eligible.add(record_id)
        return eligible

    def _candidate_hits(
        self,
        processed,
        allowed_record_ids: set[str],
        sparse_k: int,
        dense_k: int,
        candidate_k: int,
        query_vector=None,
    ) -> list[SearchHit]:
        method = self.cfg["retrieval"]["method"]
        lists: dict[str, list[SearchHit]] = {}
        if method in {"sparse", "hybrid"}:
            lists["sparse"] = self.sparse.search(processed.sparse, sparse_k, allowed_record_ids)
        if method in {"dense", "hybrid"}:
            qvec = query_vector if query_vector is not None else self.encoder.encode_query(processed.dense)
            lists["dense"] = self.dense.search(qvec, dense_k, allowed_record_ids)

        if method == "sparse":
            return lists["sparse"][:candidate_k]
        if method == "dense":
            return lists["dense"][:candidate_k]

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
        return hits[:candidate_k]

    def _parent_to_child_hits(self, processed, eligible: set[str], query_vector=None) -> list[SearchHit]:
        pcfg = self.cfg["retrieval"]["parent_child"]
        parent_ids = {
            record_id for record_id in eligible
            if self.records[record_id].metadata.get("retrieval_role") == "parent"
        }
        parent_top_k = int(pcfg.get("parent_top_k", 10))
        parents = self._candidate_hits(
            processed,
            parent_ids,
            parent_top_k,
            parent_top_k,
            parent_top_k,
            query_vector,
        )
        parent_ranks = {hit.record_id: rank for rank, hit in enumerate(parents, start=1)}
        parent_scores = {hit.record_id: hit.score for hit in parents}

        child_ids = {
            record_id for record_id in eligible
            if self.records[record_id].metadata.get("retrieval_role") == "child"
            and self.records[record_id].metadata.get("parent_id") in parent_ranks
        }
        child_top_k = int(pcfg.get("child_top_k", self.cfg["retrieval"].get("candidate_top_k", 50)))
        children = self._candidate_hits(
            processed,
            child_ids,
            child_top_k,
            child_top_k,
            child_top_k,
            query_vector,
        )

        rrf_k = int(pcfg.get("rrf_k", 60))
        parent_weight = float(pcfg.get("parent_weight", 0.7))
        child_weight = float(pcfg.get("child_weight", 1.0))
        for child_rank, hit in enumerate(children, start=1):
            parent_id = self.records[hit.record_id].metadata["parent_id"]
            parent_rank = parent_ranks[parent_id]
            child_rrf = child_weight / (rrf_k + child_rank)
            parent_rrf = parent_weight / (rrf_k + parent_rank)
            hit.component_scores.update({
                "child_raw": hit.score,
                "child_rrf": child_rrf,
                "parent_raw": parent_scores[parent_id],
                "parent_rrf": parent_rrf,
            })
            hit.score = child_rrf + parent_rrf
            hit.source = "parent_to_child"
        children.sort(key=lambda hit: hit.score, reverse=True)
        return children

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
        eligible = self._eligible_record_ids(processed)
        self.last_query_context = {
            "temporal": processed.temporal.to_dict() if processed.temporal else None,
            "eligible_record_count": len(eligible),
            "filters": processed.filters,
        }
        query_vector = self.encoder.encode_query(processed.dense) if method in {"dense", "hybrid"} else None
        candidate_k = int(rcfg.get("candidate_top_k", 50))
        pcfg = rcfg.get("parent_child", {})
        if pcfg.get("enabled"):
            hits = self._parent_to_child_hits(processed, eligible, query_vector)
        else:
            hits = self._candidate_hits(
                processed,
                eligible,
                int(rcfg.get("sparse_top_k", 50)),
                int(rcfg.get("dense_top_k", 50)),
                candidate_k,
                query_vector,
            )
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
