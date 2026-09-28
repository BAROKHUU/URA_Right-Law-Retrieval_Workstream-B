from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .text import normalize_whitespace, sparse_normalize
from .temporal import TemporalConstraint, parse_temporal_query


@dataclass
class ProcessedQuery:
    raw: str
    dense: str
    sparse: str
    filters: dict[str, Any]
    expansions: list[str]
    temporal: TemporalConstraint | None = None


class QueryProcessor:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg.get("query_processing", {})
        self.temporal_cfg = cfg.get("temporal", {})

    def process(self, query: str, filters: dict[str, Any] | None = None) -> ProcessedQuery:
        normalized = normalize_whitespace(query) if self.cfg.get("normalize_whitespace", True) else query
        temporal_cfg = self.temporal_cfg
        temporal = parse_temporal_query(normalized, temporal_cfg) if temporal_cfg.get("enabled") else None
        semantic_query = temporal.semantic_query if temporal else normalized
        expansion_cfg = self.cfg.get("query_expansion", {})
        expansions: list[str] = []
        if expansion_cfg.get("enabled") and expansion_cfg.get("method") == "static_map":
            lower = semantic_query.lower()
            for trigger, terms in expansion_cfg.get("static_map", {}).items():
                if trigger.lower() in lower:
                    expansions.extend(str(x) for x in terms)
        # Stable deduplication.
        expansions = list(dict.fromkeys(expansions))
        expanded = semantic_query if not expansions else semantic_query + " " + " ".join(expansions)
        sparse = sparse_normalize(
            expanded,
            lowercase=self.cfg.get("lowercase_for_sparse", True),
            fold=self.cfg.get("fold_vietnamese_for_sparse", True),
        )
        filter_cfg = self.cfg.get("metadata_filter", {})
        applied_filters = (filters or {}) if filter_cfg.get("enabled", True) else {}
        return ProcessedQuery(
            raw=query,
            dense=expanded,
            sparse=sparse,
            filters=applied_filters,
            expansions=expansions,
            temporal=temporal,
        )
