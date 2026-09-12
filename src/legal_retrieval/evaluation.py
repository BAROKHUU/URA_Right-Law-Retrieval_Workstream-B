from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SearchHit
from .paths import resolve_project_path


@dataclass
class BenchmarkQuery:
    query_id: str
    query: str
    relevant_ids: list[str]
    filters: dict[str, Any]


def load_benchmark(cfg: dict[str, Any]) -> list[BenchmarkQuery]:
    ecfg = cfg["evaluation"]
    raw_path = ecfg.get("benchmark_path")
    if not raw_path:
        return []
    path = resolve_project_path(raw_path)
    assert path is not None
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("queries", [])
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("benchmark_path must be .jsonl, .json, or .csv")

    qf = ecfg.get("query_field", "query")
    qid = ecfg.get("query_id_field", "query_id")
    rf = ecfg.get("relevant_ids_field", "relevant_unit_ids")
    ff = ecfg.get("filters_field", "filters")
    out = []
    for i, row in enumerate(rows):
        relevant = row.get(rf, []) or []
        filters = row.get(ff, {}) or {}
        if isinstance(relevant, str):
            try:
                parsed = json.loads(relevant)
                relevant = parsed if isinstance(parsed, list) else [relevant]
            except json.JSONDecodeError:
                relevant = [x.strip() for x in relevant.split("|") if x.strip()]
        if isinstance(filters, str):
            filters = json.loads(filters) if filters.strip() else {}
        out.append(BenchmarkQuery(
            query_id=str(row.get(qid, i)),
            query=str(row[qf]),
            relevant_ids=[str(x) for x in relevant],
            filters=filters,
        ))
    return out


def _hit_relevant_ids(hit: SearchHit, match_by: str) -> set[str]:
    if match_by == "record_id":
        return {hit.record_id}
    if match_by == "source_unit":
        if hit.record:
            return set(hit.record.source_unit_ids)
        return {hit.record_id}
    if match_by == "doc_id":
        if hit.record:
            value = hit.record.metadata.get("doc_id")
            return {str(value)} if value is not None else set()
        return set()
    raise ValueError(f"Unknown evaluation.match_by={match_by}")


def relevance_trace(hits: list[SearchHit], relevant: set[str], match_by: str) -> tuple[list[int], list[int]]:
    # Overlapping chunks can point to the same legal unit. Count a relevant
    # entity only on its first occurrence so metrics cannot be inflated by
    # returning several chunks that cover the same ground truth.
    seen: set[str] = set()
    binary: list[int] = []
    newly_covered: list[int] = []
    for hit in hits:
        matched = (_hit_relevant_ids(hit, match_by) & relevant) - seen
        binary.append(1 if matched else 0)
        newly_covered.append(len(matched))
        seen.update(matched)
    return binary, newly_covered


def binary_relevance(hits: list[SearchHit], relevant: set[str], match_by: str) -> list[int]:
    return relevance_trace(hits, relevant, match_by)[0]


def metrics_at_k(
    binary: list[int], relevant_count: int, k: int, newly_covered: list[int] | None = None
) -> dict[str, float]:
    rel = binary[:k]
    retrieved_relevant = sum(rel)
    precision = retrieved_relevant / k if k else 0.0
    covered = sum((newly_covered or binary)[:k])
    recall = min(covered / relevant_count, 1.0) if relevant_count else 0.0
    hit_rate = 1.0 if retrieved_relevant else 0.0
    rr = next((1.0 / (i + 1) for i, x in enumerate(rel) if x), 0.0)
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(rel))
    ideal_n = min(relevant_count, k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    ndcg = min(dcg / idcg, 1.0) if idcg else 0.0
    running = 0
    ap_sum = 0.0
    for i, x in enumerate(rel, start=1):
        if x:
            running += 1
            ap_sum += running / i
    ap = min(ap_sum / min(relevant_count, k), 1.0) if relevant_count else 0.0
    return {
        "hit_rate": hit_rate,
        "precision": precision,
        "recall": recall,
        "mrr": rr,
        "ndcg": ndcg,
        "map": ap,
    }


def evaluate_query(hits: list[SearchHit], relevant_ids: list[str], cfg: dict[str, Any]) -> dict[str, float]:
    ecfg = cfg["evaluation"]
    relevant = set(relevant_ids)
    binary, newly_covered = relevance_trace(hits, relevant, ecfg.get("match_by", "source_unit"))
    selected = set(ecfg.get("metrics", []))
    out: dict[str, float] = {}
    for k in ecfg.get("k_values", [10]):
        values = metrics_at_k(binary, len(relevant), int(k), newly_covered)
        for name, value in values.items():
            if name in selected:
                out[f"{name}@{k}"] = value
    return out


def aggregate_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = sorted(set().union(*(row.keys() for row in per_query)))
    return {key: sum(row.get(key, 0.0) for row in per_query) / len(per_query) for key in keys}
