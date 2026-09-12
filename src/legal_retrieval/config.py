from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import yaml

from .paths import project_root


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _load_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at: {path}")
    seen.add(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        base = _load_yaml(parent_path, seen)
        data = deep_merge(base, data)
    seen.remove(path)
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        # First respect a path relative to CWD; otherwise use project root.
        p = p if p.exists() else project_root() / p
    cfg = _load_yaml(p, set())
    validate_config(cfg)
    cfg["_config_path"] = str(p.resolve())
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    mode = cfg.get("representation", {}).get("mode")
    if mode not in {"hierarchical_units", "fixed_length"}:
        raise ValueError(f"representation.mode must be hierarchical_units or fixed_length, got {mode!r}")

    method = cfg.get("retrieval", {}).get("method")
    if method not in {"sparse", "dense", "hybrid"}:
        raise ValueError(f"retrieval.method must be sparse/dense/hybrid, got {method!r}")

    sparse_enabled = bool(cfg.get("indexing", {}).get("sparse", {}).get("enabled"))
    dense_enabled = bool(cfg.get("indexing", {}).get("dense", {}).get("enabled"))
    sparse_backend = cfg.get("indexing", {}).get("sparse", {}).get("backend")
    if sparse_enabled and sparse_backend != "bm25":
        raise ValueError(f"Unsupported indexing.sparse.backend={sparse_backend!r}; currently only bm25 is available")
    if method in {"sparse", "hybrid"} and not sparse_enabled:
        raise ValueError(f"retrieval.method={method} requires indexing.sparse.enabled=true")
    if method in {"dense", "hybrid"} and not dense_enabled:
        raise ValueError(f"retrieval.method={method} requires indexing.dense.enabled=true")

    fusion = cfg.get("fusion", {}).get("method", "none")
    if fusion not in {"none", "rrf", "weighted"}:
        raise ValueError(f"Unsupported fusion.method={fusion!r}")
    if method == "hybrid" and fusion == "none":
        raise ValueError("Hybrid retrieval requires fusion.method=rrf or weighted")

    rerank = cfg.get("reranking", {})
    if rerank.get("enabled") and not rerank.get("models"):
        raise ValueError("reranking.enabled=true but reranking.models is empty")

    metrics = cfg.get("evaluation", {}).get("metrics", [])
    allowed = {"hit_rate", "precision", "recall", "mrr", "ndcg", "map"}
    unknown = set(metrics) - allowed
    if unknown:
        raise ValueError(f"Unsupported evaluation metrics: {sorted(unknown)}")

    def positive(path: str, value: Any) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{path} must be a positive integer, got {value!r}")

    retrieval = cfg.get("retrieval", {})
    for key in ("sparse_top_k", "dense_top_k", "candidate_top_k"):
        positive(f"retrieval.{key}", retrieval.get(key))
    positive("ranking.final_top_k", cfg.get("ranking", {}).get("final_top_k"))
    for value in cfg.get("evaluation", {}).get("k_values", []):
        positive("evaluation.k_values[]", value)

    dedupe = cfg.get("ranking", {}).get("deduplicate_by")
    if dedupe not in {"none", "record_id", "source_unit"}:
        raise ValueError(f"Unsupported ranking.deduplicate_by={dedupe!r}")
    match_by = cfg.get("evaluation", {}).get("match_by")
    if match_by not in {"record_id", "source_unit", "doc_id"}:
        raise ValueError(f"Unsupported evaluation.match_by={match_by!r}")

    fixed = cfg.get("representation", {}).get("fixed_length", {})
    size, overlap = fixed.get("chunk_size"), fixed.get("overlap")
    if mode == "fixed_length":
        positive("representation.fixed_length.chunk_size", size)
        if not isinstance(overlap, int) or isinstance(overlap, bool) or not 0 <= overlap < size:
            raise ValueError("representation.fixed_length.overlap must satisfy 0 <= overlap < chunk_size")

    if fusion == "weighted":
        normalization = cfg.get("fusion", {}).get("score_normalization", "minmax")
        if normalization != "minmax":
            raise ValueError("Only fusion.score_normalization=minmax is currently supported")
    fusion_cfg = cfg.get("fusion", {})
    weights = [float(fusion_cfg.get("sparse_weight", 0.0)), float(fusion_cfg.get("dense_weight", 0.0))]
    if any(weight < 0 for weight in weights) or (method == "hybrid" and not any(weights)):
        raise ValueError("Fusion weights must be non-negative and at least one must be positive")

    if rerank.get("enabled") and rerank.get("aggregation", "weighted") not in {"weighted", "rrf"}:
        raise ValueError("reranking.aggregation must be weighted or rrf")
