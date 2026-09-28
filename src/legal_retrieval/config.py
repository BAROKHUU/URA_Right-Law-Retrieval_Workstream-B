from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any
import yaml

from .paths import project_root


# These fields are expected to differ between experiment files but do not alter
# the scientific treatment. They are therefore excluded from hypothesis diffing.
_HYPOTHESIS_DIFF_IGNORES = (
    "hypothesis",
    "indexing.output_dir",
)
_HF_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


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


def _resolve_config_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        # First respect a path relative to CWD; otherwise use project root.
        p = p if p.exists() else project_root() / p
    return p.resolve()


def _immediate_parent_path(path: Path) -> Path | None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.get("extends")
    return (path.parent / parent).resolve() if parent else None


def _is_ignored_diff(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") or path.startswith(prefix + "[")
               for prefix in _HYPOTHESIS_DIFF_IGNORES)


def _diff_paths(baseline: Any, current: Any, prefix: str = "") -> list[str]:
    """Return concrete leaf paths whose resolved values differ."""
    if isinstance(baseline, dict) and isinstance(current, dict):
        changes: list[str] = []
        for key in sorted(set(baseline) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in baseline or key not in current:
                if not _is_ignored_diff(path):
                    changes.append(path)
                continue
            changes.extend(_diff_paths(baseline[key], current[key], path))
        return changes

    if isinstance(baseline, list) and isinstance(current, list):
        changes = []
        common = min(len(baseline), len(current))
        for i in range(common):
            path = f"{prefix}[{i}]"
            changes.extend(_diff_paths(baseline[i], current[i], path))
        for i in range(common, max(len(baseline), len(current))):
            path = f"{prefix}[{i}]"
            if not _is_ignored_diff(path):
                changes.append(path)
        return changes

    if baseline != current and prefix and not _is_ignored_diff(prefix):
        return [prefix]
    return []


def _change_is_declared(change: str, declared: str) -> bool:
    """A declaration may name a leaf or intentionally authorize a subtree/list."""
    return (
        change == declared
        or change.startswith(declared + ".")
        or change.startswith(declared + "[")
    )


def validate_hypothesis_changes(cfg: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    declared = list(cfg.get("hypothesis", {}).get("variables", []) or [])
    if not all(isinstance(x, str) and x.strip() for x in declared):
        raise ValueError("hypothesis.variables must be a list of non-empty config paths")
    declared = sorted(dict.fromkeys(x.strip() for x in declared))

    actual = sorted(_diff_paths(baseline or {}, cfg)) if baseline is not None else []
    undeclared = [c for c in actual if not any(_change_is_declared(c, d) for d in declared)]

    report = {
        "declared_changes": declared,
        "actual_changes": actual,
        "undeclared_changes": undeclared,
    }
    if undeclared:
        raise ValueError(
            "Hypothesis config changed variables outside hypothesis.variables.\n"
            f"Declared changes = {declared}\n"
            f"Actual changes = {actual}\n"
            f"Undeclared changes = {undeclared}"
        )
    return report


def load_config(path: str | Path) -> dict[str, Any]:
    p = _resolve_config_path(path)
    cfg = _load_yaml(p, set())
    validate_config(cfg)

    parent_path = _immediate_parent_path(p)
    baseline = _load_yaml(parent_path, set()) if parent_path is not None else None
    diff_report = validate_hypothesis_changes(cfg, baseline)

    cfg["_config_path"] = str(p)
    cfg["_baseline_config_path"] = str(parent_path) if parent_path is not None else None
    cfg["_hypothesis_diff"] = diff_report
    return cfg


def _require_hf_commit(path: str, revision: Any) -> None:
    if not isinstance(revision, str) or not _HF_COMMIT_RE.fullmatch(revision):
        raise ValueError(
            f"{path} must be pinned to a 40-character Hugging Face commit hash "
            "when runtime.reproducibility_mode=final"
        )


def validate_config(cfg: dict[str, Any]) -> None:
    mode = cfg.get("representation", {}).get("mode")
    if mode not in {"hierarchical_units", "fixed_length", "parent_child"}:
        raise ValueError(
            "representation.mode must be hierarchical_units, fixed_length, or parent_child, "
            f"got {mode!r}"
        )

    method = cfg.get("retrieval", {}).get("method")
    if method not in {"sparse", "dense", "hybrid"}:
        raise ValueError(f"retrieval.method must be sparse/dense/hybrid, got {method!r}")

    sparse_enabled = bool(cfg.get("indexing", {}).get("sparse", {}).get("enabled"))
    dense_cfg = cfg.get("indexing", {}).get("dense", {})
    dense_enabled = bool(dense_cfg.get("enabled"))
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

    parent_child = cfg.get("representation", {}).get("parent_child", {})
    if mode == "parent_child":
        if not parent_child.get("child_unit_types"):
            raise ValueError("representation.parent_child.child_unit_types must not be empty")
        positive(
            "representation.parent_child.context.parent_heading_max_words",
            parent_child.get("context", {}).get("parent_heading_max_words"),
        )
        if parent_child.get("orphan_policy", "error") not in {"error", "skip"}:
            raise ValueError("representation.parent_child.orphan_policy must be error or skip")

    parent_retrieval = retrieval.get("parent_child", {})
    if parent_retrieval.get("enabled"):
        if mode != "parent_child":
            raise ValueError("retrieval.parent_child.enabled requires representation.mode=parent_child")
        if not parent_child.get("include_parent_records"):
            raise ValueError(
                "retrieval.parent_child.enabled requires representation.parent_child.include_parent_records=true"
            )
        for key in ("parent_top_k", "child_top_k", "rrf_k"):
            positive(f"retrieval.parent_child.{key}", parent_retrieval.get(key))
        for key in ("parent_weight", "child_weight"):
            if float(parent_retrieval.get(key, 0.0)) < 0:
                raise ValueError(f"retrieval.parent_child.{key} must be non-negative")

    temporal = cfg.get("temporal", {})
    if temporal.get("missing_date_policy", "error") not in {"error", "exclude", "include"}:
        raise ValueError("temporal.missing_date_policy must be error, exclude, or include")

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

    reproducibility_mode = cfg.get("runtime", {}).get("reproducibility_mode", "development")
    if reproducibility_mode not in {"development", "final"}:
        raise ValueError("runtime.reproducibility_mode must be development or final")
    if reproducibility_mode == "final":
        if dense_enabled:
            _require_hf_commit("indexing.dense.revision", dense_cfg.get("revision"))
        if rerank.get("enabled"):
            for i, spec in enumerate(rerank.get("models", [])):
                _require_hf_commit(f"reranking.models[{i}].revision", spec.get("revision"))
