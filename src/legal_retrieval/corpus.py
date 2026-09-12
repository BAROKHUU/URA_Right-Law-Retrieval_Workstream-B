from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .models import RetrievalRecord
from .paths import resolve_project_path
from .text import sparse_normalize, normalize_whitespace


def load_enriched_units(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    corpus_cfg = cfg["corpus"]
    pattern_path = resolve_project_path(corpus_cfg["input_glob"])
    assert pattern_path is not None
    files = sorted(glob.glob(str(pattern_path)))
    if not files:
        raise FileNotFoundError(f"No enriched JSON files match: {pattern_path}")

    required = corpus_cfg.get("required_fields", [])
    include_types = set(corpus_cfg.get("include_unit_types", []))
    units: list[dict[str, Any]] = []
    for file in files:
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {file}")
        for row in data:
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(f"{file}: record missing fields {missing}")
            if include_types and row.get("unit_type") not in include_types:
                continue
            units.append(row)
    return units


def build_index_text(unit: dict[str, Any], rep_cfg: dict[str, Any]) -> str:
    parts: list[str] = []
    for spec in rep_cfg.get("fields", []):
        if not spec.get("enabled", True):
            continue
        value = unit.get(spec["name"])
        if value is None or str(value).strip() == "":
            continue
        parts.append(f"{spec.get('prefix', '')}{normalize_whitespace(str(value))}")
    return rep_cfg.get("separator", "\n").join(parts)


def _metadata_from_unit(unit: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in unit.items() if k not in {"text", "text_folded"}}


def _hierarchical_records(units: list[dict[str, Any]], cfg: dict[str, Any]) -> list[RetrievalRecord]:
    rep_cfg = cfg["representation"]
    qp = cfg.get("query_processing", {})
    records: list[RetrievalRecord] = []
    for unit in units:
        text = build_index_text(unit, rep_cfg)
        records.append(
            RetrievalRecord(
                record_id=unit["unit_id"],
                text=text,
                sparse_text=sparse_normalize(
                    text,
                    lowercase=qp.get("lowercase_for_sparse", True),
                    fold=qp.get("fold_vietnamese_for_sparse", True),
                ),
                source_unit_ids=[unit["unit_id"]],
                metadata=_metadata_from_unit(unit),
            )
        )
    return records


def _windows(words: list[tuple[str, str]], size: int, overlap: int):
    if size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            yield start, chunk
        if start + size >= len(words):
            break


def _fixed_length_records(units: list[dict[str, Any]], cfg: dict[str, Any]) -> list[RetrievalRecord]:
    rep_cfg = cfg["representation"]
    fixed = rep_cfg["fixed_length"]
    if fixed.get("unit", "words") != "words":
        raise ValueError("Current fixed_length implementation supports unit=words")
    allowed_types = set(fixed.get("source_unit_types", []))
    size = int(fixed.get("chunk_size", 220))
    overlap = int(fixed.get("overlap", 40))
    qp = cfg.get("query_processing", {})

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        if not allowed_types or unit.get("unit_type") in allowed_types:
            by_doc[unit["doc_id"]].append(unit)

    records: list[RetrievalRecord] = []
    for doc_id, doc_units in by_doc.items():
        stream: list[tuple[str, str]] = []
        for unit in doc_units:
            text = build_index_text(unit, rep_cfg)
            # Attach each word to the original legal unit for evaluation mapping.
            stream.extend((word, unit["unit_id"]) for word in text.split())
        for idx, (_, chunk) in enumerate(_windows(stream, size, overlap), start=1):
            chunk_text = " ".join(word for word, _ in chunk)
            source_ids = list(dict.fromkeys(uid for _, uid in chunk))
            representative = next((u for u in doc_units if u["unit_id"] == source_ids[0]), doc_units[0])
            metadata = _metadata_from_unit(representative)
            metadata.update({
                "doc_id": doc_id,
                "unit_type": "fixed_chunk",
                "chunk_index": idx,
                "source_unit_count": len(source_ids),
            })
            rid = f"{doc_id}__fixedchunk-{idx:05d}"
            records.append(
                RetrievalRecord(
                    record_id=rid,
                    text=chunk_text,
                    sparse_text=sparse_normalize(
                        chunk_text,
                        lowercase=qp.get("lowercase_for_sparse", True),
                        fold=qp.get("fold_vietnamese_for_sparse", True),
                    ),
                    source_unit_ids=source_ids,
                    metadata=metadata,
                )
            )
    return records


def build_records(cfg: dict[str, Any]) -> list[RetrievalRecord]:
    units = load_enriched_units(cfg)
    mode = cfg["representation"]["mode"]
    if mode == "hierarchical_units":
        return _hierarchical_records(units, cfg)
    if mode == "fixed_length":
        return _fixed_length_records(units, cfg)
    raise ValueError(mode)
