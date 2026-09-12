from __future__ import annotations

import hashlib
import json
import logging
import glob
import os
import shutil
import uuid
from pathlib import Path
from typing import Any
import yaml

from ..corpus import build_records
from ..embedding import SentenceTransformerEncoder
from ..io_utils import save_records, load_records
from ..models import RetrievalRecord
from ..paths import resolve_project_path
from .sparse import BM25Index
from .dense import create_dense_backend, load_dense_backend

logger = logging.getLogger(__name__)
INDEX_ARTIFACT_FORMAT_VERSION = 2


class IndexManager:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        out = resolve_project_path(cfg["indexing"]["output_dir"])
        assert out is not None
        self.index_root = out
        # Indexes are content-addressed and immutable. Changing the corpus,
        # representation, tokenizer settings, model, or backend creates a new
        # sibling directory instead of replacing an earlier experiment.
        self.index_signature = self.signature()
        self.index_dir = self.index_root / self.index_signature

    def _signature_payload(self) -> dict[str, Any]:
        return {
            # Bump this when record/index serialization or construction
            # semantics change, so older artifacts remain untouched.
            "artifact_format_version": INDEX_ARTIFACT_FORMAT_VERSION,
            "corpus": self.cfg.get("corpus"),
            "representation": self.cfg.get("representation"),
            "query_processing": {
                k: v for k, v in self.cfg.get("query_processing", {}).items()
                if k in {"lowercase_for_sparse", "fold_vietnamese_for_sparse"}
            },
            "sparse": self.cfg.get("indexing", {}).get("sparse"),
            "dense": self.cfg.get("indexing", {}).get("dense"),
        }

    def _corpus_digest(self) -> str:
        pattern = resolve_project_path(self.cfg["corpus"]["input_glob"])
        assert pattern is not None
        h = hashlib.sha256()
        for filename in sorted(glob.glob(str(pattern))):
            path = Path(filename)
            h.update(path.name.encode("utf-8"))
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        return h.hexdigest()

    def signature(self) -> str:
        payload = self._signature_payload()
        payload["corpus_digest"] = self._corpus_digest()
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _is_complete(self) -> bool:
        if not self.manifest_path.exists() or not (self.index_dir / "records.jsonl").exists():
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if manifest.get("signature") != self.index_signature:
            return False
        sparse = self.cfg["indexing"]["sparse"]
        dense = self.cfg["indexing"]["dense"]
        if sparse.get("enabled") and not (self.index_dir / "sparse" / "bm25.pkl").exists():
            return False
        if dense.get("enabled") and not (self.index_dir / "dense").exists():
            return False
        return True

    def needs_build(self) -> bool:
        return not self._is_complete()

    def ensure(self) -> None:
        if self.needs_build():
            if not self.cfg.get("runtime", {}).get("auto_build_index", True) and not self.cfg.get("runtime", {}).get("force_rebuild", False):
                raise FileNotFoundError(f"Index is missing/stale: {self.index_dir}")
            self.build()

    def build(self) -> None:
        if self._is_complete():
            logger.info("Immutable index already exists; reusing %s", self.index_dir)
            return
        if self.index_dir.exists():
            # Preserve an interrupted/corrupt artifact for inspection while
            # freeing the canonical content-addressed path for an atomic build.
            preserved = self.index_root / f".{self.index_signature}.incomplete-{uuid.uuid4().hex}"
            os.replace(self.index_dir, preserved)
            logger.warning("Preserved incomplete index at %s", preserved)

        logger.info("Building retrieval records...")
        records = build_records(self.cfg)
        self.index_root.mkdir(parents=True, exist_ok=True)
        build_dir = self.index_root / f".{self.index_signature}.building-{uuid.uuid4().hex}"
        build_dir.mkdir(parents=False, exist_ok=False)
        save_records(build_dir / "records.jsonl", records)
        logger.info("Built %d retrieval records", len(records))

        sparse_cfg = self.cfg["indexing"]["sparse"]
        if sparse_cfg.get("enabled"):
            logger.info("Building BM25 index...")
            index = BM25Index(k1=sparse_cfg.get("k1", 1.5), b=sparse_cfg.get("b", 0.75))
            index.build(records)
            index.save(build_dir / "sparse" / "bm25.pkl")

        dense_cfg = self.cfg["indexing"]["dense"]
        if dense_cfg.get("enabled"):
            logger.info("Encoding %d records with %s...", len(records), dense_cfg["model_name"])
            encoder = SentenceTransformerEncoder(dense_cfg)
            embeddings = encoder.encode_documents([r.text for r in records])
            backend = create_dense_backend(dense_cfg.get("backend", "faiss"), dense_cfg)
            backend.build(embeddings, [r.record_id for r in records])
            backend.save(build_dir / "dense")

        manifest = {
            "signature": self.signature(),
            "artifact_format_version": INDEX_ARTIFACT_FORMAT_VERSION,
            "record_count": len(records),
            "representation_mode": self.cfg["representation"]["mode"],
            "sparse_enabled": bool(sparse_cfg.get("enabled")),
            "dense_enabled": bool(dense_cfg.get("enabled")),
            "dense_backend": dense_cfg.get("backend") if dense_cfg.get("enabled") else None,
            "dense_model": dense_cfg.get("model_name") if dense_cfg.get("enabled") else None,
        }
        (build_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (build_dir / "resolved_index_config.yaml").write_text(
            yaml.safe_dump(self._signature_payload(), allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        try:
            os.replace(build_dir, self.index_dir)
        except OSError:
            # Another process may have completed the same content-addressed
            # artifact first. Never replace that completed artifact.
            if not self.manifest_path.exists():
                raise
            shutil.rmtree(build_dir)
        logger.info("Index ready at %s", self.index_dir)

    def load_records(self) -> list[RetrievalRecord]:
        return load_records(self.index_dir / "records.jsonl")

    def load_sparse(self) -> BM25Index:
        return BM25Index.load(self.index_dir / "sparse" / "bm25.pkl")

    def load_dense(self):
        dense_cfg = self.cfg["indexing"]["dense"]
        return load_dense_backend(dense_cfg.get("backend", "faiss"), self.index_dir / "dense", dense_cfg)
