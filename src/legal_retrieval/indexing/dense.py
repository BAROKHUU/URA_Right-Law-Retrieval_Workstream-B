from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json
import numpy as np

from ..models import SearchHit


class BaseDenseIndex(ABC):
    @abstractmethod
    def build(self, embeddings: np.ndarray, record_ids: list[str]) -> None: ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path, cfg: dict) -> "BaseDenseIndex": ...


_DENSE_BACKENDS: dict[str, type[BaseDenseIndex]] = {}


def register_dense_backend(name: str, cls: type[BaseDenseIndex]) -> None:
    _DENSE_BACKENDS[name] = cls


def create_dense_backend(name: str, cfg: dict) -> BaseDenseIndex:
    if name not in _DENSE_BACKENDS:
        raise ValueError(f"Unknown dense backend {name!r}. Registered: {sorted(_DENSE_BACKENDS)}")
    return _DENSE_BACKENDS[name](cfg)


def load_dense_backend(name: str, directory: Path, cfg: dict) -> BaseDenseIndex:
    if name not in _DENSE_BACKENDS:
        raise ValueError(f"Unknown dense backend {name!r}. Registered: {sorted(_DENSE_BACKENDS)}")
    return _DENSE_BACKENDS[name].load(directory, cfg)


class NumpyFlatIndex(BaseDenseIndex):
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.embeddings: np.ndarray | None = None
        self.record_ids: list[str] = []

    def build(self, embeddings: np.ndarray, record_ids: list[str]) -> None:
        self.embeddings = np.asarray(embeddings, dtype="float32")
        self.record_ids = list(record_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        if self.embeddings is None:
            raise RuntimeError("Index not built")
        q = np.asarray(query_embedding, dtype="float32").reshape(-1)
        scores = self.embeddings @ q
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [SearchHit(record_id=self.record_ids[i], score=float(scores[i]), rank=r + 1, source="dense")
                for r, i in enumerate(idx)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.embeddings is None:
            raise RuntimeError("Index not built")
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "ids.json").write_text(json.dumps(self.record_ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, cfg: dict) -> "NumpyFlatIndex":
        obj = cls(cfg)
        obj.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        obj.record_ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return obj


class FaissDenseIndex(BaseDenseIndex):
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.index = None
        self.record_ids: list[str] = []

    @staticmethod
    def _faiss():
        try:
            import faiss
            return faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS backend requested but faiss-cpu is not installed. "
                "Run `pip install -e \".[dense]\"` or set indexing.dense.backend=numpy_flat."
            ) from exc

    def build(self, embeddings: np.ndarray, record_ids: list[str]) -> None:
        faiss = self._faiss()
        x = np.ascontiguousarray(embeddings, dtype="float32")
        dim = x.shape[1]
        index_type = self.cfg.get("index_type", "FlatIP")
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(dim)
        elif index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            # Allows future FAISS factory strings such as HNSW32 or IVF256,Flat.
            self.index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)
            if not self.index.is_trained:
                self.index.train(x)
        self.index.add(x)
        self.record_ids = list(record_ids)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchHit]:
        if self.index is None:
            raise RuntimeError("Index not built")
        q = np.ascontiguousarray(query_embedding, dtype="float32").reshape(1, -1)
        k = min(top_k, len(self.record_ids))
        scores, indices = self.index.search(q, k)
        hits = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            # SearchHit scores consistently mean "higher is better". FAISS L2
            # returns distances (lower is better), so expose their negative.
            value = -float(score) if self.cfg.get("index_type", "FlatIP") == "FlatL2" else float(score)
            hits.append(SearchHit(record_id=self.record_ids[idx], score=value, rank=rank, source="dense"))
        return hits

    def save(self, directory: Path) -> None:
        faiss = self._faiss()
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is None:
            raise RuntimeError("Index not built")
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "ids.json").write_text(json.dumps(self.record_ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, cfg: dict) -> "FaissDenseIndex":
        faiss = cls._faiss()
        obj = cls(cfg)
        obj.index = faiss.read_index(str(directory / "index.faiss"))
        obj.record_ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return obj


register_dense_backend("numpy_flat", NumpyFlatIndex)
register_dense_backend("faiss", FaissDenseIndex)
