from __future__ import annotations

from typing import Any
import numpy as np


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class SentenceTransformerEncoder:
    def __init__(self, cfg: dict[str, Any]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Dense retrieval/reranking dependencies are not installed. "
                "Run `pip install -e \".[dense]\"`."
            ) from exc
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "auto"))
        self.model = SentenceTransformer(
            cfg["model_name"],
            device=self.device,
            trust_remote_code=cfg.get("trust_remote_code", False),
            revision=cfg.get("revision"),
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        prefix = self.cfg.get("document_prefix", "")
        payload = [prefix + t for t in texts]
        return np.asarray(self.model.encode(
            payload,
            batch_size=int(self.cfg.get("batch_size", 8)),
            normalize_embeddings=bool(self.cfg.get("normalize_embeddings", True)),
            show_progress_bar=True,
            convert_to_numpy=True,
        ), dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        prefix = self.cfg.get("query_prefix", "")
        return np.asarray(self.model.encode(
            [prefix + text],
            batch_size=1,
            normalize_embeddings=bool(self.cfg.get("normalize_embeddings", True)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0], dtype="float32")
