from __future__ import annotations

from collections import Counter
import math
import pickle
from pathlib import Path

from ..models import RetrievalRecord, SearchHit
from ..text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avgdl = 0.0
        self.term_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}

    def build(self, records: list[RetrievalRecord]) -> None:
        self.doc_ids = [r.record_id for r in records]
        tokenized = [tokenize(r.sparse_text) for r in records]
        self.doc_lengths = [len(tokens) for tokens in tokenized]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in tokenized]
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(records)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        q_tokens = tokenize(query)
        scores: list[tuple[float, str]] = []
        for i, tf in enumerate(self.term_freqs):
            dl = self.doc_lengths[i]
            score = 0.0
            for term in q_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-12))
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((score, self.doc_ids[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [SearchHit(record_id=doc_id, score=score, rank=i + 1, source="sparse")
                for i, (score, doc_id) in enumerate(scores[:top_k])]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        obj = cls()
        with path.open("rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
