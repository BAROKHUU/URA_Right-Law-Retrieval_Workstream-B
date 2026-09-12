from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RetrievalRecord:
    record_id: str
    text: str
    sparse_text: str
    source_unit_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalRecord":
        return cls(**data)


@dataclass
class SearchHit:
    record_id: str
    score: float
    rank: int = 0
    source: str = ""
    record: RetrievalRecord | None = None
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "rank": self.rank,
            "score": float(self.score),
            "source": self.source,
            "component_scores": {k: float(v) for k, v in self.component_scores.items()},
        }
        if self.record:
            payload["unit_id"] = self.record.record_id
            payload["source_unit_ids"] = self.record.source_unit_ids
            payload["metadata"] = self.record.metadata
            if include_text:
                payload["text"] = self.record.text
        return payload
