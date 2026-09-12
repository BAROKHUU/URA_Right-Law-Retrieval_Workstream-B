from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any

from .models import RetrievalRecord


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_records(path: Path, records: list[RetrievalRecord]) -> None:
    write_jsonl(path, (r.to_dict() for r in records))


def load_records(path: Path) -> list[RetrievalRecord]:
    return [RetrievalRecord.from_dict(row) for row in read_jsonl(path)]
