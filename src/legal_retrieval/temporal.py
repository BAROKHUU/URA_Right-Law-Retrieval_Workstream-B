from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import Any


_FULL_DATE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|2100)(?!\d)")
_TEMPORAL_YEAR_CUE = re.compile(
    r"\b(?:năm|nam|vào|vao|tại|tai|thời điểm|thoi diem|trước|truoc|sau|từ|tu|đến|den)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalConstraint:
    mode: str
    start: date | None
    end: date | None
    explicit: bool
    semantic_query: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "explicit": self.explicit,
        }


def parse_legal_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported legal date {value!r}; use DD/MM/YYYY or YYYY-MM-DD")


def _reference_date(cfg: dict[str, Any]) -> date:
    configured = cfg.get("reference_date")
    return parse_legal_date(configured) if configured else date.today()


def parse_temporal_query(query: str, cfg: dict[str, Any]) -> TemporalConstraint:
    """Extract a conservative temporal constraint from a Vietnamese query.

    A bare four-digit year is only temporal when the query has a temporal cue,
    preventing document names such as "Luật Đất đai 2024" from automatically
    becoming an as-of query.
    """
    lowered = query.lower()
    dates: list[tuple[date, tuple[int, int]]] = []
    for match in _FULL_DATE.finditer(query):
        day, month, year = map(int, match.groups())
        dates.append((date(year, month, day), match.span()))
    for match in _ISO_DATE.finditer(query):
        year, month, day = map(int, match.groups())
        dates.append((date(year, month, day), match.span()))

    years: list[tuple[int, tuple[int, int]]] = []
    if not dates and _TEMPORAL_YEAR_CUE.search(lowered):
        years = [(int(m.group(1)), m.span()) for m in _YEAR.finditer(query)]

    spans = [span for _, span in dates] + [span for _, span in years]
    semantic_query = query
    if cfg.get("strip_time_from_query", True):
        for start, end in sorted(spans, reverse=True):
            semantic_query = semantic_query[:start] + " " + semantic_query[end:]
        semantic_query = " ".join(semantic_query.split())

    if not dates and not years:
        ref = _reference_date(cfg)
        return TemporalConstraint("current", ref, ref, False, semantic_query)

    before = bool(re.search(r"\b(?:trước|truoc)\b", lowered))
    after = bool(re.search(r"\b(?:sau)\b", lowered))
    between = len(dates) >= 2 or len(years) >= 2

    if dates:
        values = sorted(value for value, _ in dates)
        if between:
            return TemporalConstraint("between", values[0], values[-1], True, semantic_query)
        point = values[0]
        mode = "before" if before else "after" if after else "as_of"
        return TemporalConstraint(mode, point, point, True, semantic_query)

    values = sorted(year for year, _ in years)
    if between:
        return TemporalConstraint("between", date(values[0], 1, 1), date(values[-1], 12, 31), True, semantic_query)
    year = values[0]
    if before:
        point = date(year, 1, 1)
        return TemporalConstraint("before", point, point, True, semantic_query)
    if after:
        point = date(year, 12, 31)
        return TemporalConstraint("after", point, point, True, semantic_query)
    return TemporalConstraint("as_of", date(year, 1, 1), date(year, 12, 31), True, semantic_query)


def record_matches_temporal(metadata: dict[str, Any], constraint: TemporalConstraint, missing_policy: str) -> bool:
    try:
        effective_from = parse_legal_date(metadata.get("effective_from"))
        effective_to = parse_legal_date(metadata.get("effective_to"))
    except ValueError:
        if missing_policy == "error":
            raise
        return missing_policy == "include"

    if effective_from is None:
        if missing_policy == "error":
            raise ValueError(f"Record {metadata.get('unit_id')} has no effective_from")
        return missing_policy == "include"

    start = constraint.start
    end = constraint.end
    assert start is not None and end is not None
    if constraint.mode == "before":
        return effective_from < start
    if constraint.mode == "after":
        return effective_from > end
    # current, as_of and between mean that the validity interval overlaps the
    # requested interval. For a precise date start == end.
    return effective_from <= end and (effective_to is None or effective_to >= start)
