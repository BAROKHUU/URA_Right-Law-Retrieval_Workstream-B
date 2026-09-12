from __future__ import annotations

import re
import unicodedata


_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def fold_vietnamese(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def sparse_normalize(text: str, lowercase: bool = True, fold: bool = True) -> str:
    out = normalize_whitespace(text)
    if fold:
        out = fold_vietnamese(out)
    if lowercase:
        out = out.lower()
    return out


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())
