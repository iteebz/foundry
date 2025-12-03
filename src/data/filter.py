from typing import Iterator
import hashlib


def dedupe(texts: Iterator[str]) -> Iterator[str]:
    seen: set[str] = set()
    for text in texts:
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            yield text


def length_filter(texts: Iterator[str], min_len: int = 1, max_len: int | None = None) -> Iterator[str]:
    for text in texts:
        if len(text) >= min_len and (max_len is None or len(text) <= max_len):
            yield text
