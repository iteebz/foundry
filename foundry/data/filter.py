import hashlib
from collections.abc import Iterator


def dedupe(texts: Iterator[str], hash_algo: str = "sha256") -> Iterator[str]:
    seen: set[str] = set()
    hashlib.new(hash_algo)

    for text in texts:
        h = hashlib.new(hash_algo, text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            yield text


def length_filter(
    texts: Iterator[str], min_len: int = 1, max_len: int | None = None, count_tokens: bool = False
) -> Iterator[str]:
    for text in texts:
        length = len(text.split()) if count_tokens else len(text)
        if length >= min_len and (max_len is None or length <= max_len):
            yield text
