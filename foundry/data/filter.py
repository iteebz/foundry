"""Reference-grade data quality filtering.

Design principles:
- Filters return scores (float 0-1), not bools — information preservation
- Filters are pure functions — state injected, not hidden
- Composition is AND over thresholds — order-independent
- Observable by default — every decision logged
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass
class FilterResult:
    sample: str
    scores: dict[str, float]
    passed: bool
    failed_filters: list[str] = field(default_factory=list)


@dataclass
class FilterStats:
    total: int = 0
    passed: int = 0
    failed: int = 0
    by_filter: dict[str, int] = field(default_factory=dict)
    score_sums: dict[str, float] = field(default_factory=dict)

    def record(self, result: FilterResult) -> None:
        self.total += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1
            for name in result.failed_filters:
                self.by_filter[name] = self.by_filter.get(name, 0) + 1
        for name, score in result.scores.items():
            self.score_sums[name] = self.score_sums.get(name, 0.0) + score

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def mean_scores(self) -> dict[str, float]:
        return {k: v / self.total for k, v in self.score_sums.items()} if self.total > 0 else {}

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "mean_scores": self.mean_scores(),
            "rejection_counts": self.by_filter,
        }


class Filter(ABC):
    name: str
    threshold: float

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abstractmethod
    def score(self, sample: str) -> float:
        """Pure function: sample -> score in [0, 1]. Higher = better."""
        ...

    def __call__(self, sample: str) -> float:
        return self.score(sample)


class LengthFilter(Filter):
    name = "length"

    def __init__(
        self,
        min_len: int = 1,
        max_len: int = 100_000,
        count_tokens: bool = False,
        threshold: float = 0.5,
    ):
        super().__init__(threshold)
        self.min_len = min_len
        self.max_len = max_len
        self.count_tokens = count_tokens

    def score(self, sample: str) -> float:
        length = len(sample.split()) if self.count_tokens else len(sample)
        if length < self.min_len:
            return length / self.min_len if self.min_len > 0 else 0.0
        if length > self.max_len:
            return self.max_len / length if length > 0 else 0.0
        return 1.0


class RepetitionFilter(Filter):
    """Detect pathological repetition (the the the, character runs, etc.)."""

    name = "repetition"

    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)

    def score(self, sample: str) -> float:
        if not sample:
            return 0.0

        scores = []

        words = sample.split()
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            scores.append(unique_ratio)

        chars = list(sample)
        if len(chars) > 1:
            runs = 1
            for i in range(1, len(chars)):
                if chars[i] != chars[i - 1]:
                    runs += 1
            char_diversity = runs / len(chars)
            scores.append(char_diversity)

        ngram_scores = []
        for n in [2, 3, 4]:
            if len(words) >= n:
                ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
                if ngrams:
                    unique_ngrams = len(set(ngrams)) / len(ngrams)
                    ngram_scores.append(unique_ngrams)
        if ngram_scores:
            scores.append(min(ngram_scores))

        return min(scores) if scores else 1.0


class CharacterDistributionFilter(Filter):
    """Detect encoding errors, binary data, excessive special chars."""

    name = "char_distribution"

    def __init__(self, threshold: float = 0.6):
        super().__init__(threshold)

    def score(self, sample: str) -> float:
        if not sample:
            return 0.0

        printable = sum(1 for c in sample if c.isprintable() or c in "\n\t\r")
        printable_ratio = printable / len(sample)

        alpha_numeric = sum(1 for c in sample if c.isalnum() or c.isspace())
        alpha_ratio = alpha_numeric / len(sample)

        null_count = sample.count("\x00")
        null_penalty = 1.0 - (null_count / len(sample)) if null_count else 1.0

        return min(printable_ratio, alpha_ratio * 1.5, null_penalty)


class LanguageFilter(Filter):
    """Heuristic language detection (no dependencies)."""

    name = "language"

    COMMON_WORDS: ClassVar[dict[str, set[str]]] = {
        "en": {
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "is",
            "was",
            "are",
            "as",
        },
        "code": {
            "def",
            "class",
            "import",
            "return",
            "if",
            "else",
            "for",
            "while",
            "function",
            "const",
            "let",
            "var",
            "fn",
            "pub",
            "impl",
        },
    }

    def __init__(self, allow: list[str] | None = None, threshold: float = 0.3):
        super().__init__(threshold)
        self.allow = set(allow) if allow else {"en", "code"}

    def score(self, sample: str) -> float:
        words = set(sample.lower().split())
        if not words:
            return 0.0

        best_score = 0.0
        for lang in self.allow:
            if lang in self.COMMON_WORDS:
                common = self.COMMON_WORDS[lang]
                overlap = len(words & common)
                score = min(1.0, overlap / 5)
                best_score = max(best_score, score)

        return best_score


class PerplexityFilter(Filter):
    """Entropy-based gibberish detection (no external model needed)."""

    name = "perplexity"

    def __init__(self, max_entropy: float = 5.0, threshold: float = 0.5):
        super().__init__(threshold)
        self.max_entropy = max_entropy

    def score(self, sample: str) -> float:
        if len(sample) < 10:
            return 0.5

        char_counts: dict[str, int] = {}
        for c in sample.lower():
            char_counts[c] = char_counts.get(c, 0) + 1

        total = len(sample)
        entropy = -sum((count / total) * math.log2(count / total) for count in char_counts.values())

        if entropy < 1.0:
            return 0.3
        if entropy > self.max_entropy:
            return max(0.0, 1.0 - (entropy - self.max_entropy) / self.max_entropy)
        return 1.0


class URLDensityFilter(Filter):
    """Penalize samples that are mostly URLs."""

    name = "url_density"

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

    def __init__(self, max_url_ratio: float = 0.3, threshold: float = 0.5):
        super().__init__(threshold)
        self.max_url_ratio = max_url_ratio

    def score(self, sample: str) -> float:
        if not sample:
            return 1.0

        urls = self.URL_PATTERN.findall(sample)
        url_chars = sum(len(url) for url in urls)
        url_ratio = url_chars / len(sample)

        if url_ratio <= self.max_url_ratio:
            return 1.0
        return max(0.0, 1.0 - (url_ratio - self.max_url_ratio) / (1.0 - self.max_url_ratio))


class DedupFilter(Filter):
    """Exact dedup via hash. State (seen set) must be injected."""

    name = "dedup"

    def __init__(
        self, seen: set[str] | None = None, hash_algo: str = "sha256", threshold: float = 1.0
    ):
        super().__init__(threshold)
        self.seen = seen if seen is not None else set()
        self.hash_algo = hash_algo

    def score(self, sample: str) -> float:
        h = hashlib.new(self.hash_algo, sample.encode()).hexdigest()
        if h in self.seen:
            return 0.0
        self.seen.add(h)
        return 1.0


class MinHashDedupFilter(Filter):
    """Fuzzy near-duplicate detection via MinHash. State injected."""

    name = "minhash_dedup"

    def __init__(
        self,
        seen: list[set[int]] | None = None,
        num_hashes: int = 128,
        similarity_threshold: float = 0.8,
        ngram_size: int = 3,
        threshold: float = 1.0,
    ):
        super().__init__(threshold)
        self.seen = seen if seen is not None else []
        self.num_hashes = num_hashes
        self.similarity_threshold = similarity_threshold
        self.ngram_size = ngram_size

    def _stable_hash(self, shingle: str, seed: int) -> int:
        data = f"{seed}:{shingle}".encode("utf-8")
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return int.from_bytes(digest, "little")

    def _minhash(self, text: str) -> set[tuple[int, int]]:
        words = text.lower().split()
        if len(words) < self.ngram_size:
            shingles = {text.lower()}
        else:
            shingles = {
                " ".join(words[i : i + self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)
            }

        signature: set[tuple[int, int]] = set()
        for i in range(self.num_hashes):
            min_hash = min(self._stable_hash(shingle, i) for shingle in shingles) if shingles else 0
            signature.add((i, min_hash))
        return signature

    def _similarity(self, sig1: set[tuple[int, int]], sig2: set[tuple[int, int]]) -> float:
        return len(sig1 & sig2) / self.num_hashes

    def score(self, sample: str) -> float:
        sig = self._minhash(sample)

        for existing_sig in self.seen:
            if self._similarity(sig, existing_sig) >= self.similarity_threshold:
                return 0.0

        self.seen.append(sig)
        return 1.0


def default_filters() -> list[Filter]:
    return [
        LengthFilter(min_len=50, max_len=100_000),
        RepetitionFilter(),
        CharacterDistributionFilter(),
        LanguageFilter(),
        PerplexityFilter(),
        URLDensityFilter(),
    ]


@dataclass
class FilterPipeline:
    filters: list[Filter]
    stats: FilterStats = field(default_factory=FilterStats)
    log_fn: Callable[[FilterResult], None] | None = None

    def evaluate(self, sample: str) -> FilterResult:
        scores = {f.name: f.score(sample) for f in self.filters}
        failed = [f.name for f in self.filters if scores[f.name] < f.threshold]
        result = FilterResult(
            sample=sample, scores=scores, passed=len(failed) == 0, failed_filters=failed
        )
        self.stats.record(result)
        if self.log_fn:
            self.log_fn(result)
        return result

    def __call__(self, samples: Iterator[str]) -> Iterator[str]:
        for sample in samples:
            result = self.evaluate(sample)
            if result.passed:
                yield sample


def filter_samples(
    samples: Iterator[str],
    filters: list[Filter] | None = None,
    log_fn: Callable[[FilterResult], None] | None = None,
) -> tuple[Iterator[str], FilterStats]:
    pipeline = FilterPipeline(filters=filters or default_filters(), log_fn=log_fn)
    return pipeline(samples), pipeline.stats


def dedupe(texts: Iterator[str], hash_algo: str = "sha256") -> Iterator[str]:
    seen: set[str] = set()
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
