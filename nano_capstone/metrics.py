"""Quantitative text metrics for the next-character suppression experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Mapping


WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


def character_entropy(text: str) -> float:
    """Return Shannon entropy in bits per character."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def word_tokens(text: str) -> list[str]:
    """Extract lowercase word tokens while retaining internal apostrophes."""
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def vocabulary_match(text: str, reference_vocabulary: set[str]) -> float:
    """Return the fraction of word tokens found in a reference vocabulary."""
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    return sum(token in reference_vocabulary for token in tokens) / len(tokens)


def type_token_diversity(text: str) -> float:
    """Return unique word types divided by total word tokens."""
    tokens = word_tokens(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def rank_frequency_fit(text: str) -> tuple[float, float]:
    """Fit log word frequency against log rank and return slope and R-squared.

    Word types are ranked from most to least frequent. Ties are sorted
    alphabetically so the calculation is deterministic, although tied types have
    identical frequency values and therefore do not change the fitted line.
    """
    counts = Counter(word_tokens(text))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) < 2:
        return 0.0, 1.0

    log_ranks = [math.log(rank) for rank in range(1, len(ranked) + 1)]
    log_frequencies = [math.log(count) for _, count in ranked]
    mean_rank = sum(log_ranks) / len(log_ranks)
    mean_frequency = sum(log_frequencies) / len(log_frequencies)
    rank_variance = sum((value - mean_rank) ** 2 for value in log_ranks)
    slope = sum(
        (rank - mean_rank) * (frequency - mean_frequency)
        for rank, frequency in zip(log_ranks, log_frequencies)
    ) / rank_variance
    intercept = mean_frequency - slope * mean_rank
    residual_sum_squares = sum(
        (frequency - (intercept + slope * rank)) ** 2
        for rank, frequency in zip(log_ranks, log_frequencies)
    )
    total_sum_squares = sum(
        (frequency - mean_frequency) ** 2 for frequency in log_frequencies
    )
    r_squared = (
        1.0
        if math.isclose(total_sum_squares, 0.0)
        else 1.0 - (residual_sum_squares / total_sum_squares)
    )
    return slope, max(0.0, min(1.0, r_squared))


def character_ngrams(text: str, n: int) -> list[str]:
    if n < 1:
        raise ValueError("n must be at least 1")
    return [text[index : index + n] for index in range(len(text) - n + 1)]


def ngram_repetition(text: str, n: int = 4) -> float:
    """Return the fraction of character n-grams that repeat an earlier n-gram."""
    ngrams = character_ngrams(text, n)
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def ngram_distribution(text: str, n: int) -> dict[str, float]:
    """Return a normalized character n-gram frequency distribution."""
    counts = Counter(character_ngrams(text, n))
    total = sum(counts.values())
    if not total:
        return {}
    return {ngram: count / total for ngram, count in counts.items()}


def jensen_shannon_divergence(
    first: Mapping[str, float], second: Mapping[str, float]
) -> float:
    """Return base-2 Jensen-Shannon divergence, bounded between zero and one."""
    if not first and not second:
        return 0.0
    if not first or not second:
        return 1.0

    support = set(first) | set(second)
    midpoint = {
        item: (first.get(item, 0.0) + second.get(item, 0.0)) / 2
        for item in support
    }

    def kl_divergence(distribution: Mapping[str, float]) -> float:
        return sum(
            probability * math.log2(probability / midpoint[item])
            for item, probability in distribution.items()
            if probability > 0
        )

    return (kl_divergence(first) + kl_divergence(second)) / 2


@dataclass(frozen=True)
class TextMetrics:
    """Evaluate generated samples against one frozen reference text."""

    reference_text: str
    repetition_ngram: int = 4
    divergence_ngram: int = 3
    reference_vocabulary: frozenset[str] = field(init=False)
    reference_distribution: Mapping[str, float] = field(init=False)

    def __post_init__(self) -> None:
        if self.repetition_ngram < 1 or self.divergence_ngram < 1:
            raise ValueError("n-gram sizes must be at least 1")
        object.__setattr__(
            self,
            "reference_vocabulary",
            frozenset(word_tokens(self.reference_text)),
        )
        object.__setattr__(
            self,
            "reference_distribution",
            ngram_distribution(self.reference_text, self.divergence_ngram),
        )

    def evaluate(self, sample: str) -> dict[str, float]:
        """Calculate all preregistered core metrics for one generated sample."""
        sample_distribution = ngram_distribution(sample, self.divergence_ngram)
        rank_frequency_slope, rank_frequency_r_squared = rank_frequency_fit(sample)
        return {
            "character_entropy": character_entropy(sample),
            "vocabulary_match": vocabulary_match(
                sample, set(self.reference_vocabulary)
            ),
            "type_token_diversity": type_token_diversity(sample),
            "ngram_repetition": ngram_repetition(sample, self.repetition_ngram),
            "jensen_shannon_divergence": jensen_shannon_divergence(
                sample_distribution, self.reference_distribution
            ),
            "rank_frequency_slope": rank_frequency_slope,
            "rank_frequency_r_squared": rank_frequency_r_squared,
        }
