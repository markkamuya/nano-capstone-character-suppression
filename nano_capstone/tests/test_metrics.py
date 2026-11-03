from __future__ import annotations

import math
import unittest

from nano_capstone.metrics import (
    TextMetrics,
    character_entropy,
    jensen_shannon_divergence,
    ngram_distribution,
    ngram_repetition,
    rank_frequency_fit,
    type_token_diversity,
    vocabulary_match,
)


class MetricTests(unittest.TestCase):
    def test_character_entropy_known_values(self) -> None:
        self.assertEqual(character_entropy(""), 0.0)
        self.assertEqual(character_entropy("aaaa"), 0.0)
        self.assertTrue(math.isclose(character_entropy("abab"), 1.0))

    def test_word_metrics(self) -> None:
        vocabulary = {"the", "cat", "sat"}
        sample = "The cat met the yak."
        self.assertEqual(vocabulary_match(sample, vocabulary), 3 / 5)
        self.assertEqual(type_token_diversity(sample), 4 / 5)

    def test_ngram_repetition(self) -> None:
        self.assertEqual(ngram_repetition("abc", n=4), 0.0)
        self.assertEqual(ngram_repetition("aaaaa", n=2), 0.75)

    def test_rank_frequency_fit_recovers_inverse_rank_law(self) -> None:
        sample = " ".join(
            ["first"] * 12
            + ["second"] * 6
            + ["third"] * 4
            + ["fourth"] * 3
        )
        slope, r_squared = rank_frequency_fit(sample)
        self.assertAlmostEqual(slope, -1.0)
        self.assertAlmostEqual(r_squared, 1.0)

    def test_rank_frequency_fit_handles_degenerate_samples(self) -> None:
        self.assertEqual(rank_frequency_fit(""), (0.0, 1.0))
        self.assertEqual(rank_frequency_fit("same same same"), (0.0, 1.0))
        slope, r_squared = rank_frequency_fit("a a b b")
        self.assertEqual(slope, 0.0)
        self.assertEqual(r_squared, 1.0)

    def test_jensen_shannon_divergence(self) -> None:
        first = {"a": 1.0}
        second = {"a": 1.0}
        disjoint = {"b": 1.0}
        self.assertEqual(jensen_shannon_divergence(first, second), 0.0)
        self.assertEqual(jensen_shannon_divergence(first, disjoint), 1.0)

    def test_metric_suite_uses_frozen_reference(self) -> None:
        evaluator = TextMetrics("the cat sat on the mat", divergence_ngram=2)
        result = evaluator.evaluate("the cat sat")
        self.assertEqual(set(result), {
            "character_entropy",
            "vocabulary_match",
            "type_token_diversity",
            "ngram_repetition",
            "jensen_shannon_divergence",
            "rank_frequency_slope",
            "rank_frequency_r_squared",
        })
        self.assertEqual(result["vocabulary_match"], 1.0)
        self.assertGreaterEqual(result["jensen_shannon_divergence"], 0.0)
        self.assertLessEqual(result["jensen_shannon_divergence"], 1.0)
        self.assertAlmostEqual(sum(ngram_distribution("ababa", 2).values()), 1.0)


if __name__ == "__main__":
    unittest.main()
