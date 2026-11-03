import math
import unittest

from nano_capstone.ngram import CharacterNGramModel


class CharacterNGramModelTests(unittest.TestCase):
    def test_orders_three_four_and_five_are_supported(self) -> None:
        for order in (3, 4, 5):
            model = CharacterNGramModel(order=order).fit("abracadabra")
            self.assertEqual(model.order, order)
            self.assertAlmostEqual(sum(model.probability_distribution("abra").values()), 1.0)

    def test_smoothed_distribution_is_positive_and_normalized(self) -> None:
        model = CharacterNGramModel(order=3, alpha=0.5).fit("abc")
        distribution = model.probability_distribution("ab")

        self.assertEqual(set(distribution), {"a", "b", "c"})
        self.assertTrue(all(probability > 0 for probability in distribution.values()))
        self.assertTrue(math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12))
        self.assertAlmostEqual(distribution["a"], 0.2)
        self.assertAlmostEqual(distribution["b"], 0.2)
        self.assertAlmostEqual(distribution["c"], 0.6)

    def test_unseen_context_backs_off_to_empty_context(self) -> None:
        model = CharacterNGramModel(order=5).fit("ababa")
        self.assertEqual(
            model.probability_distribution("zzzz"),
            model.probability_distribution(""),
        )

    def test_generation_is_deterministic_for_each_condition(self) -> None:
        model = CharacterNGramModel(order=4).fit("to be or not to be ")
        for intervention in ("none", "suppress_top", "random_remove"):
            first = model.generate(
                100,
                seed=19,
                intervention=intervention,
                intervention_rate=0.5,
            )
            second = model.generate(
                100,
                seed=19,
                intervention=intervention,
                intervention_rate=0.5,
            )
            self.assertEqual(first, second)

    def test_zero_rate_matches_unmodified_generation(self) -> None:
        model = CharacterNGramModel(order=3).fit("abcabcabc")
        baseline = model.generate(80, seed=4)
        self.assertEqual(
            baseline,
            model.generate(80, seed=4, intervention="suppress_top", intervention_rate=0),
        )
        self.assertEqual(
            baseline,
            model.generate(80, seed=4, intervention="random_remove", intervention_rate=0),
        )

    def test_full_top_suppression_never_emits_unigram_mode(self) -> None:
        model = CharacterNGramModel(order=1, alpha=0.1).fit("aaaaab")
        result = model.generate_with_trace(
            30,
            seed=7,
            intervention="suppress_top",
            intervention_rate=1,
        )

        self.assertEqual(result.text, "b" * 30)
        self.assertEqual(result.interventions_applied, 30)
        self.assertEqual(set(result.removed_characters), {"a"})

    def test_random_removal_is_recorded_at_every_full_rate_step(self) -> None:
        model = CharacterNGramModel(order=2).fit("abcabcabc")
        result = model.generate_with_trace(
            25,
            seed=11,
            intervention="random_remove",
            intervention_rate=1,
        )
        self.assertEqual(result.interventions_applied, 25)
        self.assertEqual(len(result.removed_characters), 25)
        self.assertTrue(set(result.removed_characters) <= set(model.alphabet))

    def test_cross_entropy_prefers_matching_sequence(self) -> None:
        model = CharacterNGramModel(order=2, alpha=0.1).fit("abababababab")
        matching = model.cross_entropy("abababab")
        unlikely = model.cross_entropy("aaaaaaaa")
        self.assertLess(matching, unlikely)

    def test_cross_entropy_rejects_unknown_characters(self) -> None:
        model = CharacterNGramModel().fit("abc")
        with self.assertRaisesRegex(ValueError, "outside the training alphabet"):
            model.cross_entropy("abcd")

    def test_invalid_configuration_and_generation_arguments_fail(self) -> None:
        with self.assertRaises(ValueError):
            CharacterNGramModel(order=0)
        with self.assertRaises(ValueError):
            CharacterNGramModel(alpha=0)

        model = CharacterNGramModel().fit("abc")
        with self.assertRaises(ValueError):
            model.generate(10, seed=1, intervention_rate=1.1)
        with self.assertRaises(ValueError):
            model.generate(10, seed=1, intervention="unsupported")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
