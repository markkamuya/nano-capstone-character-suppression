"""Interpretable character-level n-gram language model.

The model deliberately uses a small, inspectable implementation for the Nano
Capstone experiment.  An order-N model conditions on at most N - 1 previous
characters.  It backs off to shorter suffixes when a context was not observed
during training and applies additive smoothing at the selected context.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import random
from typing import Literal, Mapping


Intervention = Literal["none", "suppress_top", "random_remove"]


@dataclass(frozen=True)
class GenerationResult:
    """Generated text and enough metadata to audit the intervention."""

    text: str
    interventions_applied: int
    removed_characters: tuple[str, ...]


class CharacterNGramModel:
    """A smoothed character n-gram model with explicit backoff.

    Parameters
    ----------
    order:
        N-gram order.  For example, order 5 uses up to four preceding
        characters.  The study is designed to compare orders 3, 4, and 5,
        though any positive integer is supported.
    alpha:
        Additive-smoothing strength.  It must be strictly positive so held-out
        cross-entropy is finite for every character in the training alphabet.
    """

    def __init__(self, order: int = 5, alpha: float = 0.1) -> None:
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ValueError("order must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite number greater than zero")

        self.order = order
        self.alpha = float(alpha)
        self._counts: dict[str, Counter[str]] = {}
        self._alphabet: tuple[str, ...] = ()
        self._fitted = False

    @property
    def alphabet(self) -> tuple[str, ...]:
        """The sorted training alphabet."""

        self._require_fitted()
        return self._alphabet

    def fit(self, text: str) -> "CharacterNGramModel":
        """Count every context length from zero through ``order - 1``."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            raise ValueError("training text must not be empty")

        counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        max_context = self.order - 1
        for index, character in enumerate(text):
            available = min(index, max_context)
            for context_length in range(available + 1):
                context = text[index - context_length : index]
                counts[context][character] += 1

        self._counts = dict(counts)
        self._alphabet = tuple(sorted(set(text)))
        self._fitted = True
        return self

    def probability_distribution(self, context: str = "") -> dict[str, float]:
        """Return a smoothed distribution after backing off when necessary.

        Only the last ``order - 1`` characters can affect the prediction.  If
        that suffix was unseen, the model repeatedly removes its first
        character until it finds a recorded context.  The empty context is
        always available after fitting.
        """

        self._require_fitted()
        if not isinstance(context, str):
            raise TypeError("context must be a string")

        suffix = context[-(self.order - 1) :] if self.order > 1 else ""
        while suffix not in self._counts:
            suffix = suffix[1:]

        counts = self._counts[suffix]
        denominator = sum(counts.values()) + self.alpha * len(self._alphabet)
        return {
            character: (counts.get(character, 0) + self.alpha) / denominator
            for character in self._alphabet
        }

    def probability(self, character: str, context: str = "") -> float:
        """Return the probability of one next character."""

        if not isinstance(character, str) or len(character) != 1:
            raise ValueError("character must contain exactly one character")
        distribution = self.probability_distribution(context)
        if character not in distribution:
            return 0.0
        return distribution[character]

    def cross_entropy(self, text: str, initial_context: str = "") -> float:
        """Return mean held-out cross-entropy in bits per character.

        The held-out text must use the training alphabet.  Rejecting unseen
        characters makes corpus/preprocessing mismatches explicit rather than
        silently assigning them an arbitrary probability.
        """

        self._require_fitted()
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            raise ValueError("held-out text must not be empty")

        unknown = sorted(set(text) - set(self._alphabet))
        if unknown:
            rendered = ", ".join(repr(character) for character in unknown)
            raise ValueError(
                "held-out text contains characters outside the training alphabet: "
                f"{rendered}"
            )

        history = initial_context
        negative_log_likelihood = 0.0
        for character in text:
            probability = self.probability(character, history)
            negative_log_likelihood -= math.log2(probability)
            history += character
        return negative_log_likelihood / len(text)

    def generate(
        self,
        length: int,
        *,
        seed: int,
        intervention: Intervention = "none",
        intervention_rate: float = 0.0,
        prompt: str = "",
    ) -> str:
        """Generate ``length`` new characters deterministically from ``seed``."""

        return self.generate_with_trace(
            length,
            seed=seed,
            intervention=intervention,
            intervention_rate=intervention_rate,
            prompt=prompt,
        ).text

    def generate_with_trace(
        self,
        length: int,
        *,
        seed: int,
        intervention: Intervention = "none",
        intervention_rate: float = 0.0,
        prompt: str = "",
    ) -> GenerationResult:
        """Generate text and record each removed candidate.

        Three independent pseudo-random streams control intervention timing,
        random-candidate removal, and sampling.  Consequently, matched runs
        with the same seed use the same intervention decisions and sampling
        quantiles even though their probability distributions differ.
        """

        self._require_fitted()
        self._validate_generation_arguments(length, intervention, intervention_rate, prompt)

        decision_rng = random.Random(seed)
        removal_rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
        sampling_rng = random.Random(seed ^ 0xD1B54A32D192ED03)

        history = prompt
        output: list[str] = []
        removed: list[str] = []

        for _ in range(length):
            distribution = self.probability_distribution(history)
            should_intervene = (
                intervention != "none"
                and intervention_rate > 0
                and (intervention_rate >= 1 or decision_rng.random() < intervention_rate)
            )
            if should_intervene and len(distribution) > 1:
                distribution, removed_character = self._remove_candidate(
                    distribution,
                    intervention,
                    removal_rng,
                )
                removed.append(removed_character)

            character = _sample_distribution(distribution, sampling_rng.random())
            output.append(character)
            history += character

        return GenerationResult(
            text="".join(output),
            interventions_applied=len(removed),
            removed_characters=tuple(removed),
        )

    def _remove_candidate(
        self,
        distribution: Mapping[str, float],
        intervention: Intervention,
        removal_rng: random.Random,
    ) -> tuple[dict[str, float], str]:
        candidates = sorted(distribution)
        if intervention == "suppress_top":
            # The lexicographic rule makes ties reproducible and testable.
            removed = min(candidates, key=lambda character: (-distribution[character], character))
        elif intervention == "random_remove":
            removed = candidates[removal_rng.randrange(len(candidates))]
        else:  # Defensive guard; public callers are validated earlier.
            raise ValueError(f"unknown intervention: {intervention!r}")

        retained = {
            character: probability
            for character, probability in distribution.items()
            if character != removed
        }
        total = sum(retained.values())
        normalized = {character: probability / total for character, probability in retained.items()}
        return normalized, removed

    def _validate_generation_arguments(
        self,
        length: int,
        intervention: Intervention,
        intervention_rate: float,
        prompt: str,
    ) -> None:
        if not isinstance(length, int) or isinstance(length, bool) or length < 0:
            raise ValueError("length must be a non-negative integer")
        if intervention not in {"none", "suppress_top", "random_remove"}:
            raise ValueError("intervention must be 'none', 'suppress_top', or 'random_remove'")
        if not math.isfinite(intervention_rate) or not 0 <= intervention_rate <= 1:
            raise ValueError("intervention_rate must be between zero and one")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit the model before requesting probabilities or generating text")


def _sample_distribution(distribution: Mapping[str, float], quantile: float) -> str:
    """Sample from a distribution using a supplied uniform quantile."""

    cumulative = 0.0
    last_character: str | None = None
    for character, probability in distribution.items():
        cumulative += probability
        last_character = character
        if quantile < cumulative:
            return character

    # Floating-point addition can finish a few ulps below one.
    if last_character is None:
        raise ValueError("cannot sample an empty distribution")
    return last_character
