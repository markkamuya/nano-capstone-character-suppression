"""Calibrate candidate n-gram orders using frozen held-out text."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .ngram import CharacterNGramModel


def calibrate_orders(
    train_text: str,
    held_out_text: str,
    *,
    orders: Iterable[int] = (3, 4, 5),
    alpha: float = 0.1,
) -> tuple[CharacterNGramModel, list[dict[str, float | int]]]:
    """Fit each candidate order and return the lowest-cross-entropy model.

    Calibration uses only the frozen training and held-out texts. The selected
    model can therefore be fixed before any suppression experiment is run.
    """

    candidates = tuple(orders)
    if not candidates:
        raise ValueError("at least one candidate order is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate orders must be unique")

    scored_models: list[tuple[float, int, CharacterNGramModel]] = []
    results: list[dict[str, float | int]] = []
    for order in candidates:
        model = CharacterNGramModel(order=order, alpha=alpha).fit(train_text)
        cross_entropy = model.cross_entropy(held_out_text)
        results.append(
            {
                "order": order,
                "alpha": alpha,
                "held_out_cross_entropy_bits_per_character": cross_entropy,
            }
        )
        scored_models.append((cross_entropy, order, model))

    _, _, selected_model = min(scored_models, key=lambda item: (item[0], item[1]))
    return selected_model, results


def write_calibration(
    output_directory: str | Path,
    model: CharacterNGramModel,
    results: list[dict[str, float | int]],
) -> None:
    """Write machine-readable calibration evidence."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / "calibration.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    payload = {
        "selection_rule": "lowest held-out cross-entropy; lower order breaks ties",
        "selected_order": model.order,
        "selected_alpha": model.alpha,
        "candidates": results,
    }
    (output_directory / "calibration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
