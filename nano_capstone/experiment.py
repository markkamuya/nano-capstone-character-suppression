"""Configuration-driven experiment runner with auditable saved outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import fmean, stdev
from typing import Any, Protocol, Sequence

from .metrics import TextMetrics


class CharacterGenerator(Protocol):
    """Interface implemented by the character n-gram model."""

    def generate(
        self,
        length: int,
        seed: int,
        intervention: str = "none",
        intervention_rate: float = 0.0,
        prompt: str = "",
    ) -> str: ...


@dataclass(frozen=True)
class ExperimentConfig:
    sample_length: int = 2_000
    suppression_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    seeds: tuple[int, ...] = tuple(range(20))
    interventions: tuple[str, ...] = ("suppress_top", "random_remove")
    repetition_ngram: int = 4
    divergence_ngram: int = 3
    prompt: str = ""

    def __post_init__(self) -> None:
        if self.sample_length < 1:
            raise ValueError("sample_length must be positive")
        if not self.suppression_rates or not self.seeds or not self.interventions:
            raise ValueError("rates, seeds, and interventions cannot be empty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if any(rate < 0 or rate > 1 for rate in self.suppression_rates):
            raise ValueError("suppression rates must be between 0 and 1")
        allowed = {"suppress_top", "random_remove"}
        if not set(self.interventions) <= allowed:
            raise ValueError(f"interventions must be drawn from {sorted(allowed)}")

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("suppression_rates", "seeds", "interventions"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


METRIC_NAMES = (
    "character_entropy",
    "vocabulary_match",
    "type_token_diversity",
    "ngram_repetition",
    "jensen_shannon_divergence",
    "rank_frequency_slope",
    "rank_frequency_r_squared",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rate_label(rate: float) -> str:
    return f"{rate:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each intervention-rate condition across repeated seeds."""
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["intervention"]), float(row["suppression_rate"]))
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (intervention, rate), group in sorted(groups.items()):
        summary: dict[str, Any] = {
            "intervention": intervention,
            "suppression_rate": rate,
            "repetitions": len(group),
        }
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in group]
            summary[f"{metric}_mean"] = fmean(values)
            summary[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def run_experiment(
    model: CharacterGenerator,
    reference_text: str,
    output_directory: str | Path,
    config: ExperimentConfig | None = None,
) -> list[dict[str, Any]]:
    """Run every configured condition and save samples, results, and metadata."""
    config = config or ExperimentConfig()
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_directory}. "
            "Choose a new directory to preserve the existing run."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    sample_directory = output_directory / "samples"
    sample_directory.mkdir()

    evaluator = TextMetrics(
        reference_text,
        repetition_ngram=config.repetition_ngram,
        divergence_ngram=config.divergence_ngram,
    )
    rows: list[dict[str, Any]] = []

    for intervention in config.interventions:
        for rate in config.suppression_rates:
            condition_directory = (
                sample_directory / intervention / f"rate_{_rate_label(rate)}"
            )
            condition_directory.mkdir(parents=True)
            for seed in config.seeds:
                sample = model.generate(
                    length=config.sample_length,
                    seed=seed,
                    intervention=intervention,
                    intervention_rate=rate,
                    prompt=config.prompt,
                )
                if len(sample) != config.sample_length:
                    raise ValueError(
                        f"model returned {len(sample)} characters; "
                        f"expected {config.sample_length}"
                    )
                relative_sample_path = (
                    Path("samples")
                    / intervention
                    / f"rate_{_rate_label(rate)}"
                    / f"seed_{seed}.txt"
                )
                (output_directory / relative_sample_path).write_text(
                    sample, encoding="utf-8"
                )
                row: dict[str, Any] = {
                    "intervention": intervention,
                    "suppression_rate": rate,
                    "seed": seed,
                    "sample_length": len(sample),
                    "sample_path": relative_sample_path.as_posix(),
                    "sample_sha256": _sha256_text(sample),
                }
                row.update(evaluator.evaluate(sample))
                rows.append(row)

    _write_csv(output_directory / "results.csv", rows)
    with (output_directory / "results.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    summaries = summarize_results(rows)
    _write_csv(output_directory / "summary.csv", summaries)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "expected_run_count": (
            len(config.interventions)
            * len(config.suppression_rates)
            * len(config.seeds)
        ),
        "completed_run_count": len(rows),
        "reference_sha256": _sha256_text(reference_text),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "python_version": platform.python_version(),
        "artifacts": {
            "results_csv": "results.csv",
            "results_jsonl": "results.jsonl",
            "summary_csv": "summary.csv",
            "samples": "samples/",
        },
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    from .ngram import CharacterNGramModel

    config = ExperimentConfig.from_json(args.config)
    model = CharacterNGramModel(order=args.order, alpha=args.alpha)
    model.fit(args.train.read_text(encoding="utf-8"))
    run_experiment(
        model,
        args.reference.read_text(encoding="utf-8"),
        args.output,
        config,
    )


if __name__ == "__main__":
    main()
