"""Run preprocessing, calibration, and the suppression experiment end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

from .calibrate import calibrate_orders, write_calibration
from .corpus import prepare_corpus
from .experiment import ExperimentConfig, run_experiment


def run_pipeline(
    source: str | Path,
    output_directory: str | Path,
    *,
    train_fraction: float = 0.9,
    alpha: float = 0.1,
    sample_length: int = 2_000,
    seeds: tuple[int, ...] = tuple(range(20)),
    suppression_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> int:
    """Execute the reproducible minimum study and return the selected order."""

    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {output_directory}. "
            "Choose a new directory so an earlier run is never mixed or replaced."
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    corpus_directory = output_directory / "corpus"
    prepare_corpus(
        Path(source),
        corpus_directory,
        train_fraction=train_fraction,
    )
    train_text = (corpus_directory / "train.txt").read_text(encoding="utf-8")
    held_out_text = (corpus_directory / "held_out.txt").read_text(encoding="utf-8")

    model, calibration = calibrate_orders(
        train_text,
        held_out_text,
        orders=(3, 4, 5),
        alpha=alpha,
    )
    write_calibration(output_directory / "calibration", model, calibration)

    config = ExperimentConfig(
        sample_length=sample_length,
        suppression_rates=suppression_rates,
        seeds=seeds,
    )
    run_experiment(
        model,
        held_out_text,
        output_directory / "experiment",
        config,
    )
    return model.order


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local public-domain text file")
    parser.add_argument("output_directory", type=Path, help="new output directory")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--sample-length", type=int, default=2_000)
    parser.add_argument("--seed-count", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed_count < 1:
        raise SystemExit("--seed-count must be positive")
    selected_order = run_pipeline(
        args.source,
        args.output_directory,
        train_fraction=args.train_fraction,
        alpha=args.alpha,
        sample_length=args.sample_length,
        seeds=tuple(range(args.seed_count)),
    )
    print(f"Completed pipeline in {args.output_directory}")
    print(f"Selected model order: {selected_order}")
    print(f"Results: {args.output_directory / 'experiment' / 'results.csv'}")


if __name__ == "__main__":
    main()
