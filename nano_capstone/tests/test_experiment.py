from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

from nano_capstone.experiment import ExperimentConfig, run_experiment


class DummyGenerator:
    def generate(
        self,
        length: int,
        seed: int,
        intervention: str = "none",
        intervention_rate: float = 0.0,
        prompt: str = "",
    ) -> str:
        rng = random.Random((seed, intervention, intervention_rate).__repr__())
        alphabet = "abc " if intervention == "suppress_top" else "abd "
        return "".join(rng.choice(alphabet) for _ in range(length))


class ExperimentTests(unittest.TestCase):
    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(suppression_rates=(1.1,))
        with self.assertRaises(ValueError):
            ExperimentConfig(seeds=(1, 1))

    def test_runner_saves_complete_auditable_run(self) -> None:
        config = ExperimentConfig(
            sample_length=50,
            suppression_rates=(0.0, 1.0),
            seeds=(3, 7),
            interventions=("suppress_top", "random_remove"),
            repetition_ngram=2,
            divergence_ngram=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run"
            rows = run_experiment(
                DummyGenerator(), "abc abd cab dab " * 20, output, config
            )

            self.assertEqual(len(rows), 8)
            self.assertTrue((output / "results.csv").is_file())
            self.assertTrue((output / "results.jsonl").is_file())
            self.assertTrue((output / "summary.csv").is_file())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["expected_run_count"], 8)
            self.assertEqual(manifest["completed_run_count"], 8)

            with (output / "summary.csv").open(newline="") as source:
                summary = list(csv.DictReader(source))
            self.assertEqual(len(summary), 4)
            self.assertTrue(all(int(row["repetitions"]) == 2 for row in summary))
            self.assertTrue(
                all((output / row["sample_path"]).is_file() for row in rows)
            )

    def test_runner_refuses_to_mix_with_existing_output(self) -> None:
        config = ExperimentConfig(
            sample_length=10,
            suppression_rates=(0.0,),
            seeds=(0,),
            interventions=("suppress_top",),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "run"
            output.mkdir()
            (output / "keep.txt").write_text("do not replace")
            with self.assertRaises(FileExistsError):
                run_experiment(DummyGenerator(), "reference", output, config)


if __name__ == "__main__":
    unittest.main()
