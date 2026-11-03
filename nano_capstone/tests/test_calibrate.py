from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from nano_capstone.calibrate import calibrate_orders, write_calibration


class CalibrationTests(unittest.TestCase):
    def test_calibration_scores_and_selects_candidate_orders(self) -> None:
        train = "to be or not to be, that is the question. " * 20
        held_out = "to be or not to be, that is the question. " * 3
        model, results = calibrate_orders(train, held_out, orders=(3, 4, 5))

        self.assertIn(model.order, {3, 4, 5})
        self.assertEqual([row["order"] for row in results], [3, 4, 5])
        self.assertTrue(
            all(row["held_out_cross_entropy_bits_per_character"] > 0 for row in results)
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            write_calibration(destination, model, results)
            self.assertTrue((destination / "calibration.csv").is_file())
            self.assertTrue((destination / "calibration.json").is_file())

    def test_candidate_orders_must_be_present_and_unique(self) -> None:
        with self.assertRaises(ValueError):
            calibrate_orders("abc", "abc", orders=())
        with self.assertRaises(ValueError):
            calibrate_orders("abc", "abc", orders=(3, 3))


if __name__ == "__main__":
    unittest.main()
