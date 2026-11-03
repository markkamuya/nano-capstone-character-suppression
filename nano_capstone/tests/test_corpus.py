from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nano_capstone.corpus import (
    normalize_text,
    prepare_corpus,
    split_corpus,
    strip_gutenberg_boilerplate,
)


class GutenbergBoilerplateTests(unittest.TestCase):
    def test_strips_header_and_footer_markers(self) -> None:
        source = """Metadata that is not part of the book.
*** START OF THE PROJECT GUTENBERG EBOOK A SMALL TEST ***
Actual BOOK text.
Second line.
*** END OF THE PROJECT GUTENBERG EBOOK A SMALL TEST ***
License text.
"""
        result = strip_gutenberg_boilerplate(source)

        self.assertEqual(result.text, "Actual BOOK text.\nSecond line.")
        self.assertTrue(result.start_marker_found)
        self.assertTrue(result.end_marker_found)

    def test_leaves_unmarked_text_intact_except_outer_whitespace(self) -> None:
        result = strip_gutenberg_boilerplate("\n  An ordinary local text.  \n")

        self.assertEqual(result.text, "An ordinary local text.")
        self.assertFalse(result.start_marker_found)
        self.assertFalse(result.end_marker_found)


class NormalizationTests(unittest.TestCase):
    def test_normalizes_case_punctuation_and_whitespace(self) -> None:
        source = "  HELLO\u2014World!\r\n\r\n\r\n'Test'\t42 \u2603  "

        self.assertEqual(normalize_text(source), "hello-world!\n\n'test'")

    def test_split_is_contiguous_and_complete(self) -> None:
        train, held_out, split_index = split_corpus("abcdefghij", 0.8)

        self.assertEqual((train, held_out, split_index), ("abcdefgh", "ij", 8))
        self.assertEqual(train + held_out, "abcdefghij")


class PipelineTests(unittest.TestCase):
    def test_writes_reproducible_artifacts_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "book.txt"
            output = root / "prepared"
            source.write_text(
                "Header\n"
                "*** START OF THIS PROJECT GUTENBERG EBOOK TEST ***\n"
                "One fish. Two fish! Red fish? Blue fish.\n"
                "*** END OF THIS PROJECT GUTENBERG EBOOK TEST ***\n"
                "Footer",
                encoding="utf-8",
            )

            first_manifest = prepare_corpus(source, output, train_fraction=0.75)
            first_manifest_bytes = (output / "manifest.json").read_bytes()
            second_manifest = prepare_corpus(
                source, output, train_fraction=0.75, force=True
            )

            normalized = (output / "normalized.txt").read_text(encoding="utf-8")
            train = (output / "train.txt").read_text(encoding="utf-8")
            held_out = (output / "held_out.txt").read_text(encoding="utf-8")
            saved_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_manifest_bytes, (output / "manifest.json").read_bytes())
            self.assertEqual(train + held_out, normalized)
            self.assertNotIn("header", normalized)
            self.assertNotIn("footer", normalized)
            self.assertEqual(
                saved_manifest["artifacts"]["train"]["sha256"],
                hashlib.sha256(train.encode("utf-8")).hexdigest(),
            )
            self.assertTrue(
                saved_manifest["gutenberg_boilerplate"]["start_marker_found"]
            )
            self.assertTrue(
                saved_manifest["gutenberg_boilerplate"]["end_marker_found"]
            )

    def test_refuses_to_overwrite_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "book.txt"
            output = root / "prepared"
            source.write_text("A corpus long enough to split.", encoding="utf-8")
            prepare_corpus(source, output)

            with self.assertRaises(FileExistsError):
                prepare_corpus(source, output)


if __name__ == "__main__":
    unittest.main()
