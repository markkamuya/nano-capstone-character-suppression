"""Prepare a reproducible character-model corpus from a local text file.

The pipeline deliberately uses only Python's standard library. It removes common
Project Gutenberg boilerplate, normalizes the remaining English text, makes a
contiguous train/held-out split, and records cryptographic checksums for every
artifact it writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


SCHEMA_VERSION = 1
DEFAULT_TRAIN_FRACTION = 0.90
DEFAULT_ALLOWED_CHARACTERS = "abcdefghijklmnopqrstuvwxyz .,;:!?\"'()-\n"

_GUTENBERG_START = re.compile(
    r"^\s*\*{0,3}\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_GUTENBERG_END = re.compile(
    r"^\s*\*{0,3}\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class NormalizationSettings:
    lowercase: bool = True
    allowed_characters: str = DEFAULT_ALLOWED_CHARACTERS
    collapse_horizontal_whitespace: bool = True
    maximum_consecutive_newlines: int = 2


@dataclass(frozen=True)
class BoilerplateResult:
    text: str
    start_marker_found: bool
    end_marker_found: bool


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest for *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 encoded *text*."""

    return sha256_bytes(text.encode("utf-8"))


def strip_gutenberg_boilerplate(text: str) -> BoilerplateResult:
    """Remove common Project Gutenberg header and footer marker sections.

    Text is trimmed only when a marker is actually present. The function also
    reports which boundaries it found so that the manifest exposes, rather than
    hides, what preprocessing occurred.
    """

    start_match = _GUTENBERG_START.search(text)
    content_start = start_match.end() if start_match else 0

    end_match = _GUTENBERG_END.search(text, content_start)
    content_end = end_match.start() if end_match else len(text)

    return BoilerplateResult(
        text=text[content_start:content_end].strip(),
        start_marker_found=start_match is not None,
        end_marker_found=end_match is not None,
    )


def normalize_text(
    text: str, settings: NormalizationSettings = NormalizationSettings()
) -> str:
    """Normalize text using explicit, manifest-recorded rules."""

    text = unicodedata.normalize("NFKC", text).replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    if settings.lowercase:
        text = text.lower()

    allowed = set(settings.allowed_characters)
    text = "".join(character if character in allowed else " " for character in text)

    if settings.collapse_horizontal_whitespace:
        text = "\n".join(
            _HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")
        )
    if settings.maximum_consecutive_newlines == 2:
        text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    else:
        limit = max(1, settings.maximum_consecutive_newlines)
        text = re.sub(rf"\n{{{limit + 1},}}", "\n" * limit, text)

    return text.strip()


def split_corpus(text: str, train_fraction: float) -> tuple[str, str, int]:
    """Make a deterministic contiguous train/held-out split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(text) < 2:
        raise ValueError("normalized corpus must contain at least two characters")

    split_index = round(len(text) * train_fraction)
    split_index = min(max(split_index, 1), len(text) - 1)
    return text[:split_index], text[split_index:], split_index


def _artifact_record(text: str, relative_path: str) -> dict[str, object]:
    encoded = text.encode("utf-8")
    return {
        "path": relative_path,
        "characters": len(text),
        "utf8_bytes": len(encoded),
        "lines": text.count("\n") + (1 if text else 0),
        "sha256": sha256_bytes(encoded),
    }


def _write_utf8(path: Path, text: str) -> None:
    """Write UTF-8 with Unix newlines on all supported Python versions."""

    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(text)


def prepare_corpus(
    input_path: Path,
    output_directory: Path,
    *,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    encoding: str = "utf-8",
    settings: NormalizationSettings = NormalizationSettings(),
    force: bool = False,
) -> dict[str, object]:
    """Run the pipeline and return the manifest written to disk."""

    input_path = Path(input_path)
    output_directory = Path(output_directory)
    if not input_path.is_file():
        raise FileNotFoundError(f"input corpus does not exist: {input_path}")

    output_paths = {
        "normalized": output_directory / "normalized.txt",
        "train": output_directory / "train.txt",
        "held_out": output_directory / "held_out.txt",
        "manifest": output_directory / "manifest.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing artifacts: {joined}")

    raw_bytes = input_path.read_bytes()
    raw_text = raw_bytes.decode(encoding)
    stripped = strip_gutenberg_boilerplate(raw_text)
    normalized = normalize_text(stripped.text, settings)
    train, held_out, split_index = split_corpus(normalized, train_fraction)

    output_directory.mkdir(parents=True, exist_ok=True)
    _write_utf8(output_paths["normalized"], normalized)
    _write_utf8(output_paths["train"], train)
    _write_utf8(output_paths["held_out"], held_out)

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "filename": input_path.name,
            "encoding": encoding,
            "bytes": len(raw_bytes),
            "sha256": sha256_bytes(raw_bytes),
        },
        "gutenberg_boilerplate": {
            "start_marker_found": stripped.start_marker_found,
            "end_marker_found": stripped.end_marker_found,
        },
        "normalization": asdict(settings),
        "split": {
            "method": "contiguous_prefix_train_suffix_held_out",
            "requested_train_fraction": train_fraction,
            "split_index": split_index,
            "actual_train_fraction": len(train) / len(normalized),
        },
        "artifacts": {
            "normalized": _artifact_record(normalized, "normalized.txt"),
            "train": _artifact_record(train, "train.txt"),
            "held_out": _artifact_record(held_out, "held_out.txt"),
        },
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_utf8(output_paths["manifest"], manifest_json)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Strip Project Gutenberg boilerplate, normalize a local corpus, "
            "create train/held-out files, and record checksums."
        )
    )
    parser.add_argument("input", type=Path, help="local source text file")
    parser.add_argument("output_directory", type=Path, help="directory for artifacts")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
        help=f"fraction assigned to training (default: {DEFAULT_TRAIN_FRACTION})",
    )
    parser.add_argument("--encoding", default="utf-8", help="source encoding")
    parser.add_argument(
        "--keep-case", action="store_true", help="preserve letter case during normalization"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing pipeline artifacts"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    allowed_characters = DEFAULT_ALLOWED_CHARACTERS
    if args.keep_case:
        allowed_characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    settings = NormalizationSettings(
        lowercase=not args.keep_case,
        allowed_characters=allowed_characters,
    )
    try:
        manifest = prepare_corpus(
            args.input,
            args.output_directory,
            train_fraction=args.train_fraction,
            encoding=args.encoding,
            settings=settings,
            force=args.force,
        )
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    artifacts = manifest["artifacts"]
    print(f"Wrote corpus artifacts to {args.output_directory}")
    print(f"Training characters: {artifacts['train']['characters']}")
    print(f"Held-out characters: {artifacts['held_out']['characters']}")
    print(f"Manifest: {args.output_directory / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

