# Next Character Suppression Nano Capstone

## Research question

What happens when a text-generating model is repeatedly prevented from choosing
the character it considers most likely to come next?

This Nano Capstone trains a character-level language model on public-domain
English text. During generation, the experiment removes the model's most
probable next character at controlled rates from 0% to 100%. A matched control
removes a random candidate instead. The comparison tests whether the observed
changes come specifically from denying the model its first choice rather than
from merely reducing its options.

The final study will compare recognizable-word frequency, vocabulary diversity,
repetition, character entropy, statistical distance from held-out English, and
word rank-frequency structure. The final experiment has not yet been run.

## Current progress

- Selected and froze Jane Austen's *Pride and Prejudice*, Project Gutenberg
  ebook 1342, as the corpus.
- Created a contiguous 90/10 split with 647,044 training characters and 71,894
  held-out characters.
- Compared character n-gram orders 3, 4, and 5 on held-out text.
- Selected order 5 because it produced the lowest held-out cross-entropy:
  1.954 bits per character.
- Implemented most-probable-character suppression and the matched random-removal
  control at five intervention rates.
- Added deterministic seeds, auditable outputs, six metric families, and 28
  automated tests.
- Verified that the demonstration pipeline produces all expected samples.

The demonstration proves that the system runs; it is not evidence for the
research hypotheses.

## Repository structure

- `nano_capstone/corpus.py`: corpus cleaning, normalization, splitting, and
  checksums.
- `nano_capstone/ngram.py`: smoothed character n-gram model with backoff.
- `nano_capstone/calibrate.py`: model-order comparison on held-out text.
- `nano_capstone/experiment.py`: suppression and control experiment runner.
- `nano_capstone/metrics.py`: predefined text measurements.
- `nano_capstone/pipeline.py`: one-command end-to-end pipeline.
- `nano_capstone/tests/`: automated tests.
- `evidence/`: the frozen corpus manifest and calibration results, without the
  copyrighted/downloaded text files themselves.
- `PROCESS_LOG.md`: decisions, validation, AI use, and remaining work.

## Run the demonstration

The project uses only the Python standard library.

```bash
python3 -m nano_capstone.pipeline \
  nano_capstone/data/demo_source.txt \
  /tmp/nano_capstone_demo \
  --sample-length 200 \
  --seed-count 3
```

Run all tests:

```bash
python3 -m unittest discover -s nano_capstone/tests -t . -v
```

## Reproduce the corpus preparation

Download the official Project Gutenberg text:

```bash
curl -L https://www.gutenberg.org/cache/epub/1342/pg1342.txt \
  -o nano_capstone/data/pride_and_prejudice_pg1342.txt
```

Then prepare and freeze it:

```bash
python3 -m nano_capstone.corpus \
  nano_capstone/data/pride_and_prejudice_pg1342.txt \
  artifacts/corpus
```

The expected source checksum and all processed-file checksums are recorded in
`evidence/corpus_manifest.json`.

## Run the final experiment

Only run this after the protocol has been reviewed and frozen:

```bash
python3 -m nano_capstone.pipeline \
  nano_capstone/data/pride_and_prejudice_pg1342.txt \
  outputs/final_run
```

The pipeline refuses to overwrite or mix results from an earlier run.

