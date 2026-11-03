# Process Log

## 3 November 2026 Baseline System Build

### Objective

Turn the proposal and feasibility pilot into a reproducible system that can run
the planned experiment from a local public-domain text file. This entry records
the work completed; actual time should be recorded separately by the student.

### Work completed

- Built deterministic corpus cleaning that strips common Project Gutenberg
  boilerplate, normalizes text, creates a contiguous training and held-out split,
  and records SHA-256 checksums in a manifest.
- Built a character-level n-gram model supporting orders 3, 4, and 5, additive
  smoothing, suffix backoff, seeded generation, and held-out cross-entropy.
- Implemented most-probable-character suppression and a matched random-removal
  control using separate random streams for intervention timing, candidate
  removal, and character sampling.
- Built calibration that selects the candidate order with the lowest held-out
  cross-entropy before the suppression experiment begins.
- Implemented five suppression rates, two intervention types, and configurable
  repeated seeds.
- Implemented six metric families: character entropy, reference-vocabulary
  match, type-token diversity, character n-gram repetition, Jensen-Shannon
  divergence, and word rank-frequency structure. The last reports both slope
  and R².
- Added auditable outputs: every generated sample, per-run CSV and JSONL files,
  grouped summaries, configurations, paths, run counts, and checksums.
- Added a one-command pipeline and a synthetic demonstration corpus.

### Validation

- All 28 automated tests passed.
- Static checks passed with no reported issues.
- A full demonstration completed in approximately 2.3 seconds.
- The demonstration produced all 200 planned samples: 5 rates × 2 intervention
  types × 20 seeds.
- Every sample contained exactly 2,000 characters.
- The manifest recorded 200 expected and 200 completed runs with 200 unique run
  keys.

These checks establish that the pipeline functions. They do not test the
research hypotheses because the included corpus is synthetic.

### Corpus Selection and Model Calibration

- Selected Jane Austen's *Pride and Prejudice*, Project Gutenberg ebook 1342,
  as the frozen corpus: https://www.gutenberg.org/ebooks/1342
- Downloaded the official UTF-8 plain-text edition on 3 November 2026. Its
  SHA-256 checksum is
  `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`.
- Removed Project Gutenberg boilerplate, normalized the text, and created a
  contiguous 90/10 split containing 647,044 training characters and 71,894
  held-out characters. The manifest records the files and their checksums.
- Compared orders 3, 4, and 5 using held-out cross-entropy. Their scores were
  2.703, 2.134, and 1.954 bits per character, respectively. Order 5 was frozen
  because it achieved the lowest score under the predefined selection rule.

This calibration used no suppression results. The final intervention experiment
therefore remains unrun until the protocol is reviewed and frozen.

### Decisions and rationale

- The corpus is split contiguously so held-out text remains genuinely unseen
  while preserving the sequence structure of the source.
- Orders 3–5 are compared on held-out cross-entropy rather than on final
  intervention outcomes, preventing the experimental results from influencing
  model selection.
- Existing output directories are never overwritten or mixed with a later run.
  This preserves provenance and makes accidental result contamination visible.
- Targeted and control conditions share seeds and model settings so the removed
  candidate is their principal systematic difference.
- Metrics remain separate rather than being compressed into one quality score,
  allowing disagreement among validity, diversity, repetition, and statistical
  distance to remain visible.

### Remaining work

1. Freeze the final protocol before running the study.
2. Run the actual experiment in a new output directory.
3. Add uncertainty intervals, targeted-versus-control comparisons, figures, and
   sensitivity checks to the analysis.

### AI use

The implementation was produced with multi-agent AI assistance. Separate agents
worked on corpus preparation, model behavior, and experiment metrics; the
components were then integrated and tested together. The student should review
the code, verify the methodological choices, and document any revisions before
using the system for the final study.
