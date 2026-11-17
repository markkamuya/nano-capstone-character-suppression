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

This calibration used no suppression results. The model choice was therefore
frozen independently of the intervention outcomes.

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

## 17 November 2026 Frozen Full Run and Analysis

### Work completed

- Verified the source-text checksum and reran all 28 automated tests.
- Ran the frozen experiment with five intervention rates, two removal rules,
  twenty seeds per condition, and 2,000 generated characters per sample.
- Completed all 200 expected runs and preserved their configurations, generated
  samples, measurements, and manifest.
- Added a reproducible analysis command that calculates condition means, 95%
  confidence intervals, and paired targeted-versus-control differences.
- Produced a three-panel results figure and a written findings summary for the
  paper and presentation.

### Results and interpretation

First-choice suppression produced a strong dose-response pattern while random
removal remained nearly flat. From baseline to full suppression, the
recognizable-word rate fell from 0.613 to 0.105 and Jensen-Shannon distance from
held-out text rose from 0.334 to 0.784. Vocabulary diversity rose from 0.682 to
0.848, but that increase accompanied a collapse in recognizable words. The
repetition hypothesis was not supported: repeated four-character sequences
became less common rather than more common.

### Limitation and decision

The baseline text is already noisy and only partly recognizable. I will not
present this run as evidence about modern language models generally. It shows
how the intervention changes one simple character model trained on one novel.
The next methodological decision is whether to strengthen the baseline before
adding a second corpus or additional model configuration.

### AI use

Multi-agent AI assistance produced code components for corpus preparation,
model behavior, experiment metrics, analysis, and presentation materials. I
selected the research question and experimental constraints and directed the
integration. I remain responsible for reviewing the code and method, checking
the reported values against the saved outputs, interpreting the evidence, and
presenting the limitations accurately.
