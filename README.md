# Hallucination Risk Evaluation for Video Question Answering

This repository is the reproducibility package for **Hallucination Risk
Evaluation Framework for Question-Answering in Video-Language Models**.

The method is a black-box reliability check for multiple-choice VideoQA. It
does not retrain the video-language models or change their selected answers.
Instead, it estimates how likely each fixed answer is to be correct so that a
system can accept higher-reliability answers first and send less reliable ones
for further review.

For each question, three model families return option probabilities under four
fixed temporal sampling strategies. The method combines agreement across
models with changes across temporal views to build two learned
answer-correctness scores. A small labeled sample from the target videos
compares those scores with each model's original selected-option probability.
A learned score is used only when repeated video-level evaluations show a
stable improvement; otherwise, the original probability is retained.

The code and archived result schemas use the historical internal identifier
`TRACE-RC`. Those filenames and schema values are kept unchanged so that the
archived checksums and result files remain verifiable.

## Main results

The repository contains two independent held-out evaluations:

| Test set | Questions | Videos | Mean relative AURC reduction | Smallest model-level reduction | Error reduction at 50% coverage |
|---|---:|---:|---:|---:|---:|
| HERBench Lite-v2 | 963 | 34 | 16.64% | 11.39% | 6.50 percentage points |
| Perception Test | 911 | 250 | 18.87% | 0.00% | 6.58 percentage points |

The Perception Test target check retains the original score for InternVL3.5,
which accounts for the 0.00% smallest model-level reduction. Across both test
sets, the target-checked method has the largest mean AURC reduction among seven
training-free confidence and consistency baselines. Results are reported
separately for the two datasets rather than pooled.

## Repository map

```text
src/                    analysis, audit, and plotting code
tests/                  integrity and unit tests
katana/                 inference runners and PBS wrappers
protocols/              fixed pre-inference protocol records
input/                  manifests and separated pilot/test labels
results/
  herbench_confirmation/    fixed-protocol HERBench result archive
  perception_confirmation/  fixed-protocol Perception Test result archive
  posthoc_robustness/        robustness, efficiency, and baseline audits
  vhd/                       annotation agreement audit for VHD
docs/                   execution notes
legacy/vhd_pilot/       superseded first-version VHD experiments
```

The `legacy/vhd_pilot/` directory is retained for provenance. Its earlier
four-model Accuracy/PDI/HCE experiment is not evidence for the current paper's
main claims.

## Verify the held-out result archives

Create a Python environment and install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Then verify both archive manifests and rebuild the two-dataset summary:

```bash
python src/summarize_two_confirmations.py \
  --herbench-dir results/herbench_confirmation/8918264.kman.restech.unsw.edu.au \
  --perception-dir results/perception_confirmation/8919110.kman.restech.unsw.edu.au \
  --output-dir reproduced/two_confirmations
```

The command stops if a required file is absent, an SHA-256 digest differs, or
a reported result cannot be reconstructed from the archived model metrics.

## Reproduce the training-free baseline audit

The seven baselines use only cached option probabilities: top-two margin,
negative entropy, base-view cross-model vote and support, within-model temporal
stability, and full-panel vote and support. With the raw response archives
available in the workspace, reproduce the model metrics and 10,000 paired
complete-video bootstrap samples with:

```bash
python src/analyze_trace_rc_training_free_baselines.py \
  --workspace /path/to/paper-workspace \
  --output-dir results/posthoc_robustness \
  --bootstrap-iterations 10000

python src/audit_trace_rc_training_free_baselines.py \
  --workspace /path/to/paper-workspace \
  --result-dir results/posthoc_robustness
```

## Rebuild the public result figures

The main held-out and post-hoc figures can be rebuilt from the files included
in this repository:

```bash
python src/render_trace_rc_plots.py \
  --herbench-dir results/herbench_confirmation/8918264.kman.restech.unsw.edu.au \
  --perception-dir results/perception_confirmation/8919110.kman.restech.unsw.edu.au \
  --posthoc-dir results/posthoc_robustness \
  --output-dir reproduced/figures
```

The VHD model-consensus figure additionally requires the model-consensus table
derived from the omitted raw response archive. Pass that file with `--vhd-csv`
when it is available; it is not required for the other figures.

## Audit the VHD annotations

Video Hallucination Diagnosis (VHD) is a controlled diagnostic dataset used to
study shared high-confidence errors during method development. It is not one of
the two final held-out test sets. After downloading VHD and the reviewer
exports, reproduce the double-review audit with:

```bash
python src/audit_vhd_annotations.py \
  --dataset-root /path/to/VHD/Dataset \
  --output-dir results/vhd
```

The final VHD v1.1.2 question bank retains 801 of 1,155 candidate questions
after two independent reviews and adjudication. The audit reports
task-appropriate agreement and does not treat intentionally blank labels as
missing data.

## Models, views, and data

The held-out evaluations use Qwen3-VL-8B-Instruct,
InternVL3.5-14B-Instruct, and LLaVA-OneVision-2-8B-Instruct.
Each question is evaluated under a base view, a shifted sampling view, a denser
view, and a reverse-order view. In the full held-out configuration, these use
16, 8, 24, and 8 frames, respectively. Across the three model families, this
corresponds to 12 model calls and 168 model-input frames per question. Frame
counts in development experiments differ and are recorded in the scripts and
protocol files.

Data resources:

- [VHD pilot package on Kaggle](https://www.kaggle.com/datasets/mlopssss/video-hallucination-diagnosis)
- [VHD v1.1.2 metadata, gold labels, and evaluation tools](https://github.com/transitional-ai/hallucinationdataset/tree/master/releases/v1.1.2-metadata-code)
- [VHD annotation and inference tools](https://github.com/transitional-ai/hallucinationdataset)
- The Kaggle page currently hosts the earlier 1,155-question pilot package.
  The frozen v1.1.2 archive contains the 801 adjudicated questions reported in
  the paper and does not redistribute MP4 files.
- HERBench and Perception Test remain subject to their upstream licenses and
  distribution terms. This repository stores derived scores and hashes, not
  copies of their videos.

The raw JSONL response archives are omitted because they are large and include
upstream media metadata. Their SHA-256 digests are retained in the result
archives and analysis manifests. The included scored predictions are sufficient
to verify the reported held-out tables and target-pilot decisions.

## License and citation

The source code is released under the [MIT License](LICENSE). Dataset media,
benchmark files, model weights, and other third-party material remain subject
to their respective upstream terms. Software citation metadata is provided in
[`CITATION.cff`](CITATION.cff); the final article citation will be added after
publication.

Katana should be cited as:

> Katana. (2010). https://doi.org/10.26190/669X-A286
