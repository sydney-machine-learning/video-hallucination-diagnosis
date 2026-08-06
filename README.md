# TRACE-RC

This repository is the reproducibility package for **TRACE-RC: Target-Checked
Confidence Ranking from Temporal Response Graphs for Video Question Answering**.
TRACE-RC keeps a VideoLLM's base answer fixed and changes only the reliability
score used to rank or defer that answer.  A labeled target pilot decides whether
a learned response-graph score is safe to promote; otherwise the implementation
returns raw confidence exactly.

The current release contains the two sealed confirmation archives used in the
paper:

| Confirmation | Final-test questions | Final-test videos | Macro relative AURC reduction | Grouped 95% interval |
|---|---:|---:|---:|---:|
| HERBench Lite-v2 | 963 | 34 | 16.64% | [12.12%, 22.61%] |
| Perception Test | 911 | 250 | 18.87% | [15.43%, 22.14%] |

These effects are reported separately rather than pooled.  The archived gates
also verify that every rejected model environment used the exact raw-confidence
fallback.

## Repository map

```text
src/                    TRACE-RC analysis, audit, and plotting code
tests/                  lightweight integrity and unit tests
katana/                 frozen inference runners and PBS wrappers
protocols/              pre-inference seals
input/                  frozen manifests and separated pilot/test labels
results/
  herbench_confirmation/    sealed HERBench result archive
  perception_confirmation/  sealed Perception Test result archive
  posthoc_robustness/        fixed-test robustness and compute audits
  vhd/                       annotation agreement audit for VHD
docs/                   execution notes
legacy/vhd_pilot/       superseded first-version VHD experiments
```

The `legacy/vhd_pilot/` directory is retained for provenance.  Its four-model
Accuracy/PDI/HCE experiment is not evidence for the current TRACE-RC claims.

## Verify the published confirmation archives

Create a Python environment and install the analysis dependencies:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Then verify both archive manifests and rebuild the non-pooled summary:

```bash
python src/summarize_two_confirmations.py \
  --herbench-dir results/herbench_confirmation/8918264.kman.restech.unsw.edu.au \
  --perception-dir results/perception_confirmation/8919110.kman.restech.unsw.edu.au \
  --output-dir reproduced/two_confirmations
```

The command stops if a required file is absent, an SHA-256 digest differs, or a
reported gate value cannot be reconstructed from the archived model metrics.

## Audit the VHD annotations

VHD is a controlled mechanism-diagnostic dataset used in source development;
it is not one of the two final confirmation sets.  After downloading the VHD
dataset and reviewer exports, reproduce the double-review audit with:

```bash
python src/audit_vhd_annotations.py \
  --dataset-root /path/to/VHD/Dataset \
  --output-dir results/vhd
```

The final v1.1.2 bank retains 801 of 1,155 candidate questions after two
independent reviews and adjudication.  The audit reports task-appropriate
agreement only; it does not treat intentionally blank labels as missing data.

## Data and linked resources

- [VHD v1.1.2 on Kaggle](https://www.kaggle.com/datasets/mlopssss/video-hallucination-diagnosis)
- [VHD dataset and annotation tools](https://github.com/transitional-ai/hallucinationdataset)
- HERBench and Perception Test remain subject to their upstream licenses and
  distribution terms.  This repository stores derived scores and hashes, not
  copies of their videos.

The raw JSONL response archives are omitted because they are large and contain
upstream media metadata.  Their SHA-256 digests are retained in the sealed
archives and post-hoc analysis manifest.  The included scored predictions are
sufficient to verify the reported confirmation tables and promotion outcomes.

## Inference scope

The registered full panel evaluates three model families under four temporal
views, giving 12 calls and 168 input frames per question.  The `katana/`
directory records the exact model revisions, deterministic frame samplers,
checkpointing, and job wrappers used on the UNSW Katana cluster.  Model weights
are not redistributed here.

## License and citation

The source code in this repository is released under the
[MIT License](LICENSE).  Dataset media, benchmark files, model weights, and
other third-party material remain subject to their respective upstream terms.
The final article citation will be added after publication.  Katana should be
cited as:

> Katana. (2010). https://doi.org/10.26190/669X-A286
