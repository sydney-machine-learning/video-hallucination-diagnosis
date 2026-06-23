# Video Hallucination Diagnosis

This repository contains the analysis code, plotting scripts, and lightweight CSV experiment logs/summaries for the VH-Probe video hallucination diagnosis study.

## Contents

- `main.tex`: manuscript source included for context.
- `scripts/`: metric aggregation and figure-generation scripts.
- `data/experiment_logs/`: lightweight model-response logs used by the scripts.
- `data/metric_summaries/`: CSV summaries used for paper tables and plots.
- `data/movie_segments_manifest.csv`: local timestamp manifest for the four movie examples.
- `*.png`: generated figures used by the analysis and manuscript.

Large video files are intentionally excluded from the GitHub repository and should be archived separately, for example through Zenodo.

## Reproducing figures

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all figure-generation scripts:

```bash
python scripts/generate_ablation_figures.py
```

Note: `data/metric_summaries/` contains the current paper-facing summary values. Recomputing summaries from raw logs may overwrite manually aligned CSV values.

To recreate local movie snippets from locally available source videos, run:

```bash
python scripts/extract_movie_segments.py
```

The generated clips are written to `data/missing_files_v1-2_test/movie_segments/` and are not distributed in this repository.
