# Video Hallucination Diagnosis

This repository contains the code, lightweight evaluation logs, summary CSV files, and generated figures for the VH-Probe video hallucination diagnosis study.

VH-Probe is designed to evaluate whether video-language models rely on visual evidence or language priors when answering questions about physically challenging video scenarios.

## Repository Structure

```text
.
|-- data/
|   |-- experiment_logs/        # Lightweight model-response logs
|   |-- metric_summaries/       # Summary CSV files used for tables and figures
|   `-- movie_segments_manifest.csv
|-- scripts/
|   |-- compute_experiment_metrics.py
|   |-- plot_paper_figures.py
|   |-- plot_ipv_bench_mcqa.py
|   |-- generate_paper_figures.py
|   `-- extract_movie_segments.py
|-- *.png                      # Generated analysis figures
|-- main.tex                   # Manuscript source included for reference
`-- requirements.txt
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

The plotting scripts use:

```text
matplotlib
numpy
scipy
```

## Reproducing the Figures

To regenerate the main analysis figures, run:

```bash
python scripts/generate_paper_figures.py
```

This script calls the plotting utilities and writes the generated figures to the repository root.

## Data Files

The repository includes lightweight CSV files only:

- `data/experiment_logs/` contains model-response logs.
- `data/metric_summaries/` contains summary statistics used by the figures and tables.
- `data/movie_segments_manifest.csv` records the local timestamp ranges used for the movie examples.

Large video files are not included in this repository.

The archived data package is available on Zenodo:

```text
https://doi.org/10.5281/zenodo.20806194
```

## Movie Segment Extraction

The script below can recreate local movie snippets if the corresponding source videos are available locally:

```bash
python scripts/extract_movie_segments.py
```

The generated clips are written to:

```text
data/missing_files_v1-2_test/movie_segments/
```

These generated clips are intentionally excluded from GitHub.

## Notes on Reproducibility

The CSV files in `data/metric_summaries/` are the paper-facing summary values used for the current figures and tables. Recomputing summaries from raw logs may overwrite these values, so check the outputs before replacing the committed CSV files.

## Citation

If you use this code or data, please cite the associated VH-Probe paper.

A formal citation entry will be added after publication.
