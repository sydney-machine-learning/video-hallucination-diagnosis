# Video Hallucination Diagnosis

This repository contains the evaluation outputs, metric implementation, and
paper figures for the Video Hallucination Diagnostics (VHD) pilot benchmark.
VHD evaluates four video-language models with three measures:

- **Accuracy** on the original video questions.
- **Prior Dominance Index (PDI)** on temporally reversed videos.
- **Hallucination Calibration Error (HCE)** on masked-evidence videos.

The evaluated models are Qwen2.5-Omni-3B, GPT-5.4, GPT-5.5, and GPT-5.6.

## Related resources

- [VHD dataset on Kaggle](https://www.kaggle.com/datasets/mlopssss/video-hallucination-diagnosis): videos, question sets, human-verified labels, and the labelled evaluation manifest.
- [Dataset and annotation tools](https://github.com/transitional-ai/hallucinationdataset): dataset download, human-in-the-loop review, and model-response collection.

Videos and annotation utilities are not duplicated in this repository. The
manuscript source is maintained separately in Overleaf while the paper is under
revision.

## Repository structure

```text
.
|-- data/
|   `-- model_responses/       # One response CSV per evaluated model
|-- results/
|   |-- figures/               # Main, source-split, and category plots
|   |-- tables/                # Item-, clip-, and aggregate-level results
|   `-- case_examples/         # Real and AI-generated fire examples
|-- scripts/
|   |-- generate_all_results.py
|   |-- make_main_results_figure.py
|   |-- make_source_split_figures.py
|   `-- make_category_heatmaps.py
`-- requirements.txt
```

## Reproduce the results

Create an environment and install the plotting dependencies:

```bash
python -m pip install -r requirements.txt
```

Download and extract the Kaggle dataset, then pass its labelled
`evaluation_manifest.csv` to the driver script:

```bash
python scripts/generate_all_results.py \
  --manifest /path/to/evaluation_manifest.csv \
  --bootstrap-repeats 1000
```

The script regenerates all aggregate CSV files in `results/tables/` and all
quantitative plots in `results/figures/`. Progress is printed every 100
bootstrap samples.

## Evaluation details

All four response files contain 807 completed responses covering 340 short
video clips. The repository stores model predictions and confidence values,
but not the gold labels. During scoring, predictions are joined to the
human-verified Kaggle manifest by `question_id`.

Scores are first computed at short-clip level. Accuracy and PDI uncertainty is
reported with 95% percentile bootstrap confidence intervals over clips. HCE is
recomputed after each clip-level bootstrap sample using five confidence bins.
The real-versus-AI analysis uses the same procedure within each video source.

## Citation

Please cite the associated VHD paper when using these results or scripts. A
formal citation entry will be added after publication.
