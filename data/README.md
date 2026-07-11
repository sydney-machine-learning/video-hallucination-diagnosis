# Model response files

`model_responses/` contains the final outputs used for the paper results. Each
CSV has one row per question and preserves the selected choice, confidence,
parsing status, refusal flag, model metadata, and raw model response. The
`question_id` field is the only dataset identifier included in these files.

Gold answers and derived correctness columns are deliberately excluded. Obtain
the human-verified `evaluation_manifest.csv` from the
[Kaggle dataset](https://www.kaggle.com/datasets/mlopssss/video-hallucination-diagnosis)
and provide it to `scripts/generate_all_results.py` when reproducing the scores.

The four files should each contain exactly 807 unique `question_id` values and
340 unique `clip_id` values.
