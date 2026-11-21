# Two families, four views

All three pairs survive separately for each target, with two reference-model rows per pair. Mean and minimum below are across the two included models, not across three pairs, videos or repeated splits. Gains are relative AURC reduction percentages against each reference model’s original confidence.

Source: `results/posthoc_robustness/compute_budget_ablation.csv`. Filter: `audit_family=model_panel`, `configuration=Two-family four views`. Each configuration uses 8 model/view calls and 112 model-input frames per question.

| Target | Pair | Mean gain (%) | Minimum gain (%) | CSV lines |
| --- | --- | --- | --- | --- |
| HERBench | InternVL+Qwen | 17.534755 | 10.095504 | 11,12 |
| HERBench | InternVL+LLaVA | 10.052165 | 9.448836 | 13,14 |
| HERBench | Qwen+LLaVA | 10.098873 | 2.740005 | 15,16 |
| Perception Test | InternVL+Qwen | 15.047914 | 1.385169 | 53,54 |
| Perception Test | InternVL+LLaVA | 14.837176 | 0.989612 | 55,56 |
| Perception Test | Qwen+LLaVA | 5.769737 | 2.996730 | 57,58 |

These are saved posthoc budget ablations. The implementation calls fit_target_scores and score_gain directly; this extraction does not apply or rerun the promotion bootstrap. Do not describe these rows as newly validated checked selections. No inference, fitting or GPU job was run.
