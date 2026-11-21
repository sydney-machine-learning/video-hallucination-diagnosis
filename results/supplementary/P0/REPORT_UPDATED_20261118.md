# P0 updated report: requested CPU recomputations

This report supersedes the earlier statement that no further recomputation was requested. The user subsequently authorized selection sensitivity and all trapezoidal test intervals to be recomputed with prefix-risk mean. Historical files and selections remain unchanged.

## 1. Original-split selection sensitivity

Not completed: target pilot response vectors, complete features, and out-of-fold candidate scores remain unavailable in the searched repository, D:/paper 3 and Katana locations. The additional D-drive check covers binary feature/model containers and CSV dataset fields. Katana inspection covers experiment logs, candidate tables and checkpoints; it found labels, fitted bundles and aggregate candidate metrics, not the missing row-level pilot scores. Checkpoint mentions in logs refer to model weight loading rather than saved pilot score vectors.

The requested prefix pilot gains, lower bounds and candidate pass/fail therefore remain unknown. We do not report unchanged selections or infer rejected-candidate bounds from fallback zeros. Known archived values and blank recomputation fields are saved in selection_sensitivity_status.csv. Original model choices have not been altered.

## 2. Test confidence intervals

Completed for both target confirmations. We reused analyze_herbench_trace_rc.primary_bootstrap unchanged, with 10,000 draws and original seeds 20260802/20260803. We first reran the archived trapezoidal definition. Every replicate equals the archived distribution exactly: maximum absolute error is zero for both datasets. Only the metric callable was then changed to evaluate_probe_escalation.aurc. The same video blocks, sampled order, within-block row order and strict input-order tie rule are retained.

| Dataset | Old trapezoidal CI (%) | New prefix mean CI (%) | Point gain (%) | Primary success check |
| --- | --- | --- | --- | --- |
| HERBench | [12.121000, 22.609653] | [12.112518, 22.590944] | 16.644639 | Pass under both |
| Perception | [15.428924, 22.139513] | [15.413039, 22.115259] | 18.868028 | Pass under both |

Outputs below are located in ../P0_recompute/. Saved outputs: test_ci_old_vs_new.csv, each target's prefix/trapezoid bootstrap CSV, verification JSON, and resamples.npz. The NPZ stores the ordered video list, bootstrap group indices and seed. The original estimates and selections stay fixed. Both revised lower bounds remain above zero, so neither primary success check changes. This is not evidence about counterfactual pilot selection.

## 3. Paired-difference intervals

The 26 target/comparator intervals in the two posthoc archives already use prefix-risk mean. The call chain is analyze_trace_rc_posthoc_robustness.reported_relative_gain → evaluate_response_graph_features.aurc → evaluate_probe_escalation.aurc. The training-free baseline routine calls the same reported_relative_gain. They were not trapezoidal intervals and do not require conversion.

We verified each interval against quantiles of its saved replicate distribution and retained it unchanged. The saved seeds are dataset seed + 4409 for standard paired comparisons and + 8123 for training-free comparisons. These differ from the primary interval seeds by design; the primary and paired-versus-raw intervals need not be numerically identical. paired_ci_old_vs_new.csv records the unchanged old/new endpoints and explicitly labels them already_prefix_saved_quantiles_verified, not newly recomputed distributions.

| Dataset | Family | Comparator | Old CI | New CI |
| --- | --- | --- | --- | --- |
| HERBench | paired_bootstrap | raw_confidence | [12.091415, 22.541547] | [12.091415, 22.541547] |
| HERBench | paired_bootstrap | target_scalar_logistic | [11.817307, 22.361565] | [11.817307, 22.361565] |
| HERBench | paired_bootstrap | target_base_graph_logistic | [0.336360, 5.481977] | [0.336360, 5.481977] |
| HERBench | paired_bootstrap | source_trace_logistic | [-1.011706, 1.350825] | [-1.011706, 1.350825] |
| HERBench | paired_bootstrap | source_trace_hgb | [0.150670, 4.843772] | [0.150670, 4.843772] |
| HERBench | paired_bootstrap | target_full_graph_logistic | [0.000000, 0.000000] | [0.000000, 0.000000] |
| Perception Test | paired_bootstrap | raw_confidence | [15.509268, 22.080692] | [15.509268, 22.080692] |
| Perception Test | paired_bootstrap | target_scalar_logistic | [15.511563, 22.220240] | [15.511563, 22.220240] |
| Perception Test | paired_bootstrap | target_base_graph_logistic | [-0.995909, 2.097622] | [-0.995909, 2.097622] |
| Perception Test | paired_bootstrap | source_trace_logistic | [0.539110, 7.530031] | [0.539110, 7.530031] |
| Perception Test | paired_bootstrap | source_trace_hgb | [1.412231, 6.314558] | [1.412231, 6.314558] |
| Perception Test | paired_bootstrap | target_full_graph_logistic | [-2.273831, 0.920880] | [-2.273831, 0.920880] |
| HERBench | training_free_paired_bootstrap | confidence_margin | [13.297369, 24.157089] | [13.297369, 24.157089] |
| HERBench | training_free_paired_bootstrap | negative_entropy | [12.050199, 23.145952] | [12.050199, 23.145952] |
| HERBench | training_free_paired_bootstrap | base_vote_then_confidence | [4.398484, 12.205561] | [4.398484, 12.205561] |
| HERBench | training_free_paired_bootstrap | base_support | [1.347722, 6.736268] | [1.347722, 6.736268] |
| HERBench | training_free_paired_bootstrap | own_temporal_stability | [24.916131, 39.750672] | [24.916131, 39.750672] |
| HERBench | training_free_paired_bootstrap | all_view_vote_then_confidence | [1.085569, 7.211977] | [1.085569, 7.211977] |
| HERBench | training_free_paired_bootstrap | all_view_support | [-0.697181, 2.910530] | [-0.697181, 2.910530] |
| Perception Test | training_free_paired_bootstrap | confidence_margin | [15.436289, 22.343195] | [15.436289, 22.343195] |
| Perception Test | training_free_paired_bootstrap | negative_entropy | [15.802259, 23.039775] | [15.802259, 23.039775] |
| Perception Test | training_free_paired_bootstrap | base_vote_then_confidence | [6.083775, 15.413001] | [6.083775, 15.413001] |
| Perception Test | training_free_paired_bootstrap | base_support | [2.763354, 11.857986] | [2.763354, 11.857986] |
| Perception Test | training_free_paired_bootstrap | own_temporal_stability | [42.313135, 69.763984] | [42.313135, 69.763984] |
| Perception Test | training_free_paired_bootstrap | all_view_vote_then_confidence | [6.449958, 18.393274] | [6.449958, 18.393274] |
| Perception Test | training_free_paired_bootstrap | all_view_support | [3.036442, 12.202227] | [3.036442, 12.202227] |

## 4. Additional Katana and D-drive checks

katana_logs_and_readiness.json contains source paths, hashes, relevant log excerpts, candidate-table schemas and asset existence checks. The target raw records were not recovered. All target video paths referenced by the archived manifests are currently absent, as are the original Qwen/InternVL and LLaVA Python environment executables. The user-level Hugging Face cache directory exists; presence of the required pinned model revisions has not been established.

The D-drive check in D_drive_feature_check.json found no target dataset table or binary feature/model container that fills the pilot-data gap. The prior search reports record scope and inaccessible locations; absence in the searched locations is not proof that all possible backups are gone.

## 5. Progress after the CPU work

P2 now has split 0 plus 30 additional whole-video 50/50 splits per target. HERBench is balanced by source collection. No labels were used to construct splits, no video crosses its pilot/test boundary, and split 0 retains archived roles. These are prepared splits, not performance results. No selection model was retuned or refitted.

P1 cannot complete the requested archived-probability equivalence check without the target raw responses; its original media and inference environments also need restoration. P2, P3A and P4 require the missing target option probabilities or equivalent feature archives. Test correctness and scalar confidence alone cannot reconstruct cross-model option distributions or pilot features. These are execution dependencies, not additional approval checkpoints. No GPU jobs were submitted, and S2 has not been reached.

## 6. Changes required in paper numbers

Replace the two primary confidence intervals with the new endpoints above when reporting prefix-risk mean consistently for fixed archived choices. Keep point gains 16.64% and 18.87%. Paired-difference intervals need no numerical changes. Do not claim prefix-based pilot selection has been validated: that computation is blocked by missing inputs. Historical selections still used the original helper and are retained for split 0.

The reusable snippet is p0_ci_update.tex. The previous P0 report remains preserved; this is its dated update. Environment versions, input/output hashes and original-file preservation are recorded in MANIFEST.json.
