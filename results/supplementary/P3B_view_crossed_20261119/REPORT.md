# Two observation protocols with shared answers

Complete. This is exploratory evidence on the same previously inspected Perception cohort. Both protocols and all 30 additional splits are retained.

## Primary feature comparison

Both feature sets use the exact same fixed pilot LR and target labels. Values are paired AURC differences: our features minus reconstructed PhaseGuard features. Negative favors our features. Intervals are pointwise video bootstrap intervals, without multiplicity adjustment.

| policy | estimate | lo | hi |
| --- | --- | --- | --- |
| original_views | -0.00735 | -0.02007 | 0.00119 |
| phase_views | 0.00383 | -0.00508 | 0.01438 |

## Original split

| policy | method | accuracy | aurc | auroc | risk_at_50pct | gain_percent |
| --- | --- | --- | --- | --- | --- | --- |
| original_views | raw | 0.70472 | 0.12528 | 0.78956 | 0.08772 | 0.00000 |
| original_views | label_free_support | 0.70472 | 0.11543 | 0.79479 | 0.10307 | 7.86597 |
| original_views | ours_unchecked_lr | 0.70472 | 0.10442 | 0.83006 | 0.08553 | 16.65387 |
| original_views | ours_checked | 0.70472 | 0.10442 | 0.83006 | 0.08553 | 16.65387 |
| original_views | pg_single | 0.70472 | 0.11936 | 0.79652 | 0.08991 | 4.72528 |
| original_views | pg_full | 0.70472 | 0.10525 | 0.82608 | 0.08333 | 15.98764 |
| original_views | pg_fixed_rf | 0.70472 | 0.10324 | 0.83136 | 0.08333 | 17.59527 |
| original_views | pg_same_lr | 0.70472 | 0.11177 | 0.82008 | 0.08772 | 10.78423 |
| phase_views | raw | 0.70472 | 0.12528 | 0.78956 | 0.08772 | 0.00000 |
| phase_views | label_free_support | 0.70472 | 0.14986 | 0.70396 | 0.17105 | -19.61876 |
| phase_views | ours_unchecked_lr | 0.70472 | 0.13024 | 0.77061 | 0.12939 | -3.95983 |
| phase_views | ours_checked | 0.70472 | 0.12528 | 0.78956 | 0.08772 | 0.00000 |
| phase_views | pg_single | 0.70472 | 0.12000 | 0.79322 | 0.09211 | 4.21144 |
| phase_views | pg_full | 0.70472 | 0.11365 | 0.80287 | 0.09649 | 9.28231 |
| phase_views | pg_fixed_rf | 0.70472 | 0.12128 | 0.78556 | 0.10746 | 3.19056 |
| phase_views | pg_same_lr | 0.70472 | 0.12641 | 0.77075 | 0.11404 | -0.90061 |

## Additional split variability

These percentiles describe 30 splits of the same cohort, not independent-dataset uncertainty.

| policy | method | mean_gain | p05_gain | p95_gain | harm_rate |
| --- | --- | --- | --- | --- | --- |
| original_views | raw | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| original_views | label_free_support | 9.96114 | 6.14006 | 15.05051 | 0.00000 |
| original_views | ours_unchecked_lr | 18.69469 | 14.85893 | 23.39792 | 0.00000 |
| original_views | ours_checked | 17.75201 | 13.65937 | 23.01430 | 0.00000 |
| original_views | pg_single | 3.89071 | 0.48187 | 6.54321 | 0.03333 |
| original_views | pg_full | 18.14881 | 14.11352 | 22.14522 | 0.00000 |
| original_views | pg_fixed_rf | 17.52123 | 14.19666 | 21.23453 | 0.00000 |
| original_views | pg_same_lr | 18.06933 | 12.53777 | 24.79802 | 0.00000 |
| phase_views | raw | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| phase_views | label_free_support | -18.78826 | -26.30781 | -12.57121 | 1.00000 |
| phase_views | ours_unchecked_lr | -2.80645 | -10.91375 | 5.71062 | 0.76667 |
| phase_views | ours_checked | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| phase_views | pg_single | 1.24463 | -0.00918 | 2.61253 | 0.06667 |
| phase_views | pg_full | 9.57154 | 6.55708 | 12.44776 | 0.00000 |
| phase_views | pg_fixed_rf | 3.08339 | -2.45858 | 9.21279 | 0.30000 |
| phase_views | pg_same_lr | 1.37285 | -6.71569 | 8.85291 | 0.40000 |

## Dependence on the observation protocol

Difference of the two paired feature-comparison differences, original views minus phase views. The same video resamples and answers are used.

| subset | difference_of_AURC_differences | lo | hi |
| --- | --- | --- | --- |
| all | -0.01119 | -0.02535 | 0.00085 |
| physics | -0.01534 | -0.03701 | 0.00663 |

## Measured inference cost

| protocol | full_scorer_calls_including_common_answer | frame_exposures_including_common_answer | mean_call_wall_seconds_including_common_answer | common_answer_extra_seconds |
| --- | --- | --- | --- | --- |
| original_views | 12 | 168 | 24.33037 | 0.00000 |
| phase_views | 13 | 40 | 4.36530 | 1.20695 |

Call times sum saved preprocessing and generation wall times; they are not pure GPU-kernel time or end-to-end latency. Decoding, model loading, scorer fitting, queueing and failed allocations are separate. Both full scorers share the same cost within a protocol. Actual PBS allocations, including failures, are retained in accounting/. The phase protocol is charged one extra base16 call to obtain the common answer.

## Interpretation and reproduction limits

The fixed answer is the newly inferred Qwen3-VL-8B-Instruct base16 option argmax. Accuracy is identical across all core comparisons by construction. This isolates reliability ranking, not answer generation or rescue. The source HGB uses extra source labels and encounters changed inference/view distributions; its result is a transfer control, not the primary feature attribution.

Original frame indices follow the archived sampler. The new shared prompt/generation/runtime differs from the missing historical option-logit-only run, so these are not replacements for archived paper numbers. On original views, PhaseGuard adjacent/opposite features refer to fixed slots, not literal phase geometry. On phase views, our original addition/removal labels lose their literal intervention meaning. Full PhaseGuard features are a documented reconstruction. See PROTOCOL.md and FEATURE_SPEC.json.

The native phase-vote Qwen3-VL-8B-Instruct reliability reproduction remains separately in ../P3B_full_reference_20261119. It excludes 32B rescue and uses our target cohort. Its different answers and budget prevent treating its absolute AURC as a matched core comparison.

## Sources

matched_metrics.csv; original_split_intervals.csv; repeated_summary.csv; cross_view_interaction.csv; cost_by_model_view.csv; cost_by_protocol.csv; analysis/*/split_*/test_predictions.csv; COMMON_ANSWERS.csv; MANIFEST.json; ANALYSIS_LOCK.json; AUDIT.json; accounting/.

All claims about superiority must follow the saved intervals and both view conditions; an interval containing zero does not establish a difference. No Overleaf edits are made.
