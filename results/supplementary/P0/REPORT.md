# P0 preflight report

Status: S1, awaiting review. No GPU jobs were submitted. Archived files and decisions were preserved.

## Scope clarification

The user's follow-up asks whether archived code and Katana records identify the AURC definition, making further recomputation unnecessary. They do. No additional AURC or bootstrap recomputation was run after that clarification. Test point estimates had already been recomputed. Pilot counterfactual gains, bounds and changed decisions remain unevaluated, rather than being reported as unchanged.

## Which AURC was used

| Use | Definition | Code |
| --- | --- | --- |
| Pilot candidate gain and bootstrap lower bound | Tie-grouped trapezoid | evaluate_v2_target_pilot.py: aurc, relative_gain, grouped_bootstrap_gains, promote |
| Test AURC and reported mean gain | Mean prefix risk | analyze_herbench_trace_rc.py: method_metrics → evaluate_response_graph_features.aurc → evaluate_probe_escalation.aurc |
| Test bootstrap confidence interval | Tie-grouped trapezoid | analyze_herbench_trace_rc.py: primary_bootstrap → evaluate_v2_target_pilot.relative_gain |

The pilot helper groups exact score ties and integrates group endpoint risks with the trapezoidal rule. The paper function averages cumulative error rates over every retained prefix. It sorts by descending score, with original input row order breaking ties. These definitions can differ even without ties. During bootstrap, the input order is the concatenation of sampled whole-video blocks, with the original within-video row order preserved.

The helper's SHA256 matches both target seals. The current final-analysis source matches the Perception seal. The older HERBench seal names a different final-analysis hash, discussed below. The saved test point estimates independently reproduce the prefix definition. Evidence and function excerpts: [aurc_provenance.json](aurc_provenance.json).

Identifying the historical definition does not require recomputation. Determining whether a *different* definition would change candidate eligibility or selection does. Existing selected-candidate lower bounds do not supply the missing candidate score vectors or the bounds for other candidates.

## Test results already recomputed

All rows keep the archived choices fixed. Each benchmark mean averages model-specific relative reductions, without pooling questions or taking a ratio of averaged AURCs.

| Dataset | Paper prefix mean (%) | Helper trapezoid (%) | Difference (percentage points) |
| --- | --- | --- | --- |
| HERBench | 16.644639 | 16.657901 | 0.013262 |
| Perception | 18.868028 | 18.889766 | 0.021738 |

| Dataset | Model | Archived choice | Prefix gain (%) | Trapezoid gain (%) |
| --- | --- | --- | --- | --- |
| HERBench | InternVL | target_logistic | 11.392735 | 11.402687 |
| HERBench | Qwen | target_logistic | 25.255787 | 25.274251 |
| HERBench | LLaVA | target_logistic | 13.285395 | 13.296763 |
| Perception | InternVL | raw_fallback | 0.000000 | 0.000000 |
| Perception | Qwen | target_logistic | 28.141382 | 28.173923 |
| Perception | LLaVA | target_logistic | 28.462701 | 28.495376 |

The paper's 16.64% and 18.87% point gains are reproduced. These small test-point differences do not establish that candidate choices or confidence bounds would stay unchanged. All candidate test scores are in [test_metrics_both_definitions.csv](test_metrics_both_definitions.csv). No confidence interval under the prefix definition was computed here.

## Pilot records and decision limits

| Dataset | Model | Candidate | Logged pilot gain (%) | Logged lower bound (%) | Selected |
| --- | --- | --- | --- | --- | --- |
| HERBench | InternVL | source_trace_hgb | 3.776449 | Not saved | False |
| HERBench | InternVL | target_logistic | 12.703782 | 6.783693 | True |
| HERBench | Qwen | source_trace_hgb | 25.775610 | Not saved | False |
| HERBench | Qwen | target_logistic | 31.394530 | 23.198038 | True |
| HERBench | LLaVA | source_trace_hgb | 7.269445 | Not saved | False |
| HERBench | LLaVA | target_logistic | 12.252639 | 4.866150 | True |
| Perception | InternVL | source_trace_hgb | -8.190513 | Not saved | False |
| Perception | InternVL | target_logistic | -1.379104 | Not saved | False |
| Perception | Qwen | source_trace_hgb | 22.795643 | Not saved | False |
| Perception | Qwen | target_logistic | 27.185236 | 21.046780 | True |
| Perception | LLaVA | source_trace_hgb | 26.919620 | Not saved | False |
| Perception | LLaVA | target_logistic | 29.615954 | 23.911790 | True |

These are transcribed archived values, not newly recomputed values. For a fallback decision, the lock's zero gain and zero bound describe the fallback itself; they are not rejected-candidate bounds. An unselected candidate can have passed its threshold and lost to another candidate, so unselected is not automatically failed.

The target response JSONL files are absent from the repository and were not found in the surveyed Katana home and scratch directories. The search excluded hidden directories, environments and caches; see [remote_inventory.json](remote_inventory.json) for scope. This is an availability finding, not a claim that no backup exists anywhere. The model bundles contain fitted estimators and feature-column lists, but no pilot response vectors, features or out-of-fold scores. Test scores cannot reconstruct those pilot observations.

[pilot_candidate_audit.csv](pilot_candidate_audit.csv) records the archived gains, available bounds, fixed seeds and thresholds; recomputed fields are blank. [decision_comparison.csv](decision_comparison.csv) marks all counterfactual decisions unknown. Later repeated-split analyses and archived-probability equivalence checks need the target raw responses or a verified equivalent feature/score archive.

## Split and checksum inventory

| Dataset | Split | Questions | Videos | Verified raw response rows |
| --- | --- | --- | --- | --- |
| HERBench | pilot | 1008 | 34 | Unavailable |
| HERBench | test | 963 | 34 | Unavailable |
| Perception | pilot | 977 | 250 | Unavailable |
| Perception | test | 911 | 250 | Unavailable |

All question IDs are unique in each manifest. Label joins are complete, video groups do not cross pilot/test, and each test question has exactly the three model rows. HERBench collection balance matches its lock; see [herbench_source_balance.csv](herbench_source_balance.csv).

The initial local/reference audit contains 129 checksum comparisons: {'git_blob_match_worktree_crlf': 55, 'content_mismatch': 3, 'exact_match': 29, 'missing': 42}. These counts are comparisons, not unique files; missing local paths are not evidence that remote copies are missing. Git blob checks separate Windows CRLF conversion from substantive byte differences. No archived file was rewritten to normalize line endings.

Follow-up resolved two initially unmatched files. The original HERBench seal was recovered with the exact archived SHA256; its local difference is mixed versus normalized line endings. The persistent worker was recovered with its sealed hash; the local version omits a trailing blank line. The current HERBench analysis script agrees with Katana and the Perception seal but does not match the older HERBench seal. That historical source version is unresolved. See [checksum_followup.json](checksum_followup.json).

All 19 development input references in the posthoc manifest were located on Katana by exact checksum, including raw responses and metadata, with row counts saved in [development_dependency_resolution.json](development_dependency_resolution.json). The preliminary remote inventory's NExT-GQA duplicate counts use the wrong ID preference and are explicitly excluded; its byte hashes and total row counts remain valid. The existing feature builder correctly prioritizes pilot_id. No development features or models were rebuilt.

## MVBench raw records

All three raw archives were found, copied into P0 and matched the posthoc SHA256 values. Each has 7200 rows and 1800 questions with four views. Each model has 1723 distinct media paths; these are path counts, not a newly reconstructed group split.

| View | Frame budget |
| --- | --- |
| base | 8 |
| invariant_nuisance | 8 |
| dense_acquisition | 16 |
| temporal_order_destruction | 8 |

The MVBench model panel is Qwen 4B, InternVL 4B and LLaVA 8B. It differs from the target panel and its 16/8/24/8 frame budgets. Each MVBench raw record reports one prefix-logit forward call and generated=false. See [mvbench_views.csv](mvbench_views.csv) and [mvbench_raw_audit.json](mvbench_raw_audit.json).

## Inference and prompt placement

The examined adapters score option-label logits directly. With single-token labels they use one forward pass; otherwise they use teacher-forced continuation scoring. They do not call generate(). Video/frame content comes before the question and options in all three examined prompt builders. InternVL also places its system message before the user message.

The existing runner already decodes the union of requested frame indices and reuses decoded frames for the same video segment. The target wrapper groups work by segment. The examined forward calls use use_cache=False. Thus P1 must measure additional changes against this existing implementation, rather than attributing existing frame reuse or single-token scoring to a new pipeline.

The public repository omits the imported LLaVA adapter. Its current Katana copy is saved under inputs/remote_code. Its historical target-run identity is not independently sealed here. Missing target raw records also prevent verifying how often those runs took a multi-token fallback. See [inference_audit.json](inference_audit.json).

## Environment, anomalies and later phases

Metric calculations used the environment recorded in [environment.json](environment.json). The fitted bundles were created with scikit-learn 1.7.2, whereas the available inspection environment has 1.6.1. Bundles were inspected only; no predictions, training or refitting used them. A compatible environment is needed before later model-based reproduction. The default Anaconda environment could not load these NumPy 2-era bundles; this did not affect the saved metric calculations.

The first audit attempts encountered different split-lock field names and were corrected before emitting final analytical outputs. No model parameters, thresholds, feature code or archived selections were changed.

The archived PBS scripts request allocations exceeding the user's five GPU-hour stop rule. They were not submitted. [pbs_budget_guard.json](pbs_budget_guard.json) records their configured allocation limits; these are not measured runtime costs.

For P2 onward, [canonical_metric.py](canonical_metric.py) imports the exact paper prefix-risk function. It is not wired into any running experiment. New splits must use that one function for selection, bootstrap and reporting, with the existing resampling, LR/HGB settings and thresholds unchanged. Split 0 must retain the archived choices. Whole-video grouping remains mandatory.

## Manuscript implications

The two published point gains need no numerical change based on this audit. The methods description must distinguish the helper's trapezoidal AURC used for pilot selection and bootstrap from the prefix-risk mean used for reported test point gains. Do not describe the archived confidence intervals as newly computed under the prefix definition, or claim that changing definitions preserves the candidate decisions. MVBench uses the smaller development panel and 8/8/16/8 frames. Do not describe the existing single-token scoring path as text generation.

Copyable LaTeX is in [p0_metric_note.tex](p0_metric_note.tex). This report does not modify the paper or historical result archives.

## Checkpoint

Stopped at S1. The historical metric question is answered. Counterfactual pilot decisions remain unknown. P1 and later phases have not started.

