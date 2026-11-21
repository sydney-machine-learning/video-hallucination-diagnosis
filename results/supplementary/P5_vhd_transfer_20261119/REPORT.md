# Existing record analysis: VHD reliability and fixed checkpoint transfer

Completed locally with CPU only. No VLM inference, GPU jobs, or Overleaf edits. Original results and selections are unchanged. All results below are new, exploratory reconstructions from archived probabilities, not newly discovered historical metrics.

## Findings

The VHD analysis connects the diagnostic dataset to the learned scores, with a negative result on the reversed subset. It does not demonstrate successful rejection of shared errors. The fixed checkpoint analysis shows that changing model size is not necessary for transfer failure. It does not identify the separate effects of dataset, model size and views on the final targets.

## Inputs and reconstruction

The original development job arguments are preserved in katana/recover_llava_and_analyze.pbs, lines 65 to 86, and katana/run_perception_valid500.pbs. We reused those response banks and metadata. source_manifest.json records absolute source paths, row counts, byte sizes and SHA256 checksums; download_manifest.json records remote paths and matching local checksums for eight downloaded files. Remote storage was read only.

The main remote sources recovered were:

* /home/z5611125/sicr_route/results_fresh/rava_vhd_confirmation/8905351.kman.restech.unsw.edu.au/
* /home/z5611125/sicr_route/results_fresh/prave_tempcompass_confirmation/8905479.kman.restech.unsw.edu.au/
* /home/z5611125/probe_route/input/development_scored/

NExT GQA, STAR and MVBench responses were read from their surviving D:/paper 3 archives. VHD task labels and source clip IDs came from D:/paper 3/caver_route/results/vhd_mechanisms/vhd_primary_predictions.csv and were joined by exact question IDs. Gold labels and source clip IDs were checked against the development metadata. No scores or answers were borrowed from that separate mechanism run.

| Dataset | Questions | Source videos |
| --- | --- | --- |
| MVBench | 1800 | 1723 |
| NExT-GQA | 500 | 500 |
| STAR | 500 | 500 |
| TempCompass | 410 | 410 |
| VHD | 801 | 341 |

The same checkpoints occur in all five datasets:

| Model | Revision |
| --- | --- |
| Qwen/Qwen3-VL-4B-Instruct | ebb281ec70b05090aa6165b016eac8ec08e71b17 |
| OpenGVLab/InternVL3_5-4B-Instruct | a3fd3158be4027880d91ccee80ea8089559e0896 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | 802f16c8346a062fef7ddf92b2bd64a770a50a1a |

There are 40,866 archived response records, 4,011 questions and 12,033 answer rows. The unchanged graph builder creates 53 features. VHD has base and dense views with 8 and 16 frames, TempCompass has 6 and 12 frames, STAR and MVBench have 8/8/16/8 frames, and NExT GQA uses four 16 frame views. Thus the fixed checkpoint control still includes differences in tasks and view definitions. Missing view groups retain the original availability indicators and zero comparisons.

## Protocol

Entire datasets are held out. The VHD model is fitted on NExT GQA, STAR, MVBench and TempCompass, excluding all VHD labels and every VHD variant. This is a test of transferred development scores, not target pilot LR, promotion checks or the complete selected pipeline.

The existing functions graph_table, fit_graph_logistic, equal_domain_weights and aurc are imported from repository source. Development LR uses C=0.2 and standardization. HGB uses learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=30 and l2_regularization=2.0. Both retain seed 20260801 and equal total weight per source dataset. No hyperparameter search or target tuning was performed. The development LR is distinct from the target pilot LR with C=0.1.

AURC is the paper's prefix risk mean with the archived input order tie rule. Risk at 50% accepts ceil(N/2) highest scores. Positive relative gain means lower AURC. The main gain is the arithmetic mean of three model specific relative gains; it is not the relative change in the average AURC. AUROC uses correctness as the positive class and is computed over both correct and incorrect answers in each subset.

Bootstrap intervals use 10,000 paired source video resamples, seed 20261119, shared across methods and answering models. Every question and model response for a sampled video is included together. Scorers are held fixed; these intervals measure evaluation sample variation, not variation in scorer training or across independently collected datasets. Bootstrap index files and distributions are saved. They are exploratory intervals without a multiple comparison correction.

## VHD reversed subset

178 questions from 178 source clips. All are labeled reversed in the archived metadata. The stored dataset builders use ffmpeg's reverse filter for this manipulation. The models have mean answering accuracy 24.91%; uniform random choice between two options has expected accuracy 50%. Scores reorder fixed answers and cannot change this accuracy.

| Score | AURC | Mean relative gain and 95% CI | AUROC | Risk at 50% | Shared errors deferred at 50% |
| --- | --- | --- | --- | --- | --- |
| Original confidence | 0.7923 | reference | 0.384 | 80.90% | 39.73% |
| Development LR | 0.8231 | -7.20% [-13.28, -2.64] | 0.323 | 83.90% | 24.58% |
| Development HGB | 0.8098 | -5.48% [-9.16, -1.27] | 0.437 | 79.40% | 31.99% |

99 of the 178 questions are answered incorrectly by all three models. Shared error deferral is the fraction of these 99 questions placed in the rejected half, averaged over the three answering models. The selection budget is applied within this subset. These rates are not precision, accuracy, or an AUROC calculated on only wrong answers. Under random half selection its expected value is 50%.

HGB improves risk at exactly 50% slightly, despite worse overall AURC. The overall negative result is driven by InternVL; Qwen and LLaVA have positive AURC gains. Neither fact is suppressed.

| Answering model | Accuracy | Original AURC | LR gain | HGB gain |
| --- | --- | --- | --- | --- |
| InternVL3.5 4B Instruct | 37.64% | 0.5878 | -32.51% | -30.32% |
| Qwen3-VL 4B Instruct | 20.79% | 0.9188 | 7.81% | 9.95% |
| LLaVA OneVision 2 8B Instruct | 16.29% | 0.8703 | 3.09% | 3.92% |

The same VHD holdout model is used for all subsets; no subset tuning is performed:

| VHD subset | Questions | Source videos | LR gain | HGB gain |
| --- | --- | --- | --- | --- |
| all | 801 | 341 | -3.77% [-7.46, -0.37] | -2.14% [-14.52, -0.09] |
| original | 333 | 333 | -17.59% [-28.26, -8.71] | -19.17% [-45.46, -10.02] |
| reversed | 178 | 178 | -7.20% [-13.28, -2.64] | -5.48% [-9.16, -1.27] |
| masked | 290 | 290 | 10.78% [-2.44, 22.42] | 12.48% [-10.19, 22.37] |

Correctness must follow the sequence actually displayed. The reversal experiment is a controlled visual grounding test, not an estimate of hallucination prevalence in unedited videos. Metadata and construction code establish the manipulation; this CPU analysis does not revalidate every video annotation.

## Fixed checkpoint transfer control

The following HGB results use source video grouped internal validation wherever the archived default activates it. VHD and MVBench have no internal validation split under that default and are unchanged.

| Held out dataset | Development LR gain | Development HGB gain |
| --- | --- | --- |
| MVBench | 13.76% [10.60, 16.85] | 1.49% [1.36, 12.24] |
| NExT-GQA | 26.83% [20.03, 34.04] | 26.71% [11.88, 32.24] |
| STAR | 9.01% [2.98, 15.41] | 11.32% [2.23, 20.71] |
| TempCompass | 12.24% [6.24, 18.28] | 10.61% [1.71, 17.64] |
| VHD | -3.77% [-7.46, -0.37] | -2.14% [-14.52, -0.09] |

This demonstrates performance variation and VHD transfer failure with identical checkpoints. Dataset content, label distribution, task mix and view budgets remain coupled. It does not prove that model scaling is irrelevant to HERBench or Perception Test.

## Identity indicator sensitivity

Relative gains in percent:

| Dataset | LR full | LR remove three ID | HGB full | HGB remove three ID |
| --- | --- | --- | --- | --- |
| MVBench | 13.7606 | 13.7557 | 1.4909 | 1.4909 |
| NExT-GQA | 26.8272 | 26.8850 | 26.7115 | 26.7115 |
| STAR | 9.0093 | 9.0076 | 11.3214 | 11.3214 |
| TempCompass | 12.2361 | 12.2660 | 10.6074 | 10.6074 |
| VHD | -3.7697 | -3.7769 | -2.1352 | -2.1352 |

HGB scores are exactly unchanged in all five holdouts. Each model_identity_i column exactly duplicates base_peer_i_is_self in this builder; the named peer slots also retain family information. Therefore removing only the three model_identity columns is a redundant column ablation, not an identity free experiment. It cannot dismiss the reviewer's identity concern. LR changes slightly because duplicated standardized columns interact with regularization. No model size causal inference follows from this ablation.

## Internal validation audit and deviations

The frozen HGB constructor leaves early_stopping='auto'. For NExT GQA, STAR and TempCompass holdouts, the source table exceeds 10,000 rows, so sklearn's default makes a row based internal validation split. That can separate answers belonging to the same source video. The entire test dataset remains excluded, so no held out labels are leaked, but the default does not satisfy the requested source video grouping rule.

We first saved the exact constructor reconstruction for audit, then made one fixed CPU control with GroupShuffleSplit(test_size=0.1, random_state=20260801) over dataset plus source clip. All model rows and variants stay together. All fitting and regularization settings are retained. early_stopping is explicitly set to True in these three controls to preserve its original active mode after passing the validation rows separately; there is no threshold tuning. Sample weights are computed on the original source pool and subset into the two partitions. The primary supplemental table uses this grouped control. No archived evaluation or selection is replaced.

| Dataset | Original default validation gain | Grouped validation gain |
| --- | --- | --- |
| NExT-GQA | 27.65% | 26.71% |
| STAR | 11.15% | 11.32% |
| TempCompass | 7.82% | 10.61% |

Each grouped model used all 200 boosting iterations. Internal partition files and audit JSON are saved. The VHD result is unaffected by this issue. The first sandboxed fit could not create a Windows thread communication pipe; the authorized local CPU fit was rerun with the necessary process permission. One preliminary grouped attempt left auto early stopping off after reducing the explicit training rows; the assertion stopped it before any outputs were used. It was corrected to preserve the original active mode.

## Other checks and limitations

The separate surviving VHD mechanism file agrees with this response bank on 2287/2403 answers overall and 503/534 reversed subset answers. It has 98 shared wrong reversed questions, versus 99 here, with 93 in common. Its task labels are used, not its answers. Mixing its answers with these learned scores would be invalid. alignment_summary.json and vhd_primary_alignment.csv retain the comparison. The reconstructed current bank gives 99/178 shared errors and 99/116 errors among unanimous answers, matching the manuscript's rounded 55.6% and 85.3%; that numerical match alone does not certify the figure's full provenance.

HGB produces tied scores. The specified stable input order rule is preserved, including when video bootstrap draws reorder the sample. Some percentile intervals are asymmetric; no tie rule or interval was chosen after inspecting performance. score_tie_audit.csv records the number of distinct scores. Training uncertainty, additional VHD annotations and matched model scaling experiments are outside this analysis.

## Deliverables and review

* DRAFT_FOR_REVIEW.md: Chinese and English manuscript paragraphs, placement and limitations replacement.
* tables.tex: optional compact LaTeX tables. No new section is required.
* model_metrics.csv and mean_metrics.csv: exact archived constructor reconstruction, all subsets.
* model_metrics_grouped_validation.csv and mean_metrics_grouped_validation.csv: source video grouped controls, all five holdouts.
* gain_intervals_grouped_validation.csv: corresponding paired intervals; VHD subset rows are unchanged from the exact reconstruction.
* predictions.csv, predictions_grouped_validation.csv and vhd_scored_subsets.csv: all saved answer scores and labels.
* models/: fitted CPU scorers and feature columns.
* MANIFEST.json: commit, software environment, seeds, source and output checksums, GPU usage none.

Recommended edits: add the negative VHD holdout finding to RQ2, replace the unresolved VHD sentence, state the two option chance level and reversal interpretation, and qualify model transfer attribution. Do not claim these analyses prove the complete target checked method succeeds on VHD or disentangle model scale from dataset shift. Await the author's review before editing Overleaf.
