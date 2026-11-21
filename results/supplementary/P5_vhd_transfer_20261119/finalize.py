import analyze as a
import json
import numpy as np
import pandas as pd

p=a.HERE
means=pd.read_csv(p/'mean_metrics.csv')
grouped=pd.read_csv(p/'mean_metrics_grouped_validation.csv')
cis=pd.read_csv(p/'gain_intervals_grouped_validation.csv')
metrics=pd.read_csv(p/'model_metrics.csv')
alignment=json.loads((p/'alignment_summary.json').read_text())
names={'raw_confidence':'Original confidence','graph_lr':'Development LR','graph_hgb':'Development HGB','lr_without_three_id':'LR without three ID columns','hgb_without_three_id':'HGB without three ID columns'}
models={'OpenGVLab/InternVL3_5-4B-Instruct':'InternVL3.5 4B Instruct','Qwen/Qwen3-VL-4B-Instruct':'Qwen3-VL 4B Instruct','lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct':'LLaVA OneVision 2 8B Instruct'}

def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def ci_text(d,s,m):
    r=cis[(cis.dataset==d)&(cis.subset==s)&(cis.method==m)].iloc[0]
    return f'{r.gain_pct:.2f}% [{r.ci_low:.2f}, {r.ci_high:.2f}]'

vr=means[(means.dataset=='VHD')&(means.subset=='reversed')&means.method.isin(names)].set_index('method')
vtable=md(['Score','AURC','Mean relative gain and 95% CI','AUROC','Risk at 50%','Shared errors deferred at 50%'],[
    [names[m],f'{vr.loc[m,"aurc"]:.4f}', 'reference' if m=='raw_confidence' else ci_text('VHD','reversed',m),f'{vr.loc[m,"auroc"]:.3f}',f'{100*vr.loc[m,"risk50"]:.2f}%',f'{100*vr.loc[m,"shared_wrong_deferral_rate"]:.2f}%'] for m in a.METHODS[:3]])
lodo=md(['Held out dataset','Development LR gain','Development HGB gain'],[[d,ci_text(d,'all','graph_lr'),ci_text(d,'all','graph_hgb')] for d in sorted(grouped.dataset.unique())])
permodel=md(['Answering model','Accuracy','Original AURC','LR gain','HGB gain'],[[models[m],f'{100*g.iloc[0].accuracy:.2f}%',f'{g[g.method.eq("raw_confidence")].aurc.iloc[0]:.4f}',f'{g[g.method.eq("graph_lr")].relative_gain_pct.iloc[0]:.2f}%',f'{g[g.method.eq("graph_hgb")].relative_gain_pct.iloc[0]:.2f}%'] for m,g in metrics[(metrics.dataset=='VHD')&(metrics.subset=='reversed')].groupby('model_id')])
subsets=md(['VHD subset','Questions','Source videos','LR gain','HGB gain'],[[s,int(metrics[(metrics.dataset=='VHD')&(metrics.subset==s)].iloc[0]['n']),int(metrics[(metrics.dataset=='VHD')&(metrics.subset==s)].iloc[0].videos),ci_text('VHD',s,'graph_lr'),ci_text('VHD',s,'graph_hgb')] for s in ['all','original','reversed','masked']])
idrows=[]
for d,g in grouped.groupby('dataset'):
    g=g.set_index('method')
    idrows.append([d,f'{g.loc["graph_lr","relative_gain_pct"]:.4f}',f'{g.loc["lr_without_three_id","relative_gain_pct"]:.4f}',f'{g.loc["graph_hgb","relative_gain_pct"]:.4f}',f'{g.loc["hgb_without_three_id","relative_gain_pct"]:.4f}'])
idtable=md(['Dataset','LR full','LR remove three ID','HGB full','HGB remove three ID'],idrows)
reference=means[means.subset.eq('all')&means.method.eq('graph_hgb')].set_index('dataset')
grouped_hgb=grouped[grouped.method.eq('graph_hgb')].set_index('dataset')
earlytable=md(['Dataset','Original default validation gain','Grouped validation gain'],[[d,f'{reference.loc[d,"relative_gain_pct"]:.2f}%',f'{grouped_hgb.loc[d,"relative_gain_pct"]:.2f}%'] for d in ['NExT-GQA','STAR','TempCompass']])

zh_vhd='为检验评分能否识别共同错误，我们将整个 VHD 留作测试，用其余四个开发数据集训练评分器。在 178 道倒放视频题上，LR 和 HGB 相对原始置信度的平均 AURC 增益分别为 −7.20% 和 −5.48%，95% 置信区间分别为 [−13.28, −2.64] 和 [−9.16, −1.27]。这些结果表明，从其他数据集学到的评分未能改善该子集的整体排序。该子集采用两个选项，模型平均准确率为 24.91%，随机选择的期望准确率为 50%。正确答案依据画面中实际呈现的过程确定；因此，这项测试衡量倒放条件下的视觉判断，不代表自然视频中的幻觉发生率。'
en_vhd='We held out all of VHD and trained the reliability scores on the other four development datasets. On its 178 questions about reversed videos, mean relative AURC gains were −7.20% for LR and −5.48% for HGB, with 95% confidence intervals of [−13.28, −2.64] and [−9.16, −1.27]. The transferred scores therefore did not improve overall ranking on this subset. Mean answering accuracy was 24.91%, compared with 50% expected under random choice between the two options. Correctness follows the sequence shown in the video. These results assess visual judgments under reversal and do not estimate the prevalence of hallucination in natural videos.'
zh_m7='我们进一步固定三个开发模型及其权重，逐一留出完整数据集进行测试。HGB 的平均相对 AURC 增益在 NExT GQA、STAR、MVBench 和 TempCompass 上分别为 26.71%、11.32%、1.49% 和 10.61%，在 VHD 上为 −2.14%。这说明，模型尺寸不变时，迁移仍可能失败。删除三个模型身份指示量后，HGB 预测保持不变，但其余特征仍含模型身份信息。最终目标实验同时改变了数据集、部分模型尺寸和视图配置，现有结果不能分别确定这些变化的影响。'
en_m7='We also held out each development dataset while keeping all three answering models and their checkpoints fixed. Mean relative AURC gains for HGB were 26.71% on NExT GQA, 11.32% on STAR, 1.49% on MVBench, 10.61% on TempCompass, and −2.14% on VHD. Transfer can therefore fail without a change in model size. Removing the three model identity indicators left HGB predictions unchanged, although other features still retain identity information. The final target experiments change the dataset, some model sizes, and the view configuration together, so they do not isolate the effect of each change.'
en_limit='The VHD holdout exposes a limitation of scores transferred from other datasets: they did not improve ranking on the reversed video subset. This analysis does not evaluate whether fitting and checking a score with VHD pilot labels would help.'
en_methods='These supplementary analyses reuse archived option probabilities and require no new VLM inference. Each test dataset is excluded in full when fitting its development scores. We retain the existing feature builder and fitting settings, and keep source videos together in the internal HGB validation split where it is active. VHD supplies an eight frame base view and a sixteen frame dense view. We use the mean of prefix risks for AURC, preserve input order to resolve ties, and obtain paired 95% percentile intervals from 10,000 source video bootstrap samples with the scores held fixed. Gains are calculated for each answering model and then averaged.'
draft=f'''# Proposed manuscript changes for review

Status: draft only. Overleaf has not been edited.

## RQ2: add after the VHD error analysis

{zh_vhd}

{en_vhd}

Place the following compact table in the appendix. The full report gives individual models and every subset.

{vtable}

## Transfer discussion: add the fixed checkpoint control

{zh_m7}

{en_m7}

{lodo}

## Replace the unresolved VHD sentence in limitations

原句：Finally, VHD measures errors shared by the answering models; whether learned scores can identify these errors remains open.

建议：VHD 留出实验显示，从其他数据集迁移来的评分未能改善倒放子集的排序。使用 VHD 的 pilot 标签拟合并检查评分是否有效，仍需单独验证。

{en_limit}

## Appendix methods

{en_methods}

## Placement and interpretation

Retain the existing RQ2 subsection. Add the VHD test there and put its detailed table and the fixed checkpoint results in the appendix. Replace only the unresolved VHD sentence and clarify the transfer attribution. Do not claim that the target check solves the VHD failure, that model size has no effect, or that deleting three redundant ID columns removes all model identity information.

All numeric draft claims map to model_metrics.csv, mean_metrics.csv, mean_metrics_grouped_validation.csv, gain_intervals_grouped_validation.csv, vhd_question_subsets.csv and identity_audit.json in this directory. Human approval is pending.
'''
(p/'DRAFT_FOR_REVIEW.md').write_text(draft,encoding='utf-8')

report=f'''# Existing record analysis: VHD reliability and fixed checkpoint transfer

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

{md(['Dataset','Questions','Source videos'],[[r.dataset,r.questions,r.videos] for r in pd.read_csv(p/'dataset_inventory.csv').itertuples()])}

The same checkpoints occur in all five datasets:

{md(['Model','Revision'],pd.read_csv(p/'view_inventory.csv')[['model_id','revision']].drop_duplicates().values.tolist())}

There are 40,866 archived response records, 4,011 questions and 12,033 answer rows. The unchanged graph builder creates 53 features. VHD has base and dense views with 8 and 16 frames, TempCompass has 6 and 12 frames, STAR and MVBench have 8/8/16/8 frames, and NExT GQA uses four 16 frame views. Thus the fixed checkpoint control still includes differences in tasks and view definitions. Missing view groups retain the original availability indicators and zero comparisons.

## Protocol

Entire datasets are held out. The VHD model is fitted on NExT GQA, STAR, MVBench and TempCompass, excluding all VHD labels and every VHD variant. This is a test of transferred development scores, not target pilot LR, promotion checks or the complete selected pipeline.

The existing functions graph_table, fit_graph_logistic, equal_domain_weights and aurc are imported from repository source. Development LR uses C=0.2 and standardization. HGB uses learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=30 and l2_regularization=2.0. Both retain seed 20260801 and equal total weight per source dataset. No hyperparameter search or target tuning was performed. The development LR is distinct from the target pilot LR with C=0.1.

AURC is the paper's prefix risk mean with the archived input order tie rule. Risk at 50% accepts ceil(N/2) highest scores. Positive relative gain means lower AURC. The main gain is the arithmetic mean of three model specific relative gains; it is not the relative change in the average AURC. AUROC uses correctness as the positive class and is computed over both correct and incorrect answers in each subset.

Bootstrap intervals use 10,000 paired source video resamples, seed 20261119, shared across methods and answering models. Every question and model response for a sampled video is included together. Scorers are held fixed; these intervals measure evaluation sample variation, not variation in scorer training or across independently collected datasets. Bootstrap index files and distributions are saved. They are exploratory intervals without a multiple comparison correction.

## VHD reversed subset

178 questions from 178 source clips. All are labeled reversed in the archived metadata. The stored dataset builders use ffmpeg's reverse filter for this manipulation. The models have mean answering accuracy 24.91%; uniform random choice between two options has expected accuracy 50%. Scores reorder fixed answers and cannot change this accuracy.

{vtable}

99 of the 178 questions are answered incorrectly by all three models. Shared error deferral is the fraction of these 99 questions placed in the rejected half, averaged over the three answering models. The selection budget is applied within this subset. These rates are not precision, accuracy, or an AUROC calculated on only wrong answers. Under random half selection its expected value is 50%.

HGB improves risk at exactly 50% slightly, despite worse overall AURC. The overall negative result is driven by InternVL; Qwen and LLaVA have positive AURC gains. Neither fact is suppressed.

{permodel}

The same VHD holdout model is used for all subsets; no subset tuning is performed:

{subsets}

Correctness must follow the sequence actually displayed. The reversal experiment is a controlled visual grounding test, not an estimate of hallucination prevalence in unedited videos. Metadata and construction code establish the manipulation; this CPU analysis does not revalidate every video annotation.

## Fixed checkpoint transfer control

The following HGB results use source video grouped internal validation wherever the archived default activates it. VHD and MVBench have no internal validation split under that default and are unchanged.

{lodo}

This demonstrates performance variation and VHD transfer failure with identical checkpoints. Dataset content, label distribution, task mix and view budgets remain coupled. It does not prove that model scaling is irrelevant to HERBench or Perception Test.

## Identity indicator sensitivity

Relative gains in percent:

{idtable}

HGB scores are exactly unchanged in all five holdouts. Each model_identity_i column exactly duplicates base_peer_i_is_self in this builder; the named peer slots also retain family information. Therefore removing only the three model_identity columns is a redundant column ablation, not an identity free experiment. It cannot dismiss the reviewer's identity concern. LR changes slightly because duplicated standardized columns interact with regularization. No model size causal inference follows from this ablation.

## Internal validation audit and deviations

The frozen HGB constructor leaves early_stopping='auto'. For NExT GQA, STAR and TempCompass holdouts, the source table exceeds 10,000 rows, so sklearn's default makes a row based internal validation split. That can separate answers belonging to the same source video. The entire test dataset remains excluded, so no held out labels are leaked, but the default does not satisfy the requested source video grouping rule.

We first saved the exact constructor reconstruction for audit, then made one fixed CPU control with GroupShuffleSplit(test_size=0.1, random_state=20260801) over dataset plus source clip. All model rows and variants stay together. All fitting and regularization settings are retained. early_stopping is explicitly set to True in these three controls to preserve its original active mode after passing the validation rows separately; there is no threshold tuning. Sample weights are computed on the original source pool and subset into the two partitions. The primary supplemental table uses this grouped control. No archived evaluation or selection is replaced.

{earlytable}

Each grouped model used all 200 boosting iterations. Internal partition files and audit JSON are saved. The VHD result is unaffected by this issue. The first sandboxed fit could not create a Windows thread communication pipe; the authorized local CPU fit was rerun with the necessary process permission. One preliminary grouped attempt left auto early stopping off after reducing the explicit training rows; the assertion stopped it before any outputs were used. It was corrected to preserve the original active mode.

## Other checks and limitations

The separate surviving VHD mechanism file agrees with this response bank on {alignment['all_answer_matches']}/{alignment['all_response_count']} answers overall and {alignment['reversed_answer_matches']}/{alignment['reversed_response_count']} reversed subset answers. It has {alignment['old_shared_wrong']} shared wrong reversed questions, versus {alignment['new_shared_wrong']} here, with {alignment['shared_wrong_intersection']} in common. Its task labels are used, not its answers. Mixing its answers with these learned scores would be invalid. alignment_summary.json and vhd_primary_alignment.csv retain the comparison. The reconstructed current bank gives 99/178 shared errors and 99/116 errors among unanimous answers, matching the manuscript's rounded 55.6% and 85.3%; that numerical match alone does not certify the figure's full provenance.

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
'''
(p/'REPORT.md').write_text(report,encoding='utf-8')

latex=r'''% Proposed appendix tables. Review before inserting. Mean over three models.
\begin{table}[t]
\centering
\caption{Transferred development scores on 178 VHD questions about reversed videos. Lower AURC and risk are better. Answers are fixed.}
\label{tab:vhd-transferred-scores}
\begin{tabular}{lrrr}
\toprule
Score & AURC & AUROC & Risk at 50\% \\
\midrule
'''
for m in a.METHODS[:3]:
    r=vr.loc[m];latex+=f'{names[m]} & {r.aurc:.4f} & {r.auroc:.3f} & {100*r.risk50:.2f}\\% '+r'\\'+'\n'
latex+=r'''\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Mean relative AURC gain when each development dataset is held out. The three answering model checkpoints remain fixed. Active internal HGB validation is grouped by source video.}
\label{tab:fixed-checkpoint-transfer}
\begin{tabular}{lrr}
\toprule
Dataset & Development LR & Development HGB \\
\midrule
'''
for d,g in grouped.groupby('dataset'):
    g=g.set_index('method');latex+=f'{d} & {g.loc["graph_lr","relative_gain_pct"]:.2f}\\% & {g.loc["graph_hgb","relative_gain_pct"]:.2f}\\% '+r'\\'+'\n'
latex+=r'''\bottomrule
\end{tabular}
\end{table}
'''
(p/'tables.tex').write_text(latex,encoding='utf-8')
print('Wrote REPORT.md, DRAFT_FOR_REVIEW.md, tables.tex')
