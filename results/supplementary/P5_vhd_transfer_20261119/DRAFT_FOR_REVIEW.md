# Proposed manuscript changes for review

Status: draft only. Overleaf has not been edited.

## RQ2: add after the VHD error analysis

为检验评分能否识别共同错误，我们将整个 VHD 留作测试，用其余四个开发数据集训练评分器。在 178 道倒放视频题上，LR 和 HGB 相对原始置信度的平均 AURC 增益分别为 −7.20% 和 −5.48%，95% 置信区间分别为 [−13.28, −2.64] 和 [−9.16, −1.27]。这些结果表明，从其他数据集学到的评分未能改善该子集的整体排序。该子集采用两个选项，模型平均准确率为 24.91%，随机选择的期望准确率为 50%。正确答案依据画面中实际呈现的过程确定；因此，这项测试衡量倒放条件下的视觉判断，不代表自然视频中的幻觉发生率。

We held out all of VHD and trained the reliability scores on the other four development datasets. On its 178 questions about reversed videos, mean relative AURC gains were −7.20% for LR and −5.48% for HGB, with 95% confidence intervals of [−13.28, −2.64] and [−9.16, −1.27]. The transferred scores therefore did not improve overall ranking on this subset. Mean answering accuracy was 24.91%, compared with 50% expected under random choice between the two options. Correctness follows the sequence shown in the video. These results assess visual judgments under reversal and do not estimate the prevalence of hallucination in natural videos.

Place the following compact table in the appendix. The full report gives individual models and every subset.

| Score | AURC | Mean relative gain and 95% CI | AUROC | Risk at 50% | Shared errors deferred at 50% |
| --- | --- | --- | --- | --- | --- |
| Original confidence | 0.7923 | reference | 0.384 | 80.90% | 39.73% |
| Development LR | 0.8231 | -7.20% [-13.28, -2.64] | 0.323 | 83.90% | 24.58% |
| Development HGB | 0.8098 | -5.48% [-9.16, -1.27] | 0.437 | 79.40% | 31.99% |

## Transfer discussion: add the fixed checkpoint control

我们进一步固定三个开发模型及其权重，逐一留出完整数据集进行测试。HGB 的平均相对 AURC 增益在 NExT GQA、STAR、MVBench 和 TempCompass 上分别为 26.71%、11.32%、1.49% 和 10.61%，在 VHD 上为 −2.14%。这说明，模型尺寸不变时，迁移仍可能失败。删除三个模型身份指示量后，HGB 预测保持不变，但其余特征仍含模型身份信息。最终目标实验同时改变了数据集、部分模型尺寸和视图配置，现有结果不能分别确定这些变化的影响。

We also held out each development dataset while keeping all three answering models and their checkpoints fixed. Mean relative AURC gains for HGB were 26.71% on NExT GQA, 11.32% on STAR, 1.49% on MVBench, 10.61% on TempCompass, and −2.14% on VHD. Transfer can therefore fail without a change in model size. Removing the three model identity indicators left HGB predictions unchanged, although other features still retain identity information. The final target experiments change the dataset, some model sizes, and the view configuration together, so they do not isolate the effect of each change.

| Held out dataset | Development LR gain | Development HGB gain |
| --- | --- | --- |
| MVBench | 13.76% [10.60, 16.85] | 1.49% [1.36, 12.24] |
| NExT-GQA | 26.83% [20.03, 34.04] | 26.71% [11.88, 32.24] |
| STAR | 9.01% [2.98, 15.41] | 11.32% [2.23, 20.71] |
| TempCompass | 12.24% [6.24, 18.28] | 10.61% [1.71, 17.64] |
| VHD | -3.77% [-7.46, -0.37] | -2.14% [-14.52, -0.09] |

## Replace the unresolved VHD sentence in limitations

原句：Finally, VHD measures errors shared by the answering models; whether learned scores can identify these errors remains open.

建议：VHD 留出实验显示，从其他数据集迁移来的评分未能改善倒放子集的排序。使用 VHD 的 pilot 标签拟合并检查评分是否有效，仍需单独验证。

The VHD holdout exposes a limitation of scores transferred from other datasets: they did not improve ranking on the reversed video subset. This analysis does not evaluate whether fitting and checking a score with VHD pilot labels would help.

## Appendix methods

These supplementary analyses reuse archived option probabilities and require no new VLM inference. Each test dataset is excluded in full when fitting its development scores. We retain the existing feature builder and fitting settings, and keep source videos together in the internal HGB validation split where it is active. VHD supplies an eight frame base view and a sixteen frame dense view. We use the mean of prefix risks for AURC, preserve input order to resolve ties, and obtain paired 95% percentile intervals from 10,000 source video bootstrap samples with the scores held fixed. Gains are calculated for each answering model and then averaged.

## Placement and interpretation

Retain the existing RQ2 subsection. Add the VHD test there and put its detailed table and the fixed checkpoint results in the appendix. Replace only the unresolved VHD sentence and clarify the transfer attribution. Do not claim that the target check solves the VHD failure, that model size has no effect, or that deleting three redundant ID columns removes all model identity information.

All numeric draft claims map to model_metrics.csv, mean_metrics.csv, mean_metrics_grouped_validation.csv, gain_intervals_grouped_validation.csv, vhd_question_subsets.csv and identity_audit.json in this directory. Human approval is pending.
