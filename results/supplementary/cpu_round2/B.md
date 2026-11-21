# B：VHD 留出数据集评分诊断

复用 [predictions.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/predictions.csv>) 中 held-out VHD 的预测及 [VHD.joblib](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/models/VHD.joblib>)。这些文件来自 2026 年 11 月 19 日对归档响应的 CPU 重建。本次没有重新训练，没有 GPU 推理。

## 输入与统计口径

训练数据为 MVBench、NExT-GQA、STAR、TempCompass，共 9,630 个模型与问题组合；VHD 的全部变体均留出。VHD 有 801 题、341 个源视频、2,403 个模型与问题组合。训练不使用 VHD 标签。审计：[B_reuse_audit.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/B_reuse_audit.json>)、[holdout_audit.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/holdout_audit.json>)。

开发模型是 InternVL3.5 4B Instruct、Qwen3-VL 4B Instruct、LLaVA-OneVision-2 8B Instruct，区别于 A 的 14B/8B/8B。VHD 仅有 base 8 帧、dense 16 帧；缺失视角遵循原特征构建器处理。

LR 沿用 C=0.2、max_iter=3000、标准化和等数据集权重。HGB 沿用 learning_rate=0.05、max_iter=200、max_leaf_nodes=15、min_samples_leaf=30、l2_regularization=2.0 及原随机种子。实际预测直接复用，参数来源：[analyze.py](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/analyze.py:98>)、[evaluate_response_graph_features.py](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/src/evaluate_response_graph_features.py:245>)。

原始置信度为固定答案的 base 概率。Base vote 为三个 base 答案支持固定答案的比例，base support 为三个 base 响应给该答案的平均概率。两者均不改变答案，不等同于 A 的改答案多数投票。

AURC 一律前缀均值，同分保持输入顺序。Base vote 大量同分，另列原归档的置信度破同分版本，即 vote + 1e-9 × raw confidence，作为敏感性结果。AUROC 按标准半权处理相等分数。

复用 P5 保存的 10,000 次源视频抽样及种子 20261119。每次把同一源视频的全部题目、变体及三个模型一起抽取；各评分共享抽样，固定已拟合评分器。CI 为 2.5% 和 97.5% 分位数。每次先算各模型指标，再等权平均；增益先逐模型算相对下降，再取平均。这里的 CI 不包含训练评分器的随机性。

## 子集结果

| 子集 | 题数 | 源视频数 |
| --- | --- | --- |
| all | 801 | 341 |
| visible | 333 | 333 |
| counterintuitive | 178 | 178 |
| hidden | 290 | 290 |

visible 对应 original，counterintuitive 对应 reversed，hidden 对应 masked。这里 reversed 指素材变体，不是另一个推理视角。全集源视频数小于子集数之和，因为变体共享源视频。每题两个选项，随机作答准确率为 50%；AUROC 的无判别参照为 0.5，二者不是同一指标。

| 子集 | 评分 | AUROC [95% CI] | 相对 AURC 下降 % [95% CI] |
| --- | --- | --- | --- |
| 全集 | 原始置信度 | 0.619 [0.590, 0.646] | 0.00 [0.00, 0.00] |
| 全集 | Base vote | 0.558 [0.543, 0.573] | -19.11 [-38.75, -15.45] |
| 全集 | Base vote，原始置信度破同分 | 0.624 [0.599, 0.647] | -0.06 [-2.62, 1.83] |
| 全集 | Base support | 0.624 [0.594, 0.652] | -3.31 [-7.02, 0.11] |
| 全集 | 开发集 LR | 0.620 [0.591, 0.648] | -3.77 [-7.46, -0.37] |
| 全集 | 开发集 HGB | 0.633 [0.601, 0.662] | -2.14 [-14.52, -0.09] |
| visible | 原始置信度 | 0.636 [0.592, 0.680] | 0.00 [0.00, 0.00] |
| visible | Base vote | 0.574 [0.545, 0.606] | -23.21 [-65.16, -15.46] |
| visible | Base vote，原始置信度破同分 | 0.639 [0.600, 0.678] | -1.28 [-7.29, 2.15] |
| visible | Base support | 0.615 [0.567, 0.664] | -16.00 [-26.79, -7.11] |
| visible | 开发集 LR | 0.603 [0.554, 0.651] | -17.59 [-28.26, -8.71] |
| visible | 开发集 HGB | 0.603 [0.556, 0.652] | -19.17 [-45.46, -10.02] |
| counterintuitive | 原始置信度 | 0.384 [0.315, 0.453] | 0.00 [0.00, 0.00] |
| counterintuitive | Base vote | 0.327 [0.262, 0.385] | -11.84 [-17.96, -1.48] |
| counterintuitive | Base vote，原始置信度破同分 | 0.311 [0.236, 0.382] | -6.36 [-10.52, -3.03] |
| counterintuitive | Base support | 0.317 [0.236, 0.396] | -7.83 [-14.04, -3.16] |
| counterintuitive | 开发集 LR | 0.323 [0.245, 0.397] | -7.20 [-13.28, -2.64] |
| counterintuitive | 开发集 HGB | 0.437 [0.364, 0.511] | -5.48 [-9.16, -1.27] |
| hidden | 原始置信度 | 0.642 [0.592, 0.693] | 0.00 [0.00, 0.00] |
| hidden | Base vote | 0.658 [0.616, 0.706] | 1.71 [-50.25, 15.74] |
| hidden | Base vote，原始置信度破同分 | 0.702 [0.653, 0.753] | 10.88 [4.85, 18.00] |
| hidden | Base support | 0.705 [0.647, 0.764] | 8.37 [-4.78, 19.99] |
| hidden | 开发集 LR | 0.715 [0.656, 0.773] | 10.78 [-2.44, 22.42] |
| hidden | 开发集 HGB | 0.706 [0.645, 0.768] | 12.48 [-10.19, 22.37] |

主表为三个开发模型的等权平均。每个模型、每个子集的完整 AUROC、AURC、相对下降和 CI 均在 [B_metrics.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/B_metrics.csv>)，可直接转为分模型表格；全部版本的 LaTeX 表格在 [B_tables.tex](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/B_tables.tex>)。

## 三个模型一致答错的反常识题

同一响应面板中，counterintuitive 的 178 题里有 99 题三个模型都答错。因每题只有两个选项，这些题也必然三个模型答案一致。此计数以本报告响应为准，不混用其他推理版本的共同错误计数。

每个模型分别在 178 个 counterintuitive 题内计算分数的升序百分位：100 × 平均秩 / 178。百分位越高表示越被评分器信任。下表统计 99 题的百分位，再对三个模型等权平均；中位数取合并的 297 个百分位。均值 CI 和与原始置信度的配对差异 CI 都按源视频抽样，并在每次样本内重算百分位。

| 评分 | 平均百分位 [95% CI] | 中位百分位 | 相对原始置信度的百分位差 [95% CI] |
| --- | --- | --- | --- |
| 原始置信度 | 56.37 [53.54, 59.39] | 58.71 | 0.00 [0.00, 0.00] |
| Base vote | 67.70 [64.33, 71.07] | 67.70 | 11.32 [8.07, 14.90] |
| Base vote，原始置信度破同分 | 67.48 [63.94, 71.08] | 66.85 | 11.11 [8.15, 14.35] |
| Base support | 65.60 [62.05, 69.27] | 69.10 | 9.22 [6.95, 11.66] |
| 开发集 LR | 66.87 [63.32, 70.54] | 67.98 | 10.50 [7.83, 13.42] |
| 开发集 HGB | 61.96 [58.53, 65.68] | 68.54 | 5.59 [3.54, 7.75] |

这些共同错误在 LR 和 HGB 下的平均百分位都高于原始置信度，即仍被赋予较高可信度。本分析不支持“学习评分已能识别反常识共同错误”。HGB 的平均 AUROC 高于原始置信度，但 AURC 仍更差；两个指标关注的排序位置不同，不能互相替代。

这检验的是开发集评分直接迁移到完整留出的 VHD，未使用目标 pilot 或替换检查，不能等同于 A 的完整目标检查流程。

逐题百分位另含以全部 801 个 VHD 问题为参照的版本，见 [B_joint_error_percentiles.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/B_joint_error_percentiles.csv>)。分模型百分位及 CI：[B_joint_error_percentile_summary.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/B_joint_error_percentile_summary.csv>)。评分和标签：[B_scores.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/B_scores.csv>)。抽样分布：B_bootstrap_*.npz。

LR/HGB 的八组子集点估计及 CI 与 P5 保存结果一致；独立 AUROC 与前缀 AURC 校验通过，见 [VALIDATION.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/VALIDATION.json>)。
