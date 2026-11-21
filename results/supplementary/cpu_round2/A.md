# A：Perception Test 多数投票与 30 次划分

仅用已有响应，未运行 GPU 推理，未修改归档结果。所有 AURC 使用前缀风险均值，按置信度降序排序，同分保留输入顺序。相对下降为 100 ×（原始 AURC 减去新 AURC）/ 原始 AURC；负值表示变差。

## 输入与范围

InternVL3.5 14B Instruct：[responses.jsonl](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P3B_view_crossed_20261119/merged_responses_internvl14/responses.jsonl>)，7,552 行；对应 COMPLETE.json 的 SHA256 已匹配。
Qwen3-VL 8B Instruct：[responses.jsonl](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P3B_view_crossed_20261119/merged_responses_qwen8/responses.jsonl>)，7,552 行；对应 COMPLETE.json 的 SHA256 已匹配。
LLaVA-OneVision-2 8B Instruct：[responses.jsonl](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P3B_view_crossed_20261119/merged_responses_llava8/responses.jsonl>)，7,552 行；对应 COMPLETE.json 的 SHA256 已匹配。

共 1,888 题、500 个视频，每个模型都有 base 16、shifted 8、dense 24、reversed 8 帧四个视角和逐题三个选项概率。模型版本见 [PREFLIGHT.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/PREFLIGHT.json>)。划分来自 [video_assignments.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P2/video_assignments.csv>)；使用其 Perception 的 split 1 至 30，每次 250 个 pilot 视频、250 个测试视频。split 0 只作原始测试划分参照，不混入 30 次平均。未重新抽取划分，也未筛选结果。

特征使用原有 53 维构建器；以每个作答模型自己的 base 概率最大选项为固定答案。Qwen 的特征与归档 PhaseGuard 对比矩阵逐项匹配。来源：[evaluate_response_graph_features.py](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/src/evaluate_response_graph_features.py:127>)、[original_views_features.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P3B_view_crossed_20261119/original_views_features.csv>)。

## 多数投票准确率

三个模型在 base 上投票。票数相同时，在并列选项中取三个模型平均概率最大的选项；若仍完全相同，取选项序列中的首项。每个视频整组有放回抽样 10,000 次，模型和投票共享抽样，报告百分位 95% CI。准确率按题计算，而非先平均每个视频。

| 方法 | 原始测试划分：911 题，250 视频，准确率 % [95% CI] | 完整响应面板：1888 题，500 视频，准确率 % [95% CI] |
| --- | --- | --- |
| InternVL3.5 14B Instruct | 72.67 [69.29, 75.86] | 73.68 [71.50, 75.81] |
| Qwen3-VL 8B Instruct | 70.47 [67.10, 73.71] | 71.08 [68.80, 73.29] |
| LLaVA-OneVision-2 8B Instruct | 63.01 [59.53, 66.37] | 63.77 [61.37, 66.06] |
| 多数投票 | 70.80 [67.34, 74.16] | 72.25 [69.89, 74.54] |

完整面板包含 pilot，仅作描述；论文测试对比应使用左列。

| 原始测试划分的配对差异 | 准确率差 pp | 95% CI |
| --- | --- | --- |
| 多数投票减去 InternVL3.5 14B Instruct | -1.87 | [-4.40, 0.73] |
| 多数投票减去 Qwen3-VL 8B Instruct | 0.33 | [-1.50, 2.20] |
| 多数投票减去 LLaVA-OneVision-2 8B Instruct | 7.79 | [5.15, 10.51] |

多数投票显著优于 LLaVA；相对 InternVL 和 Qwen 的配对区间包含 0，不能声称优于全部单模型。

每个随机划分的测试准确率与 CI 均保存在 [A_accuracy_intervals.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_accuracy_intervals.csv>)；对应配对差异在 [A_vote_paired_intervals.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_vote_paired_intervals.csv>)，抽样索引保存在 A_accuracy_bootstrap_*.npz。种子为 20261120 + split，完整面板 split 为负一。

## 检查流程在两种作答模型上的结果

复用原有 pilot LR、冻结的开发集 HGB 和选择逻辑。pilot LR 按视频做五折 OOF；标准化、缺失值处理、C=0.1、balanced 权重、liblinear、max_iter=3000 均不变。HGB 直接读取 [pilot_model_bundle.joblib](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/perception_confirmation/8919110.kman.restech.unsw.edu.au/pilot_model_bundle.joblib>)，没有再训练或调参。

两个候选各做 2,000 次 pilot 视频 bootstrap，取 2.5% 分位数作为下界。pilot LR 的下界须严格大于 0，开发集 HGB 须严格大于 0.5 pp；多个候选通过时取下界最大的候选，否则退回原始置信度。0.5 pp 指相对 AURC 下降百分比的百分点。仅在本次进程内将原选择器的 AURC 替换为前缀均值函数，原文件不变。来源：[evaluate_v2_target_pilot.py](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/src/evaluate_v2_target_pilot.py:197>)。

| 作答模型 | 平均相对下降 % | 最小相对下降 % | 变差划分 | 退回原始置信度 | 选择 pilot LR 次数 | 选择开发集 HGB 次数 |
| --- | --- | --- | --- | --- | --- | --- |
| InternVL3.5 14B Instruct | 0.26 | 0.00 | 0/30 | 27/30 | 3 | 0 |
| LLaVA-OneVision-2 8B Instruct | 26.61 | 22.24 | 0/30 | 0/30 | 17 | 13 |

InternVL 的小平均增益伴随 27 次退回；这说明检查较保守，不能把没有变差解释成优于未检查的候选。未检查候选的同批结果如下。

| 作答模型 | 未检查候选 | 平均下降 % | 最小下降 % | 变差次数 |
| --- | --- | --- | --- | --- |
| InternVL3.5 14B Instruct | pilot_lr | 5.75 | 1.04 | 0 |
| InternVL3.5 14B Instruct | development_hgb | 2.71 | -2.52 | 3 |
| LLaVA-OneVision-2 8B Instruct | pilot_lr | 28.14 | 22.24 | 0 |
| LLaVA-OneVision-2 8B Instruct | development_hgb | 25.99 | 22.37 | 0 |

逐划分结果：[A_selection_splits.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_selection_splits.csv>)。各次 pilot OOF 分数、选择下界、测试分数和拟合 LR 位于 [A_runs](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_runs>)。抽样种子、候选偏移和实现保存在 [ANALYSIS_PLAN.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/ANALYSIS_PLAN.json>)、[run_A.py](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/run_A.py>)。

## 无标签 support

Base support 是三个 base 响应对该作答模型固定答案的平均概率；all-view support 是全部十二个响应对同一答案的平均概率。均含作答模型自身，不使用 pilot 标签，也不改变答案。

| 作答模型 | All-view support 平均下降 % | Base support 平均下降 % |
| --- | --- | --- |
| InternVL3.5 14B Instruct | -0.43 | -0.75 |
| Qwen3-VL 8B Instruct | 9.96 | 11.51 |
| LLaVA-OneVision-2 8B Instruct | 26.00 | 26.84 |
| 三个模型等权平均 | 11.84 | 12.53 |

最后一行先对每个划分的三个模型等权平均，再对 30 次划分平均，不能用合并所有预测后的 AURC 替代。逐划分结果：[A_support_splits.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_support_splits.csv>)。

## 答案字段差异

本报告主分析沿用论文构建器的选项概率 argmax 定义。响应中的生成字母 answer 与 argmax 并非处处相同：

| 模型 | base 不一致题数 | 四视角不一致记录数 |
| --- | --- | --- |
| InternVL3.5 14B Instruct | 2 | 9 |
| Qwen3-VL 8B Instruct | 0 | 0 |
| LLaVA-OneVision-2 8B Instruct | 36 | 139 |

另按生成字母投票，复用相同视频抽样，原始测试划分得到：

| 方法 | 生成字母准确率 % [95% CI] |
| --- | --- |
| InternVL3.5 14B Instruct | 72.67 [69.29, 75.86] |
| Qwen3-VL 8B Instruct | 70.47 [67.10, 73.71] |
| LLaVA-OneVision-2 8B Instruct | 63.45 [60.05, 66.81] |
| 多数投票 | 71.02 [67.54, 74.38] |

这项敏感性分析不替换主分析的固定答案。来源：[A_generated_answer_sensitivity.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_generated_answer_sensitivity.csv>)。

30 次划分重复使用同一批视频，不是 30 个独立数据集；这里的最小值与变差次数只描述这 30 次划分，不构成对新目标的保证。检查组件也未重新设计。

可转入论文的表格见 [A_tables.tex](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_tables.tex>)。数值验证见 [VALIDATION.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/VALIDATION.json>)，输入哈希见 [INPUT_MANIFEST.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/INPUT_MANIFEST.json>)。
