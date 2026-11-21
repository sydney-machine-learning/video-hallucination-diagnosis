# D：Perception Test 准确率与训练数据声明核对

核对日期：2026 年 11 月 20 日。仅阅读已有结果及官方论文、模型卡、仓库配置和上游数据说明，没有运行模型。下载的是文档及文件名清单，未下载视频、权重或训练样本。

## 准确率与技术报告

| 模型 | 原论文归档准确率 % | PhaseGuard 对比响应的 base 准确率 % | 同型号技术报告的 Perception Test 数值 |
| --- | --- | --- | --- |
| OpenGVLab/InternVL3_5-14B-Instruct | 72.78 | 72.67 | 未找到 |
| Qwen/Qwen3-VL-8B-Instruct | 67.29 | 70.47 | 未找到 |
| lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct | 61.58 | 63.01 | 未找到 |

两个本地列都对应原始测试划分的 911 题、250 个视频；它们是公开 validation 面板中的测试部分，并非官方隐藏 test 集。第一列来源：[test_model_metrics.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/perception_confirmation/8919110.kman.restech.unsw.edu.au/test_model_metrics.csv>) 的 raw_confidence 三行；第二列来源：[A_accuracy_intervals.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/A_accuracy_intervals.csv>) 的 split 0。两批推理响应的准确率不同，应保留批次标识，不能混用。新响应按选项概率 argmax 作答，生成字母敏感性结果另见 A。保存表：[D_accuracy_comparison.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/D_accuracy_comparison.csv>)。

检查的主要官方来源：

1. [Qwen3-VL Technical Report，2511.21631v1](https://arxiv.org/html/2511.21631v1)，视频评测及附录；[8B Instruct 实验版本模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/README.md)。
2. [InternVL3.5 技术报告，2508.18265v2](https://arxiv.org/html/2508.18265v2)，第 3.10 节视频评测；[14B Instruct 实验版本模型卡](https://huggingface.co/OpenGVLab/InternVL3_5-14B-Instruct/blob/72c82460a2c05a3bc2b47be230c44edc88e86ed4/README.md)。
3. [LLaVA-OneVision-2 技术报告，2605.25979v1](https://arxiv.org/html/2605.25979v1)，视频评测表及附录；[8B Instruct 实验版本模型卡](https://huggingface.co/lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct/blob/802f16c8346a062fef7ddf92b2bd64a770a50a1a/README.md)。

这三份技术报告及对应模型卡中均未找到同型号的 Perception Test 准确率。因此没有可核验的官方数值可作数值差异对比，不能填入旧型号或另一评测的分数。

相关的 [OneVision-Encoder 报告，2602.08683v1](https://arxiv.org/html/2602.08683v1) 确实出现 Perception Test，但其中语言模型配对评测采用 Qwen3-4B，另有视觉编码器的 attentive probing 实验。这些都不是本实验的 LLaVA-OneVision-2-8B-Instruct，未移植其分数到上表。

官方数据集对 validation 与 test 分开提供下载和评测入口，见 [Perception Test 官方仓库](https://github.com/google-deepmind/perception_test)。即使另有完整 validation 或隐藏 test 的官方模型成绩，也需匹配视频集合、抽帧、分辨率、提示词、作答与评分方式，才适合直接比较。

## 是否声明训练使用 Perception Test

| 模型 | 核对所得证据 | 能得出的结论 |
| --- | --- | --- |
| Qwen3-VL 8B Instruct | 报告第 3.2.7 节描述视频来源类别、标注和采样，没有逐一列出完整训练数据；模型卡未点名 Perception Test | 在核对范围内未找到包含或明确排除该数据集的声明；训练重叠未知 |
| InternVL3.5 14B Instruct | 报告第 2.2、2.3 节说明继承 InternVL3 语料；两份报告及模型卡没有点名 Perception Test；公开 14B 训练脚本指向 debug 数据配置 | 未获得足以确认训练包含或排除该数据集的完整清单 |
| LLaVA-OneVision-2 8B Instruct | 报告第 3.1 节声明采用 VideoChat-Flash-Training-Data 等语料的相关子集；该上游仓库确有 Perception Test 文件 | 有上游包含证据，但该检查点选用子集和具体视频未公开，不能确认使用了我们的 validation 视频 |

Qwen 的 [公开微调数据配置](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune/qwenvl/data/__init__.py) 包含示例路径占位符，不能当作发布检查点的完整训练清单。InternVL 的 [14B SFT 脚本](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat_gpt_oss/shell/internvl3_5_qwen3/internvl3_5_14b_sft.sh) 第 48 行引用 debug_sft.json，[14B MPO 脚本](https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat_gpt_oss/shell/internvl3_5_qwen3/internvl3_5_14b_mpo.sh) 同样引用 debug_mpo.json；其 debug 示例不能证明完整训练语料。[InternVL3 报告](https://arxiv.org/html/2504.10479v1) 和 [InternVL-Data 卡](https://huggingface.co/datasets/OpenGVLab/InternVL-Data) 也未提供可核对该检查点的完整视频 ID 清单。

LLaVA 的 [公开 SFT 数据 YAML](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/blob/8fcc665d3ef113eff21df6572f6a2536f98ecd2d/aiak_megatron/examples/multimodal/sft_dataset.yaml) 使用占位路径；[旧 configs 目录说明](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/blob/8fcc665d3ef113eff21df6572f6a2536f98ecd2d/configs/README.md) 标为 deprecated，不将其视为完整发布配方。报告只说从上游语料选取相关部分，未声明全部使用。

上游 VideoChat-Flash-Training-Data 的官方文件名清单含以下三个条目：

[annotations/video/llava-video_0_30_s_perceptiontest_mc_qa_processed_1785_with_duration.jsonl](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/blob/be87f5516a709be079cec8b727dd2287bf2dd70f/annotations/video/llava-video_0_30_s_perceptiontest_mc_qa_processed_1785_with_duration.jsonl)

[annotations/video/llava-video_30_60_s_perceptiontest_mc_qa_processed_618_with_duration.jsonl](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/blob/be87f5516a709be079cec8b727dd2287bf2dd70f/annotations/video/llava-video_30_60_s_perceptiontest_mc_qa_processed_618_with_duration.jsonl)

[annotations/video/vqa_perception_train-mc_question_train_forchoice_8k.json](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/blob/be87f5516a709be079cec8b727dd2287bf2dd70f/annotations/video/vqa_perception_train-mc_question_train_forchoice_8k.json)

第三个文件名明确指向 train；前两个文件名不能单独证明官方划分。这里没有下载样本、没有进行视频重叠审计。上游存在这些文件，不等于该模型一定使用了全部文件，更不等于使用了本实验的 validation 标签。原始文件名清单与版本 SHA 已保存于 [videochat_training_inventory.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/D_sources/videochat_training_inventory.json>)，匹配条目见 [videochat_perception_paths.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/D_sources/videochat_perception_paths.json>)。

因此不能写“三个模型都没有见过 Perception Test”，也不能根据这些文档直接指控 validation 泄漏。当前证据只支持：公开资料不足以排除重叠；LLaVA 的一个已声明上游语料含相关数据，实际选用范围未知。

## 核对范围与留档

审阅三个模型的官方主报告、当前模型卡、与本地响应 model_revision 一致的模型卡、官方仓库 README、公开训练数据配置，并追查 InternVL3 和 VideoChat-Flash 上游说明。检索 Perception Test、PerceptionTest、perception_test、perception-test；普通名词 perception 不算命中。

网页及官方仓库快照位于 [D_sources](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/D_sources>)。来源 URL 和 SHA256 分别在 SOURCES.json、CONFIG_SOURCES.json、UPSTREAM_SOURCES.json；官方仓库 tree SHA 在 repo_inventory.json。已把“未查到”“上游包含”和“检查点实际使用”分开，不将文档缺失当作无污染证明。本轮未修改论文。
