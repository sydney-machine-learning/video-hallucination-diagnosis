# C：开发阶段视角与帧数核对

仅读取配置、推理脚本、归档响应日志和论文副本，未执行这些实验脚本，未改动原文件。逐条记录核对结果保存于 [C_raw_view_audit.csv](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/cpu_round2/C_raw_view_audit.csv>)。本报告只核对每次推理的输入帧数，不把图像裁块数或视觉 token 数当作帧数。

| 数据集 | 实际视角与帧数 | 配置或原始记录证据 | 与论文附录核对 |
| --- | --- | --- | --- |
| MVBench | base 8；shifted 8；dense 16；reversed 8 | [mvbench_multimodel_runner.py](<D:/paper 3/caver_route/katana/mvbench_multimodel_runner.py:47>)，第 47 至 64 行 | 附录第 558 行未列出 MVBench，建议补充 |
| NExT-GQA | base 16；shifted 16；evidence intervals 16；complement 16 | [build_nextgqa_pilot_manifest.py](<D:/paper 3/sicr_route/src/build_nextgqa_pilot_manifest.py:23>)，第 23 至 44 行；[build_nextgqa_pilot_manifest.py](<D:/paper 3/sicr_route/src/build_nextgqa_pilot_manifest.py:232>) | 一致；后两个视角不是 dense/reversed |
| STAR | base 8；shifted 8；dense 16；reversed 8 | [build_star_pilot_manifest.py](<D:/paper 3/sicr_route/src/build_star_pilot_manifest.py:23>)，第 23 至 52 行 | 一致 |
| VHD | base 8；evidence_addition/dense 16 | [vhd_qwen_predictions.jsonl](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_qwen_predictions.jsonl:1>)，第 1、2 行 | 一致；仅两个视角 |
| TempCompass | base 6；evidence_addition/dense 12 | [tempcompass_qwen_predictions.jsonl](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_qwen_predictions.jsonl:1>)，第 1、2 行 | 一致；仅两个视角 |

论文依据：[verified.tex](<C:/Users/87385/Desktop/paper 3/overleaf_review_points_234_20261119/verified.tex:558>)。该行已正确描述 NExT-GQA、STAR、VHD、TempCompass，缺少 MVBench。可补一句：

> MVBench uses eight frames for the base, shifted, and reversed views, and sixteen frames for the dense view.

## 原始响应交叉核对

下表列出三个模型各自的来源文件与首个对应记录行号。逐行读取的全部记录在每个视角上的 frame_budget 均一致，不只检查示例行。NExT-GQA 和 STAR 每模型 500 题 × 4 视角，MVBench 每模型 1,800 题 × 4 视角，VHD 每模型 801 题 × 2 视角，TempCompass 每模型 410 题 × 2 视角。

| 数据集 | 模型 | 视角=帧数 | 总行数 | 原始文件及行号，顺序与视角列一致 |
| --- | --- | --- | --- | --- |
| NExT-GQA | Qwen3-VL-4B-Instruct | <bound method Series.view of dataset                                                  NExT-GQA
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                                         base
frame_budget                                                   16
generator                            uniform_full_video_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\qw...
first_line                                                      1
Name: 0, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                           invariant_nuisance
frame_budget                                                   16
generator                           uniform_phase_shift_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\qw...
first_line                                                      2
Name: 1, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                temporal_evidence_sufficiency
frame_budget                                                   16
generator                    nextgqa_evidence_intervals_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\qw...
first_line                                                      3
Name: 2, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                             evidence_removal
frame_budget                                                   16
generator       complement_of_nextgqa_evidence_intervals_16_fr...
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\qw...
first_line                                                      4
Name: 3, dtype: object>=16 | 2000 | [第 1 行](<D:/paper 3/sicr_route/results/katana_phase2/qwen3_vl_4b_predictions.jsonl:1>); [第 2 行](<D:/paper 3/sicr_route/results/katana_phase2/qwen3_vl_4b_predictions.jsonl:2>); [第 3 行](<D:/paper 3/sicr_route/results/katana_phase2/qwen3_vl_4b_predictions.jsonl:3>); [第 4 行](<D:/paper 3/sicr_route/results/katana_phase2/qwen3_vl_4b_predictions.jsonl:4>) |
| NExT-GQA | InternVL3_5-4B-Instruct | <bound method Series.view of dataset                                                  NExT-GQA
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                                         base
frame_budget                                                   16
generator                            uniform_full_video_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\in...
first_line                                                      1
Name: 4, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                           invariant_nuisance
frame_budget                                                   16
generator                           uniform_phase_shift_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\in...
first_line                                                      2
Name: 5, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                temporal_evidence_sufficiency
frame_budget                                                   16
generator                    nextgqa_evidence_intervals_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\in...
first_line                                                      3
Name: 6, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                             evidence_removal
frame_budget                                                   16
generator       complement_of_nextgqa_evidence_intervals_16_fr...
rows                                                          500
path            D:\paper 3\sicr_route\results\katana_phase2\in...
first_line                                                      4
Name: 7, dtype: object>=16 | 2000 | [第 1 行](<D:/paper 3/sicr_route/results/katana_phase2/internvl3_5_4b_predictions.jsonl:1>); [第 2 行](<D:/paper 3/sicr_route/results/katana_phase2/internvl3_5_4b_predictions.jsonl:2>); [第 3 行](<D:/paper 3/sicr_route/results/katana_phase2/internvl3_5_4b_predictions.jsonl:3>); [第 4 行](<D:/paper 3/sicr_route/results/katana_phase2/internvl3_5_4b_predictions.jsonl:4>) |
| NExT-GQA | LLaVA-OneVision-2-8B-Instruct | <bound method Series.view of dataset                                                  NExT-GQA
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                                         base
frame_budget                                                   16
generator                            uniform_full_video_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      1
Name: 8, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                           invariant_nuisance
frame_budget                                                   16
generator                           uniform_phase_shift_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      2
Name: 9, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                temporal_evidence_sufficiency
frame_budget                                                   16
generator                    nextgqa_evidence_intervals_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      3
Name: 10, dtype: object>=16; <bound method Series.view of dataset                                                  NExT-GQA
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                             evidence_removal
frame_budget                                                   16
generator       complement_of_nextgqa_evidence_intervals_16_fr...
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      4
Name: 11, dtype: object>=16 | 2000 | [第 1 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/nextgqa_llava/llava_ov2_8b_predictions.jsonl:1>); [第 2 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/nextgqa_llava/llava_ov2_8b_predictions.jsonl:2>); [第 3 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/nextgqa_llava/llava_ov2_8b_predictions.jsonl:3>); [第 4 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/nextgqa_llava/llava_ov2_8b_predictions.jsonl:4>) |
| STAR | Qwen3-VL-4B-Instruct | <bound method Series.view of dataset                                                      STAR
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                                         base
frame_budget                                                    8
generator                       uniform_official_segment_8_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      1
Name: 12, dtype: object>=8; <bound method Series.view of dataset                                                      STAR
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                           invariant_nuisance
frame_budget                                                    8
generator                   phase_shift_official_segment_8_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      2
Name: 13, dtype: object>=8; <bound method Series.view of dataset                                                      STAR
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                            dense_acquisition
frame_budget                                                   16
generator                      uniform_official_segment_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      3
Name: 14, dtype: object>=16; <bound method Series.view of dataset                                                      STAR
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                   temporal_order_destruction
frame_budget                                                    8
generator                    deterministic_reverse_of_base_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      4
Name: 15, dtype: object>=8 | 2000 | [第 1 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/qwen_predictions.jsonl:1>); [第 2 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/qwen_predictions.jsonl:2>); [第 3 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/qwen_predictions.jsonl:3>); [第 4 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/qwen_predictions.jsonl:4>) |
| STAR | InternVL3_5-4B-Instruct | <bound method Series.view of dataset                                                      STAR
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                                         base
frame_budget                                                    8
generator                       uniform_official_segment_8_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      1
Name: 16, dtype: object>=8; <bound method Series.view of dataset                                                      STAR
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                           invariant_nuisance
frame_budget                                                    8
generator                   phase_shift_official_segment_8_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      2
Name: 17, dtype: object>=8; <bound method Series.view of dataset                                                      STAR
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                            dense_acquisition
frame_budget                                                   16
generator                      uniform_official_segment_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      3
Name: 18, dtype: object>=16; <bound method Series.view of dataset                                                      STAR
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                   temporal_order_destruction
frame_budget                                                    8
generator                    deterministic_reverse_of_base_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      4
Name: 19, dtype: object>=8 | 2000 | [第 1 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/internvl_predictions.jsonl:1>); [第 2 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/internvl_predictions.jsonl:2>); [第 3 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/internvl_predictions.jsonl:3>); [第 4 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/internvl_predictions.jsonl:4>) |
| STAR | LLaVA-OneVision-2-8B-Instruct | <bound method Series.view of dataset                                                      STAR
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                                         base
frame_budget                                                    8
generator                       uniform_official_segment_8_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      1
Name: 20, dtype: object>=8; <bound method Series.view of dataset                                                      STAR
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                           invariant_nuisance
frame_budget                                                    8
generator                   phase_shift_official_segment_8_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      2
Name: 21, dtype: object>=8; <bound method Series.view of dataset                                                      STAR
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                            dense_acquisition
frame_budget                                                   16
generator                      uniform_official_segment_16_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      3
Name: 22, dtype: object>=16; <bound method Series.view of dataset                                                      STAR
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                   temporal_order_destruction
frame_budget                                                    8
generator                    deterministic_reverse_of_base_frames
rows                                                          500
path            D:\paper 3\sicr_route\results_fresh\download_2...
first_line                                                      4
Name: 23, dtype: object>=8 | 2000 | [第 1 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/llava_predictions.jsonl:1>); [第 2 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/llava_predictions.jsonl:2>); [第 3 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/llava_predictions.jsonl:3>); [第 4 行](<D:/paper 3/sicr_route/results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au/star_three_model/llava_predictions.jsonl:4>) |
| MVBench | Qwen3-VL-4B-Instruct | <bound method Series.view of dataset                                                   MVBench
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                                         base
frame_budget                                                    8
generator                         uniform_official_bound_8_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      1
Name: 24, dtype: object>=8; <bound method Series.view of dataset                                                   MVBench
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                           invariant_nuisance
frame_budget                                                    8
generator                     phase_shift_official_bound_8_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      2
Name: 25, dtype: object>=8; <bound method Series.view of dataset                                                   MVBench
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                            dense_acquisition
frame_budget                                                   16
generator                        uniform_official_bound_16_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      3
Name: 26, dtype: object>=16; <bound method Series.view of dataset                                                   MVBench
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                   temporal_order_destruction
frame_budget                                                    8
generator                    deterministic_reverse_of_base_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      4
Name: 27, dtype: object>=8 | 7200 | [第 1 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/qwen_predictions.jsonl:1>); [第 2 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/qwen_predictions.jsonl:2>); [第 3 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/qwen_predictions.jsonl:3>); [第 4 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/qwen_predictions.jsonl:4>) |
| MVBench | InternVL3_5-4B-Instruct | <bound method Series.view of dataset                                                   MVBench
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                                         base
frame_budget                                                    8
generator                         uniform_official_bound_8_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      1
Name: 28, dtype: object>=8; <bound method Series.view of dataset                                                   MVBench
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                           invariant_nuisance
frame_budget                                                    8
generator                     phase_shift_official_bound_8_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      2
Name: 29, dtype: object>=8; <bound method Series.view of dataset                                                   MVBench
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                            dense_acquisition
frame_budget                                                   16
generator                        uniform_official_bound_16_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      3
Name: 30, dtype: object>=16; <bound method Series.view of dataset                                                   MVBench
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                   temporal_order_destruction
frame_budget                                                    8
generator                    deterministic_reverse_of_base_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      4
Name: 31, dtype: object>=8 | 7200 | [第 1 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/internvl_predictions.jsonl:1>); [第 2 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/internvl_predictions.jsonl:2>); [第 3 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/internvl_predictions.jsonl:3>); [第 4 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/internvl_predictions.jsonl:4>) |
| MVBench | LLaVA-OneVision-2-8B-Instruct | <bound method Series.view of dataset                                                   MVBench
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                                         base
frame_budget                                                    8
generator                         uniform_official_bound_8_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      1
Name: 32, dtype: object>=8; <bound method Series.view of dataset                                                   MVBench
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                           invariant_nuisance
frame_budget                                                    8
generator                     phase_shift_official_bound_8_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      2
Name: 33, dtype: object>=8; <bound method Series.view of dataset                                                   MVBench
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                            dense_acquisition
frame_budget                                                   16
generator                        uniform_official_bound_16_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      3
Name: 34, dtype: object>=16; <bound method Series.view of dataset                                                   MVBench
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                   temporal_order_destruction
frame_budget                                                    8
generator                    deterministic_reverse_of_base_frames
rows                                                         1800
path            D:\paper 3\caver_route\results_fresh\mvbench_i...
first_line                                                      4
Name: 35, dtype: object>=8 | 7200 | [第 1 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/llava_predictions.jsonl:1>); [第 2 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/llava_predictions.jsonl:2>); [第 3 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/llava_predictions.jsonl:3>); [第 4 行](<D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au/llava_predictions.jsonl:4>) |
| VHD | Qwen3-VL-4B-Instruct | <bound method Series.view of dataset                                                       VHD
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                                         base
frame_budget                                                    8
generator                         locked_uniform_base8_or_dense16
rows                                                          801
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      1
Name: 36, dtype: object>=8; <bound method Series.view of dataset                                                       VHD
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                            evidence_addition
frame_budget                                                   16
generator                         locked_uniform_base8_or_dense16
rows                                                          801
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      2
Name: 37, dtype: object>=16 | 1602 | [第 1 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_qwen_predictions.jsonl:1>); [第 2 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_qwen_predictions.jsonl:2>) |
| VHD | InternVL3_5-4B-Instruct | <bound method Series.view of dataset                                                       VHD
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                                         base
frame_budget                                                    8
generator                         locked_uniform_base8_or_dense16
rows                                                          801
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      1
Name: 38, dtype: object>=8; <bound method Series.view of dataset                                                       VHD
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                            evidence_addition
frame_budget                                                   16
generator                         locked_uniform_base8_or_dense16
rows                                                          801
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      2
Name: 39, dtype: object>=16 | 1602 | [第 1 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_internvl_predictions.jsonl:1>); [第 2 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_internvl_predictions.jsonl:2>) |
| VHD | LLaVA-OneVision-2-8B-Instruct | <bound method Series.view of dataset                                                       VHD
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                                         base
frame_budget                                                    8
generator                         locked_uniform_base8_or_dense16
rows                                                          801
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      1
Name: 40, dtype: object>=8; <bound method Series.view of dataset                                                       VHD
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                            evidence_addition
frame_budget                                                   16
generator                         locked_uniform_base8_or_dense16
rows                                                          801
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      2
Name: 41, dtype: object>=16 | 1602 | [第 1 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_llava_predictions.jsonl:1>); [第 2 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/vhd_llava_predictions.jsonl:2>) |
| TempCompass | Qwen3-VL-4B-Instruct | <bound method Series.view of dataset                                               TempCompass
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                                         base
frame_budget                                                    6
generator                         locked_uniform_base6_or_dense12
rows                                                          410
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      1
Name: 42, dtype: object>=6; <bound method Series.view of dataset                                               TempCompass
model_id                                Qwen/Qwen3-VL-4B-Instruct
view                                            evidence_addition
frame_budget                                                   12
generator                         locked_uniform_base6_or_dense12
rows                                                          410
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      2
Name: 43, dtype: object>=12 | 820 | [第 1 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_qwen_predictions.jsonl:1>); [第 2 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_qwen_predictions.jsonl:2>) |
| TempCompass | InternVL3_5-4B-Instruct | <bound method Series.view of dataset                                               TempCompass
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                                         base
frame_budget                                                    6
generator                         locked_uniform_base6_or_dense12
rows                                                          410
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      1
Name: 44, dtype: object>=6; <bound method Series.view of dataset                                               TempCompass
model_id                        OpenGVLab/InternVL3_5-4B-Instruct
view                                            evidence_addition
frame_budget                                                   12
generator                         locked_uniform_base6_or_dense12
rows                                                          410
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      2
Name: 45, dtype: object>=12 | 820 | [第 1 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_internvl_predictions.jsonl:1>); [第 2 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_internvl_predictions.jsonl:2>) |
| TempCompass | LLaVA-OneVision-2-8B-Instruct | <bound method Series.view of dataset                                               TempCompass
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                                         base
frame_budget                                                    6
generator                         locked_uniform_base6_or_dense12
rows                                                          410
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      1
Name: 46, dtype: object>=6; <bound method Series.view of dataset                                               TempCompass
model_id           lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct
view                                            evidence_addition
frame_budget                                                   12
generator                         locked_uniform_base6_or_dense12
rows                                                          410
path            C:\Users\87385\Desktop\paper 3\video-hallucina...
first_line                                                      2
Name: 47, dtype: object>=12 | 820 | [第 1 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_llava_predictions.jsonl:1>); [第 2 行](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/inputs/tempcompass_llava_predictions.jsonl:2>) |

MVBench 的实际调用脚本为 [run_mvbench_three_model_bundle.pbs](<D:/paper 3/caver_route/katana/run_mvbench_three_model_bundle.pbs:94>)，调用 katana/mvbench_multimodel_runner.py。采样实现见 [mvbench_multimodel_runner.py](<D:/paper 3/caver_route/katana/mvbench_multimodel_runner.py:75>)：base 和 dense 使用官方片段边界内的采样，shifted 改变采样位置，reversed 翻转 base 帧序列。

这些响应确实被开发评分流程读取，而非另一个同名实验：[run_perception_valid500.pbs](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/katana/run_perception_valid500.pbs:198>) 的第 198 至 218 行列出了 NExT-GQA、STAR、MVBench、VHD 和 TempCompass 的 development-result 路径。MVBench 指向任务 8904762，VHD 指向 8905351，TempCompass 指向 8905479。VHD 和 TempCompass 的本地副本与原归档路径、哈希映射见 [download_manifest.json](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/results/supplementary/P5_vhd_transfer_20261119/download_manifest.json>)。

不能把目标集的统一四视角配置回填到全部开发数据。NExT-GQA 的后两个视角是有证据区间与其补集；VHD 与 TempCompass 没有 shifted/reversed 推理记录。VHD 的 counterintuitive/reversed 素材子集与 reversed 推理视角不是一回事。缺失特征的零填充和可用性标记见 [evaluate_response_graph_features.py](<C:/Users/87385/Desktop/paper 3/video-hallucination-diagnosis/src/evaluate_response_graph_features.py:156>)。

本轮未修改论文，以上 MVBench 句子仅供审核。
