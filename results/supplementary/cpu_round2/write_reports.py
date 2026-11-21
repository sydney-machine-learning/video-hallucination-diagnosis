from common import *
import csv,re,subprocess,platform,datetime

def link(p,label=None,line=None):
 p=Path(p).resolve();target=p.as_posix()+(f':{line}' if line else '')
 return f'[{label or p.name}](<{target}>)'
def file(name):return link(ROOT/name)
def ci(r,key='accuracy',scale=100,digits=2):
 low='ci_low' if key=='accuracy' else key+'_ci_low';high='ci_high' if key=='accuracy' else key+'_ci_high'
 return f'{r[key]*scale:.{digits}f} [{r[low]*scale:.{digits}f}, {r[high]*scale:.{digits}f}]'
names={'internvl14':'InternVL3.5 14B Instruct','qwen8':'Qwen3-VL 8B Instruct','llava8':'LLaVA-OneVision-2 8B Instruct','majority_vote':'多数投票','mean across three':'三个模型等权平均'}
bname={'raw_confidence':'原始置信度','base_vote':'Base vote','base_vote_then_confidence':'Base vote，原始置信度破同分','base_support':'Base support','graph_lr':'开发集 LR','graph_hgb':'开发集 HGB'}
def latex(headers,rows):
 def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
 return '\\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}\n\\hline\n'+' & '.join(map(esc,headers))+r' \\'+'\n\\hline\n'+'\n'.join(' & '.join(map(esc,r))+r' \\' for r in rows)+'\n\\hline\n\\end{tabular}\n'

# A: tables are rendered only from saved results.
acc=pd.read_csv(ROOT/'A_accuracy_intervals.csv');pair=pd.read_csv(ROOT/'A_vote_paired_intervals.csv');sel=pd.read_csv(ROOT/'A_selection_summary.csv');counts=pd.read_csv(ROOT/'A_selection_counts.csv').set_index('model');support=pd.read_csv(ROOT/'A_support_summary.csv')
at=[]
for m in MODELS+['majority_vote']:
 a=acc[(acc.split==0)&(acc.model==m)].iloc[0];b=acc[(acc.split==-1)&(acc.model==m)].iloc[0]
 at.append([names[m],ci(a),ci(b)])
ct=[]
for m in ['internvl14','llava8']:
 r=sel[(sel.model==m)&(sel.method=='checked')].iloc[0];n=counts.loc[m]
 ct.append([names[m],f'{r.mean_gain:.2f}',f'{r.min_gain:.2f}',f'{int(r.harm_splits)}/30',f'{int(n.raw_fallback)}/30',int(n.target_logistic),int(n.source_trace_hgb)])
st=[]
for m in MODELS+['mean across three']:
 row=[names[m]]
 for method in ['all_view_support','base_support']:row.append(f'{support[(support.model==m)&(support.method==method)].iloc[0].mean_gain:.2f}')
 st.append(row)
pt=[[r.comparison.replace('vote minus ','多数投票减去 ').replace('internvl14',names['internvl14']).replace('qwen8',names['qwen8']).replace('llava8',names['llava8']),f'{r.difference_pp:.2f}',f'[{r.ci_low:.2f}, {r.ci_high:.2f}]'] for _,r in pair[pair.split==0].iterrows()]
audit=pd.read_csv(ROOT/'A_answer_definition_audit.csv');sens=pd.read_csv(ROOT/'A_generated_answer_sensitivity.csv');sensrows=[[names[r.model],ci(r)] for _,r in sens[sens.split==0].iterrows()]
unchecked=[[names[m],method,f'{r.mean_gain:.2f}',f'{r.min_gain:.2f}',int(r.harm_splits)] for m in ['internvl14','llava8'] for method in ['pilot_lr','development_hgb'] for _,r in sel[(sel.model==m)&(sel.method==method)].iterrows()]
araw='\n'.join(f'{names[m]}：{link(OLD/f"merged_responses_{m}/responses.jsonl")}，7,552 行；对应 COMPLETE.json 的 SHA256 已匹配。' for m in MODELS)
A=f'''# A：Perception Test 多数投票与 30 次划分

仅用已有响应，未运行 GPU 推理，未修改归档结果。所有 AURC 使用前缀风险均值，按置信度降序排序，同分保留输入顺序。相对下降为 100 ×（原始 AURC 减去新 AURC）/ 原始 AURC；负值表示变差。

## 输入与范围

{araw}

共 1,888 题、500 个视频，每个模型都有 base 16、shifted 8、dense 24、reversed 8 帧四个视角和逐题三个选项概率。模型版本见 {file('PREFLIGHT.json')}。划分来自 {link(ASSIGN)}；使用其 Perception 的 split 1 至 30，每次 250 个 pilot 视频、250 个测试视频。split 0 只作原始测试划分参照，不混入 30 次平均。未重新抽取划分，也未筛选结果。

特征使用原有 53 维构建器；以每个作答模型自己的 base 概率最大选项为固定答案。Qwen 的特征与归档 PhaseGuard 对比矩阵逐项匹配。来源：{link(REPO/'src/evaluate_response_graph_features.py',line=127)}、{link(OLD/'original_views_features.csv')}。

## 多数投票准确率

三个模型在 base 上投票。票数相同时，在并列选项中取三个模型平均概率最大的选项；若仍完全相同，取选项序列中的首项。每个视频整组有放回抽样 10,000 次，模型和投票共享抽样，报告百分位 95% CI。准确率按题计算，而非先平均每个视频。

{md(['方法','原始测试划分：911 题，250 视频，准确率 % [95% CI]','完整响应面板：1888 题，500 视频，准确率 % [95% CI]'],at)}

完整面板包含 pilot，仅作描述；论文测试对比应使用左列。

{md(['原始测试划分的配对差异','准确率差 pp','95% CI'],pt)}

多数投票显著优于 LLaVA；相对 InternVL 和 Qwen 的配对区间包含 0，不能声称优于全部单模型。

每个随机划分的测试准确率与 CI 均保存在 {file('A_accuracy_intervals.csv')}；对应配对差异在 {file('A_vote_paired_intervals.csv')}，抽样索引保存在 A_accuracy_bootstrap_*.npz。种子为 20261120 + split，完整面板 split 为负一。

## 检查流程在两种作答模型上的结果

复用原有 pilot LR、冻结的开发集 HGB 和选择逻辑。pilot LR 按视频做五折 OOF；标准化、缺失值处理、C=0.1、balanced 权重、liblinear、max_iter=3000 均不变。HGB 直接读取 {link(BUNDLE)}，没有再训练或调参。

两个候选各做 2,000 次 pilot 视频 bootstrap，取 2.5% 分位数作为下界。pilot LR 的下界须严格大于 0，开发集 HGB 须严格大于 0.5 pp；多个候选通过时取下界最大的候选，否则退回原始置信度。0.5 pp 指相对 AURC 下降百分比的百分点。仅在本次进程内将原选择器的 AURC 替换为前缀均值函数，原文件不变。来源：{link(REPO/'src/evaluate_v2_target_pilot.py',line=197)}。

{md(['作答模型','平均相对下降 %','最小相对下降 %','变差划分','退回原始置信度','选择 pilot LR 次数','选择开发集 HGB 次数'],ct)}

InternVL 的小平均增益伴随 27 次退回；这说明检查较保守，不能把没有变差解释成优于未检查的候选。未检查候选的同批结果如下。

{md(['作答模型','未检查候选','平均下降 %','最小下降 %','变差次数'],unchecked)}

逐划分结果：{file('A_selection_splits.csv')}。各次 pilot OOF 分数、选择下界、测试分数和拟合 LR 位于 {link(ROOT/'A_runs')}。抽样种子、候选偏移和实现保存在 {file('ANALYSIS_PLAN.json')}、{file('run_A.py')}。

## 无标签 support

Base support 是三个 base 响应对该作答模型固定答案的平均概率；all-view support 是全部十二个响应对同一答案的平均概率。均含作答模型自身，不使用 pilot 标签，也不改变答案。

{md(['作答模型','All-view support 平均下降 %','Base support 平均下降 %'],st)}

最后一行先对每个划分的三个模型等权平均，再对 30 次划分平均，不能用合并所有预测后的 AURC 替代。逐划分结果：{file('A_support_splits.csv')}。

## 答案字段差异

本报告主分析沿用论文构建器的选项概率 argmax 定义。响应中的生成字母 answer 与 argmax 并非处处相同：

{md(['模型','base 不一致题数','四视角不一致记录数'],[[names[r.model],r.base_mismatch,r.all_views_mismatch] for _,r in audit.iterrows()])}

另按生成字母投票，复用相同视频抽样，原始测试划分得到：

{md(['方法','生成字母准确率 % [95% CI]'],sensrows)}

这项敏感性分析不替换主分析的固定答案。来源：{file('A_generated_answer_sensitivity.csv')}。

30 次划分重复使用同一批视频，不是 30 个独立数据集；这里的最小值与变差次数只描述这 30 次划分，不构成对新目标的保证。检查组件也未重新设计。

可转入论文的表格见 {file('A_tables.tex')}。数值验证见 {file('VALIDATION.json')}，输入哈希见 {file('INPUT_MANIFEST.json')}。
'''
(ROOT/'A.md').write_text(A,encoding='utf-8')
(ROOT/'A_tables.tex').write_text('% Accuracy percent; brackets are 95% video bootstrap CIs.\n'+latex(['Model','Reference test','Full panel'],[[('Majority vote' if m=='majority_vote' else names[m]),ci(acc[(acc.split==0)&(acc.model==m)].iloc[0]),ci(acc[(acc.split==-1)&(acc.model==m)].iloc[0])] for m in MODELS+['majority_vote']])+'\n% Thirty fixed video splits; relative AURC reduction percent.\n'+latex(['Answering model','Mean','Minimum','Harm','Fallback','Pilot LR','Dev HGB'],ct)+'\n'+latex(['Answering model','All view support','Base support'],[[('Mean across models' if m=='mean across three' else names[m]),*st[i][1:]] for i,m in enumerate(MODELS+['mean across three'])]),encoding='utf-8')

# B
bm=pd.read_csv(ROOT/'B_metrics.csv');bp=pd.read_csv(ROOT/'B_joint_error_percentile_summary.csv');subset_names={'all':'全集','visible':'visible','counterintuitive':'counterintuitive','hidden':'hidden'}
brows=[];tex=[]
for subset in ['all','visible','counterintuitive','hidden']:
 f=bm[(bm.subset==subset)&(bm.model_id=='mean across three')]
 for _,r in f.iterrows():brows.append([subset_names[subset],bname[r.method],ci(r,'auroc',1,3),ci(r,'gain_pct',1,2)])
 tex.append('% '+subset+'; equal mean across three development models.\n'+latex(['Score','AUROC [95% CI]','AURC reduction [95% CI]'],[[r.method,ci(r,'auroc',1,3),ci(r,'gain_pct',1,2)] for _,r in f.iterrows()]))
pr=[]
for _,r in bp[bp.model_id=='mean across three'].iterrows():pr.append([bname[r.method],f'{r.mean_percentile:.2f} [{r.mean_ci_low:.2f}, {r.mean_ci_high:.2f}]',f'{r.median_percentile:.2f}',f'{r.delta_vs_raw_pp:.2f} [{r.delta_ci_low:.2f}, {r.delta_ci_high:.2f}]'])
B=f'''# B：VHD 留出数据集评分诊断

复用 {link(VHD/'predictions.csv')} 中 held-out VHD 的预测及 {link(VHD/'models/VHD.joblib')}。这些文件来自 2026 年 11 月 19 日对归档响应的 CPU 重建。本次没有重新训练，没有 GPU 推理。

## 输入与统计口径

训练数据为 MVBench、NExT-GQA、STAR、TempCompass，共 9,630 个模型与问题组合；VHD 的全部变体均留出。VHD 有 801 题、341 个源视频、2,403 个模型与问题组合。训练不使用 VHD 标签。审计：{file('B_reuse_audit.json')}、{link(VHD/'holdout_audit.json')}。

开发模型是 InternVL3.5 4B Instruct、Qwen3-VL 4B Instruct、LLaVA-OneVision-2 8B Instruct，区别于 A 的 14B/8B/8B。VHD 仅有 base 8 帧、dense 16 帧；缺失视角遵循原特征构建器处理。

LR 沿用 C=0.2、max_iter=3000、标准化和等数据集权重。HGB 沿用 learning_rate=0.05、max_iter=200、max_leaf_nodes=15、min_samples_leaf=30、l2_regularization=2.0 及原随机种子。实际预测直接复用，参数来源：{link(VHD/'analyze.py',line=98)}、{link(REPO/'src/evaluate_response_graph_features.py',line=245)}。

原始置信度为固定答案的 base 概率。Base vote 为三个 base 答案支持固定答案的比例，base support 为三个 base 响应给该答案的平均概率。两者均不改变答案，不等同于 A 的改答案多数投票。

AURC 一律前缀均值，同分保持输入顺序。Base vote 大量同分，另列原归档的置信度破同分版本，即 vote + 1e-9 × raw confidence，作为敏感性结果。AUROC 按标准半权处理相等分数。

复用 P5 保存的 10,000 次源视频抽样及种子 20261119。每次把同一源视频的全部题目、变体及三个模型一起抽取；各评分共享抽样，固定已拟合评分器。CI 为 2.5% 和 97.5% 分位数。每次先算各模型指标，再等权平均；增益先逐模型算相对下降，再取平均。这里的 CI 不包含训练评分器的随机性。

## 子集结果

{md(['子集','题数','源视频数'],[[s,int(bm[bm.subset==s].iloc[0].questions),int(bm[bm.subset==s].iloc[0].videos)] for s in ['all','visible','counterintuitive','hidden']])}

visible 对应 original，counterintuitive 对应 reversed，hidden 对应 masked。这里 reversed 指素材变体，不是另一个推理视角。全集源视频数小于子集数之和，因为变体共享源视频。每题两个选项，随机作答准确率为 50%；AUROC 的无判别参照为 0.5，二者不是同一指标。

{md(['子集','评分','AUROC [95% CI]','相对 AURC 下降 % [95% CI]'],brows)}

主表为三个开发模型的等权平均。每个模型、每个子集的完整 AUROC、AURC、相对下降和 CI 均在 {file('B_metrics.csv')}，可直接转为分模型表格；全部版本的 LaTeX 表格在 {file('B_tables.tex')}。

## 三个模型一致答错的反常识题

同一响应面板中，counterintuitive 的 178 题里有 99 题三个模型都答错。因每题只有两个选项，这些题也必然三个模型答案一致。此计数以本报告响应为准，不混用其他推理版本的共同错误计数。

每个模型分别在 178 个 counterintuitive 题内计算分数的升序百分位：100 × 平均秩 / 178。百分位越高表示越被评分器信任。下表统计 99 题的百分位，再对三个模型等权平均；中位数取合并的 297 个百分位。均值 CI 和与原始置信度的配对差异 CI 都按源视频抽样，并在每次样本内重算百分位。

{md(['评分','平均百分位 [95% CI]','中位百分位','相对原始置信度的百分位差 [95% CI]'],pr)}

这些共同错误在 LR 和 HGB 下的平均百分位都高于原始置信度，即仍被赋予较高可信度。本分析不支持“学习评分已能识别反常识共同错误”。HGB 的平均 AUROC 高于原始置信度，但 AURC 仍更差；两个指标关注的排序位置不同，不能互相替代。

这检验的是开发集评分直接迁移到完整留出的 VHD，未使用目标 pilot 或替换检查，不能等同于 A 的完整目标检查流程。

逐题百分位另含以全部 801 个 VHD 问题为参照的版本，见 {file('B_joint_error_percentiles.csv')}。分模型百分位及 CI：{file('B_joint_error_percentile_summary.csv')}。评分和标签：{file('B_scores.csv')}。抽样分布：B_bootstrap_*.npz。

LR/HGB 的八组子集点估计及 CI 与 P5 保存结果一致；独立 AUROC 与前缀 AURC 校验通过，见 {file('VALIDATION.json')}。
'''
(ROOT/'B.md').write_text(B,encoding='utf-8')
for model in bm.model_id.unique():
 if model=='mean across three':continue
 for subset in ['all','visible','counterintuitive','hidden']:
  f=bm[(bm.subset==subset)&(bm.model_id==model)]
  tex.append('% '+model+' / '+subset+'\n'+latex(['Score','AUROC [95% CI]','AURC reduction [95% CI]'],[[r.method,ci(r,'auroc',1,3),ci(r,'gain_pct',1,2)] for _,r in f.iterrows()]))
(ROOT/'B_tables.tex').write_text('\n'.join(tex),encoding='utf-8')

# C: archived files are read, never executed.
raw=pd.read_csv(ROOT/'C_raw_view_audit.csv');paper=REPO.parent/'overleaf_review_points_234_20261119/verified.tex'
scripts={'MVBench':Path('D:/paper 3/caver_route/katana/mvbench_multimodel_runner.py'),'NExT-GQA':Path('D:/paper 3/sicr_route/src/build_nextgqa_pilot_manifest.py'),'STAR':Path('D:/paper 3/sicr_route/src/build_star_pilot_manifest.py')}
ctable=[['MVBench','base 8；shifted 8；dense 16；reversed 8',link(scripts['MVBench'],line=47)+'，第 47 至 64 行','附录第 558 行未列出 MVBench，建议补充'],['NExT-GQA','base 16；shifted 16；evidence intervals 16；complement 16',link(scripts['NExT-GQA'],line=23)+'，第 23 至 44 行；'+link(scripts['NExT-GQA'],line=232),'一致；后两个视角不是 dense/reversed'],['STAR','base 8；shifted 8；dense 16；reversed 8',link(scripts['STAR'],line=23)+'，第 23 至 52 行','一致'],['VHD','base 8；evidence_addition/dense 16',link(VHD/'inputs/vhd_qwen_predictions.jsonl',line=1)+'，第 1、2 行','一致；仅两个视角'],['TempCompass','base 6；evidence_addition/dense 12',link(VHD/'inputs/tempcompass_qwen_predictions.jsonl',line=1)+'，第 1、2 行','一致；仅两个视角']]
rawrows=[]
for (dataset,model,path),g in raw.groupby(['dataset','model_id','path'],sort=False):
 rawrows.append([dataset,model.split('/')[-1],'; '.join(f'{r.view}={int(r.frame_budget)}' for _,r in g.iterrows()),int(g.rows.sum()),'; '.join(link(path,f'第 {int(r.first_line)} 行',int(r.first_line)) for _,r in g.iterrows())])
C=f'''# C：开发阶段视角与帧数核对

仅读取配置、推理脚本、归档响应日志和论文副本，未执行这些实验脚本，未改动原文件。逐条记录核对结果保存于 {file('C_raw_view_audit.csv')}。本报告只核对每次推理的输入帧数，不把图像裁块数或视觉 token 数当作帧数。

{md(['数据集','实际视角与帧数','配置或原始记录证据','与论文附录核对'],ctable)}

论文依据：{link(paper,line=558)}。该行已正确描述 NExT-GQA、STAR、VHD、TempCompass，缺少 MVBench。可补一句：

> MVBench uses eight frames for the base, shifted, and reversed views, and sixteen frames for the dense view.

## 原始响应交叉核对

下表列出三个模型各自的来源文件与首个对应记录行号。逐行读取的全部记录在每个视角上的 frame_budget 均一致，不只检查示例行。NExT-GQA 和 STAR 每模型 500 题 × 4 视角，MVBench 每模型 1,800 题 × 4 视角，VHD 每模型 801 题 × 2 视角，TempCompass 每模型 410 题 × 2 视角。

{md(['数据集','模型','视角=帧数','总行数','原始文件及行号，顺序与视角列一致'],rawrows)}

MVBench 的实际调用脚本为 {link(Path('D:/paper 3/caver_route/katana/run_mvbench_three_model_bundle.pbs'),line=94)}，调用 katana/mvbench_multimodel_runner.py。采样实现见 {link(scripts['MVBench'],line=75)}：base 和 dense 使用官方片段边界内的采样，shifted 改变采样位置，reversed 翻转 base 帧序列。

这些响应确实被开发评分流程读取，而非另一个同名实验：{link(REPO/'katana/run_perception_valid500.pbs',line=198)} 的第 198 至 218 行列出了 NExT-GQA、STAR、MVBench、VHD 和 TempCompass 的 development-result 路径。MVBench 指向任务 8904762，VHD 指向 8905351，TempCompass 指向 8905479。VHD 和 TempCompass 的本地副本与原归档路径、哈希映射见 {link(VHD/'download_manifest.json')}。

不能把目标集的统一四视角配置回填到全部开发数据。NExT-GQA 的后两个视角是有证据区间与其补集；VHD 与 TempCompass 没有 shifted/reversed 推理记录。VHD 的 counterintuitive/reversed 素材子集与 reversed 推理视角不是一回事。缺失特征的零填充和可用性标记见 {link(REPO/'src/evaluate_response_graph_features.py',line=156)}。

本轮未修改论文，以上 MVBench 句子仅供审核。
'''
(ROOT/'C.md').write_text(C,encoding='utf-8')

# D: existing accuracies, exact model revisions, official documents.
oldmetrics=REPO/'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/test_model_metrics.csv'
om=pd.read_csv(oldmetrics);om=om[om.method.eq('raw_confidence')]
pre=json.loads((ROOT/'PREFLIGHT.json').read_text());dr=[]
for entry in pre['models']:
 m=entry['model'];oldrow=om[om.model_id==entry['model_ids'][0]].iloc[0];new=acc[(acc.split==0)&(acc.model==m)].iloc[0]
 dr.append(dict(model=entry['model_ids'][0],revision=entry['revisions'][0],paper_archive_accuracy_pct=100*oldrow.accuracy,phaseguard_bank_accuracy_pct=100*new.accuracy,questions=911,videos=250,official_same_checkpoint_perception_score='not found in audited primary sources'))
pd.DataFrame(dr).to_csv(ROOT/'D_accuracy_comparison.csv',index=False)
paths=json.loads((ROOT/'D_sources/videochat_perception_paths.json').read_text(encoding='utf-8'));hfinv=json.loads((ROOT/'D_sources/videochat_training_inventory.json').read_text(encoding='utf-8'));hfsha=hfinv.get('sha','main')
ups='\n\n'.join(f'[{p}](https://huggingface.co/datasets/OpenGVLab/VideoChat-Flash-Training-Data/blob/{hfsha}/{p})' for p in paths)
configsources=json.loads((ROOT/'D_sources/CONFIG_SOURCES.json').read_text());sourceurl=lambda pattern:next(x['source'] for x in configsources if pattern in x['file'])
D=f'''# D：Perception Test 准确率与训练数据声明核对

核对日期：2026 年 11 月 20 日。仅阅读已有结果及官方论文、模型卡、仓库配置和上游数据说明，没有运行模型。下载的是文档及文件名清单，未下载视频、权重或训练样本。

## 准确率与技术报告

{md(['模型','原论文归档准确率 %','PhaseGuard 对比响应的 base 准确率 %','同型号技术报告的 Perception Test 数值'],[[x['model'],f"{x['paper_archive_accuracy_pct']:.2f}",f"{x['phaseguard_bank_accuracy_pct']:.2f}",'未找到'] for x in dr])}

两个本地列都对应原始测试划分的 911 题、250 个视频；它们是公开 validation 面板中的测试部分，并非官方隐藏 test 集。第一列来源：{link(oldmetrics)} 的 raw_confidence 三行；第二列来源：{file('A_accuracy_intervals.csv')} 的 split 0。两批推理响应的准确率不同，应保留批次标识，不能混用。新响应按选项概率 argmax 作答，生成字母敏感性结果另见 A。保存表：{file('D_accuracy_comparison.csv')}。

检查的主要官方来源：

1. [Qwen3-VL Technical Report，2511.21631v1](https://arxiv.org/html/2511.21631v1)，视频评测及附录；[8B Instruct 实验版本模型卡]({sourceurl('qwen8_archived_card')})。
2. [InternVL3.5 技术报告，2508.18265v2](https://arxiv.org/html/2508.18265v2)，第 3.10 节视频评测；[14B Instruct 实验版本模型卡]({sourceurl('internvl14_archived_card')})。
3. [LLaVA-OneVision-2 技术报告，2605.25979v1](https://arxiv.org/html/2605.25979v1)，视频评测表及附录；[8B Instruct 实验版本模型卡]({sourceurl('llava8_archived_card')})。

这三份技术报告及对应模型卡中均未找到同型号的 Perception Test 准确率。因此没有可核验的官方数值可作数值差异对比，不能填入旧型号或另一评测的分数。

相关的 [OneVision-Encoder 报告，2602.08683v1](https://arxiv.org/html/2602.08683v1) 确实出现 Perception Test，但其中语言模型配对评测采用 Qwen3-4B，另有视觉编码器的 attentive probing 实验。这些都不是本实验的 LLaVA-OneVision-2-8B-Instruct，未移植其分数到上表。

官方数据集对 validation 与 test 分开提供下载和评测入口，见 [Perception Test 官方仓库](https://github.com/google-deepmind/perception_test)。即使另有完整 validation 或隐藏 test 的官方模型成绩，也需匹配视频集合、抽帧、分辨率、提示词、作答与评分方式，才适合直接比较。

## 是否声明训练使用 Perception Test

| 模型 | 核对所得证据 | 能得出的结论 |
| --- | --- | --- |
| Qwen3-VL 8B Instruct | 报告第 3.2.7 节描述视频来源类别、标注和采样，没有逐一列出完整训练数据；模型卡未点名 Perception Test | 在核对范围内未找到包含或明确排除该数据集的声明；训练重叠未知 |
| InternVL3.5 14B Instruct | 报告第 2.2、2.3 节说明继承 InternVL3 语料；两份报告及模型卡没有点名 Perception Test；公开 14B 训练脚本指向 debug 数据配置 | 未获得足以确认训练包含或排除该数据集的完整清单 |
| LLaVA-OneVision-2 8B Instruct | 报告第 3.1 节声明采用 VideoChat-Flash-Training-Data 等语料的相关子集；该上游仓库确有 Perception Test 文件 | 有上游包含证据，但该检查点选用子集和具体视频未公开，不能确认使用了我们的 validation 视频 |

Qwen 的 [公开微调数据配置]({sourceurl('data____init__')}) 包含示例路径占位符，不能当作发布检查点的完整训练清单。InternVL 的 [14B SFT 脚本]({sourceurl('14b_sft')}) 第 48 行引用 debug_sft.json，[14B MPO 脚本]({sourceurl('14b_mpo')}) 同样引用 debug_mpo.json；其 debug 示例不能证明完整训练语料。[InternVL3 报告](https://arxiv.org/html/2504.10479v1) 和 [InternVL-Data 卡](https://huggingface.co/datasets/OpenGVLab/InternVL-Data) 也未提供可核对该检查点的完整视频 ID 清单。

LLaVA 的 [公开 SFT 数据 YAML]({sourceurl('multimodal__sft_dataset.yaml')}) 使用占位路径；[旧 configs 目录说明]({sourceurl('configs__README')}) 标为 deprecated，不将其视为完整发布配方。报告只说从上游语料选取相关部分，未声明全部使用。

上游 VideoChat-Flash-Training-Data 的官方文件名清单含以下三个条目：

{ups}

第三个文件名明确指向 train；前两个文件名不能单独证明官方划分。这里没有下载样本、没有进行视频重叠审计。上游存在这些文件，不等于该模型一定使用了全部文件，更不等于使用了本实验的 validation 标签。原始文件名清单与版本 SHA 已保存于 {file('D_sources/videochat_training_inventory.json')}，匹配条目见 {file('D_sources/videochat_perception_paths.json')}。

因此不能写“三个模型都没有见过 Perception Test”，也不能根据这些文档直接指控 validation 泄漏。当前证据只支持：公开资料不足以排除重叠；LLaVA 的一个已声明上游语料含相关数据，实际选用范围未知。

## 核对范围与留档

审阅三个模型的官方主报告、当前模型卡、与本地响应 model_revision 一致的模型卡、官方仓库 README、公开训练数据配置，并追查 InternVL3 和 VideoChat-Flash 上游说明。检索 Perception Test、PerceptionTest、perception_test、perception-test；普通名词 perception 不算命中。

网页及官方仓库快照位于 {link(ROOT/'D_sources')}。来源 URL 和 SHA256 分别在 SOURCES.json、CONFIG_SOURCES.json、UPSTREAM_SOURCES.json；官方仓库 tree SHA 在 repo_inventory.json。已把“未查到”“上游包含”和“检查点实际使用”分开，不将文档缺失当作无污染证明。本轮未修改论文。
'''
(ROOT/'D.md').write_text(D,encoding='utf-8')
print('Wrote A.md, B.md, C.md, D.md, A_tables.tex, B_tables.tex, D_accuracy_comparison.csv')
