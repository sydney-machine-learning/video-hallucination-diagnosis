"""Aggregate surviving pair ablations; no inference or fitting."""
import csv,hashlib,json,platform,subprocess,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

out=Path(__file__).resolve().parent
root=out.parents[2]
source=root/'results/posthoc_robustness/compute_budget_ablation.csv'
groups=defaultdict(list)
with source.open(newline='',encoding='utf-8') as f:
    for line,row in enumerate(csv.DictReader(f),2):
        if row['audit_family']=='model_panel' and row['configuration']=='Two-family four views':
            assert row['model_calls_per_question']=='8'
            assert row['roles']=='base+invariant+addition+removal'
            groups[row['dataset'],row['panel_models']].append((line,row))
assert len(groups)==6
records=[]
for (dataset,pair),rows in groups.items():
    assert len(rows)==2
    assert set(pair.split('+'))=={r['reference_family'] for _,r in rows}
    values=[float(r['relative_aurc_reduction_percent']) for _,r in rows]
    records.append({'dataset':dataset,'pair':pair,'mean_relative_aurc_gain_percent':sum(values)/2,
        'min_relative_aurc_gain_percent':min(values),'reference_models':2,
        'source':str(source.relative_to(root)).replace('\\','/'),
        'source_lines':','.join(str(line) for line,_ in rows),
        'individual_gains':json.dumps({r['reference_family']:v for (_,r),v in zip(rows,values)})})
with (out/'pair_gains.csv').open('x',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
report=['# Two families, four views',
        'All three pairs survive separately for each target, with two reference-model rows per pair. Mean and minimum below are across the two included models, not across three pairs, videos or repeated splits. Gains are relative AURC reduction percentages against each reference model’s original confidence.',
        'Source: `results/posthoc_robustness/compute_budget_ablation.csv`. Filter: `audit_family=model_panel`, `configuration=Two-family four views`. Each configuration uses 8 model/view calls and 112 model-input frames per question.',
        '| Target | Pair | Mean gain (%) | Minimum gain (%) | CSV lines |',
        '| --- | --- | --- | --- | --- |']
for r in records:
    report.append(f"| {r['dataset']} | {r['pair']} | {r['mean_relative_aurc_gain_percent']:.6f} | {r['min_relative_aurc_gain_percent']:.6f} | {r['source_lines']} |")
report+=['','These are saved posthoc budget ablations. The implementation calls fit_target_scores and score_gain directly; this extraction does not apply or rerun the promotion bootstrap. Do not describe these rows as newly validated checked selections. No inference, fitting or GPU job was run.']
(out/'REPORT.md').write_text('\n\n'.join(report[:3])+'\n\n'+'\n'.join(report[3:])+'\n',encoding='utf-8')
manifest={'phase':'pair_ablation','utc':datetime.now(timezone.utc).isoformat(),
 'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
 'seeds':None,'gpu_used':False,'jobs_submitted':0,
 'environment':{'python':sys.version,'platform':platform.platform()},
 'input_sha256':{str(source.relative_to(root)).replace('\\','/'):hashlib.sha256(source.read_bytes()).hexdigest()},
 'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='MANIFEST.json'}}
with (out/'MANIFEST.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
for r in records:print(r['dataset'],r['pair'],f"mean={r['mean_relative_aurc_gain_percent']:.6f}",f"min={r['min_relative_aurc_gain_percent']:.6f}",'lines='+r['source_lines'])
