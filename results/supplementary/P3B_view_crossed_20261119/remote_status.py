"""Read-only job/output status, without exposing job environments."""
import json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
state=json.loads((ROOT/'STATE.json').read_text());out=[]
for task in state['tasks']:
    directory=ROOT if task['protocol']=='original_views' else ROOT.parent/'P3B_original_panel_20261119'
    row=dict(task);raw=directory/('responses_'+task['model']);job=task['job']
    if task['protocol']=='original_views' and (directory/('merged_responses_'+task['model'])/'COMPLETE.json').exists():raw=directory/('merged_responses_'+task['model'])
    row['bank_directory']=raw.name
    result=subprocess.run(['qstat','-xf',job],capture_output=True,text=True)
    fields={}
    for line in result.stdout.splitlines():
        m=re.match(r'^    (job_state|Exit_status|resources_used.walltime) = (.*)$',line)
        if m:fields[m[1]]=m[2]
    row['scheduler']=fields;row['qstat_ok']=result.returncode==0
    validation=directory/(task['model']+'_FULL_VALIDATION.json')
    row['complete']=validation.exists() and (raw/'COMPLETE.json').exists()
    if row['complete']:row['validation']=json.loads(validation.read_text())
    pilot=directory/(task['model']+'_PILOT_VALIDATION.json')
    if pilot.exists():row['pilot']=json.loads(pilot.read_text())
    logfile=directory/('pilot.'+job+'.log')
    if logfile.exists():
        with logfile.open('rb') as f:
            f.seek(max(0,logfile.stat().st_size-4096));tail=f.read().decode(errors='replace')
        matches=re.findall(r'REFERENCE_PHASE_DONE (\d+)',tail)
        row['latest_logged_calls']=int(matches[-1]) if matches else None
        row['error_tail']=tail[-1400:] if 'Traceback' in tail or 'Error' in tail else None
    gate=directory/(task['model']+'_BUDGET_GATE.'+job+'.json')
    if gate.exists():row['budget_gate']=json.loads(gate.read_text())
    plan=directory/('SHARD_PLAN_'+task['model']+'.json')
    if plan.exists():
        shards=json.loads(plan.read_text())['shards']
        row['shards_total']=len(shards)
        row['shards_complete']=sum((directory/'shards'/task['model']/f'shard_{s["index"]:02d}'/'COMPLETE.json').exists() for s in shards)
    out.append(row)
print(json.dumps(out),flush=True)
