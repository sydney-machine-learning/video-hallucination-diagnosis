"""Copy validated outputs and overlap local CPU fits with remaining GPU jobs.

Submits timing-gated native-view shard arrays, with one GPU per 110-minute task
and at most two concurrent tasks per model. Does not cancel existing jobs.
Stops after both protocols are audited and reported, or after 12 hours.
State and failures remain reviewable.
"""
import hashlib,json,os,re,shlex,subprocess,sys,time
from importlib.metadata import version
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/srv/scratch/z5611125/video_hallucination_supplementary/results/supplementary/'
HOST='z5611125@129.94.146.61'
COMMON=['-i','C:/Users/87385/.ssh/codex_katana_ed25519','-o','HostKeyAlias=katana.restech.unsw.edu.au','-o','BatchMode=yes','-o','ConnectTimeout=20']
SSH=['C:/Windows/System32/OpenSSH/ssh.exe',*COMMON]
SCP=['C:/Windows/System32/OpenSSH/scp.exe',*COMMON]
def save(name,value):(ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def resume_timed_out_llava_pilot(task):
    """One bounded retry for the measured 55-minute pilot, preserving responses."""
    if task['job']!='9829910.kman.restech.unsw.edu.au':return False
    state=task['scheduler']
    if state.get('job_state')!='F' or state.get('Exit_status') in [None,'0']:return False
    if task.get('pilot',{}).get('passed') or task.get('error_tail'):return False
    parts=state.get('resources_used.walltime','00:00:00').split(':')
    if len(parts)!=3:return False
    h,m,s=map(int,parts)
    if h*3600+m*60+s<3290:return False
    receipt=ROOT/'PILOT_TIMEOUT_RETRIES.json'
    if receipt.exists():return False
    retry={'previous_job':task['job'],'model':'llava8','walltime_limit_seconds':4500,
           'status':'submitting','reason':'Original pilot reached its 55-minute allocation without a model error; resume saved responses once.'}
    save(receipt.name,retry)
    cmd='cd '+REMOTE+ROOT.name+' && qsub -l walltime=01:15:00 -v MODEL_KEY=llava8 native_views.pbs'
    result=subprocess.run([*SSH,HOST,cmd],check=True,capture_output=True,text=True,timeout=60)
    job=result.stdout.strip();assert re.fullmatch(r'[0-9]+\.kman\.restech\.unsw\.edu\.au',job),job
    retry.update(status='submitted',job=job);save(receipt.name,retry)
    code="from pathlib import Path; import json; p=Path("+repr(REMOTE+ROOT.name+'/STATE.json')+"); s=json.loads(p.read_text()); t=next(x for x in s['tasks'] if x['protocol']=='original_views' and x['model']=='llava8'); assert t['job']=="+repr(task['job'])+"; s.setdefault('timeout_retries',[]).append("+repr(retry)+"); t['job']="+repr(job)+"; q=p.with_suffix('.tmp'); q.write_text(json.dumps(s,indent=2)+'\\n'); q.replace(p); print(json.dumps(s))"
    updated=subprocess.run([*SSH,HOST,'/srv/scratch/z5611125/venvs/pilot/bin/python -c '+shlex.quote(code)],check=True,capture_output=True,text=True,timeout=60)
    save('STATE.json',json.loads(updated.stdout))
    print('PILOT_TIMEOUT_RESUMED',job,flush=True)
    return True
def transfer(task):
    phase=ROOT.name if task['protocol']=='original_views' else 'P3B_original_panel_20261119'
    dest=ROOT if task['protocol']=='original_views' else ROOT.parent/phase
    model=task['model'];remote=REMOTE+phase+'/';bank=task.get('bank_directory','responses_'+model)
    assert bank in ['responses_'+model,'merged_responses_'+model]
    subprocess.run([*SCP,'-r',HOST+':'+remote+bank,str(dest)],check=True,timeout=240)
    for name in [model+'_FULL_VALIDATION.json',model+'_MODEL_READY.json',model+'_metadata.json',model+'_PILOT_VALIDATION.json']:
        subprocess.run([*SCP,HOST+':'+remote+name,str(dest/name)],check=True,timeout=90)
    actual=sha(dest/bank/'responses.jsonl')
    assert actual==task['validation']['responses_sha256']
    assert actual==json.loads((dest/bank/'COMPLETE.json').read_text())['responses_sha256']
    return {'protocol':task['protocol'],'model':model,'responses_sha256':actual,'local_directory':str(dest/bank)}

def main():
    versions={n:version(n) for n in ['scikit-learn','numpy','pandas','scipy']}
    save('LOCAL_CPU_ENV.json',{'python':sys.executable,'prefix':sys.prefix,'versions':versions})
    assert versions['scikit-learn']=='1.7.2' and versions['numpy']=='2.2.6'
    start=time.monotonic();copied={};running={};completed=set();logs=[]
    submissions=json.loads((ROOT/'SHARD_SUBMISSIONS.json').read_text()) if (ROOT/'SHARD_SUBMISSIONS.json').exists() else {}
    save('COLLECTOR_PID.json',{'pid':os.getpid(),'start_unix':time.time()})
    receipt=ROOT/'TRANSFER_RECEIPTS.json'
    if receipt.exists():copied=json.loads(receipt.read_text())
    while time.monotonic()-start<12*3600:
        result=subprocess.run([*SSH,HOST,'/srv/scratch/z5611125/venvs/pilot/bin/python -B '+REMOTE+ROOT.name+'/remote_status.py'],capture_output=True,text=True,timeout=120)
        if result.returncode:
            save('MONITOR_NETWORK_ERROR.json',{'returncode':result.returncode,'stderr':result.stderr[-2000:]});time.sleep(30);continue
        tasks=json.loads(result.stdout);save('REMOTE_STATUS.json',{'checked_unix':time.time(),'tasks':tasks})
        for t in tasks:
            key=t['protocol']+'/'+t['model']
            if resume_timed_out_llava_pilot(t):continue
            if t['complete'] and key not in copied:
                copied[key]=transfer(t);save('TRANSFER_RECEIPTS.json',copied)
                print('VALIDATED_COPY',key,flush=True)
            if t['protocol']=='original_views' and not t['complete']:
                if t.get('shards_total') and t.get('shards_complete')==t['shards_total']:
                    cmd='/srv/scratch/z5611125/venvs/pilot/bin/python -B '+REMOTE+ROOT.name+'/merge_shards.py '+t['model']
                    subprocess.run([*SSH,HOST,cmd],check=True,timeout=180)
                    print('SHARDS_MERGED',t['model'],flush=True)
                    continue
                if t['scheduler'].get('Exit_status')=='0' and t.get('budget_gate',{}).get('continue') is False and t['model'] not in submissions:
                    cmd='/srv/scratch/z5611125/venvs/pilot/bin/python -B '+REMOTE+ROOT.name+'/plan_shards.py '+t['model']
                    planned=subprocess.run([*SSH,HOST,cmd],check=True,capture_output=True,text=True,timeout=120)
                    plan=json.loads(planned.stdout);n=len(plan['shards']);assert 1<=n<=450 and plan['job_cap_seconds']==6600
                    (ROOT/('SHARD_PLAN_'+t['model']+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
                    submissions[t['model']]={'status':'submitting','plan':plan};save('SHARD_SUBMISSIONS.json',submissions)
                    cmd='cd '+REMOTE+ROOT.name+' && qsub -J 0-'+str(n-1)+'%'+str(min(2,n))+' -v MODEL_KEY='+t['model']+' native_shards.pbs'
                    submitted=subprocess.run([*SSH,HOST,cmd],check=True,capture_output=True,text=True,timeout=60)
                    job=submitted.stdout.strip();assert re.fullmatch(r'[0-9]+\[\]\.kman\.restech\.unsw\.edu\.au',job),job
                    submissions[t['model']].update(status='submitted',job=job);save('SHARD_SUBMISSIONS.json',submissions)
                    print('BOUNDED_SHARD_ARRAY_SUBMITTED',t['model'],job,n,flush=True)
            if t['scheduler'].get('Exit_status') not in [None,'0'] and not t['complete']:
                save('NEEDS_ATTENTION.json',t)
        for protocol in ['phase_views','original_views']:
            needed={protocol+'/'+m for m in ['internvl14','llava8']}
            needed.add('original_views/qwen8')
            if protocol=='original_views':needed.add('original_views/llava8')
            if needed<=set(copied) and protocol not in running and protocol not in completed:
                subprocess.run([sys.executable,'-B',str(ROOT/'prepare_features.py'),'--protocol',protocol],check=True)
                log=(ROOT/('analysis_'+protocol+'.log')).open('a');logs.append(log)
                running[protocol]=subprocess.Popen([sys.executable,'-B',str(ROOT/'analyze_matched.py'),'--protocol',protocol,'--workers','2'],stdout=log,stderr=subprocess.STDOUT)
                print('CPU_ANALYSIS_STARTED',protocol,flush=True)
        for protocol,process in list(running.items()):
            code=process.poll()
            if code is not None:
                if code:raise RuntimeError(f'CPU analysis failed: {protocol}, exit {code}; inspect analysis log')
                completed.add(protocol);del running[protocol];print('CPU_ANALYSIS_COMPLETE',protocol,flush=True)
        if len(completed)==2:
            ids=[t['job'] for t in tasks]+['9829050.kman.restech.unsw.edu.au','9829051.kman.restech.unsw.edu.au','9829635.kman.restech.unsw.edu.au','9829909.kman.restech.unsw.edu.au']
            if (ROOT/'PILOT_TIMEOUT_RETRIES.json').exists():ids.append(json.loads((ROOT/'PILOT_TIMEOUT_RETRIES.json').read_text())['previous_job'])
            for submission in submissions.values():
                if submission.get('status')=='submitted':ids.extend(submission['job'].replace('[]','['+str(i)+']') for i in range(len(submission['plan']['shards'])))
            cmd='/srv/scratch/z5611125/venvs/pilot/bin/python -B '+REMOTE+ROOT.name+'/collect_accounting.py '+' '.join(shlex.quote(j) for j in ids)
            subprocess.run([*SSH,HOST,cmd],check=True,timeout=120)
            subprocess.run([*SCP,'-r',HOST+':'+REMOTE+ROOT.name+'/accounting',str(ROOT)],check=True,timeout=180)
            for script in ['summarize_matched.py','audit_matched.py','report_crossed.py']:
                subprocess.run([sys.executable,'-B',str(ROOT/script)],check=True)
            save('PIPELINE_COMPLETE.json',{'complete':True,'seconds':time.monotonic()-start,'protocols':2,'splits_each':31})
            print('CROSSED_EXPERIMENT_COMPLETE',flush=True);return
        save('LOCAL_STATUS.json',{'copied':list(copied),'running_cpu':list(running),'completed_cpu':sorted(completed),'elapsed_seconds':time.monotonic()-start})
        time.sleep(45)
    raise TimeoutError('12-hour monitoring window reached; saved work is resumable')
if __name__=='__main__':main()
