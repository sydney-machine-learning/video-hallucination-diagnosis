"""Prepare outcome-blind, whole-video splits. No fits or performance estimates."""
import hashlib,json,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[2]
SPECS={'HERBench':('herbench/herbench_lite_v2_inference.jsonl',20260802),
       'Perception':('perception_test/frozen/perception_test_valid500_inference.jsonl',20260803)}
rows=[];validation=[];inputs={}
for name,(relative,base_seed) in SPECS.items():
    path=ROOT/'input'/relative
    inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    frame=pd.read_json(path,lines=True)
    assert frame.groupby('video_id').role.nunique().max()==1
    columns=['video_id','role']+(['source_dataset'] if name=='HERBench' else [])
    videos=frame[columns].drop_duplicates('video_id').sort_values('video_id').reset_index(drop=True)
    n=len(videos); seen=set()
    for split in range(31):
        seed=base_seed+1000003*split
        if split==0:
            pilot=set(videos.loc[videos.role.eq('calibration'),'video_id'])
        else:
            rng=np.random.default_rng(seed);pilot=set()
            if name=='HERBench':
                strata={str(k):sorted(g.video_id) for k,g in videos.groupby('source_dataset',sort=True)}
                odd=[k for k,ids in strata.items() if len(ids)%2]
                deficit=n//2-sum(len(ids)//2 for ids in strata.values())
                ceil=set(rng.choice(odd,size=deficit,replace=False))
                for k,ids in strata.items():
                    pilot.update(rng.permutation(ids)[:len(ids)//2+(k in ceil)])
            else:pilot=set(rng.permutation(videos.video_id.to_numpy())[:n//2])
        assert len(pilot)==n//2
        fingerprint=hashlib.sha256('\n'.join(sorted(pilot)).encode()).hexdigest()
        assert fingerprint not in seen,'Unexpected duplicate split; do not silently reroll'
        seen.add(fingerprint)
        for r in videos.to_dict('records'):
            rows.append({'dataset':name,'split_id':split,'seed':seed if split else None,'video_id':r['video_id'],
                         'role':'calibration' if r['video_id'] in pilot else 'test',
                         'source_collection':r.get('source_dataset','Perception Test')})
        for source,g in videos.groupby('source_dataset' if name=='HERBench' else lambda i:'Perception Test'):
            p=sum(v in pilot for v in g.video_id);t=len(g)-p
            assert abs(p-t)<=1
            validation.append({'dataset':name,'split_id':split,'source':source,'pilot_videos':p,'test_videos':t,
                               'pilot_questions':int(frame.video_id.isin(pilot).sum()),
                               'test_questions':int((~frame.video_id.isin(pilot)).sum()),'no_video_leakage':True})
pd.DataFrame(rows).to_csv(OUT/'video_assignments.csv',index=False,mode='x')
pd.DataFrame(validation).to_csv(OUT/'split_checks.csv',index=False,mode='x')
report='''# P2 preparation

Prepared split 0 and 30 additional 50/50 video splits per target. Split 0 exactly
retains the original manifest roles and must retain the original model choices.
New HERBench splits are balanced within source collection, with floor/ceiling
allocation randomized for odd-sized collections and exactly half the videos in
the pilot. Perception splits select half the videos. All questions inherit their
video assignment. No labels were read during split generation. No duplicate
pilot subsets or video leakage were found.

New split seed = target base seed + 1000003 * split_id. Base seeds are 20260802
and 20260803. These are new predetermined split seeds, not modifications of an
archived run. No fits, retuning, bootstrap evaluations, or gain distributions have
run. Execution needs the missing target probability archives (or verified full
features) and the compatible estimator environment. The canonical metric is
../P0/canonical_metric.py. Current output is preparation only.

No performance table or LaTeX result snippet is available before execution.
'''
(OUT/'REPORT.md').write_text(report,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={'phase':'P2','status':'splits_prepared_execution_blocked_missing_target_responses',
 'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),'utc':datetime.now(timezone.utc).isoformat(),
 'seeds':{'base':{k:v[1] for k,v in SPECS.items()},'formula':'base + 1000003 * split_id'},
 'gpu':{'used':False},'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()},
 'input_sha256':inputs,'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'}}
with (OUT/'MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
print('Prepared 31 video splits per target; no model evaluation executed.')
