"""Read Katana experiment logs/checkpoints and P1 asset readiness, no jobs."""
import json,subprocess,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'P0'))
from collect_remote import SSH
CODE=r'''
import csv,hashlib,json,os,re
from pathlib import Path
root=Path('/home/z5611125/trace_rc_route')
out={'log_hits':[],'candidate_tables':[],'checkpoint_files':[],'assets':{},'errors':[]}
for p in root.rglob('*'):
 if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
 low=p.name.lower()
 if ('checkpoint' in low or 'state' in low) and p.suffix in {'.json','.jsonl','.csv','.npz','.pkl','.joblib'}:
  out['checkpoint_files'].append({'path':str(p),'bytes':p.stat().st_size})
 if (re.search(r'\.o\d+$',low) or p.suffix in {'.log','.out'} or 'EXECUTION_LOG' in p.name) and p.stat().st_size<10000000:
  lines=p.read_text(errors='replace').splitlines()
  matches=[]
  for i,line in enumerate(lines):
   if any(w in line.lower() for w in ('pilot','candidate_probabilities','candidate_scores','oof','features.csv','audited_raw','trapezoid','prefix','archive','checkpoint')):
    matches.append({'line':i+1,'text':line[:1800]})
  out['log_hits'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'total_matching_lines':len(matches),'matches':matches})
 if p.suffix=='.csv' and any(w in low for w in ('pilot','split_model','feature')):
  with p.open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f); rows=list(reader)
  out['candidate_tables'].append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'columns':reader.fieldnames,'rows':len(rows),'first_rows':rows[:2],
   'datasets':sorted({r.get('dataset',r.get('held_out','')) for r in rows})})
for label,manifest in [('HERBench',root/'input/herbench_lite_v2_inference.jsonl'),('Perception',root/'input/perception_test/frozen/perception_test_valid500_inference.jsonl')]:
 rows=[json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
 paths=sorted({r['video_path'] for r in rows})
 out['assets'][label]={'manifest':str(manifest),'videos_referenced':len(paths),'existing_videos':sum(Path(p).is_file() for p in paths),'missing_examples':[p for p in paths if not Path(p).is_file()][:3]}
for p in ['/srv/scratch/z5611125/sicr/envs/qwen-smoke/bin/python','/srv/scratch/z5611125/sicr/envs/llava-ov2/bin/python','/srv/scratch/z5611125/sicr/huggingface/hub','/home/z5611125/.cache/huggingface/hub']:
 out['assets'][p]={'exists':Path(p).exists()}
print(json.dumps(out,indent=2))
'''
run=subprocess.run(SSH+['python3 -'],input=CODE.encode(),capture_output=True,timeout=120)
assert run.returncode==0,run.stderr.decode()
with (OUT/'katana_logs_and_readiness.json').open('xb') as f:f.write(run.stdout)
d=json.loads(run.stdout)
print(json.dumps({'log_files':len(d['log_hits']),'candidate_tables':len(d['candidate_tables']),'checkpoints':d['checkpoint_files'],'assets':d['assets']},indent=2))
