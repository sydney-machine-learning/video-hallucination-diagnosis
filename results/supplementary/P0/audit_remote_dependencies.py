"""Inventory remote development inputs and recover provenance by checksum."""
import json
import subprocess
from collect_remote import SSH,OUT

inventory=json.loads((OUT/"remote_inventory.json").read_text())
paths=[r["path"] for r in inventory["files"] if
       ("/sicr_route/" in r["path"] or "/caver_route/" in r["path"] or "/probe_route/input/development_scored/" in r["path"])
       and "vhd_three_model" not in r["path"] and "mvbench_test_predictions" not in r["path"]]
code=r'''
import csv,hashlib,json,collections
from pathlib import Path
paths=PATHS
out=[]
for name in paths:
 p=Path(name); h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 row={'path':name,'sha256':h.hexdigest(),'bytes':p.stat().st_size}
 if p.suffix=='.csv':
  with p.open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f); row['columns']=reader.fieldnames; row['rows']=sum(1 for r in reader)
 else:
  views=collections.Counter(); states=collections.Counter(); qids=set(); keys=set(); dup=0; count=0; models=set(); datasets=set()
  with p.open(encoding='utf-8') as f:
   for line in f:
    if not line.strip():continue
    r=json.loads(line); count+=1; states[str(r.get('status'))]+=1
    q=r.get('question_id',r.get('pilot_id')); qids.add(str(q)); models.add(str(r.get('model_id'))); datasets.add(str(r.get('dataset')))
    key=(str(q),str(r.get('model_id')),str(r.get('view_id')))
    dup+=key in keys;keys.add(key)
    views[(str(r.get('view_id')),str(r.get('frame_budget')),str(r.get('input_visual_frames')))]+=1
  row.update(rows=count,questions=len(qids),duplicate_question_model_view_keys=dup,status_counts=dict(states),models=sorted(models),datasets=sorted(datasets),
             views=[{'view':k[0],'frame_budget':k[1],'input_visual_frames':k[2],'rows':v} for k,v in views.items()])
 out.append(row)
print(json.dumps(out,indent=2))
'''.replace('PATHS',repr(paths))
run=subprocess.run(SSH+["python3 -"],input=code.encode(),capture_output=True,timeout=180)
assert run.returncode==0,run.stderr.decode()
with (OUT/"remote_dependency_audit.json").open("xb") as f:f.write(run.stdout)
print("Audited remote dependencies:",len(json.loads(run.stdout)))
