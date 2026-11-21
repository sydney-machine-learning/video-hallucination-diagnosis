import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--protocol',default='RUN_PROTOCOL.json');p.add_argument('--runner',default='run_peer.py');args=p.parse_args()
protocol=json.loads((ROOT/args.protocol).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'full_manifest.jsonl')==protocol['manifest_sha256']
assert sha(ROOT/args.runner)==protocol['runner_sha256']
rows=[json.loads(s) for s in (ROOT/'full_manifest.jsonl').read_text().splitlines()]
assert len(rows)==1888 and len({r['video_id'] for r in rows})==500
assert all(not ({'answer','answer_id','gold','gold_candidate_index'}&r.keys()) for r in rows)
assert len({r['question_id'] for r in rows})==1888
pilot=json.loads((ROOT/'pilot_video_ids.json').read_text());assert len(pilot)==len(set(pilot))==50
assert set(pilot)<={r['video_id'] for r in rows}
paths={r['video_path'] for r in rows};assert all(Path(p).is_file() and Path(p).stat().st_size>0 for p in paths)
for name in [args.runner,'prepare_models.py','archived_internvl_adapter.py','tokenizer_guard.py']:
 compile((ROOT/name).read_text(),name,'exec')
receipt={'passed':True,'questions':1888,'videos':500,'pilot_videos':50,'labels_absent':True,'video_paths_exist':True,'runner':args.runner,'runner_sha256':sha(ROOT/args.runner),'manifest_sha256':sha(ROOT/'full_manifest.jsonl')}
(ROOT/'ASSET_PREFLIGHT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
