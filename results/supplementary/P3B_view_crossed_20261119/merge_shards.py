"""New merged bank; preserve every pilot and shard source file unchanged."""
import argparse,json,hashlib,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args();model=args.model
    plan=json.loads((ROOT/('SHARD_PLAN_'+model+'.json')).read_text())
    sources=[ROOT/('pilot_snapshot_'+model)]+[ROOT/'shards'/model/f'shard_{s["index"]:02d}' for s in plan['shards']]
    records={};decode=[];receipts=[];load_seconds=0.
    for i,source in enumerate(sources):
        summary=json.loads((source/('PILOT_COMPLETE.json' if i==0 else 'COMPLETE.json')).read_text());assert summary['complete']
        load_seconds+=summary['model_load_seconds']
        assert sha(source/'responses.jsonl')==summary['responses_sha256']
        raw=rows(source/'responses.jsonl')
        if i:
            expected=set(plan['shards'][i-1]['video_ids']);assert {r['video_id'] for r in raw}==expected
        for r in raw:
            key=(r['question_id'],r['phase']);assert key not in records;records[key]=r
        decode.extend(rows(source/'decode.jsonl'))
        receipts.append({'source':str(source),'response_sha256':sha(source/'responses.jsonl'),'decode_sha256':sha(source/'decode.jsonl')})
    manifest=rows(ROOT/'full_manifest.jsonl');ordered=[records[(q['question_id'],p)] for q in manifest for p in (0,2,4,6)]
    assert len(records)==len(ordered)==7552 and len({r['video_id'] for r in ordered})==500
    destination=ROOT/('merged_responses_'+model);destination.mkdir(exist_ok=False)
    for name,data in [('responses.jsonl',ordered),('decode.jsonl',decode)]:
        (destination/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in data))
    for i,source in enumerate(sources):
        for pattern in ['environment.*.json','model_manifest.*.json','weight_loading.*.json','tokenizer_validation.*.json']:
            for f in source.glob(pattern):shutil.copyfile(f,destination/f.name)
    complete={'complete':True,'videos':500,'questions':1888,'calls':7552,'responses_sha256':sha(destination/'responses.jsonl'),'test_labels_read':False,'model_load_seconds':load_seconds,'peak_memory_bytes':max(r['peak_memory_bytes'] for r in ordered),'source_records_preserved':True,'merge_order':'fixed full_manifest question order then view slots 0,2,4,6','sources':receipts}
    (destination/'COMPLETE.json').write_text(json.dumps(complete,indent=2)+'\n')
    subprocess.run([sys.executable,'-B',str(ROOT/'validate_native.py'),model,'--directory',str(destination)],check=True)
    print('MERGED_VALIDATED',model,flush=True)
if __name__=='__main__':main()
