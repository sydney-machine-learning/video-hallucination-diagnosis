"""Freeze disjoint remaining-video shards from timing, never correctness."""
import argparse,json,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    target=ROOT/('SHARD_PLAN_'+args.model+'.json')
    if target.exists():print(target.read_text());return
    gate=json.loads((ROOT/(args.model+'_PILOT_VALIDATION.json')).read_text());assert gate['passed']
    rows=[json.loads(s) for s in (ROOT/'full_manifest.jsonl').read_text().splitlines()]
    pilot=set(json.loads((ROOT/'pilot_video_ids.json').read_text()))
    counts=Counter(r['video_id'] for r in rows if r['video_id'] not in pilot)
    assert len(counts)==450
    # Reserve 15 minutes per job for CPU checks, pinned downloads and model load.
    count=max(1,math.ceil(1.5*gate['remaining_estimate_seconds']/(6600-900)))
    count=min(count,len(counts));bins=[[] for _ in range(count)];loads=[0]*count
    for video,n in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        index=min(range(count),key=lambda k:(loads[k],k));bins[index].append(video);loads[index]+=n
    plan={'model':args.model,'labels_used':False,'seed':'deterministic question-count balancing, lexical ties','pilot_validation_sha256':hashlib.sha256((ROOT/(args.model+'_PILOT_VALIDATION.json')).read_bytes()).hexdigest(),'job_cap_seconds':6600,'fixed_overhead_allowance_seconds':900,'safety_multiplier':1.5,'concurrency_limit':2,'shards':[{'index':i,'video_ids':sorted(v),'questions':loads[i]} for i,v in enumerate(bins)]}
    assert set(sum(bins,[]))==set(counts) and sum(map(len,bins))==450
    target.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
if __name__=='__main__':main()
