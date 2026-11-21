"""Validate native-view records without labels; estimate time from the pilot."""
import argparse,json,math
from pathlib import Path
from native_sampling import indices,FRAME_COUNTS,VIEW_IDS
from run_peer import ROOT,digest,parse_answer

def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--pilot',action='store_true');p.add_argument('--directory',type=Path);args=p.parse_args()
    out=args.directory or ROOT/('responses_'+args.model);assert out.resolve().is_relative_to(ROOT.resolve());name='PILOT_COMPLETE.json' if args.pilot else 'COMPLETE.json'
    summary=json.loads((out/name).read_text());assert summary['complete']
    assert digest(out/'responses.jsonl')==summary['responses_sha256']
    manifest=rows(ROOT/'full_manifest.jsonl')
    allowed=set(json.loads((ROOT/'pilot_video_ids.json').read_text())) if args.pilot else {r['video_id'] for r in manifest}
    questions={r['question_id']:r for r in manifest if r['video_id'] in allowed}
    raw=rows(out/'responses.jsonl');assert len(raw)==4*len(questions)
    assert {(r['question_id'],r['phase']) for r in raw}=={(q,p) for q in questions for p in FRAME_COUNTS}
    decode=rows(out/'decode.jsonl');assert {r['video_id'] for r in decode}==allowed
    ref={r['video_id']:r for r in rows(ROOT.parent/'P3B_full_reference_20261119/reference_full/decode.jsonl')}
    for r in decode:assert all(r[k]==ref[r['video_id']][k] for k in ['sha256','frame_count','duration','fps'])
    meta=json.loads((ROOT/(args.model+'_MODEL_READY.json')).read_text())
    for r in raw:
        item=questions[r['question_id']];d=ref[item['video_id']];p=r['phase'];n=FRAME_COUNTS[p]
        assert r['model_revision']==meta['revision'] and r['model_id']==meta['model_id']
        assert r['video_id']==item['video_id'] and r['role']==item['role']
        assert r['view_id']==VIEW_IDS[p] and r['frame_budget']==n
        assert r['frame_indices']==indices(item,d['frame_count'],d['duration'],p)
        prob=r['candidate_probabilities'];assert len(prob)==len(item['candidates'])
        assert all(math.isfinite(v) and v>=0 for v in prob) and abs(sum(prob)-1)<1e-6
        assert parse_answer(r['generated_text'],list('ABCDE'[:len(prob)]))==r['answer']
        if args.model=='internvl14':assert r['input_shapes']['pixel_values']==[n,3,448,448]
        elif args.model=='llava8':assert r['input_shapes']['image_grid_thw']==[n,3]
        else:assert r['video_grid'][0][0]*2==n
    inference=sum(r['reference_seconds']+r['preprocess_seconds'] for r in raw)
    decoding=sum(r['seconds'] for r in decode)
    estimate=inference*1888/len(questions)+decoding*500/len(allowed)+summary['model_load_seconds']
    snapshot=ROOT/('pilot_snapshot_'+args.model)/'responses.jsonl'
    if snapshot.exists():
        current={(r['question_id'],r['phase']):r for r in raw}
        assert all(current[(r['question_id'],r['phase'])]==r for r in rows(snapshot))
    result={'passed':True,'labels_read':False,'model':meta['model_id'],'videos':len(allowed),'questions':len(questions),'calls':len(raw),'responses_sha256':digest(out/'responses.jsonl'),'full_500_video_estimate_seconds':estimate,'remaining_estimate_seconds':estimate-inference-decoding,'peak_memory_bytes':summary['peak_memory_bytes'],'exact_archived_sampling':True}
    (ROOT/(args.model+('_PILOT_VALIDATION.json' if args.pilot else '_FULL_VALIDATION.json'))).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
