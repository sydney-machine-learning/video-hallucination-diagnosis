"""Two observation protocols, one common answer and reference confidence."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from matched_features import ROOT,graph_table,pg_features
from native_sampling import indices as native_indices,FRAME_COUNTS

MODELS=['internvl14','qwen8','llava8']
ROLES=['base','invariant','addition','removal']
POLICIES=['original_views','phase_views']
def original_bank(model):
    merged=ROOT/('merged_responses_'+model)
    return merged if (merged/'COMPLETE.json').exists() else ROOT/('responses_'+model)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def bank(path,manifest,native):
    complete=json.loads((path/'COMPLETE.json').read_text());assert complete['complete']
    assert sha(path/'responses.jsonl')==complete['responses_sha256']
    raw=rows(path/'responses.jsonl');result={(r['question_id'],r['phase']):r for r in raw}
    assert len(raw)==len(result)==7552
    refpath=ROOT.parent/'P3B_full_reference_20261119/reference_full'
    ref={r['video_id']:r for r in rows(refpath/'decode.jsonl')}
    decode=rows(path/'decode.jsonl');assert {r['video_id'] for r in decode}==set(ref)
    for d in decode:assert all(d[k]==ref[d['video_id']][k] for k in ['sha256','duration','frame_count','fps'])
    for q in manifest:
        d=ref[q['video_id']]
        for p in (0,2,4,6):
            r=result[(q['question_id'],p)];assert r['video_id']==q['video_id']
            prob=np.array(r['candidate_probabilities']);assert len(prob)==len(q['candidates']) and np.isfinite(prob).all() and (prob>=0).all() and np.isclose(prob.sum(),1)
            if native:assert r['frame_indices']==native_indices(q,d['frame_count'],d['duration'],p)
            else:
                first=max(0,int(np.ceil(float(q['start'])*d['frame_count']/d['duration']-.5)))
                last=min(d['frame_count']-1,int(np.floor(min(float(q['end']),d['duration'])*d['frame_count']/d['duration']-.5)))
                size=last-first+1
                expected=[first+min(size-1,int(np.floor(size*(i+(p+.5)/8)/2))) for i in range(2)]
                assert r['frame_indices']==expected
    return result,{'responses_sha256':sha(path/'responses.jsonl'),'decode_sha256':sha(path/'decode.jsonl'),'directory':str(path)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',choices=POLICIES);args=parser.parse_args()
    manifest=rows(ROOT/'full_manifest.jsonl');assert len(manifest)==1888
    qwen,common_receipt=bank(original_bank('qwen8'),manifest,True)
    common={q['question_id']:np.asarray(qwen[(q['question_id'],0)]['candidate_probabilities']) for q in manifest}
    common_answers={q:int(p.argmax()) for q,p in common.items()}
    pd.DataFrame([{'qid':q['question_id'],'group_id':q['video_id'],'answer_index':common_answers[q['question_id']],'confidence':float(common[q['question_id']].max())} for q in manifest]).to_csv(ROOT/'COMMON_ANSWERS.csv',index=False)
    schemas=json.loads((ROOT/'SCHEMAS.json').read_text()) if (ROOT/'SCHEMAS.json').exists() else {}
    for policy in [args.protocol] if args.protocol else POLICIES:
        native=policy=='original_views';banks={};receipts={}
        for model in MODELS:
            path=(original_bank(model) if native else ROOT.parent/('P3B_full_reference_20261119/reference_full' if model=='qwen8' else 'P3B_original_panel_20261119/responses_'+model))
            banks[model],receipts[model]=bank(path,manifest,native)
        values={};metadata=[];data=[];targets={}
        for q in manifest:
            qid=q['question_id'];target=common_answers[qid];label=chr(65+target)
            metadata.append({'dataset':'Perception','qid':qid,'group_id':q['video_id'],'gold':-1,'candidate_count':len(q['candidates'])})
            traces={m:[banks[m][(qid,p)] for p in (0,2,4,6)] for m in MODELS}
            for m in MODELS:
                for r,role in zip(traces[m],ROLES):values[('Perception',m,qid,role)]=np.array(r['candidate_probabilities'])
            targets[('Perception',qid,'qwen8')]=target
            answers=[r['answer'] for r in traces['qwen8']];prob=[r['candidate_probabilities'] for r in traces['qwen8']]
            peers=[(m,[r['answer'] for r in traces[m]],[r['candidate_probabilities'] for r in traces[m]]) for m in ['llava8','internvl14']]
            single,full=pg_features(q['question'],q['candidates'],answers,prob,label,peers)
            all_records=[r for m in MODELS for r in traces[m]]
            data.append({'qid':qid,'group_id':q['video_id'],'role':q['role'],'task':q.get('task',''),'answer_index':target,
                'score__raw_confidence':float(common[qid][target]),
                'base_model_support':float(np.mean([traces[m][0]['candidate_probabilities'][target] for m in MODELS])),
                'label_free_support':float(np.mean([r['candidate_probabilities'][target] for r in all_records])),
                'vote_support':float(np.mean([r['answer']==label for r in all_records])),**{'pg__'+k:v for k,v in full.items()}})
        graph,_=graph_table(pd.DataFrame(metadata),values,MODELS,targets)
        graph=graph[graph.model_id=='qwen8'].drop(columns=['dataset','group_id','model_id','correct'])
        frame=pd.DataFrame(data).merge(graph,on='qid',validate='one_to_one',sort=False)
        assert frame.qid.tolist()==[q['question_id'] for q in manifest]
        schemas[policy]={'pg_single':['pg__'+k for k in single],'pg_full':['pg__'+k for k in full],'trace':[c for c in graph if c!='qid']}
        assert len(schemas[policy]['trace'])==53
        frame.to_csv(ROOT/(policy+'_features.csv'),index=False)
        (ROOT/(policy+'_FEATURE_VALIDATION.json')).write_text(json.dumps({'passed':True,'labels_read':False,'common_answer_source':common_receipt,'receipts':receipts,'features_sha256':sha(ROOT/(policy+'_features.csv'))},indent=2)+'\n')
    (ROOT/'SCHEMAS.json').write_text(json.dumps(schemas,indent=2)+'\n')
    if all((ROOT/(p+'_FEATURE_VALIDATION.json')).exists() for p in POLICIES):
        a,b=[pd.read_csv(ROOT/(p+'_features.csv'),float_precision='round_trip') for p in POLICIES]
        pd.testing.assert_frame_equal(a[['qid','group_id','answer_index','score__raw_confidence']],b[['qid','group_id','answer_index','score__raw_confidence']])
        (ROOT/'FEATURE_VALIDATION.json').write_text(json.dumps({'passed':True,'same_answers_across_protocols':True,'same_reference_confidence':True,'labels_read':False})+'\n')
if __name__=='__main__':main()
