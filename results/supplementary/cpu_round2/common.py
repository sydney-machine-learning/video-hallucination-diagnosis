import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
OLD=ROOT.parent/'P3B_view_crossed_20261119'
VHD=ROOT.parent/'P5_vhd_transfer_20261119'
sys.path.insert(0,str(REPO/'src'))
import evaluate_response_graph_features as graph
from evaluate_probe_escalation import aurc
MODELS=['internvl14','qwen8','llava8']
MODEL_NAMES={'internvl14':'InternVL3.5 14B Instruct','qwen8':'Qwen3-VL 8B Instruct','llava8':'LLaVA OneVision 2 8B Instruct'}
ASSIGN=ROOT.parent/'P2/video_assignments.csv'
BUNDLE=REPO/'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/pilot_model_bundle.joblib'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')
def readjsonl(path):return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]
def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def gain(y,raw,s):
    base=aurc(y,raw)
    return 100*(base-aurc(y,s))/base if base>0 else 0.

def prepare():
    paths=[ASSIGN,BUNDLE,OLD/'full_manifest.jsonl',OLD/'original_views_features.csv',OLD/'SCHEMAS.json',VHD/'predictions.csv',VHD/'features.csv',VHD/'vhd_question_subsets.csv',VHD/'holdout_audit.json',VHD/'models/VHD.joblib']
    paths += [OLD/f'merged_responses_{m}/responses.jsonl' for m in MODELS]
    paths += [REPO/'input/perception_test/frozen'/n for n in ['pilot_labels.csv','sealed_test_labels.csv']]
    missing=[str(p) for p in paths if not p.exists()]
    if missing:
        dump(ROOT/'MISSING.json',dict(missing=missing,searched=[str(p) for p in paths]));raise RuntimeError('Missing required records. Stop.')
    dump(ROOT/'INPUT_MANIFEST.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in paths])
    manifest=readjsonl(OLD/'full_manifest.jsonl');qids=[q['question_id'] for q in manifest]
    labels=pd.concat([pd.read_csv(REPO/'input/perception_test/frozen'/n) for n in ['pilot_labels.csv','sealed_test_labels.csv']]).set_index('question_id')
    assert len(qids)==len(set(qids))==1888
    assert set(labels.index)==set(qids)
    assignments=pd.read_csv(ASSIGN);assignments=assignments[assignments.dataset.eq('Perception')]
    for s,g in assignments.groupby('split_id'):
        assert len(g)==g.video_id.nunique()==500 and g.groupby('role').size().to_dict()=={'calibration':250,'test':250}
        assert set(g.video_id)=={q['video_id'] for q in manifest}
    assert set(assignments.split_id)==set(range(31))
    assignments.to_csv(ROOT/'A_video_assignments.csv',index=False)
    banks={};values={};audits=[]
    for m in MODELS:
        path=OLD/f'merged_responses_{m}/responses.jsonl';rows=readjsonl(path)
        completion=json.loads(path.with_name('COMPLETE.json').read_text())
        assert sha(path)==completion['responses_sha256']
        bank={(r['question_id'],r['phase']):r for r in rows}
        assert len(rows)==len(bank)==7552
        answer_mismatch=0
        for q in manifest:
            for phase,role,budget in [(0,'base',16),(2,'invariant',8),(4,'addition',24),(6,'removal',8)]:
                row=bank[(q['question_id'],phase)];prob=np.array(row['candidate_probabilities'])
                assert row['video_id']==q['video_id'] and row['frame_budget']==budget
                assert prob.shape==(3,) and np.isfinite(prob).all() and np.all(prob>=0) and np.isclose(prob.sum(),1)
                values[('Perception',m,q['question_id'],role)]=prob
                answer_mismatch += row['answer']!=chr(65+int(prob.argmax()))
        audits.append(dict(model=m,rows=len(rows),questions=1888,views=4,answer_vs_argmax_mismatch=answer_mismatch,model_ids=sorted({r['model_id'] for r in rows}),revisions=sorted({r['model_revision'] for r in rows})))
        banks[m]=bank
    metadata=pd.DataFrame([dict(dataset='Perception',qid=q['question_id'],group_id=q['video_id'],gold=int(labels.loc[q['question_id'],'gold_candidate_index']),candidate_count=3) for q in manifest])
    table,_=graph.graph_table(metadata,values,MODELS)
    features=[c for c in table if c not in ['dataset','qid','group_id','model_id','correct']]
    schema=json.loads((OLD/'SCHEMAS.json').read_text())['original_views']['trace'];assert schema==features
    old=pd.read_csv(OLD/'original_views_features.csv',float_precision='round_trip');qwen=table[table.model_id.eq('qwen8')]
    assert old.qid.tolist()==qwen.qid.tolist()
    assert np.allclose(old[features],qwen[features],atol=1e-14,rtol=1e-14)
    table['score__raw_confidence']=table.own_base_confidence
    table['base_support']=table.base_selected_support_mean
    table['all_view_support']=table[[f'{r}_selected_support_mean' for r in ['base','invariant','addition','removal']]].mean(axis=1)
    table.to_csv(ROOT/'A_features.csv',index=False)
    dump(ROOT/'A_feature_columns.json',features)
    ensemble=[]
    for q in manifest:
        qid=q['question_id'];ps=np.stack([values[('Perception',m,qid,'base')] for m in MODELS]);ans=ps.argmax(axis=1)
        votes=np.bincount(ans,minlength=3);tied=np.flatnonzero(votes==votes.max());avg=ps.mean(axis=0)
        winner=int(tied[np.argmax(avg[tied])]);gold=int(labels.loc[qid,'gold_candidate_index'])
        ensemble.append(dict(qid=qid,group_id=q['video_id'],gold=gold,majority_vote_answer=winner,majority_vote=int(winner==gold),vote_tie=len(tied)>1,remaining_probability_tie=int(np.sum(avg[tied]==avg[tied].max()))>1,**{m:int(ans[i]==gold) for i,m in enumerate(MODELS)},**{m+'_answer':int(ans[i]) for i,m in enumerate(MODELS)},**{f'mean_probability_{i}':avg[i] for i in range(3)}))
    pd.DataFrame(ensemble).to_csv(ROOT/'A_answer_accuracy.csv',index=False)
    dump(ROOT/'PREFLIGHT.json',dict(status='passed',models=audits,split_ids=list(range(1,31)),split_zero='reference only',total_raw_records=22656,qwen_feature_match=True,source_results_unchanged=True))
    dump(ROOT/'ANALYSIS_PLAN.json',dict(gpu='none',A=dict(answers='base option argmax; majority; ties use largest mean option probability; exact remaining tie first option',primary_accuracy='reference split 0 test; all 30 test splits also reported; whole 500-video cohort descriptive',splits=list(range(1,31)),promotion='existing promote robust_safe; prefix mean AURC patched in memory only',bootstrap=2000,quantile=.025,thresholds={'target_logistic':0,'source_trace_hgb':.5},seed='20260803 + 1000003*split +1009*model_index; candidate offset 7919',accuracy_bootstrap=10000,accuracy_seed='20261120 + split; all cohort seed 20261119',support='mean probability assigned to own fixed answer, across 3 bases or all 12 nodes'),B=dict(predictions='Existing P5 full-VHD holdout LR/HGB reused',bootstrap=10000,seed=20261119,resamples='Reuse P5 exact saved source-video draws',vote='base_vote_fraction; also report archived confidence tie-break as sensitivity',percentile='100*midrank/N within each model and subset; high percentile means trusted; joint wrong defined from same base bank'),C='read-only source inspection; no experimental scripts run',D='primary literature and model documents only; comparisons descriptive'))
    print('PREFLIGHT PASSED: 22656 raw rows, 1888 questions, 500 videos, 30 saved splits, VHD holdout records.',flush=True)
if __name__=='__main__':prepare()
