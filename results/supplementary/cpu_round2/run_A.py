from common import *
import argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import joblib
from threadpoolctl import threadpool_limits
import evaluate_v2_target_pilot as selector
selector.aurc=aurc

def one(m,split):
    dest=ROOT/'A_runs'/m/f'split_{split:02d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'DONE.json').exists():return json.loads((dest/'DONE.json').read_text())
    frame=pd.read_csv(ROOT/'A_features.csv',float_precision='round_trip');frame=frame[frame.model_id.eq(m)].reset_index(drop=True)
    assignments=pd.read_csv(ROOT/'A_video_assignments.csv');roles=assignments[assignments.split_id.eq(split)].set_index('video_id').role
    frame['role']=frame.group_id.map(roles);pilot=frame[frame.role.eq('calibration')].copy();test=frame[frame.role.eq('test')].copy()
    assert not set(pilot.group_id)&set(test.group_id) and pilot.group_id.nunique()==test.group_id.nunique()==250
    columns=json.loads((ROOT/'A_feature_columns.json').read_text());archive=joblib.load(BUNDLE);assert columns==archive['feature_columns']
    source=archive['source_models']['source_trace_hgb']
    pilot['score__full_graph_gradient_boosting']=source.predict_proba(pilot[columns])[:,1]
    seed=20260803+1000003*split+1009*MODELS.index(m)
    distributions=[];original=selector.grouped_bootstrap_gains
    def record(*args,**kwargs):
        out=original(*args,**kwargs);distributions.append(out);return out
    selector.grouped_bootstrap_gains=record
    with threadpool_limits(limits=1):promoted=selector.promote(pilot,columns,2000,.05,seed,'robust_safe')
    selector.grouped_bootstrap_gains=original
    diagnostics=[]
    for (name,s),dist in zip(sorted(promoted.candidate_scores.items()),distributions):
        lower=float(np.quantile(dist,.025));margin=.5 if name=='source_trace_hgb' else 0
        diagnostics.append(dict(candidate=name,gain=gain(pilot.correct.to_numpy(),pilot.score__raw_confidence.to_numpy(),s),lower_bound=lower,threshold=margin,passed=lower>margin))
    eligible=sorted([r for r in diagnostics if r['passed']],key=lambda r:(-r['lower_bound'],r['candidate']))
    assert promoted.selected==(eligible[0]['candidate'] if eligible else 'raw_fallback')
    score={'original':test.score__raw_confidence.to_numpy(),'pilot_lr':promoted.target_model.predict_proba(test[columns])[:,1],'development_hgb':source.predict_proba(test[columns])[:,1],'base_support':test.base_support.to_numpy(),'all_view_support':test.all_view_support.to_numpy()}
    score['checked']=score[{'raw_fallback':'original','target_logistic':'pilot_lr','source_trace_hgb':'development_hgb'}[promoted.selected]]
    y=test.correct.to_numpy();out=test[['qid','group_id','model_id','correct']].copy();metrics=[]
    for method,s in score.items():
        out[method]=s;metrics.append(dict(model=m,split=split,method=method,accuracy=y.mean(),aurc=aurc(y,s),gain_pct=gain(y,score['original'],s),selected=promoted.selected))
    out.to_csv(dest/'test_predictions.csv',index=False)
    po=pilot[['qid','group_id','correct']].copy()
    for name,s in promoted.candidate_scores.items():po[name]=s
    po.to_csv(dest/'pilot_oof_predictions.csv',index=False)
    np.savez_compressed(dest/'promotion_distributions.npz',**dict(zip(sorted(promoted.candidate_scores),distributions)))
    pd.DataFrame(metrics).to_csv(dest/'metrics.csv',index=False)
    dump(dest/'promotion.json',dict(selected=promoted.selected,candidates=diagnostics,seed=seed,bootstrap=2000,quantile=.025,pilot_videos=sorted(pilot.group_id.unique()),test_videos=sorted(test.group_id.unique())))
    joblib.dump(promoted.target_model,dest/'pilot_lr.joblib')
    done=dict(model=m,split=split,selected=promoted.selected,gain=metrics[-1]['gain_pct']);dump(dest/'DONE.json',done)
    return done

def accuracy():
    table=pd.read_csv(ROOT/'A_answer_accuracy.csv');assign=pd.read_csv(ROOT/'A_video_assignments.csv');methods=MODELS+['majority_vote'];rows=[];pairs=[]
    for split in [-1,*range(31)]:
        keep=set(table.group_id) if split==-1 else set(assign[(assign.split_id==split)&assign.role.eq('test')].video_id)
        frame=table[table.group_id.isin(keep)];groups=sorted(keep)
        counts=frame.groupby('group_id')[methods].sum().reindex(groups).to_numpy();sizes=frame.groupby('group_id').size().reindex(groups).to_numpy()
        seed=20261120+split;rng=np.random.default_rng(seed);draws=rng.integers(0,len(groups),size=(10000,len(groups)),dtype=np.int32)
        boot=counts[draws].sum(axis=1)/sizes[draws].sum(axis=1)[:,None]
        np.savez_compressed(ROOT/f'A_accuracy_bootstrap_{split}.npz',draws=draws,groups=np.array(groups),accuracy=boot)
        for j,m in enumerate(methods):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);rows.append(dict(split=split,model=m,questions=len(frame),videos=len(groups),accuracy=frame[m].mean(),ci_low=lo,ci_high=hi))
        for j,m in enumerate(MODELS):
            dist=100*(boot[:,-1]-boot[:,j]);lo,hi=np.quantile(dist,[.025,.975]);pairs.append(dict(split=split,comparison='vote minus '+m,difference_pp=100*(frame.majority_vote.mean()-frame[m].mean()),ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(ROOT/'A_accuracy_intervals.csv',index=False);pd.DataFrame(pairs).to_csv(ROOT/'A_vote_paired_intervals.csv',index=False)
    print('A accuracy 10000 video bootstraps completed for split 0, all 30 splits, full cohort.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    accuracy()
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(one,m,s) for m in ['internvl14','llava8'] for s in range(1,31)]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();print(i,'/60',r,flush=True)
    frames=[pd.read_csv(p) for p in (ROOT/'A_runs').glob('*/split_*/metrics.csv')]
    allmetrics=pd.concat(frames,ignore_index=True)
    # Label-free comparisons for all three answering models on the same splits.
    full=pd.read_csv(ROOT/'A_features.csv',float_precision='round_trip');assign=pd.read_csv(ROOT/'A_video_assignments.csv');support=[]
    for split in range(1,31):
        testids=set(assign[(assign.split_id==split)&assign.role.eq('test')].video_id)
        for m,g in full[full.group_id.isin(testids)].groupby('model_id',sort=False):
            for method in ['base_support','all_view_support']:
                support.append(dict(split=split,model=m,method=method,gain_pct=gain(g.correct.to_numpy(),g.own_base_confidence.to_numpy(),g[method].to_numpy())))
    pd.DataFrame(support).to_csv(ROOT/'A_support_splits.csv',index=False)
    allmetrics.to_csv(ROOT/'A_selection_splits.csv',index=False)
    summary=allmetrics.groupby(['model','method']).agg(mean_gain=('gain_pct','mean'),min_gain=('gain_pct','min'),harm_splits=('gain_pct',lambda v:int((v<0).sum())),splits=('split','nunique')).reset_index()
    checks=allmetrics[allmetrics.method.eq('checked')].groupby('model').selected.value_counts().unstack(fill_value=0)
    summary.to_csv(ROOT/'A_selection_summary.csv',index=False);checks.to_csv(ROOT/'A_selection_counts.csv')
    support=pd.DataFrame(support);macro=support.groupby(['split','method']).gain_pct.mean().reset_index();macro['model']='mean across three'
    pd.concat([support,macro]).groupby(['model','method']).agg(mean_gain=('gain_pct','mean'),min_gain=('gain_pct','min'),harm_splits=('gain_pct',lambda v:int((v<0).sum()))).to_csv(ROOT/'A_support_summary.csv')
    dump(ROOT/'A_COMPLETE.json',dict(cpu_only=True,runs=60,bootstrap_selection=2000,bootstrap_accuracy=10000))
if __name__=='__main__':main()
