"""All prespecified splits, shared answers, pilot-only fitting and locked controls."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import joblib,numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
sys.path.insert(0,str(REPO/'src'))
import evaluate_v2_target_pilot as ours
from evaluate_probe_escalation import aurc
from analyze_trace_videomme_confirmation import risk_at_coverage
from matched_selector import fit_selector,predict_selector,candidates,cdf
ours.aurc=aurc

def dump(path,data):path.write_text(json.dumps(data,indent=2,default=str)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metrics(y,s):
    return {'accuracy':float(np.mean(y)),'aurc':float(aurc(y,s)),
        'auroc':float(roc_auc_score(y,s)) if len(set(y))==2 else None,
        'risk_at_50pct':float(risk_at_coverage(y,s,.5))}

def fixed_rf(x,y,groups,raw):
    estimator=next(e for n,e in candidates() if n=='RF_leaf10')
    oof=np.full(len(y),np.nan)
    for tr,va in GroupKFold(5).split(x,y,groups):
        assert not set(groups[tr])&set(groups[va])
        oof[va]=clone(estimator).fit(x[tr],y[tr]).predict_proba(x[va])[:,1]
    return {'name':'RF_leaf10','lambda':.95,'estimator':estimator.fit(x,y),'oof':oof,'token_reference':raw}

def run(policy,split):
    dest=ROOT/'analysis'/policy/f'split_{split:02d}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists():return json.loads((dest/'COMPLETE.json').read_text())
    started=time.monotonic();frame=pd.read_csv(ROOT/(policy+'_features.csv'),float_precision='round_trip')
    schema=json.loads((ROOT/'SCHEMAS.json').read_text())[policy]
    assignments=pd.read_csv(ROOT.parent/'P2/video_assignments.csv')
    assignment=assignments[(assignments.dataset=='Perception')&(assignments.split_id==split)]
    assert len(assignment)==assignment.video_id.nunique()==500
    roles=assignment.set_index('video_id').role
    frame['split_role']=frame.group_id.map(roles);assert frame.split_role.notna().all()
    dev=frame[frame.split_role=='calibration'].copy();test=frame[frame.split_role=='test'].copy()
    assert dev.group_id.nunique()==test.group_id.nunique()==250
    assert not set(dev.group_id)&set(test.group_id)
    labels=pd.concat([pd.read_csv(REPO/'input/perception_test/frozen'/n) for n in ['pilot_labels.csv','sealed_test_labels.csv']]).set_index('question_id')
    dev['correct']=(dev.answer_index.to_numpy()==labels.loc[dev.qid].gold_candidate_index.to_numpy()).astype(int)
    # Only pilot labels enter fitting. Previously inspected cohort is disclosed in protocol.
    archived=joblib.load(REPO/'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/pilot_model_bundle.joblib')
    assert archived['feature_columns']==schema['trace']
    source=archived['source_models']['source_trace_hgb']
    dev['score__full_graph_gradient_boosting']=source.predict_proba(dev[schema['trace']])[:,1]
    raw=dev.score__raw_confidence.to_numpy();y=dev.correct.to_numpy();groups=dev.group_id.to_numpy()
    bundles={};scores={'raw':test.score__raw_confidence.to_numpy(),'label_free_support':test.label_free_support.to_numpy(),'vote_support':test.vote_support.to_numpy()}
    scores['base_model_support']=test.base_model_support.to_numpy()
    with threadpool_limits(limits=1):
        promotion=ours.promote(dev,schema['trace'],2000,.05,20260803+1000003*split+1009,'robust_safe')
        scores['ours_unchecked_lr']=promotion.target_model.predict_proba(test[schema['trace']])[:,1]
        scores['frozen_source_hgb']=source.predict_proba(test[schema['trace']])[:,1]
        scores['ours_checked']={'raw_fallback':scores['raw'],'target_logistic':scores['ours_unchecked_lr'],'source_trace_hgb':scores['frozen_source_hgb']}[promotion.selected]
        bundles['ours']=promotion
        checks=[]
        for offset,(name,values) in enumerate(sorted(promotion.candidate_scores.items())):
            dist=ours.grouped_bootstrap_gains(dev,raw,values,2000,20260803+1000003*split+1009+7919*offset)
            margin=.5 if name=='source_trace_hgb' else 0.
            checks.append({'candidate':name,'gain':ours.relative_gain(y,raw,values),'lower':float(np.quantile(dist,.025)),'margin':margin,'pass':bool(np.quantile(dist,.025)>margin)})
        dump(dest/'promotion.json',{'selected':promotion.selected,'candidates':checks})
        for scope in ['single','full']:
            columns=schema['pg_'+scope]
            bundle=fit_selector(dev[columns],y,groups,raw)
            bundles['pg_'+scope]=bundle
            scores['pg_'+scope]=predict_selector(bundle,test[columns],scores['raw'])
            pd.DataFrame(bundle['search']).to_csv(dest/(scope+'_search.csv'),index=False)
        # Single wins an exact OOF AUROC tie, fixed before outcomes.
        selected=max(['single','full'],key=lambda scope:max(r['oof_auroc'] for r in bundles['pg_'+scope]['search']))
        scores['pg_scope_selected']=scores['pg_'+selected]
        published=fixed_rf(dev[schema['pg_full']].to_numpy(),y,groups,raw);bundles['pg_fixed_rf']=published
        scores['pg_fixed_rf']=predict_selector(published,test[schema['pg_full']],scores['raw'])
        _,same_lr=ours.cross_fitted_logistic(dev,schema['pg_full'],5);bundles['pg_same_lr']=same_lr
        scores['pg_same_lr']=same_lr.predict_proba(test[schema['pg_full']])[:,1]
    joblib.dump(bundles,dest/'models.joblib')
    dump(dest/'PRETEST_LOCK.json',{'policy':policy,'split':split,'models_sha256':sha(dest/'models.joblib'),
        'analysis_lock_sha256':sha(ROOT/'ANALYSIS_LOCK.json'),
        'selected_pg_scope':selected,'selected_ours':promotion.selected,'pilot_videos':sorted(set(groups)),
        'test_videos':sorted(set(test.group_id)),'test_labels_used_in_fitting':False,
        'features_sha256':sha(ROOT/(policy+'_features.csv'))})
    test=test[['qid','group_id','task','answer_index']].copy()
    test['correct']=(test.answer_index.to_numpy()==labels.loc[test.qid].gold_candidate_index.to_numpy()).astype(int)
    for name,values in scores.items():test[name]=values
    test.to_csv(dest/'test_predictions.csv',index=False)
    records=[]
    for subset,mask in [('all',np.ones(len(test),bool)),('physics',test.task.astype(str).str.lower().eq('physics').to_numpy())]:
        if not mask.any():continue
        yy=test.correct.to_numpy()[mask];base=aurc(yy,scores['raw'][mask])
        for name,values in scores.items():
            metric=metrics(yy,values[mask]);gain=100*(1-metric['aurc']/base)
            records.append({'policy':policy,'split':split,'subset':subset,'method':name,'questions':int(mask.sum()),**metric,'gain_percent':float(gain),'harm':bool(gain<0)})
    pd.DataFrame(records).to_csv(dest/'metrics.csv',index=False)
    result={'policy':policy,'split':split,'selected_pg_scope':selected,'selected_ours':promotion.selected,'seconds':time.monotonic()-started}
    dump(dest/'COMPLETE.json',result);print(result,flush=True);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--split',type=int);parser.add_argument('--protocol',choices=['original_views','phase_views']);args=parser.parse_args()
    protocols=[args.protocol] if args.protocol else ['original_views','phase_views']
    for p in protocols:assert json.loads((ROOT/(p+'_FEATURE_VALIDATION.json')).read_text())['passed']
    lock=json.loads((ROOT/'ANALYSIS_LOCK.json').read_text())
    for path,expected in lock['checksums'].items():assert sha(REPO/path)==expected,path
    splits=range(31) if args.split is None else [args.split]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run,p,s) for p in protocols for s in splits]
        for f in as_completed(jobs):f.result()
    tables=[pd.read_csv(p) for p in (ROOT/'analysis').glob('*/split_*/metrics.csv')]
    pd.concat(tables,ignore_index=True).to_csv(ROOT/'matched_metrics.csv',index=False)
if __name__=='__main__':main()
