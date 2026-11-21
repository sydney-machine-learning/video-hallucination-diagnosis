"""CPU reconstruction of fixed-panel development holdouts; no VLM inference."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse, collections, hashlib, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'src'))
import evaluate_response_graph_features as graph
from evaluate_probe_escalation import aurc

META = ['dataset', 'qid', 'group_id', 'model_id', 'correct']
FIT_SEED = graph.SEED
BOOT_SEED = 20261119
N_BOOT = 10000
METHODS = ['raw_confidence', 'graph_lr', 'graph_hgb', 'lr_without_three_id', 'hgb_without_three_id']
PRIMARY = Path('D:/paper 3/caver_route/results/vhd_mechanisms/vhd_primary_predictions.csv')

def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def input_paths():
    sicr = Path('D:/paper 3/sicr_route')
    fresh = sicr / 'results_fresh/download_2026-07-31/8900849.kman.restech.unsw.edu.au'
    mv = Path('D:/paper 3/caver_route/results_fresh/mvbench_inference/8904762.kman.restech.unsw.edu.au')
    raw = [sicr/'results/katana_phase2/qwen3_vl_4b_predictions.jsonl',
           sicr/'results/katana_phase2/internvl3_5_4b_predictions.jsonl',
           fresh/'nextgqa_llava/llava_ov2_8b_predictions.jsonl']
    raw += [fresh/'star_three_model'/f'{m}_predictions.jsonl' for m in ['qwen','internvl','llava']]
    raw += [mv/f'{m}_predictions.jsonl' for m in ['qwen','internvl','llava']]
    raw += [HERE/'inputs'/f'{d}_{m}_predictions.jsonl' for d in ['vhd','tempcompass'] for m in ['qwen','internvl','llava']]
    features = [sicr/'results_fresh/download_2026-07-31/8901064.kman.restech.unsw.edu.au/multidataset_features.csv',
                Path('D:/paper 3/caver_route/results_fresh/mvbench_confirmation/8904763.kman.restech.unsw.edu.au/mvbench_caver_features.csv')]
    return raw, features

def build():
    raw, features = input_paths()
    vhd = HERE/'inputs/vhd_vhd_scored_predictions.csv'
    temp = HERE/'inputs/tempcompass_tempcompass_scored_predictions.csv'
    sources = []
    view_rows = []
    for path in raw + features + [vhd,temp,PRIMARY]:
        record = {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
        if path.suffix == '.jsonl':
            rows = [json.loads(l) for l in path.open(encoding='utf-8') if l.strip()]
            record['rows'] = len(rows)
            counts = collections.Counter((r['dataset'],r['model_id'],r['view_id'],str(r.get('frame_budget')),str(r.get('model_revision'))) for r in rows)
            for (d,m,v,b,rev),n in counts.items():
                view_rows.append(dict(dataset=d,model_id=m,view=v,frame_budget=b,revision=rev,n=n,path=str(path)))
        sources.append(record)
    dump('source_manifest.json',sources)
    pd.DataFrame(view_rows).to_csv(HERE/'view_inventory.csv',index=False)
    values,models=graph.read_results(raw)
    metadata=graph.load_metadata(features,vhd,temp)
    assert not metadata.duplicated(['dataset','qid']).any()
    table,basecols=graph.graph_table(metadata,values,models)
    cols=[c for c in table if c not in META]
    table.to_csv(HERE/'features.csv',index=False)
    metadata.to_csv(HERE/'question_metadata.csv',index=False)
    dump('feature_spec.json',dict(models=models,columns=cols,base_columns=basecols,removed_three_id=[c for c in cols if c.startswith('model_identity_')]))
    primary=pd.read_csv(PRIMARY)
    qmeta=primary[['question_id','clip_id','stimulus_type','task','vhd_category']].drop_duplicates()
    assert not qmeta.duplicated('question_id').any()
    qmeta=qmeta.rename(columns={'question_id':'qid','clip_id':'source_clip'})
    original=primary.rename(columns={'question_id':'qid','correct':'original_correct','selected_label':'original_selected'})
    audit=table[table.dataset.eq('VHD')][META].merge(original[['qid','model_id','original_correct','original_selected']],on=['qid','model_id'],validate='one_to_one')
    audit=audit.merge(qmeta,on='qid',validate='many_to_one')
    gold=metadata[metadata.dataset.eq('VHD')].set_index('qid').gold
    audit['base_selected']=[int(values[('VHD',r.model_id,r.qid,'base')].argmax()) for r in audit.itertuples()]
    audit['original_selected_index']=audit.original_selected.map({'A':0,'B':1})
    audit['answer_matches_primary']=audit.base_selected.eq(audit.original_selected_index)
    assert audit.source_clip.eq(audit.group_id).all()
    assert all(int(primary[(primary.question_id==q)&(primary.model_id==models[0])].gold_label.iloc[0]=='B')==int(g) for q,g in gold.items())
    audit.to_csv(HERE/'vhd_primary_alignment.csv',index=False)
    qmeta.to_csv(HERE/'vhd_question_subsets.csv',index=False)
    summary=table.groupby('dataset').agg(rows=('qid','size'),questions=('qid','nunique'),videos=('group_id','nunique')).reset_index()
    summary.to_csv(HERE/'dataset_inventory.csv',index=False)
    dump('ANALYSIS_LOCK.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),gpu='none',fit_seed=FIT_SEED,bootstrap_seed=BOOT_SEED,n_boot=N_BOOT,methods=METHODS,holdout='Entire dataset including all variants, no held-out labels used for fitting or selection',aurc='Archived prefix-risk mean; stable input-order ties',risk50='ceil(0.5*N) accepted; stable ties',shared_errors='All three archived base-view answers wrong, computed within this response bank',bootstrap='Paired source-video resamples with replacement; scorers fixed; percentile CI',weights='Existing equal_domain_weights unchanged',identity_caveat='Only three model_identity columns removed; is_self and named peer slots retained, so not model-identity-free',outputs='Exploratory CPU reconstruction, not an archived evaluation result'))
    print(summary.to_string(index=False),flush=True)
    print('Features',len(cols),'VHD answer agreement with primary',audit.answer_matches_primary.mean(),flush=True)

def fit():
    table=pd.read_csv(HERE/'features.csv',dtype={'qid':str,'group_id':str})
    cols=json.loads((HERE/'feature_spec.json').read_text())['columns']
    reduced=[c for c in cols if not c.startswith('model_identity_')]
    pred=[]; splits=[]
    (HERE/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for held in sorted(table.dataset.unique()):
            source=table[table.dataset.ne(held)].reset_index(drop=True)
            target=table[table.dataset.eq(held)].reset_index(drop=True)
            assert set(source.dataset).isdisjoint(set(target.dataset))
            out=target[META].copy();out['raw_confidence']=target.own_base_confidence
            for suffix,features in [('',cols),('_without_three_id',reduced)]:
                lr=graph.fit_graph_logistic(source,features)
                hgb=HistGradientBoostingClassifier(learning_rate=.05,max_iter=200,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=2.,random_state=FIT_SEED)
                hgb.fit(source[features],source.correct,sample_weight=graph.equal_domain_weights(source))
                lrname='graph_lr' if not suffix else 'lr'+suffix
                hname='graph_hgb' if not suffix else 'hgb'+suffix
                out[lrname]=lr.predict_proba(target[features])[:,1]
                out[hname]=hgb.predict_proba(target[features])[:,1]
                joblib.dump(dict(lr=lr,hgb=hgb,features=features),HERE/'models'/f'{held}{suffix}.joblib')
            pred.append(out)
            splits.append(dict(held_out=held,train_datasets=sorted(source.dataset.unique()),train_rows=len(source),test_rows=len(target),test_videos=target.group_id.nunique()))
            print('Fitted',held,flush=True)
    pd.concat(pred,ignore_index=True).to_csv(HERE/'predictions.csv',index=False)
    dump('holdout_audit.json',splits)

def risk50(y,s):
    n=max(1,int(np.ceil(len(y)/2)))
    return float(np.mean(1-y[np.argsort(-s,kind='stable')[:n]]))

def stats():
    pred=pd.read_csv(HERE/'predictions.csv',dtype={'qid':str,'group_id':str})
    qmeta=pd.read_csv(HERE/'vhd_question_subsets.csv')
    vhd=pred[pred.dataset.eq('VHD')].merge(qmeta,on='qid',validate='many_to_one')
    wrong=vhd.groupby('qid').correct.sum().eq(0)
    vhd['shared_wrong']=vhd.qid.map(wrong).astype(bool)
    vhd.to_csv(HERE/'vhd_scored_subsets.csv',index=False)
    allrows=[];frames={}
    for d,g in pred.groupby('dataset',sort=True): frames[(d,'all')]=g
    for subset,g in vhd.groupby('stimulus_type',sort=True):frames[('VHD',subset)]=g
    for (d,subset),frame in frames.items():
        for m,g in frame.groupby('model_id',sort=True):
            y=g.correct.to_numpy();base=aurc(y,g.raw_confidence.to_numpy())
            for method in METHODS:
                s=g[method].to_numpy();a=aurc(y,s)
                row=dict(dataset=d,subset=subset,model_id=m,method=method,n=len(g),videos=g.group_id.nunique(),accuracy=y.mean(),aurc=a,relative_gain_pct=100*(base-a)/base,auroc=roc_auc_score(y,s) if len(set(y))==2 else np.nan,risk50=risk50(y,s))
                if 'shared_wrong' in g:
                    sw=g.shared_wrong.to_numpy();order=np.argsort(-s,kind='stable');deferred=order[int(np.ceil(len(g)/2)):]
                    row.update(shared_wrong_n=int(sw.sum()),shared_wrong_deferred_n=int(sw[deferred].sum()),shared_wrong_deferral_rate=float(sw[deferred].sum()/sw.sum()) if sw.sum() else np.nan)
                allrows.append(row)
    metrics=pd.DataFrame(allrows);metrics.to_csv(HERE/'model_metrics.csv',index=False)
    numeric=['accuracy','aurc','relative_gain_pct','auroc','risk50','shared_wrong_deferral_rate']
    avg=metrics.groupby(['dataset','subset','method'],sort=False)[numeric].mean().reset_index()
    avg.to_csv(HERE/'mean_metrics.csv',index=False)
    # Same paired bootstrap draws for every method and answering model within each evaluation.
    cis=[]
    for (d,subset),frame in frames.items():
        models=sorted(frame.model_id.unique())
        panels=[frame[frame.model_id.eq(m)].reset_index(drop=True) for m in models]
        assert all(p.qid.tolist()==panels[0].qid.tolist() for p in panels)
        groups=panels[0].group_id.to_numpy();unique=pd.unique(groups)
        members=[np.flatnonzero(groups==g) for g in unique]
        rng=np.random.default_rng(BOOT_SEED)
        draws=rng.integers(0,len(unique),size=(N_BOOT,len(unique)),dtype=np.int32)
        np.savez_compressed(HERE/f'bootstrap_indices_{d}_{subset}.npz',draws=draws,groups=unique.astype(str))
        ys=np.stack([p.correct.to_numpy() for p in panels]);ss=np.stack([p[METHODS].to_numpy().T for p in panels])
        bs=np.empty((N_BOOT,len(METHODS)-1))
        for b,draw in enumerate(draws):
            ix=np.concatenate([members[j] for j in draw])
            gain=[]
            for mi in range(3):
                yy=ys[mi,ix];base=aurc(yy,ss[mi,0,ix])
                gain.append([100*(base-aurc(yy,ss[mi,k,ix]))/base if base else np.nan for k in range(1,len(METHODS))])
            bs[b]=np.mean(gain,axis=0)
        pd.DataFrame(bs,columns=METHODS[1:]).to_csv(HERE/f'bootstrap_gain_{d}_{subset}.csv',index=False)
        for i,method in enumerate(METHODS[1:]):
            lo,hi=np.nanquantile(bs[:,i],[.025,.975]);point=float(avg[(avg.dataset==d)&(avg.subset==subset)&(avg.method==method)].relative_gain_pct.iloc[0])
            cis.append(dict(dataset=d,subset=subset,method=method,gain_pct=point,ci_low=lo,ci_high=hi,n_boot=N_BOOT,valid_resamples=int(np.isfinite(bs[:,i]).sum())))
        print('Bootstrap',d,subset,flush=True)
    pd.DataFrame(cis).to_csv(HERE/'gain_intervals.csv',index=False)
    print(avg[(avg.dataset=='VHD')&(avg.subset=='reversed')].to_string(index=False),flush=True)
    print(avg[(avg.subset=='all')&(avg.method=='graph_hgb')][['dataset','relative_gain_pct']].to_string(index=False),flush=True)

def manifest():
    inputs=json.loads((HERE/'source_manifest.json').read_text())
    code=[REPO/'src/evaluate_response_graph_features.py',REPO/'src/evaluate_probe_escalation.py',Path(__file__)]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),gpu='none',new_inference=False,fit_seed=FIT_SEED,bootstrap_seed=BOOT_SEED,bootstrap_n=N_BOOT,environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),inputs=inputs,code=[dict(path=str(p),sha256=sha(p)) for p in code],outputs=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'])
    dump('MANIFEST.json',result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['build','fit','stats','manifest']);args=parser.parse_args()
    globals()[args.stage]()
