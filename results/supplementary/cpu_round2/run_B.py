from common import *
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor,as_completed
METHODS=['raw_confidence','base_vote','base_vote_then_confidence','base_support','graph_lr','graph_hgb']
SUBSETS={'all':None,'visible':'original','counterintuitive':'reversed','hidden':'masked'}

def prepare_B():
    predictions=pd.read_csv(VHD/'predictions.csv');predictions=predictions[predictions.dataset.eq('VHD')]
    features=pd.read_csv(VHD/'features.csv');features=features[features.dataset.eq('VHD')]
    subset=pd.read_csv(VHD/'vhd_question_subsets.csv')
    frame=predictions.merge(features[['qid','model_id','base_vote_fraction','base_selected_support_mean']],on=['qid','model_id'],validate='one_to_one').merge(subset,on='qid',validate='many_to_one')
    assert len(frame)==2403 and frame.qid.nunique()==801 and frame.group_id.nunique()==341
    frame['base_vote']=frame.base_vote_fraction
    frame['base_vote_then_confidence']=frame.base_vote+1e-9*frame.raw_confidence
    frame['base_support']=frame.base_selected_support_mean
    joint=frame.groupby('qid').correct.sum().eq(0)
    # VHD has two choices: three incorrect answers necessarily agree on the wrong option.
    frame['joint_wrong']=frame.qid.map(joint)
    assert frame[frame.stimulus_type.eq('reversed')].drop_duplicates('qid').joint_wrong.sum()==99
    frame.to_csv(ROOT/'B_scores.csv',index=False)
    records=[]
    for model,g in frame.groupby('model_id'):
        for scope,h in [('all_VHD',g),('counterintuitive',g[g.stimulus_type.eq('reversed')])]:
            for method in METHODS:
                ranks=100*rankdata(h[method].to_numpy(),method='average')/len(h)
                for qid,s,joint_wrong,stim,pct in zip(h.qid,h[method],h.joint_wrong,h.stimulus_type,ranks):
                    if joint_wrong and stim=='reversed':records.append(dict(model_id=model,qid=qid,method=method,reference_pool=scope,score=s,percentile=pct,pool_questions=len(h)))
    pd.DataFrame(records).to_csv(ROOT/'B_joint_error_percentiles.csv',index=False)
    audit=json.loads((VHD/'holdout_audit.json').read_text());held=next(x for x in audit if x['held_out']=='VHD')
    assert set(held['train_datasets'])=={'MVBench','NExT-GQA','STAR','TempCompass'}
    dump(ROOT/'B_reuse_audit.json',dict(prediction_source=str(VHD/'predictions.csv'),model_source=str(VHD/'models/VHD.joblib'),holdout=held,refit=False,source='Existing CPU reconstruction from archived probabilities; not a new VLM inference',models=sorted(frame.model_id.unique())))

def all_metrics(y,s):
    # Vectorized exact prefix-risk AURC and Mann-Whitney AUROC, one row per model/method.
    n=y.shape[-1];yy=np.broadcast_to(y[:,None,:],s.shape)
    order=np.argsort(-s,axis=-1,kind='stable');ordered=np.take_along_axis(yy,order,axis=-1)
    a=(np.cumsum(1-ordered,axis=-1)/np.arange(1,n+1)).mean(axis=-1)
    ranks=rankdata(s,method='average',axis=-1);pos=y.sum(axis=-1)[:,None];neg=n-pos
    with np.errstate(divide='ignore',invalid='ignore'):auc=((ranks*yy).sum(axis=-1)-pos*(pos+1)/2)/(pos*neg)
    gain=100*(a[:,[0]]-a)/a[:,[0]]
    return np.stack([auc,gain,a],axis=-1),ranks

def subset_run(subset):
    frame=pd.read_csv(ROOT/'B_scores.csv');stim=SUBSETS[subset]
    if stim:frame=frame[frame.stimulus_type.eq(stim)]
    models=sorted(frame.model_id.unique());panels=[frame[frame.model_id.eq(m)].reset_index(drop=True) for m in models]
    assert all(g.qid.tolist()==panels[0].qid.tolist() for g in panels)
    y=np.stack([g.correct.to_numpy() for g in panels]);s=np.stack([g[METHODS].to_numpy().T for g in panels]);joint=panels[0].joint_wrong.to_numpy(bool)
    point,ranks=all_metrics(y,s)
    oldname=stim or 'all';bankpath=VHD/f'bootstrap_indices_VHD_{oldname}.npz';bank=np.load(bankpath)
    groups=panels[0].group_id.to_numpy();members=[np.flatnonzero(groups==v) for v in bank['groups']]
    assert all(len(x)>0 for x in members) and set(bank['groups'])==set(groups)
    assert len(bank['draws'])==10000
    results=np.empty((10000,3,len(METHODS),3));errmean=np.empty((10000,3,len(METHODS))) if subset=='counterintuitive' else None
    for i,draw in enumerate(bank['draws']):
        ix=np.concatenate([members[j] for j in draw]);metric,rnk=all_metrics(y[:,ix],s[:,:,ix]);results[i]=metric
        if errmean is not None:errmean[i]=np.mean(100*rnk[:,:,joint[ix]]/len(ix),axis=-1)
    np.savez_compressed(ROOT/f'B_bootstrap_{subset}.npz',metrics=results,methods=np.array(METHODS),models=np.array(models),**({'joint_wrong_percentile_mean':errmean} if errmean is not None else {}))
    rows=[]
    for mi,m in enumerate(models+['mean across three']):
        pt=point[mi] if mi<3 else point.mean(axis=0)
        boot=results[:,mi] if mi<3 else results.mean(axis=1)
        for j,method in enumerate(METHODS):
            row=dict(subset=subset,model_id=m,method=method,questions=len(panels[0]),videos=len(members))
            for k,name in enumerate(['auroc','gain_pct','aurc']):
                lo,hi=np.nanquantile(boot[:,j,k],[.025,.975]);row.update({name:pt[j,k],name+'_ci_low':lo,name+'_ci_high':hi})
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/f'B_metrics_{subset}.csv',index=False)
    if errmean is not None:
        output=[];percs=100*ranks[:,:,joint]/len(panels[0]);reference=percs[:,[0],:]
        for mi,m in enumerate(models+['mean across three']):
            for j,method in enumerate(METHODS):
                vals=percs[mi,j] if mi<3 else percs[:,j,:].ravel()
                draws=errmean[:,mi,j] if mi<3 else errmean[:,:,j].mean(axis=1)
                delta=errmean[:,mi,j]-errmean[:,mi,0] if mi<3 else (errmean[:,:,j]-errmean[:,:,0]).mean(axis=1)
                actual_delta=np.mean(percs[mi,j]-percs[mi,0]) if mi<3 else np.mean(percs[:,j]-percs[:,0])
                lo,hi=np.quantile(draws,[.025,.975]);dl,dh=np.quantile(delta,[.025,.975]);q1,median,q3=np.quantile(vals,[.25,.5,.75])
                output.append(dict(model_id=m,method=method,joint_wrong_questions=int(joint.sum()),mean_percentile=vals.mean(),median_percentile=median,q25=q1,q75=q3,mean_ci_low=lo,mean_ci_high=hi,delta_vs_raw_pp=actual_delta,delta_ci_low=dl,delta_ci_high=dh))
        pd.DataFrame(output).to_csv(ROOT/'B_joint_error_percentile_summary.csv',index=False)
    print('B completed',subset,flush=True)
    return subset

if __name__=='__main__':
    prepare_B()
    with ProcessPoolExecutor(3) as pool:
        for f in as_completed([pool.submit(subset_run,s) for s in SUBSETS]):f.result()
    pd.concat([pd.read_csv(ROOT/f'B_metrics_{s}.csv') for s in SUBSETS],ignore_index=True).to_csv(ROOT/'B_metrics.csv',index=False)
    dump(ROOT/'B_COMPLETE.json',dict(cpu_only=True,subsets=list(SUBSETS),bootstrap=10000,models=3))
