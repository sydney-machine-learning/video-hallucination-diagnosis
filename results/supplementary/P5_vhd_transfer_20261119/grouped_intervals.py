import analyze as a
import numpy as np
import pandas as pd

pred=pd.read_csv(a.HERE/'predictions_grouped_validation.csv',dtype={'qid':str,'group_id':str})
means=pd.read_csv(a.HERE/'mean_metrics_grouped_validation.csv')
ci=pd.read_csv(a.HERE/'gain_intervals.csv')
for d in ['NExT-GQA','STAR','TempCompass']:
    panels=[g.reset_index(drop=True) for m,g in pred[pred.dataset.eq(d)].groupby('model_id',sort=True)]
    assert all(g.qid.tolist()==panels[0].qid.tolist() for g in panels)
    bank=np.load(a.HERE/f'bootstrap_indices_{d}_all.npz')
    members=[np.flatnonzero(panels[0].group_id.to_numpy()==k) for k in bank['groups']]
    ys=[g.correct.to_numpy() for g in panels]
    raw=[g.raw_confidence.to_numpy() for g in panels]
    scores=[g.graph_hgb.to_numpy() for g in panels]
    assert all(np.array_equal(g.graph_hgb.to_numpy(),g.hgb_without_three_id.to_numpy()) for g in panels)
    gains=[]
    for draw in bank['draws']:
        ix=np.concatenate([members[j] for j in draw]);gain=[]
        for y,r,s in zip(ys,raw,scores):
            base=a.aurc(y[ix],r[ix]);gain.append(100*(base-a.aurc(y[ix],s[ix]))/base)
        gains.append(np.mean(gain))
    pd.DataFrame({'graph_hgb':gains,'hgb_without_three_id':gains}).to_csv(a.HERE/f'bootstrap_gain_grouped_{d}.csv',index=False)
    lo,hi=np.quantile(gains,[.025,.975])
    point=means[(means.dataset==d)&(means.method=='graph_hgb')].relative_gain_pct.iloc[0]
    mask=ci.dataset.eq(d)&ci.subset.eq('all')&ci.method.isin(['graph_hgb','hgb_without_three_id'])
    ci.loc[mask,['gain_pct','ci_low','ci_high']]=[point,lo,hi]
    print(d,point,lo,hi,flush=True)
ci.to_csv(a.HERE/'gain_intervals_grouped_validation.csv',index=False)
