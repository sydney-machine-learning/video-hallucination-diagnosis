"""Preserve HGB parameters, keep videos together in internal validation."""
import analyze as a
import inspect
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

table=pd.read_csv(a.HERE/'features.csv',dtype={'qid':str,'group_id':str})
pred=pd.read_csv(a.HERE/'predictions.csv',dtype={'qid':str,'group_id':str})
audit=[]
with threadpool_limits(limits=4):
    for held in sorted(table.dataset.unique()):
        source=table[table.dataset.ne(held)].reset_index(drop=True)
        target=table[table.dataset.eq(held)].reset_index(drop=True)
        old=joblib.load(a.HERE/'models'/f'{held}.joblib')['hgb']
        if not old.do_early_stopping_:
            audit.append(dict(held_out=held,changed=False,reason='Archived auto early stopping is off; no internal split'))
            continue
        groups=source.dataset+'::'+source.group_id
        train,val=next(GroupShuffleSplit(n_splits=1,test_size=old.validation_fraction,random_state=a.FIT_SEED).split(source,groups=groups))
        assert not set(groups.iloc[train]) & set(groups.iloc[val])
        membership=source[a.META].copy();membership['internal_partition']='train';membership.loc[val,'internal_partition']='validation'
        membership.to_csv(a.HERE/f'internal_groups_{held}.csv',index=False)
        audit.append(dict(held_out=held,changed=True,train_rows=len(train),validation_rows=len(val),train_videos=groups.iloc[train].nunique(),validation_videos=groups.iloc[val].nunique(),overlap=0,seed=a.FIT_SEED))
        for suffix,col in [('', 'graph_hgb'),('_without_three_id','hgb_without_three_id')]:
            bundle=joblib.load(a.HERE/'models'/f'{held}{suffix}.joblib')
            columns=bundle['features'];params=bundle['hgb'].get_params()
            # Preserve the original effective auto=True mode before the grouped
            # validation rows are removed from the explicit fit argument.
            params['early_stopping']=True
            model=a.HistGradientBoostingClassifier(**params)
            weights=a.graph.equal_domain_weights(source)
            model.fit(source.iloc[train][columns],source.iloc[train].correct,sample_weight=weights[train],X_val=source.iloc[val][columns],y_val=source.iloc[val].correct,sample_weight_val=weights[val])
            assert model.do_early_stopping_
            score=model.predict_proba(target[columns])[:,1]
            mask=pred.dataset.eq(held)
            assert pred.loc[mask,['qid','model_id']].values.tolist()==target[['qid','model_id']].values.tolist()
            pred.loc[mask,col]=score
            joblib.dump(dict(hgb=model,features=columns),a.HERE/'models'/f'{held}{suffix}_grouped_validation.joblib')
        print('Grouped internal validation',held,flush=True)
pred.to_csv(a.HERE/'predictions_grouped_validation.csv',index=False)
a.dump('internal_validation_audit.json',audit)
rows=[]
for (d,m),g in pred.groupby(['dataset','model_id']):
    y=g.correct.to_numpy();base=a.aurc(y,g.raw_confidence.to_numpy())
    for method in a.METHODS:
        s=g[method].to_numpy();value=a.aurc(y,s)
        rows.append(dict(dataset=d,model_id=m,method=method,accuracy=y.mean(),aurc=value,relative_gain_pct=100*(base-value)/base,auroc=a.roc_auc_score(y,s),risk50=a.risk50(y,s)))
metrics=pd.DataFrame(rows);metrics.to_csv(a.HERE/'model_metrics_grouped_validation.csv',index=False)
mean=metrics.groupby(['dataset','method'])[['accuracy','aurc','relative_gain_pct','auroc','risk50']].mean().reset_index()
mean.to_csv(a.HERE/'mean_metrics_grouped_validation.csv',index=False)
print(mean.pivot(index='dataset',columns='method',values='relative_gain_pct').to_string())
