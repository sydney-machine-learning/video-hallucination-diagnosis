"""Development-only selection for the independently reproduced single variant.

No test labels are accepted. The pilot inference stage does not call this file.
The search follows paper Table 8; unspecified defaults are recorded in output.
"""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED=20260820

def cdf(reference,values):
    return np.searchsorted(np.sort(reference),values,side='right')/len(reference)

def candidates():
    out=[]
    for c in [.03,.1]:
        out.append((f'LR_C{c}',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=c,max_iter=5000,random_state=SEED))))
    for leaf in [8,16]:
        out.append((f'ET_leaf{leaf}',make_pipeline(SimpleImputer(strategy='median'),ExtraTreesClassifier(n_estimators=1000,min_samples_leaf=leaf,class_weight='balanced',random_state=SEED,n_jobs=4))))
    out.append(('RF_leaf10',make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=1000,min_samples_leaf=10,class_weight='balanced',random_state=SEED,n_jobs=4))))
    out.append(('HGB_defaults',make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(class_weight='balanced',random_state=SEED))))
    return out

def fit_selector(features,correctness,video_ids,token_confidence):
    x=np.asarray(features,dtype=float); y=np.asarray(correctness,dtype=int)
    groups=np.asarray(video_ids);confidence=np.asarray(token_confidence,dtype=float)
    assert x.shape[0]==len(y) and x.shape[1] in (32,48) and len(groups)==len(y)==len(confidence)
    assert set(y)=={0,1} and len(set(groups))>=5
    folds=list(GroupKFold(n_splits=5).split(x,y,groups))
    base=cdf(confidence,confidence);search=[];best=None;best_auc=-np.inf
    for name,estimator in candidates():
        oof=np.full(len(y),np.nan)
        for train,valid in folds:
            assert not set(groups[train])&set(groups[valid])
            assert set(y[train])=={0,1}
            fitted=clone(estimator).fit(x[train],y[train])
            oof[valid]=fitted.predict_proba(x[valid])[:,1]
        assert np.isfinite(oof).all()
        ranked=cdf(oof,oof)
        for weight in np.arange(21)/20:
            score=(1-weight)*base+weight*ranked
            auc=float(roc_auc_score(y,score))
            search.append({'estimator':name,'lambda':float(weight),'oof_auroc':auc})
            # Stable predeclared candidate order and smaller blend weight break ties.
            if auc>best_auc:
                best_auc=auc
                best={'name':name,'lambda':float(weight),'estimator':clone(estimator),'oof':oof.copy()}
    best['estimator'].fit(x,y)
    best['token_reference']=confidence.copy()
    best['search']=search
    best['fold_video_ids']=[{'train':sorted(set(groups[t])),'valid':sorted(set(groups[v]))} for t,v in folds]
    best['unreported_paper_choices']={'question_length':'Unicode characters','variance':'population variance','entropy':'natural logarithm','percentile_reference':'full development token values and grouped OOF predicted values; frozen before test scoring','estimator_unspecified_settings':'installed sklearn defaults; exact get_params stored with fitted artifact','LR_max_iter':5000,'opposite_pairs':[[0,2],[1,3]],'tie_selection':'candidate order, then smaller lambda'}
    return best

def predict_selector(bundle,features,token_confidence):
    learned=bundle['estimator'].predict_proba(np.asarray(features,dtype=float))[:,1]
    weight=bundle['lambda']
    return (1-weight)*cdf(bundle['token_reference'],token_confidence)+weight*cdf(bundle['oof'],learned)
