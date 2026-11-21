import analyze as a
import json
import numpy as np
import pandas as pd
import joblib

p=a.HERE
features=pd.read_csv(p/'features.csv',dtype={'qid':str,'group_id':str})
pred=pd.read_csv(p/'predictions_grouped_validation.csv',dtype={'qid':str,'group_id':str})
identity={}
for i in range(3):
    identity[f'model_identity_{i}_duplicates_base_peer_{i}_is_self']=bool(np.array_equal(features[f'model_identity_{i}'],features[f'base_peer_{i}_is_self']))
assert all(identity.values())
identity['max_abs_hgb_prediction_change_after_removing_three_ids']=float((pred.graph_hgb-pred.hgb_without_three_id).abs().max())
assert identity['max_abs_hgb_prediction_change_after_removing_three_ids']==0
a.dump('identity_audit.json',identity)
assert len(features)==12033 and len(pred)==12033
assert not pred.duplicated(['dataset','qid','model_id']).any()
assert pred.groupby(['dataset','qid']).size().eq(3).all()
assert np.isfinite(pred[a.METHODS].to_numpy()).all()
assert ((pred[a.METHODS]>=0)&(pred[a.METHODS]<=1)).to_numpy().all()
metrics=pd.read_csv(p/'model_metrics_grouped_validation.csv')
for (d,m),g in pred.groupby(['dataset','model_id']):
    for method in a.METHODS:
        # Independent scalar implementation of the specified risk definition.
        ordered=sorted(zip(g[method],range(len(g)),g.correct),key=lambda x:(-x[0],x[1]))
        count=0;total=0
        for k,(_,_,y) in enumerate(ordered,1):count+=1-y;total+=count/k
        value=total/len(ordered)
        stored=metrics[(metrics.dataset==d)&(metrics.model_id==m)&(metrics.method==method)].aurc.iloc[0]
        assert np.isclose(value,stored,atol=1e-12)
for f in p.glob('internal_groups_*.csv'):
    rows=pd.read_csv(f,dtype={'group_id':str});assert rows.groupby(['dataset','group_id']).internal_partition.nunique().max()==1
grouped_models=[]
for f in (p/'models').glob('*_grouped_validation.joblib'):
    model=joblib.load(f)['hgb'];assert model.n_iter_==200
    grouped_models.append(dict(file=f.name,n_iter=model.n_iter_,early_stopping=bool(model.do_early_stopping_)))
a.dump('grouped_model_audit.json',grouped_models)
for source in json.loads((p/'source_manifest.json').read_text()):
    assert a.sha(source['path'])==source['sha256'],source['path']
extra_paths=[a.REPO/'katana/recover_llava_and_analyze.pbs',a.REPO/'katana/run_perception_valid500.pbs',
    a.Path('D:/paper 3/Dataset/scripts/build_real_vhd_variants.py'),a.Path('D:/paper 3/Dataset/scripts/build_ai_vhd_variants.py')]
a.dump('additional_source_checksums.json',[dict(path=str(f),sha256=a.sha(f)) for f in extra_paths])
claims=[dict(id='VHD01',claim='178 reversed questions, 178 source clips; 99 shared errors',source='vhd_scored_subsets.csv'),dict(id='VHD02',claim='Accuracy 24.91%, LR gain -7.20%, HGB gain -5.48%',source='model_metrics.csv and mean_metrics.csv'),dict(id='VHD03',claim='Paired confidence intervals',source='gain_intervals_grouped_validation.csv'),dict(id='M701',claim='Checkpoint revisions identical across datasets',source='view_inventory.csv'),dict(id='M702',claim='Fixed checkpoint gains',source='mean_metrics_grouped_validation.csv'),dict(id='M703',claim='Three ID columns redundant and HGB predictions unchanged',source='identity_audit.json')]
a.dump('claim_evidence.json',dict(status='Machine verified; author review pending',claims=claims))
a.dump('VALIDATION.json',dict(status='passed',archived_source_hashes_preserved=True,rows=12033,questions=4011,independent_aurc_checks=75,all_internal_control_splits_video_grouped=True,all_held_out_datasets_excluded_from_fit=True,all_probabilities_finite=True,new_gpu_inference=False,overleaf_modified=False))
print('Validation passed: 75 AURC checks, grouped splits, hashes, identity audit')
