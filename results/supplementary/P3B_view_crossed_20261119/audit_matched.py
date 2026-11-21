"""Independent audit of saved split outputs and source locks."""
import json
import joblib,numpy as np,pandas as pd
from analyze_matched import ROOT,REPO,sha,dump,metrics

def main():
    lock=json.loads((ROOT/'ANALYSIS_LOCK.json').read_text())
    for path,expected in lock['checksums'].items():assert sha(REPO/path)==expected,path
    assert json.loads((ROOT/'FEATURE_VALIDATION.json').read_text())['passed']
    labels=pd.concat([pd.read_csv(REPO/'input/perception_test/frozen'/n) for n in ['pilot_labels.csv','sealed_test_labels.csv']]).set_index('question_id')
    assignment=pd.read_csv(ROOT.parent/'P2/video_assignments.csv')
    common=pd.read_csv(ROOT/'COMMON_ANSWERS.csv',float_precision='round_trip').set_index('qid')
    checked=0
    for policy in ['original_views','phase_views']:
        feature=pd.read_csv(ROOT/(policy+'_features.csv'),float_precision='round_trip')
        for split in range(31):
            path=ROOT/'analysis'/policy/f'split_{split:02d}'
            pre=json.loads((path/'PRETEST_LOCK.json').read_text());complete=json.loads((path/'COMPLETE.json').read_text())
            assert pre['models_sha256']==sha(path/'models.joblib')
            assert pre['features_sha256']==sha(ROOT/(policy+'_features.csv'))
            assert pre['analysis_lock_sha256']==sha(ROOT/'ANALYSIS_LOCK.json')
            assert pre['test_labels_used_in_fitting'] is False
            assert set(pre['pilot_videos']).isdisjoint(pre['test_videos'])
            a=assignment[(assignment.dataset=='Perception')&(assignment.split_id==split)]
            assert set(pre['test_videos'])==set(a[a.role=='test'].video_id)
            assert set(pre['pilot_videos'])==set(a[a.role=='calibration'].video_id)
            test=pd.read_csv(path/'test_predictions.csv',float_precision='round_trip');assert test.qid.is_unique
            np.testing.assert_array_equal(test.answer_index,common.loc[test.qid].answer_index)
            np.testing.assert_array_equal(test.raw,common.loc[test.qid].confidence)
            assert test.qid.tolist()==feature[feature.group_id.isin(pre['test_videos'])].qid.tolist()
            np.testing.assert_array_equal(test.correct,(test.answer_index.to_numpy()==labels.loc[test.qid].gold_candidate_index.to_numpy()).astype(int))
            np.testing.assert_array_equal(test.group_id,labels.loc[test.qid].video_id)
            saved=pd.read_csv(path/'metrics.csv')
            for row in saved.itertuples():
                subset=test if row.subset=='all' else test[test.task=='physics']
                assert np.isfinite(subset[row.method]).all()
                actual=metrics(subset.correct.to_numpy(),subset[row.method].to_numpy())
                for name,value in actual.items():
                    if value is not None:assert np.isclose(value,getattr(row,name),rtol=0,atol=1e-12),(policy,split,row.method,name)
            assert (saved.groupby('subset').accuracy.nunique()==1).all()
            bundles=joblib.load(path/'models.joblib')
            for scope in ['pg_single','pg_full']:
                folds=bundles[scope]['fold_video_ids']
                assert len(folds)==5
                valid=[]
                for fold in folds:
                    assert set(fold['train']).isdisjoint(fold['valid'])
                    assert set(fold['train'])|set(fold['valid'])==set(pre['pilot_videos'])
                    valid.extend(fold['valid'])
                assert len(valid)==len(set(valid))==250
            checked+=1
    dump(ROOT/'AUDIT.json',{'passed':True,'policy_split_pairs':checked,'all_fixed_splits_present':True,
        'video_disjoint_splits_and_folds':True,'all_metrics_recomputed_from_saved_predictions':True,
        'same_answers_accuracy_and_question_order_within_policy':True,'source_locks_intact':True,
        'common_answer_verified_against_base16':True})
    print('AUDIT_PASSED',checked,flush=True)
if __name__=='__main__':main()
