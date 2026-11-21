import unittest,json
import numpy as np,pandas as pd,joblib
from matched_features import graph_table,original_graph,pg_features,majority,REPO
from phaseguard_features import build_features
from analyze_matched import aurc,ours

class MatchedTests(unittest.TestCase):
    def test_graph_default_exact_and_selected_support(self):
        rng=np.random.default_rng(48);models=['internvl14','qwen8','llava8']
        meta=pd.DataFrame([dict(dataset='Perception',qid='q',group_id='v',gold=2,candidate_count=3)])
        values={('Perception',m,'q',r):rng.dirichlet(np.ones(3)) for m in models for r in original_graph.ROLES}
        original,base=original_graph.graph_table(meta,values,models)
        current,b=graph_table(meta,values,models)
        pd.testing.assert_frame_equal(original,current);self.assertEqual(base,b)
        target=(int(values[('Perception','qwen8','q','base')].argmax())+1)%3
        overridden,_=graph_table(meta,values,models,{('Perception','q','qwen8'):target})
        row=overridden[overridden.model_id=='qwen8'].iloc[0]
        self.assertEqual(row.own_base_confidence,values[('Perception','qwen8','q','base')][target])
        self.assertEqual(row.correct,int(target==2))
        bundle=joblib.load(REPO/'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/pilot_model_bundle.joblib')
        columns=[c for c in current if c not in ['dataset','qid','group_id','model_id','correct']]
        self.assertEqual(columns,bundle['feature_columns'])
        self.assertTrue(np.isfinite(bundle['source_models']['source_trace_hgb'].predict_proba(current[columns])).all())

    def test_phase_features_native_and_permuted_labels(self):
        rng=np.random.default_rng(9);p=rng.dirichlet(np.ones(3),4);q=rng.dirichlet(np.ones(3),4)
        answers=['A','B','A','C'];peer=['B','B','C','C'];options=['abc','def','ghi']
        vote,_,native=build_features('question',options,answers,p)
        single,full=pg_features('question',options,answers,p,vote,[('llava8',peer,q),('internvl14',peer,q)])
        self.assertEqual(single,native);self.assertEqual(len(full),48)
        permutation=[2,0,1];translate={chr(65+old):chr(65+new) for new,old in enumerate(permutation)}
        a=[translate[x] for x in answers];b=[translate[x] for x in peer]
        _,other=pg_features('question',options,a,p[:,permutation],translate[vote],[('llava8',b,q[:,permutation]),('internvl14',b,q[:,permutation])])
        np.testing.assert_allclose(list(full.values()),list(other.values()),atol=1e-12)
        self.assertEqual(majority(['C','A','C','A']),'C')

    def test_prefix_metric_used_in_selection(self):
        y=np.array([1,0,0,1]);s=np.ones(4)
        expected=np.mean(np.cumsum(1-y)/np.arange(1,5))
        self.assertEqual(aurc(y,s),expected)
        self.assertIs(ours.aurc,aurc)

if __name__=='__main__':unittest.main()
