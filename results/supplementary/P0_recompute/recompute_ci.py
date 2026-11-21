"""Same archived resampling function/seeds; change only the AURC callable.
Outputs are new files, never updates to archived selections or distributions.
"""
import json,sys,time,warnings
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'src'))
import analyze_herbench_trace_rc as original
import evaluate_v2_target_pilot as helper
from evaluate_probe_escalation import aurc as prefix_aurc

SPECS={
 'HERBench':('herbench_confirmation/8918264.kman.restech.unsw.edu.au',20260802),
 'Perception':('perception_confirmation/8919110.kman.restech.unsw.edu.au',20260803),
}

def task(dataset,definition):
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    folder,seed=SPECS[dataset]; archive=ROOT/'results'/folder
    frame=pd.read_csv(archive/'test_scored_predictions.csv',float_precision='round_trip')
    gate=json.loads((archive/'confirmation_gate.json').read_text())
    original.SEED=seed
    if definition=='prefix': helper.aurc=prefix_aurc
    raw=frame.raw_confidence.to_numpy(float); score=frame.trace_rc_score.to_numpy(float)
    start=time.monotonic()
    values=original.primary_bootstrap(frame,raw,score,gate['bootstrap_iterations'])
    target=OUT/f'{dataset}_{definition}_bootstrap.csv'
    pd.DataFrame({'iteration':np.arange(len(values)),'macro_gain_percent':values}).to_csv(target,index=False,mode='x')
    check={'dataset':dataset,'definition':definition,'seconds':time.monotonic()-start,'iterations':len(values),'seed':seed}
    if definition=='trapezoid':
        saved=pd.read_csv(archive/'test_bootstrap_distribution.csv',float_precision='round_trip').macro_gain_percent.to_numpy()
        err=float(np.max(np.abs(saved-values)))
        check['max_absolute_difference_from_archived_distribution']=err
        assert err<1e-8,(dataset,err)
    with (OUT/f'{dataset}_{definition}_verification.json').open('x') as f:json.dump(check,f,indent=2)
    return check

def calculate():
    # Persist the RNG draws as group indices. Confirm index and group choice
    # consume the same random stream and preserve sampled block order.
    for name,(folder,seed) in SPECS.items():
        frame=pd.read_csv(ROOT/'results'/folder/'test_scored_predictions.csv')
        groups=np.asarray(sorted(frame.group_id.astype(str).unique()))
        a=np.random.default_rng(seed); b=np.random.default_rng(seed)
        draws=np.empty((10000,len(groups)),dtype=np.int16)
        for i in range(len(draws)):
            ids=b.choice(len(groups),size=len(groups),replace=True)
            assert np.array_equal(a.choice(groups,size=len(groups),replace=True),groups[ids])
            draws[i]=ids
        resample_path=OUT/f'{name}_resamples.npz'
        if resample_path.exists():
            with np.load(resample_path) as saved:
                assert np.array_equal(saved['groups'],groups) and np.array_equal(saved['indices'],draws)
        else:
            with resample_path.open('xb') as f:np.savez_compressed(f,groups=groups,indices=draws,seed=seed)
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(task,name,definition) for name in SPECS for definition in ('trapezoid','prefix')]
        for f in as_completed(jobs): print(json.dumps(f.result()),flush=True)
def main():
    if '--summarize-only' not in sys.argv:
        calculate()
    rows=[]
    for name,(folder,seed) in SPECS.items():
        gate=json.loads((ROOT/'results'/folder/'confirmation_gate.json').read_text())
        old=pd.read_csv(OUT/f'{name}_trapezoid_bootstrap.csv').macro_gain_percent.to_numpy()
        new=pd.read_csv(OUT/f'{name}_prefix_bootstrap.csv').macro_gain_percent.to_numpy()
        old_ci=np.quantile(old,[.025,.975]); new_ci=np.quantile(new,[.025,.975])
        assert np.allclose(old_ci,gate['video_grouped_bootstrap_95ci_percent'],rtol=0,atol=1e-8)
        rows.append({'dataset':name,'seed':seed,'iterations':len(new),'point_gain_unchanged':gate['macro_mean_relative_aurc_reduction_percent'],
             'old_ci_low':old_ci[0],'old_ci_high':old_ci[1],'new_ci_low':new_ci[0],'new_ci_high':new_ci[1],
             'lower_change_pp':new_ci[0]-old_ci[0],'upper_change_pp':new_ci[1]-old_ci[1],
             'old_lower_above_zero':old_ci[0]>0,'new_lower_above_zero':new_ci[0]>0,
             'archived_selection_unchanged':True,'original_primary_gate_pass':gate['primary_gate_pass'],
             'primary_gate_pass_with_new_ci':all(v for k,v in gate['checks'].items() if k!='grouped_ci_lower_above_zero') and new_ci[0]>0})
    result=pd.DataFrame(rows)
    target=OUT/'test_ci_old_vs_new.csv'
    if target.exists():
        assert target.read_text()==result.to_csv(index=False).replace('\r\n','\n')
    else:
        result.to_csv(target,index=False,mode='x')
    # Paired-difference distributions already use prefix AURC. Audit saved
    # quantiles and explicitly retain them rather than calling them trapezoidal.
    comparisons=[]
    for stem,key,value in [('paired_bootstrap','comparator','delta_gain_percent_points'),
                           ('training_free_paired_bootstrap','baseline','trace_minus_baseline_gain_percent_points')]:
        parent=ROOT/'results/posthoc_robustness'
        summary=pd.read_csv(parent/(stem+'_summary.csv'),float_precision='round_trip')
        distribution=pd.read_csv(parent/(stem+'_distributions.csv'),float_precision='round_trip')
        for r in summary.to_dict('records'):
            sample=distribution.loc[distribution.dataset.eq(r['dataset'])&distribution[key].eq(r[key]),value].to_numpy()
            low,high=np.quantile(sample,[.025,.975])
            assert len(sample)==r['bootstrap_iterations']
            assert abs(low-r['paired_video_bootstrap_ci95_low'])<1e-10
            assert abs(high-r['paired_video_bootstrap_ci95_high'])<1e-10
            comparisons.append({'dataset':r['dataset'],'family':stem,'comparator':r[key],
                 'old_definition':'prefix','new_definition':'prefix','status':'already_prefix_saved_quantiles_verified',
                 'old_ci_low':low,'old_ci_high':high,'new_ci_low':low,'new_ci_high':high,
                 'iterations':len(sample),'resampling_seed':SPECS['HERBench' if r['dataset']=='HERBench' else 'Perception'][1]+(4409 if key=='comparator' else 8123)})
    pd.DataFrame(comparisons).to_csv(OUT/'paired_ci_old_vs_new.csv',index=False,mode='x')
    print('All CI outputs saved.',flush=True)

if __name__=='__main__': main()
