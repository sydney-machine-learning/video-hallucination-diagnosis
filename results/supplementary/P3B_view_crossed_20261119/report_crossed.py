"""Report only values computed from saved inference and analysis outputs."""
import json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
from analyze_matched import ROOT,dump,sha
from prepare_features import original_bank

NAMES={'qwen8':'Qwen3-VL-8B-Instruct','internvl14':'InternVL3_5-14B-Instruct','llava8':'LLaVA-OneVision-2-8B-Instruct'}
def table(df):
    def fmt(x):return f'{x:.5f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(map(fmt,r))+' |' for r in df.itertuples(index=False,name=None)])

def main():
    assert json.loads((ROOT/'AUDIT.json').read_text())['passed']
    metrics=pd.read_csv(ROOT/'matched_metrics.csv');ci=pd.read_csv(ROOT/'original_split_intervals.csv')
    repeated=pd.read_csv(ROOT/'repeated_summary.csv')
    calls=[];costs=[];receipts={}
    for policy in ['original_views','phase_views']:
        records={}
        for model in NAMES:
            path=original_bank(model) if policy=='original_views' else ROOT.parent/('P3B_full_reference_20261119/reference_full' if model=='qwen8' else 'P3B_original_panel_20261119/responses_'+model)
            raw=pd.read_json(path/'responses.jsonl',lines=True);assert len(raw)==7552
            records[model]=raw;receipts[str(path/'responses.jsonl')]=sha(path/'responses.jsonl')
            for view,g in raw.groupby('phase'):
                calls.append({'protocol':policy,'model':NAMES[model],'view_slot':view,'questions':len(g),'mean_generation_seconds':g.reference_seconds.mean(),'median_generation_seconds':g.reference_seconds.median(),'p90_generation_seconds':g.reference_seconds.quantile(.9),'mean_preprocessing_seconds':g.preprocess_seconds.mean(),'peak_memory_bytes':int(g.peak_memory_bytes.max())})
        total=sum(float((r.reference_seconds+r.preprocess_seconds).sum()) for r in records.values())/1888
        extra=0.
        if policy=='phase_views':
            original=pd.read_json(original_bank('qwen8')/'responses.jsonl',lines=True)
            q=original[original.phase==0];extra=float((q.reference_seconds+q.preprocess_seconds).mean())
        costs.append({'protocol':policy,'full_scorer_calls_including_common_answer':12 if policy=='original_views' else 13,'frame_exposures_including_common_answer':168 if policy=='original_views' else 40,'mean_call_wall_seconds_including_common_answer':total+extra,'common_answer_extra_seconds':extra})
    pd.DataFrame(calls).to_csv(ROOT/'cost_by_model_view.csv',index=False)
    cost=pd.DataFrame(costs);cost.to_csv(ROOT/'cost_by_protocol.csv',index=False)
    interaction=[]
    for subset in ['all','physics']:
        banks={p:np.load(ROOT/(p+'_'+subset+'_bootstrap.npz')) for p in ['original_views','phase_views']}
        diff={p:b['ours_unchecked_lr']-b['pg_same_lr'] for p,b in banks.items()}
        distribution=diff['original_views']-diff['phase_views'];lo,hi=np.quantile(distribution,[.025,.975])
        point={}
        for p in banks:
            t=metrics[(metrics.policy==p)&(metrics.split==0)&(metrics.subset==subset)].set_index('method')
            point[p]=float(t.loc['ours_unchecked_lr','aurc']-t.loc['pg_same_lr','aurc'])
        interaction.append({'subset':subset,'difference_of_AURC_differences':point['original_views']-point['phase_views'],'lo':lo,'hi':hi})
    interaction=pd.DataFrame(interaction);interaction.to_csv(ROOT/'cross_view_interaction.csv',index=False)
    primary=ci[(ci.subset=='all')&(ci.quantity=='paired_aurc_difference_negative_favors_left')&(ci.method=='ours_unchecked_lr minus pg_same_lr')][['policy','estimate','lo','hi']]
    methods=['raw','ours_unchecked_lr','pg_same_lr','ours_checked','pg_full','pg_fixed_rf','pg_single','label_free_support']
    original=metrics[(metrics.split==0)&(metrics.subset=='all')&metrics.method.isin(methods)][['policy','method','accuracy','aurc','auroc','risk_at_50pct','gain_percent']]
    repeats=repeated[(repeated.subset=='all')&repeated.method.isin(methods)][['policy','method','mean_gain','p05_gain','p95_gain','harm_rate']]
    notes=ROOT/'EXECUTION_NOTES.md'
    if not notes.exists():notes.write_bytes((ROOT/'REPORT.md').read_bytes())
    body=['# Two observation protocols with shared answers',
          'Complete. This is exploratory evidence on the same previously inspected Perception cohort. Both protocols and all 30 additional splits are retained.',
          '## Primary feature comparison',
          'Both feature sets use the exact same fixed pilot LR and target labels. Values are paired AURC differences: our features minus reconstructed PhaseGuard features. Negative favors our features. Intervals are pointwise video bootstrap intervals, without multiplicity adjustment.',table(primary),
          '## Original split',table(original),
          '## Additional split variability',
          'These percentiles describe 30 splits of the same cohort, not independent-dataset uncertainty.',table(repeats),
          '## Dependence on the observation protocol',
          'Difference of the two paired feature-comparison differences, original views minus phase views. The same video resamples and answers are used.',table(interaction),
          '## Measured inference cost',table(cost),
          'Call times sum saved preprocessing and generation wall times; they are not pure GPU-kernel time or end-to-end latency. Decoding, model loading, scorer fitting, queueing and failed allocations are separate. Both full scorers share the same cost within a protocol. Actual PBS allocations, including failures, are retained in accounting/. The phase protocol is charged one extra base16 call to obtain the common answer.',
          '## Interpretation and reproduction limits',
          'The fixed answer is the newly inferred Qwen3-VL-8B-Instruct base16 option argmax. Accuracy is identical across all core comparisons by construction. This isolates reliability ranking, not answer generation or rescue. The source HGB uses extra source labels and encounters changed inference/view distributions; its result is a transfer control, not the primary feature attribution.',
          'Original frame indices follow the archived sampler. The new shared prompt/generation/runtime differs from the missing historical option-logit-only run, so these are not replacements for archived paper numbers. On original views, PhaseGuard adjacent/opposite features refer to fixed slots, not literal phase geometry. On phase views, our original addition/removal labels lose their literal intervention meaning. Full PhaseGuard features are a documented reconstruction. See PROTOCOL.md and FEATURE_SPEC.json.',
          'The native phase-vote Qwen3-VL-8B-Instruct reliability reproduction remains separately in ../P3B_full_reference_20261119. It excludes 32B rescue and uses our target cohort. Its different answers and budget prevent treating its absolute AURC as a matched core comparison.',
          '## Sources',
          'matched_metrics.csv; original_split_intervals.csv; repeated_summary.csv; cross_view_interaction.csv; cost_by_model_view.csv; cost_by_protocol.csv; analysis/*/split_*/test_predictions.csv; COMMON_ANSWERS.csv; MANIFEST.json; ANALYSIS_LOCK.json; AUDIT.json; accounting/.',
          'All claims about superiority must follow the saved intervals and both view conditions; an interval containing zero does not establish a difference. No Overleaf edits are made.']
    (ROOT/'REPORT.md').write_text('\n\n'.join(body)+'\n')
    original.to_csv(ROOT/'original_split_table.csv',index=False)
    tex=['\\begin{tabular}{llrrrrr}','\\hline',' & '.join(c.replace('_','\\_') for c in original.columns)+r' \\','\\hline']
    for row in original.itertuples(index=False,name=None):tex.append(' & '.join(f'{v:.4f}' if isinstance(v,(float,np.floating)) else str(v).replace('_','\\_') for v in row)+r' \\')
    (ROOT/'original_split_table.tex').write_text('\n'.join(tex+['\\hline','\\end{tabular}'])+'\n')
    subprocess.run(['D:/Anaconda/python.exe','-B',str(ROOT/'plot_matched.py')],check=True)
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    manifest.update(status='complete; both protocols audited',source_response_checksums=receipts,analysis_lock_sha256=sha(ROOT/'ANALYSIS_LOCK.json'),result_checksums={q.name:sha(q) for q in ROOT.iterdir() if q.is_file() and q.suffix in ['.csv','.npz','.tex','.png','.pdf']},audit=json.loads((ROOT/'AUDIT.json').read_text()))
    dump(ROOT/'MANIFEST.json',manifest)
    dump(ROOT/'REPORT_COMPLETE.json',{'complete':True,'protocols':2,'splits_each':31,'source_checksums':receipts})
if __name__=='__main__':main()
