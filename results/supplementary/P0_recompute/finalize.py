"""Publish actual recomputation results and explicit dependency failures."""
import hashlib,importlib.metadata,json,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)
ci=pd.read_csv(OUT/'test_ci_old_vs_new.csv',float_precision='round_trip')
paired=pd.read_csv(OUT/'paired_ci_old_vs_new.csv',float_precision='round_trip')
before=json.loads((OUT.parent/'P0/archive_preservation.json').read_text())['input_sha256_at_start']
assert all(sha(ROOT/p)==h for p,h in before.items())
verifications=[json.loads(p.read_text()) for p in OUT.glob('*_verification.json')]
assert len(verifications)==4
assert all(r.get('max_absolute_difference_from_archived_distribution',0)==0 for r in verifications)
blocked=pd.read_csv(OUT.parent/'P0/pilot_candidate_audit.csv')
blocked['status']='not_computed_target_pilot_responses_or_oof_scores_missing_after_D_drive_and_Katana_search'
blocked.to_csv(OUT/'selection_sensitivity_status.csv',index=False,mode='x')
# Further D-drive check includes binary containers and every CSV with a dataset
# field, not just files named after the targets.
inventory=json.loads((OUT.parent/'P0_record_search/local_inventory.json').read_text(encoding='utf-8'))
binary=[p for p in inventory['files'] if Path(p).suffix.lower() in {'.joblib','.pkl','.pickle','.parquet','.npz','.npy','.h5','.hdf5','.feather','.arrow'}]
datasets=[];errors=[]
for name in inventory['files']:
    p=Path(name)
    if p.suffix.lower()!='.csv':continue
    try:
        frame=pd.read_csv(p,low_memory=False)
        if 'dataset' in frame:
            datasets.append({'path':name,'sha256':sha(p),'rows':len(frame),'datasets':sorted(frame.dataset.dropna().astype(str).unique())})
    except Exception as e:errors.append({'path':name,'error':str(e)})
put(OUT/'D_drive_feature_check.json',{'root':'D:/paper 3','binary_feature_or_model_containers':binary,'dataset_tables':datasets,'read_errors':errors,
 'target_tables':[r for r in datasets if any('herbench' in s.lower() or 'perception' in s.lower() for s in r['datasets'])]})
cit=table(['Dataset','Old trapezoidal CI (%)','New prefix mean CI (%)','Point gain (%)','Primary success check'],[
 [r.dataset,f'[{r.old_ci_low:.6f}, {r.old_ci_high:.6f}]',f'[{r.new_ci_low:.6f}, {r.new_ci_high:.6f}]',f'{r.point_gain_unchanged:.6f}','Pass under both'] for r in ci.itertuples()])
pairtable=table(['Dataset','Family','Comparator','Old CI','New CI'],[
 [r.dataset,r.family,r.comparator,f'[{r.old_ci_low:.6f}, {r.old_ci_high:.6f}]',f'[{r.new_ci_low:.6f}, {r.new_ci_high:.6f}]'] for r in paired.itertuples()])
report=f'''# P0 updated report: requested CPU recomputations

This report supersedes the earlier statement that no further recomputation was requested. The user subsequently authorized selection sensitivity and all trapezoidal test intervals to be recomputed with prefix-risk mean. Historical files and selections remain unchanged.

## 1. Original-split selection sensitivity

Not completed: target pilot response vectors, complete features, and out-of-fold candidate scores remain unavailable in the searched repository, D:/paper 3 and Katana locations. The additional D-drive check covers binary feature/model containers and CSV dataset fields. Katana inspection covers experiment logs, candidate tables and checkpoints; it found labels, fitted bundles and aggregate candidate metrics, not the missing row-level pilot scores. Checkpoint mentions in logs refer to model weight loading rather than saved pilot score vectors.

The requested prefix pilot gains, lower bounds and candidate pass/fail therefore remain unknown. We do not report unchanged selections or infer rejected-candidate bounds from fallback zeros. Known archived values and blank recomputation fields are saved in selection_sensitivity_status.csv. Original model choices have not been altered.

## 2. Test confidence intervals

Completed for both target confirmations. We reused analyze_herbench_trace_rc.primary_bootstrap unchanged, with 10,000 draws and original seeds 20260802/20260803. We first reran the archived trapezoidal definition. Every replicate equals the archived distribution exactly: maximum absolute error is zero for both datasets. Only the metric callable was then changed to evaluate_probe_escalation.aurc. The same video blocks, sampled order, within-block row order and strict input-order tie rule are retained.

{cit}

Saved outputs: test_ci_old_vs_new.csv, each target's prefix/trapezoid bootstrap CSV, verification JSON, and resamples.npz. The NPZ stores the ordered video list, bootstrap group indices and seed. The original estimates and selections stay fixed. Both revised lower bounds remain above zero, so neither primary success check changes. This is not evidence about counterfactual pilot selection.

## 3. Paired-difference intervals

The {len(paired)} target/comparator intervals in the two posthoc archives already use prefix-risk mean. The call chain is analyze_trace_rc_posthoc_robustness.reported_relative_gain → evaluate_response_graph_features.aurc → evaluate_probe_escalation.aurc. The training-free baseline routine calls the same reported_relative_gain. They were not trapezoidal intervals and do not require conversion.

We verified each interval against quantiles of its saved replicate distribution and retained it unchanged. The saved seeds are dataset seed + 4409 for standard paired comparisons and + 8123 for training-free comparisons. These differ from the primary interval seeds by design; the primary and paired-versus-raw intervals need not be numerically identical. paired_ci_old_vs_new.csv records the unchanged old/new endpoints and explicitly labels them already_prefix_saved_quantiles_verified, not newly recomputed distributions.

{pairtable}

## 4. Additional Katana and D-drive checks

katana_logs_and_readiness.json contains source paths, hashes, relevant log excerpts, candidate-table schemas and asset existence checks. The target raw records were not recovered. All target video paths referenced by the archived manifests are currently absent, as are the original Qwen/InternVL and LLaVA Python environment executables. The user-level Hugging Face cache directory exists; presence of the required pinned model revisions has not been established.

The D-drive check in D_drive_feature_check.json found no target dataset table or binary feature/model container that fills the pilot-data gap. The prior search reports record scope and inaccessible locations; absence in the searched locations is not proof that all possible backups are gone.

## 5. Progress after the CPU work

P2 now has split 0 plus 30 additional whole-video 50/50 splits per target. HERBench is balanced by source collection. No labels were used to construct splits, no video crosses its pilot/test boundary, and split 0 retains archived roles. These are prepared splits, not performance results. No selection model was retuned or refitted.

P1 cannot complete the requested archived-probability equivalence check without the target raw responses; its original media and inference environments also need restoration. P2, P3A and P4 require the missing target option probabilities or equivalent feature archives. Test correctness and scalar confidence alone cannot reconstruct cross-model option distributions or pilot features. These are execution dependencies, not additional approval checkpoints. No GPU jobs were submitted, and S2 has not been reached.

## 6. Changes required in paper numbers

Replace the two primary confidence intervals with the new endpoints above when reporting prefix-risk mean consistently for fixed archived choices. Keep point gains 16.64% and 18.87%. Paired-difference intervals need no numerical changes. Do not claim prefix-based pilot selection has been validated: that computation is blocked by missing inputs. Historical selections still used the original helper and are retained for split 0.

The reusable snippet is p0_ci_update.tex. The previous P0 report remains preserved; this is its dated update. Environment versions, input/output hashes and original-file preservation are recorded in MANIFEST.json.
'''
put(OUT/'REPORT.md',report)
put(OUT.parent/'P0/REPORT_UPDATED_20261118.md',report.replace('Saved outputs:','Outputs below are located in ../P0_recompute/. Saved outputs:'))
tex=r'''% Fixed archived choices; only test CI metric changed.
\begin{tabular}{lrr}
\hline
Dataset & Archived 95\% CI & Prefix mean 95\% CI \\
\hline
'''
for r in ci.itertuples():tex+=f'{r.dataset} & [{r.old_ci_low:.4f}, {r.old_ci_high:.4f}] & [{r.new_ci_low:.4f}, {r.new_ci_high:.4f}] '+r'\\'+'\n'
tex+=r'\hline'+'\n'+r'\end{tabular}'+'\n% Candidate-selection sensitivity is not yet computed.\n'
put(OUT/'p0_ci_update.tex',tex)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
for phase,text in {
 'P1':'Readiness checked only. The original target media and inference environments are absent at archived paths. Archived response probabilities needed for equivalence are also missing. No timing, probability comparison, speedup, or GPU job has run. Existing frame reuse and single-token option scoring must remain the baseline. Restore assets and target response archives before claiming equivalence.',
 'P3A':'Blocked: majority vote, mean probabilities and relabelled pilot LR need target per-option probabilities and model answers for pilot/test. Saved test correctness and scalar confidence do not supply those. P2 video splits are prepared; no baseline metrics have run.',
 'P4':'Blocked: stage-specific LR and stopping-policy evaluation need per-stage target responses and pilot features. P2 video splits are prepared. No thresholds, policies or budget-versus-gain results were fit or measured.'}.items():
    directory=OUT.parent/phase
    put(directory/'REPORT.md',f'# {phase} execution status\n\n{text}\n\nNo result tables or LaTeX performance snippets are available. No archived files were changed.\n')
    put(directory/'MANIFEST.json',{'phase':phase,'status':'blocked_missing_inputs','commit':commit,'utc':datetime.now(timezone.utc).isoformat(),
        'seeds':None,'gpu':{'used':False,'jobs_submitted':0},'environment':{'python':sys.version},
        'input_sha256':{'../P0_recompute/katana_logs_and_readiness.json':sha(OUT/'katana_logs_and_readiness.json')},
        'output_sha256':{'REPORT.md':sha(directory/'REPORT.md')}})
put(OUT/'STATUS.json',{'selection_sensitivity':'blocked_missing_target_pilot_scores','primary_test_CIs':'completed_prefix_with_identical_resampling',
 'paired_CIs':'already_prefix_saved_quantiles_verified','archived_distribution_reconstruction_error':0,
 'P2':'31_video_splits_per_target_prepared_not_evaluated','P1_P3A_P4':'blocked_missing_inputs','gpu_jobs_submitted':0,'S2_reached':False,
 'all_original_tracked_files_unchanged':True})
put(OUT/'phaseguard_reference_check.json',{'source':'https://arxiv.org/abs/2610.23969','title':'Learning When to Trust and Reobserve Sparse-Frame VideoQA from Temporal Sampling Perturbations',
 'venue_as_listed_on_arxiv':'NeurIPS 2026 Workshop on Physical Understanding for Decision-Making (PhysUnderstand), Poster',
 'status':'bibliographic check only; no code availability conclusion, implementation, inference or pilot run'})
versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy','scikit-learn','joblib']}
put(OUT/'MANIFEST.json',{'phase':'P0_recompute','commit':commit,'utc':datetime.now(timezone.utc).isoformat(),
 'status':'test_CI_recompute_complete_selection_sensitivity_blocked',
 'seeds':{'HERBench_primary':20260802,'Perception_primary':20260803,'iterations':10000,'quantiles':[.025,.975]},
 'gpu':{'used':False,'jobs_submitted':0,'gpu_hours':0},'environment':{'python':sys.version,'platform':platform.platform(),'packages':versions},
 'input_sha256':before,'output_sha256':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='MANIFEST.json'},
 'additional_output_sha256':{'../P0/REPORT_UPDATED_20261118.md':sha(OUT.parent/'P0/REPORT_UPDATED_20261118.md')},
 'parallelism':'4 CPU worker processes for target x AURC definition; no estimator fits',
 'implementation':'original primary_bootstrap; only helper.aurc replaced in prefix worker; all source files unchanged',
 'diagnostics':'Sandbox denied Windows multiprocessing pipes; authorized escalated execution succeeded. Summary-only rerun corrected a Perception display-name alias and newline comparison without rerunning bootstrap or overwriting distributions.'})
print(ci[['dataset','old_ci_low','old_ci_high','new_ci_low','new_ci_high']].to_string(index=False))
print('Final reports saved; missing pilot input is not counted as completed.')
