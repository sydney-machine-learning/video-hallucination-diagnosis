from common import *
from run_B import all_metrics,METHODS
from sklearn.metrics import roc_auc_score

checks=[]
for item in json.loads((ROOT/'INPUT_MANIFEST.json').read_text()):
 assert sha(item['path'])==item['sha256'],item['path']
checks.append('All preflight source checksums unchanged')
b=pd.read_csv(ROOT/'B_scores.csv');old=pd.read_csv(VHD/'gain_intervals.csv')
for subset,stim in [('all',None),('visible','original'),('counterintuitive','reversed'),('hidden','masked')]:
 f=b if stim is None else b[b.stimulus_type.eq(stim)]
 panels=[g.reset_index(drop=True) for _,g in f.groupby('model_id')]
 y=np.stack([g.correct.to_numpy() for g in panels]);s=np.stack([g[METHODS].to_numpy().T for g in panels])
 calc,_=all_metrics(y,s)
 for i in range(3):
  for j in range(len(METHODS)):
   assert abs(calc[i,j,0]-roc_auc_score(y[i],s[i,j]))<1e-12
   assert abs(calc[i,j,2]-aurc(y[i],s[i,j]))<1e-12
 bank=np.load(VHD/f'bootstrap_indices_VHD_{stim or "all"}.npz');saved=np.load(ROOT/f'B_bootstrap_{subset}.npz')['metrics']
 members=[np.flatnonzero(panels[0].group_id.to_numpy()==v) for v in bank['groups']]
 for k in [0,1,77,9999]:
  ix=np.concatenate([members[j] for j in bank['draws'][k]])
  for i in range(3):
   for j in range(len(METHODS)):
    assert abs(saved[k,i,j,0]-roc_auc_score(y[i,ix],s[i,j,ix]))<1e-12
    assert abs(saved[k,i,j,2]-aurc(y[i,ix],s[i,j,ix]))<1e-12
 current=pd.read_csv(ROOT/f'B_metrics_{subset}.csv')
 for method in ['graph_lr','graph_hgb']:
  a=current[(current.model_id=='mean across three')&(current.method==method)].iloc[0]
  p=old[(old.dataset=='VHD')&(old.subset==(stim or 'all'))&(old.method==method)].iloc[0]
  assert np.allclose([a.gain_pct,a.gain_pct_ci_low,a.gain_pct_ci_high],[p.gain_pct,p.ci_low,p.ci_high],atol=1e-10)
checks.append('All 72 B point metrics and 288 sampled bootstrap metrics match independent sklearn AUROC and archived prefix AURC; eight LR/HGB point and CI rows match P5')
for d in (ROOT/'A_runs').glob('*/split_*'):
 p=json.loads((d/'promotion.json').read_text());dist=np.load(d/'promotion_distributions.npz')
 for c in p['candidates']:
  assert len(dist[c['candidate']])==2000
  assert c['lower_bound']==float(np.quantile(dist[c['candidate']],.025))
  assert c['passed']==(c['lower_bound']>c['threshold'])
 assert not set(p['pilot_videos'])&set(p['test_videos'])
 f=pd.read_csv(d/'test_predictions.csv',float_precision='round_trip');metrics=pd.read_csv(d/'metrics.csv')
 for _,r in metrics.iterrows():
  assert abs(aurc(f.correct.to_numpy(),f[r.method].to_numpy())-r.aurc)<1e-12
checks.append('All 60 A selections have correct thresholds, lower quantiles, disjoint videos, and reproducible test AURC')
# Archive-generated letters differ from option argmax for a small number of records.
# Report this separately without changing the fixed argmax answers in A selection.
frame=pd.read_csv(ROOT/'A_answer_accuracy.csv');audit=[]
for m in MODELS:
 rows=readjsonl(OLD/f'merged_responses_{m}/responses.jsonl');base={r['question_id']:r for r in rows if r['phase']==0}
 answers=[base[q]['answer'] for q in frame.qid]
 assert set(answers)<=set('ABC')
 frame[m+'_generated_answer']=[ord(a)-65 for a in answers]
 frame[m]=(frame[m+'_generated_answer']==frame.gold).astype(int)
 audit.append(dict(model=m,base_mismatch=int((frame[m+'_answer']!=frame[m+'_generated_answer']).sum()),all_views_mismatch=sum(r['answer']!=chr(65+int(np.argmax(r['candidate_probabilities']))) for r in rows)))
votes=[]
for _,r in frame.iterrows():
 counts=np.bincount([int(r[m+'_generated_answer']) for m in MODELS],minlength=3);ties=np.flatnonzero(counts==counts.max())
 probs=np.array([r[f'mean_probability_{i}'] for i in range(3)]);answer=ties[np.argmax(probs[ties])]
 votes.append(int(answer==r.gold))
frame['majority_vote']=votes;out=[]
for split in [-1,0]:
 bank=np.load(ROOT/f'A_accuracy_bootstrap_{split}.npz');f=frame[frame.group_id.isin(bank['groups'])]
 methods=MODELS+['majority_vote'];counts=f.groupby('group_id')[methods].sum().reindex(bank['groups']).to_numpy();sizes=f.groupby('group_id').size().reindex(bank['groups']).to_numpy()
 boot=counts[bank['draws']].sum(axis=1)/sizes[bank['draws']].sum(axis=1)[:,None]
 for j,m in enumerate(methods):
  lo,hi=np.quantile(boot[:,j],[.025,.975]);out.append(dict(split=split,model=m,accuracy=f[m].mean(),ci_low=lo,ci_high=hi))
pd.DataFrame(out).to_csv(ROOT/'A_generated_answer_sensitivity.csv',index=False)
pd.DataFrame(audit).to_csv(ROOT/'A_answer_definition_audit.csv',index=False)
dump(ROOT/'VALIDATION.json',dict(status='passed',checks=checks,gpu='none'))
print(json.dumps(dict(checks=checks,answer_audit=audit),indent=2))
