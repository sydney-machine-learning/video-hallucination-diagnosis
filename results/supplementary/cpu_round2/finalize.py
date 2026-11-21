from common import *
import subprocess,platform,datetime,re,html,importlib.metadata

# Verify report links, record the complete input closure, and retain source hashes.
checked_links=[]
for name in ['A.md','B.md','C.md','D.md']:
 text=(ROOT/name).read_text(encoding='utf-8')
 for target in re.findall(r'\]\(<([^>]+)>\)',text):
  match=re.match(r'^(.*?)(?::([0-9]+))?$',target);p=Path(match.group(1))
  assert p.exists(),(name,target)
  if match.group(2):assert int(match.group(2))<=len(p.read_text(encoding='utf-8').splitlines()),target
  checked_links.append(target)

sourceitems=json.loads((ROOT/'INPUT_MANIFEST.json').read_text(encoding='utf-8'))
sourceitems+=json.loads((VHD/'source_manifest.json').read_text(encoding='utf-8'))
for item in sourceitems:assert sha(item['path'])==item['sha256'],item['path']
sourcepaths={Path(x['path']) for x in sourceitems}
sourcepaths.update(VHD.glob('bootstrap_indices_VHD_*.npz'))
sourcepaths.update([VHD/'analyze.py',VHD/'gain_intervals.csv',VHD/'feature_spec.json',VHD/'download_manifest.json',REPO/'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/test_model_metrics.csv',REPO.parent/'overleaf_review_points_234_20261119/verified.tex'])
sourcepaths.update([REPO/'src'/s for s in ['evaluate_response_graph_features.py','evaluate_v2_target_pilot.py','evaluate_probe_escalation.py']])
sourcepaths.update([REPO/'katana/run_perception_valid500.pbs',Path('D:/paper 3/caver_route/katana/mvbench_multimodel_runner.py'),Path('D:/paper 3/caver_route/katana/run_mvbench_three_model_bundle.pbs'),Path('D:/paper 3/sicr_route/src/build_nextgqa_pilot_manifest.py'),Path('D:/paper 3/sicr_route/src/build_star_pilot_manifest.py')])
inputs=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(sourcepaths)]
dump(ROOT/'SOURCE_CHECKSUMS.json',inputs)

search=[]
for p in sorted((ROOT/'D_sources').iterdir()):
 if p.suffix not in ['.md','.txt','.sh','.yaml','.py','.html']:continue
 value=p.read_text(encoding='utf-8')
 if p.suffix=='.html':value=html.unescape(re.sub('<[^>]*>',' ',value))
 value=re.sub(r'\s+',' ',value)
 matches=list(re.finditer(r'perception[\s_-]*test',value,re.I))
 search.append(dict(file=str(p),matches=len(matches),contexts=[value[max(0,m.start()-60):m.end()+120] for m in matches[:12]]))
dump(ROOT/'D_document_search_complete.json',search)

plan=json.loads((ROOT/'ANALYSIS_PLAN.json').read_text(encoding='utf-8'))
assign=pd.read_csv(ROOT/'A_video_assignments.csv')
seeds=assign.groupby('split_id').seed.first().to_dict()
env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']},gpu='none',gpu_jobs_submitted=0,inference_calls=0)
dump(ROOT/'ENVIRONMENT.json',env)
validation=json.loads((ROOT/'VALIDATION.json').read_text(encoding='utf-8'));validation.update(report_local_links_verified=len(checked_links),source_checksums_verified=len(sourceitems))
dump(ROOT/'VALIDATION.json',validation)
outputs=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts]
manifest=dict(phase='cpu_round2',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repository=str(REPO),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),GPU='none',environment=env,analysis_plan=plan,split_seeds={str(k):(None if pd.isna(v) else int(v)) for k,v in seeds.items()},pilot_lr_random_state=20260802,development_fit_seed=20260801,archived_sources_unchanged=True,inputs=inputs,outputs=outputs,limitations=['Repeated splits reuse the same 500 target videos','VHD source scores are reused from the prior CPU reconstruction','No directly comparable official Perception Test score found for the exact three checkpoints','LLaVA upstream corpus contains Perception Test annotations, selected training subset unknown'])
dump(ROOT/'MANIFEST.json',manifest)
print(json.dumps(dict(status='complete',inputs=len(inputs),outputs=len(outputs),local_links=len(checked_links),output_bytes=sum(x['bytes'] for x in outputs),reports=[str(ROOT/x) for x in ['A.md','B.md','C.md','D.md']]),indent=2))
