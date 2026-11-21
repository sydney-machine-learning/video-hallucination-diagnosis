"""Finalize P0 evidence after the user's clarification: identify as-run metrics,
without further counterfactual AURC/bootstrap recomputation.
"""
import ast
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,value):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,ensure_ascii=False))
        f.write("\n")
def table(headers,rows):
    return "| "+" | ".join(headers)+" |\n| "+" | ".join(["---"]*len(headers))+" |\n"+"\n".join("| "+" | ".join(str(v) for v in r)+" |" for r in rows)
def short(model):
    return "InternVL" if "InternVL" in model else "Qwen" if "Qwen" in model else "LLaVA"

evidence=[]
for file,names in {
    "src/evaluate_v2_target_pilot.py":["aurc","relative_gain","grouped_bootstrap_gains","promote"],
    "src/evaluate_probe_escalation.py":["aurc"],
    "src/analyze_herbench_trace_rc.py":["pilot_phase","method_metrics","primary_bootstrap","final_phase"],
}.items():
    p=ROOT/file;text=p.read_text(encoding="utf-8");lines=text.splitlines()
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names:
            evidence.append({"file":file,"function":node.name,"start_line":node.lineno,"end_line":node.end_lineno,
                "file_sha256":sha(p),"source":"\n".join(lines[node.lineno-1:node.end_lineno])})
put("aurc_provenance.json",{"conclusion":{
    "pilot_gains_candidate_bootstrap_and_selection":"tie-grouped trapezoidal AURC",
    "reported_test_point_gains":"mean prefix risk, stable input-row tie breaking",
    "test_bootstrap_confidence_intervals":"tie-grouped trapezoidal AURC",
    "historical_definition_identification_requires_recomputation":False,
    "counterfactual_selection_change_requires_pilot_scores_and_recomputation":True},
    "evidence":evidence,
    "historical_caveat":"The current HERBench analysis source matches the Perception seal, but not the older HERBench seal. Its current Katana copy is identical after newline normalization. See checksum_followup.json. The pilot helper matches both seals; saved test point gains match the prefix function."})

checks=pd.read_csv(OUT/"checksum_checks.csv").fillna("")
follow=[]
names={"HERBENCH_PREINFERENCE_SEAL_20260802.json":"original_herbench_seal.json",
       "herbench_persistent_worker.py":"remote_herbench_persistent_worker.py",
       "analyze_herbench_trace_rc.py":"remote_analyze_herbench_trace_rc.py"}
for row in checks.loc[checks.status.eq("content_mismatch")].to_dict("records"):
    p=ROOT/row["local_path"]; remote=OUT/"inputs/remote_code"/names[p.name]
    rb=remote.read_bytes(); lb=p.read_bytes()
    normalized_equal=rb.replace(b"\r\n",b"\n")==lb.replace(b"\r\n",b"\n")
    trailing_equal=rb.replace(b"\r\n",b"\n").rstrip(b"\n")==lb.replace(b"\r\n",b"\n").rstrip(b"\n")
    matches=sha(remote)==row["expected_sha256"]
    explanation=("Recovered byte-exact sealed file; local difference is line endings" if matches and normalized_equal else
                 "Recovered byte-exact sealed file; local difference is a trailing blank line" if matches and trailing_equal else
                 "Current local and Katana source agree after newline normalization and match the Perception seal; older HERBench sealed version remains unavailable")
    follow.append({**row,"remote_copy":remote.relative_to(OUT).as_posix(),"remote_sha256":sha(remote),
                   "remote_matches_expected":matches,"equal_after_newline_normalization":normalized_equal,
                   "equal_after_ignoring_final_newlines":trailing_equal,"resolution":explanation})
put("checksum_followup.json",follow)

remote=load(OUT/"remote_dependency_audit.json")
expected=load(ROOT/"results/posthoc_robustness/analysis_manifest.json")["input_sha256"]
dependency=[]
for name,digest in expected.items():
    if name.startswith("trace_rc_route/"):continue
    found=[r for r in remote if r["sha256"]==digest]
    dependency.append({"archived_path":name,"expected_sha256":digest,"remote_checksum_matches":bool(found),
                       "remote_paths":[r["path"] for r in found],"rows":found[0]["rows"] if found else None})
assert all(r["remote_checksum_matches"] for r in dependency)
put("development_dependency_resolution.json",{
    "references":dependency,
    "inventory_schema_caveat":"remote_dependency_audit.json uses question_id before pilot_id for duplicate counting. This causes false duplicate flags for NExT-GQA, where question_id is local to a video. Those duplicate counts and question counts are not validated and are not used here. The archived feature builder correctly prioritizes pilot_id. Byte hashes and total row counts are unaffected.",
    "audit_scope":"SHA256 and total row counts only; no development model refit or feature reconstruction was performed"})

inference={
 "target_views":{"base":16,"invariant_nuisance":8,"dense_acquisition":24,"temporal_order_destruction":8},
 "models":[
  {"model":"Qwen","code":"katana/nextgqa_qwen_runner.py","prompt_lines":[427,440],"score_lines":[486,520],
   "video_position":"first user content item, before question and options","method":"one forward pass on final prefix token logits when each option label has one token; teacher-forced per-candidate fallback otherwise","generate_called":False},
  {"model":"InternVL","code":"katana/nextgqa_internvl_runner.py","prompt_lines":[449,474],"score_lines":[533,600],
   "video_position":"FrameN image tokens precede question and options in user message, after the model system message","method":"one forward pass for single-token contextual option labels; teacher-forced fallback otherwise","generate_called":False},
  {"model":"LLaVA","code":"results/supplementary/P0/inputs/remote_code/nextgqa_llava_ov2_runner.py","prompt_lines":[196,222],"score_lines":[274,320],
   "video_position":"image content items for all frames precede the text question and options","method":"single-prefix scoring or teacher-forced continuation according to option tokenization","generate_called":False,
   "provenance":"current Katana source copied; file is missing from public repository; not verified against a historical target-run source checksum"}],
 "decoded_frame_reuse":{"code":"katana/mvbench_multimodel_runner.py:436-467","description":"decodes union of frame indices once per video/segment/view-set key, reuses decoded frames for subsequent questions with the same key; HERBench wrapper sorts work by segment"},
 "kv_reuse":{"implemented_in_examined_adapters":False,"evidence":"use_cache=False in forward calls"},
 "raw_record_confirmation":"MVBench records all show single_prefix_next_token_logits, one forward call and generated=false. Target raw records are missing, so target branch frequencies cannot be verified.",
 "timing_implication":"P1 cannot claim frame decoding reuse and single-token option scoring as new optimizations. Measure any added caching or runtime changes against the actual existing pipeline."}
put("inference_audit.json",inference)

summary=load(OUT/"audit_summary.json")
macro=pd.read_csv(OUT/"test_macro_gains.csv")
test=pd.read_csv(OUT/"test_metrics_both_definitions.csv")
splits=pd.read_csv(OUT/"split_inventory.csv")
pilots=pd.read_csv(OUT/"pilot_candidate_audit.csv")
mv=load(OUT/"mvbench_raw_audit.json")
budget=[]
for dataset,gpus in (("HERBench",2),("Perception",4)):
    hours=5+58/60
    budget.append({"dataset":dataset,"archived_allocated_gpus":gpus,"archived_walltime_hours":hours,
                   "allocated_gpu_hours":gpus*hours,"exceeds_user_5_gpu_hour_stop":gpus*hours>5})
put("pbs_budget_guard.json",{"jobs_submitted":0,"archived_scripts":budget,
    "rule":"These are configured allocation limits, not measured utilization. Do not resubmit unchanged; future jobs exceeding five GPU-hours require the user's stop/review."})

macrotable=table(["Dataset","Paper prefix mean (%)","Helper trapezoid (%)","Difference (percentage points)"],[
    [name,f"{g.iloc[0].mean_relative_gain_percent:.6f}",f"{g.iloc[1].mean_relative_gain_percent:.6f}",
     f"{g.iloc[1].mean_relative_gain_percent-g.iloc[0].mean_relative_gain_percent:.6f}"] for name,g in macro.groupby("dataset",sort=False)])
splitstable=table(["Dataset","Split","Questions","Videos","Verified raw response rows"],
    [[r.dataset,r.split,r.questions,r.videos,"Unavailable"] for r in splits.itertuples()])
selected=test.loc[test.method.eq("TRACE-RC")]
modeltable=table(["Dataset","Model","Archived choice","Prefix gain (%)","Trapezoid gain (%)"],[
    [name,short(model),g.iloc[0].archived_selection,f"{g.iloc[0].relative_gain_percent:.6f}",f"{g.iloc[1].relative_gain_percent:.6f}"]
    for (name,model),g in selected.groupby(["dataset","model_id"],sort=False)])
old=pilots.loc[pilots.aurc_definition.eq("helper_tie_grouped_trapezoid")]
pilottable=table(["Dataset","Model","Candidate","Logged pilot gain (%)","Logged lower bound (%)","Selected"],[
    [r.dataset,short(r.model_id),r.candidate,f"{r.archived_gain_percent:.6f}",
     "Not saved" if pd.isna(r.archived_lower_bound_percent) else f"{r.archived_lower_bound_percent:.6f}",r.archived_selected]
    for r in old.itertuples()])
provenancetable=table(["Use","Definition","Code"],[
 ["Pilot candidate gain and bootstrap lower bound","Tie-grouped trapezoid","evaluate_v2_target_pilot.py: aurc, relative_gain, grouped_bootstrap_gains, promote"],
 ["Test AURC and reported mean gain","Mean prefix risk","analyze_herbench_trace_rc.py: method_metrics → evaluate_response_graph_features.aurc → evaluate_probe_escalation.aurc"],
 ["Test bootstrap confidence interval","Tie-grouped trapezoid","analyze_herbench_trace_rc.py: primary_bootstrap → evaluate_v2_target_pilot.relative_gain"]])

report=f"""# P0 preflight report

Status: S1, awaiting review. No GPU jobs were submitted. Archived files and decisions were preserved.

## Scope clarification

The user's follow-up asks whether archived code and Katana records identify the AURC definition, making further recomputation unnecessary. They do. No additional AURC or bootstrap recomputation was run after that clarification. Test point estimates had already been recomputed. Pilot counterfactual gains, bounds and changed decisions remain unevaluated, rather than being reported as unchanged.

## Which AURC was used

{provenancetable}

The pilot helper groups exact score ties and integrates group endpoint risks with the trapezoidal rule. The paper function averages cumulative error rates over every retained prefix. It sorts by descending score, with original input row order breaking ties. These definitions can differ even without ties. During bootstrap, the input order is the concatenation of sampled whole-video blocks, with the original within-video row order preserved.

The helper's SHA256 matches both target seals. The current final-analysis source matches the Perception seal. The older HERBench seal names a different final-analysis hash, discussed below. The saved test point estimates independently reproduce the prefix definition. Evidence and function excerpts: [aurc_provenance.json](aurc_provenance.json).

Identifying the historical definition does not require recomputation. Determining whether a *different* definition would change candidate eligibility or selection does. Existing selected-candidate lower bounds do not supply the missing candidate score vectors or the bounds for other candidates.

## Test results already recomputed

All rows keep the archived choices fixed. Each benchmark mean averages model-specific relative reductions, without pooling questions or taking a ratio of averaged AURCs.

{macrotable}

{modeltable}

The paper's 16.64% and 18.87% point gains are reproduced. These small test-point differences do not establish that candidate choices or confidence bounds would stay unchanged. All candidate test scores are in [test_metrics_both_definitions.csv](test_metrics_both_definitions.csv). No confidence interval under the prefix definition was computed here.

## Pilot records and decision limits

{pilottable}

These are transcribed archived values, not newly recomputed values. For a fallback decision, the lock's zero gain and zero bound describe the fallback itself; they are not rejected-candidate bounds. An unselected candidate can have passed its threshold and lost to another candidate, so unselected is not automatically failed.

The target response JSONL files are absent from the repository and were not found in the surveyed Katana home and scratch directories. The search excluded hidden directories, environments and caches; see [remote_inventory.json](remote_inventory.json) for scope. This is an availability finding, not a claim that no backup exists anywhere. The model bundles contain fitted estimators and feature-column lists, but no pilot response vectors, features or out-of-fold scores. Test scores cannot reconstruct those pilot observations.

[pilot_candidate_audit.csv](pilot_candidate_audit.csv) records the archived gains, available bounds, fixed seeds and thresholds; recomputed fields are blank. [decision_comparison.csv](decision_comparison.csv) marks all counterfactual decisions unknown. Later repeated-split analyses and archived-probability equivalence checks need the target raw responses or a verified equivalent feature/score archive.

## Split and checksum inventory

{splitstable}

All question IDs are unique in each manifest. Label joins are complete, video groups do not cross pilot/test, and each test question has exactly the three model rows. HERBench collection balance matches its lock; see [herbench_source_balance.csv](herbench_source_balance.csv).

The initial local/reference audit contains {len(checks)} checksum comparisons: {dict(Counter(checks.status))}. These counts are comparisons, not unique files; missing local paths are not evidence that remote copies are missing. Git blob checks separate Windows CRLF conversion from substantive byte differences. No archived file was rewritten to normalize line endings.

Follow-up resolved two initially unmatched files. The original HERBench seal was recovered with the exact archived SHA256; its local difference is mixed versus normalized line endings. The persistent worker was recovered with its sealed hash; the local version omits a trailing blank line. The current HERBench analysis script agrees with Katana and the Perception seal but does not match the older HERBench seal. That historical source version is unresolved. See [checksum_followup.json](checksum_followup.json).

All {len(dependency)} development input references in the posthoc manifest were located on Katana by exact checksum, including raw responses and metadata, with row counts saved in [development_dependency_resolution.json](development_dependency_resolution.json). The preliminary remote inventory's NExT-GQA duplicate counts use the wrong ID preference and are explicitly excluded; its byte hashes and total row counts remain valid. The existing feature builder correctly prioritizes pilot_id. No development features or models were rebuilt.

## MVBench raw records

All three raw archives were found, copied into P0 and matched the posthoc SHA256 values. Each has {mv[0]['rows']} rows and {mv[0]['questions']} questions with four views. Each model has {mv[0]['unique_media_paths']} distinct media paths; these are path counts, not a newly reconstructed group split.

| View | Frame budget |
| --- | --- |
| base | 8 |
| invariant_nuisance | 8 |
| dense_acquisition | 16 |
| temporal_order_destruction | 8 |

The MVBench model panel is Qwen 4B, InternVL 4B and LLaVA 8B. It differs from the target panel and its 16/8/24/8 frame budgets. Each MVBench raw record reports one prefix-logit forward call and generated=false. See [mvbench_views.csv](mvbench_views.csv) and [mvbench_raw_audit.json](mvbench_raw_audit.json).

## Inference and prompt placement

The examined adapters score option-label logits directly. With single-token labels they use one forward pass; otherwise they use teacher-forced continuation scoring. They do not call generate(). Video/frame content comes before the question and options in all three examined prompt builders. InternVL also places its system message before the user message.

The existing runner already decodes the union of requested frame indices and reuses decoded frames for the same video segment. The target wrapper groups work by segment. The examined forward calls use use_cache=False. Thus P1 must measure additional changes against this existing implementation, rather than attributing existing frame reuse or single-token scoring to a new pipeline.

The public repository omits the imported LLaVA adapter. Its current Katana copy is saved under inputs/remote_code. Its historical target-run identity is not independently sealed here. Missing target raw records also prevent verifying how often those runs took a multi-token fallback. See [inference_audit.json](inference_audit.json).

## Environment, anomalies and later phases

Metric calculations used the environment recorded in [environment.json](environment.json). The fitted bundles were created with scikit-learn 1.7.2, whereas the available inspection environment has 1.6.1. Bundles were inspected only; no predictions, training or refitting used them. A compatible environment is needed before later model-based reproduction. The default Anaconda environment could not load these NumPy 2-era bundles; this did not affect the saved metric calculations.

The first audit attempts encountered different split-lock field names and were corrected before emitting final analytical outputs. No model parameters, thresholds, feature code or archived selections were changed.

The archived PBS scripts request allocations exceeding the user's five GPU-hour stop rule. They were not submitted. [pbs_budget_guard.json](pbs_budget_guard.json) records their configured allocation limits; these are not measured runtime costs.

For P2 onward, [canonical_metric.py](canonical_metric.py) imports the exact paper prefix-risk function. It is not wired into any running experiment. New splits must use that one function for selection, bootstrap and reporting, with the existing resampling, LR/HGB settings and thresholds unchanged. Split 0 must retain the archived choices. Whole-video grouping remains mandatory.

## Manuscript implications

The two published point gains need no numerical change based on this audit. The methods description must distinguish the helper's trapezoidal AURC used for pilot selection and bootstrap from the prefix-risk mean used for reported test point gains. Do not describe the archived confidence intervals as newly computed under the prefix definition, or claim that changing definitions preserves the candidate decisions. MVBench uses the smaller development panel and 8/8/16/8 frames. Do not describe the existing single-token scoring path as text generation.

Copyable LaTeX is in [p0_metric_note.tex](p0_metric_note.tex). This report does not modify the paper or historical result archives.

## Checkpoint

Stopped at S1. The historical metric question is answered. Counterfactual pilot decisions remain unknown. P1 and later phases have not started.
"""
put("REPORT.md",report)
tex=r"""% P0 provenance note; preserve archived decisions and reported point results.
The archived pilot selection and bootstrap bounds used a trapezoidal AURC
computed at the endpoints of tied score groups. Reported test point estimates
used the mean risk across all retained prefixes, with input order breaking score
ties. Test bootstrap intervals followed the former definition. We retain the
archived decisions and do not infer that replacing the metric would leave
candidate selection unchanged.

\begin{tabular}{lrr}
\hline
Dataset & Prefix mean gain (\%) & Trapezoidal gain (\%) \\
\hline
"""
for name,g in macro.groupby("dataset",sort=False):
    tex+=f"{name} & {g.iloc[0].mean_relative_gain_percent:.4f} & {g.iloc[1].mean_relative_gain_percent:.4f} "+r"\\"+"\n"
tex+=r"\hline"+"\n"+r"\end{tabular}"+"\n% All gains retain the archived selections; these are not new selections.\n"
put("p0_metric_note.tex",tex)

preserved=load(OUT/"archive_preservation.json")
assert all(sha(ROOT/p)==h for p,h in preserved["input_sha256_at_start"].items())
status=subprocess.check_output(["git","status","--short"],cwd=ROOT).decode()
assert not subprocess.check_output(["git","diff","--name-only"],cwd=ROOT).strip()
put("final_verification.json",{"all_original_tracked_files_unchanged":True,"git_status":status,
    "checkpoint":"S1","scope_after_user_clarification":"Identify historical definitions; stop further AURC recomputation",
    "counterfactual_pilot_selection":"unknown_not_evaluated","gpu_jobs_submitted":0,
    "known_unresolved_items":["target raw responses unavailable in searched locations","older HERBench analysis source hash unresolved",
                              "scikit-learn environment mismatch for future estimator use"],
    "preliminary_outputs":"audit_summary.json and checksum_checks.csv precede remote follow-up; read REPORT.md, checksum_followup.json and development_dependency_resolution.json for final resolution"})
output_files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.name!="MANIFEST.json"}
put("MANIFEST.json",{
    "schema_version":"supplementary-P0-1.0","phase":"P0","utc":datetime.now(timezone.utc).isoformat(),
    "repository":"video-hallucination-diagnosis","commit":summary["commit"],
    "status":"S1_review_historical_metric_identified_no_further_recomputation",
    "scope_clarification":"User asked to identify historical AURC from code/records rather than continue recomputing.",
    "seeds":{"HERBench":20260802,"Perception":20260803,"pilot_bootstrap_formula":"dataset_seed + 1009 * sorted_model_index + 7919 * sorted_candidate_index",
             "pilot_bootstrap_iterations":2000,"familywise_alpha":0.05,"candidate_lower_quantile":0.025,
             "pilot_bootstrap_executed_this_phase":False,"random_computation_this_phase":False},
    "gpu":{"used":False,"jobs_submitted":0,"gpu_hours":0},"environment":load(OUT/"environment.json"),
    "input_worktree_sha256":preserved["input_sha256_at_start"],
    "remote_inputs":"remote_transfers.json, remote_source_transfers.json, remote_dependency_audit.json",
    "output_sha256":output_files,
    "archive_preservation":"Every pre-existing tracked file is byte-identical to audit start.",
    "result_limits":"Pilot gains/bounds/pass-fail and counterfactual choices were not recomputed. No equivalence or speedup measurements were made.",
    "reproduction":"Run collectors and audit in a fresh phase directory. Tools refuse to overwrite differing generated outputs. Python -B prevents bytecode writes into source directories."})
print("Finalized REPORT.md and MANIFEST.json; stopped at S1.")
