"""P0 audit. Read archived files only; refuse to overwrite generated outputs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from canonical_metric import aurc as prefix_aurc
from evaluate_v2_target_pilot import aurc as grouped_aurc

DEFINITIONS = {"paper_prefix_risk_mean": prefix_aurc,
               "helper_tie_grouped_trapezoid": grouped_aurc}
SPECS = {
    "HERBench": {"dir": "herbench_confirmation/8918264.kman.restech.unsw.edu.au",
                 "input": "herbench", "manifest": "herbench_lite_v2_inference.jsonl",
                 "seed": 20260802, "paper_gain": 16.64},
    "Perception": {"dir": "perception_confirmation/8919110.kman.restech.unsw.edu.au",
                   "input": "perception_test/frozen", "manifest": "perception_test_valid500_inference.jsonl",
                   "seed": 20260803, "paper_gain": 18.87},
}

def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(path.read_bytes())
def rel(path): return path.relative_to(ROOT).as_posix()
def readj(path): return json.loads(path.read_text(encoding="utf-8-sig"))
def writej(name, value):
    with (OUT/name).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write("\n")
def writecsv(name, rows):
    data=pd.DataFrame(rows).to_csv(index=False,lineterminator="\n").encode("utf-8")
    if (OUT/name).exists():
        assert (OUT/name).read_bytes()==data, "Refusing to replace differing prior output: "+name
        return
    with (OUT/name).open("xb") as f: f.write(data)
def git(*args):
    return subprocess.check_output(["git",*args],cwd=ROOT)
def csv(path): return pd.read_csv(path,float_precision="round_trip")
def gain(y,raw,score,fn):
    baseline=fn(y,raw)
    return 100*(baseline-fn(y,score))/baseline if baseline>0 else 0.0

def main():
    assert not (OUT/"MANIFEST.json").exists(), "Start a new phase directory for another audit"
    commit=git("rev-parse","HEAD").decode().strip()
    tracked=git("ls-files","-z").decode().split("\0")
    paths=[ROOT/p for p in tracked if p and (ROOT/p).is_file()]
    before={rel(p):sha(p) for p in paths}
    inventory=[]
    for p in paths:
        row={"path":rel(p),"bytes":p.stat().st_size,"sha256_worktree":before[rel(p)]}
        if p.suffix==".csv":
            frame=csv(p); row.update(rows=len(frame),columns=len(frame.columns))
        elif p.suffix==".jsonl":
            row["rows"]=sum(bool(line.strip()) for line in p.open(encoding="utf-8"))
        inventory.append(row)
    writecsv("repository_inventory.csv",inventory)
    checks=[]
    def check(scope, expected, p, recorded):
        item={"scope":scope,"recorded_path":recorded,"local_path":rel(p) if p else "",
              "expected_sha256":expected,"actual_sha256":"","git_blob_sha256":"","status":"missing"}
        if p and p.is_file():
            data=p.read_bytes(); actual=digest(data); item["actual_sha256"]=actual
            if actual==expected:
                item["status"]="exact_match"
            else:
                blob=None
                if rel(p) in before:
                    blob=git("show",commit+":"+rel(p)); item["git_blob_sha256"]=digest(blob)
                if blob is not None and digest(blob)==expected and data.replace(b"\r\n",b"\n")==blob:
                    item["status"]="git_blob_match_worktree_crlf"
                else:
                    item["status"]="content_mismatch"
        checks.append(item)

    splitrows=[]; balance=[]; pilotrows=[]; decisions=[]; testrows=[]; macrorows=[]; bundle_info={}
    for name,spec in SPECS.items():
        archive=ROOT/"results"/spec["dir"]
        inp=ROOT/"input"/spec["input"]
        lock=readj(archive/"promotion_decision_lock.json")
        prelock=readj(archive/"preinference_split_lock.json")
        for filename in ("archive.sha256","promotion_decision_lock.sha256","raw_predictions.sha256"):
            for line in (archive/filename).read_text().splitlines():
                expected,recorded=line.split(maxsplit=1)
                p=archive/Path(recorded.strip()).name
                if filename=="raw_predictions.sha256":
                    candidates=list((OUT/"inputs/recovered").rglob(p.name)) if (OUT/"inputs/recovered").exists() else []
                    p=next((c for c in candidates if sha(c)==expected),None)
                check(name+":"+filename,expected,p,recorded)
        for field,p in (("manifest_sha256",inp/spec["manifest"]),
                        ("pilot_labels_sha256",inp/"pilot_labels.csv"),
                        ("sealed_test_labels_sha256",inp/"sealed_test_labels.csv")):
            check(name+":split_lock",prelock[field],p,str(p))
        seal=readj(next(archive.glob("*PREINFERENCE_SEAL*.json")))
        for recorded,expected in seal["source_hashes"].items():
            p=ROOT/recorded
            if Path(recorded).name==spec["manifest"]: p=inp/spec["manifest"]
            if Path(recorded).name=="preinference_split_lock.json": p=archive/"preinference_split_lock.json"
            check(name+":sealed_source",expected,p,recorded)
        manifest=pd.read_json(inp/spec["manifest"],lines=True)
        assert manifest["question_id"].is_unique
        assert manifest.groupby("video_id")["role"].nunique().max()==1
        for role,label_file in (("calibration","pilot_labels.csv"),("test","sealed_test_labels.csv")):
            sub=manifest.loc[manifest.role.eq(role)]
            labels=csv(inp/label_file)
            joined=sub.merge(labels,on=["question_id","video_id"],validate="one_to_one")
            assert len(joined)==len(sub)==len(labels)
            key="pilot" if role=="calibration" else "test"
            lock_key=key if key+"_questions" in prelock else role
            assert len(sub)==prelock[lock_key+"_questions"]
            assert sub.video_id.nunique()==prelock[lock_key+"_videos"]
            splitrows.append({"dataset":name,"split":key,"questions":len(sub),"videos":sub.video_id.nunique(),
                              "video_overlap":0,"labels_join_exact":True,
                              "expected_raw_response_rows_per_model":len(sub)*4,
                              "raw_response_rows_verified":None})
            if "source_video_balance" in prelock:
                for source,group in sub.groupby("source_dataset"):
                    n=group.video_id.nunique()
                    assert n==prelock["source_video_balance"][source][role]
                    balance.append({"dataset":name,"split":key,"source_collection":source,"videos":n,"questions":len(group)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                bundle=joblib.load(archive/"pilot_model_bundle.joblib")
                bundle_info[name]={"keys":list(bundle),"feature_columns":bundle["feature_columns"],
                                   "base_columns":bundle["base_columns"],"scalar_columns":bundle["scalar_columns"],
                                   "target_model_ids":bundle["target_model_ids"],
                                   "pilot_row_predictions_saved":False,"pilot_features_saved":False,
                                   "inspection_only_no_prediction_or_refitting":True}
            except Exception as e:
                bundle_info[name]={"load_error":repr(e)}
            bundle_info[name]["warnings"]=sorted(set(str(w.message) for w in caught))
        old_pilot=csv(archive/"pilot_candidate_metrics.csv")
        for index,model in enumerate(sorted(lock["decisions"])):
            chosen=lock["decisions"][model]
            decisions.append({"dataset":name,"model_id":model,"archived_selection":chosen["selected"],
                              "prefix_selection":None,"would_differ":None,
                              "status":"unverifiable_missing_pilot_responses_and_scores"})
            for offset,candidate in enumerate(sorted(lock["candidate_library"])):
                old=old_pilot.loc[old_pilot.model_id.eq(model)&old_pilot.candidate.eq(candidate)].iloc[0]
                for definition in DEFINITIONS:
                    is_old=definition=="helper_tie_grouped_trapezoid"
                    selected=candidate==chosen["selected"]
                    pilotrows.append({"dataset":name,"model_id":model,"candidate":candidate,"aurc_definition":definition,
                        "bootstrap_seed":spec["seed"]+1009*index+7919*offset,"iterations":lock["bootstrap_iterations"],
                        "lower_quantile":lock["familywise_alpha"]/len(lock["candidate_library"]),
                        "strict_lower_bound_margin_percent":lock["candidate_lower_bound_margins_percent"][candidate],
                        "archived_gain_percent":float(old.pilot_gain_percent) if is_old else None,
                        "archived_lower_bound_percent":chosen["multiplicity_adjusted_lower_bound_percent"] if is_old and selected else None,
                        "archived_pass_known":True if is_old and selected else None,
                        "archived_selected":selected,"recomputed_gain_percent":None,"recomputed_lower_bound_percent":None,
                        "recomputed_pass":None,"status":"unverifiable_missing_pilot_responses_and_scores"})
        frame=csv(archive/"test_scored_predictions.csv")
        assert not frame.duplicated(["model_id","qid"]).any()
        assert frame.groupby("qid").model_id.nunique().eq(3).all()
        allowed=manifest.loc[manifest.role.eq("test"),["question_id","video_id"]]
        merged=frame.merge(allowed,left_on=["qid","group_id"],right_on=["question_id","video_id"],validate="many_to_one")
        assert len(merged)==len(frame)==len(allowed)*3
        archived_metrics=csv(archive/"test_model_metrics.csv")
        for model,group in frame.groupby("model_id",sort=True):
            y=group.correct.to_numpy(int); raw=group.raw_confidence.to_numpy(float)
            chosen=lock["decisions"][model]["selected"]
            col={"raw_fallback":"raw_confidence","target_logistic":"score__target_full_graph_logistic",
                 "source_trace_hgb":"score__source_trace_hgb"}[chosen]
            assert group.promotion_decision.eq(chosen).all()
            assert np.array_equal(group.trace_rc_score.to_numpy(),group[col].to_numpy())
            for scorecol in [c for c in frame if c.startswith("score__")]:
                method=scorecol.removeprefix("score__")
                score=group[scorecol].to_numpy(float)
                for definition,fn in DEFINITIONS.items():
                    a=fn(y,raw); b=fn(y,score); g=gain(y,raw,score,fn)
                    archived=archived_metrics.loc[archived_metrics.model_id.eq(model)&archived_metrics.method.eq(method)].iloc[0]
                    delta=g-float(archived.relative_aurc_reduction_percent)
                    if definition=="paper_prefix_risk_mean": assert abs(delta)<1e-8,(name,model,method,delta)
                    testrows.append({"dataset":name,"model_id":model,"method":method,"aurc_definition":definition,
                        "questions":len(group),"videos":group.group_id.nunique(),"accuracy":float(y.mean()),
                        "raw_aurc":a,"candidate_aurc":b,"relative_gain_percent":g,"archived_selection":chosen,
                        "raw_exact_tie_excess_rows":len(raw)-len(np.unique(raw)),
                        "candidate_exact_tie_excess_rows":len(score)-len(np.unique(score)),
                        "difference_from_archived_prefix_gain_pp":delta})
        for definition in DEFINITIONS:
            selected=[r for r in testrows if r["dataset"]==name and r["method"]=="TRACE-RC" and r["aurc_definition"]==definition]
            value=float(np.mean([r["relative_gain_percent"] for r in selected]))
            if definition=="paper_prefix_risk_mean": assert round(value,2)==spec["paper_gain"]
            macrorows.append({"dataset":name,"aurc_definition":definition,"policy":"fixed_archived_selection",
                              "mean_relative_gain_percent":value,"worst_model_gain_percent":min(r["relative_gain_percent"] for r in selected),
                              "paper_reported_gain_percent":spec["paper_gain"]})

    # Check both existing posthoc manifests without silently normalizing input bytes.
    for filename in ("analysis_manifest.json","training_free_analysis_manifest.json"):
        parent=ROOT/"results/posthoc_robustness"; data=readj(parent/filename)
        for key in ("input_sha256","output_sha256"):
            for recorded,expected in data.get(key,{}).items():
                if key=="output_sha256": p=parent/recorded
                elif recorded.startswith("trace_rc_route/"):
                    tail=recorded.removeprefix("trace_rc_route/")
                    p=ROOT/tail
                    if tail in {"input/herbench_lite_v2_inference.jsonl","input/pilot_labels.csv","input/sealed_test_labels.csv"}:
                        p=ROOT/"input/herbench"/Path(tail).name
                elif "mvbench_inference" in recorded:
                    p=OUT/"inputs/mvbench"/Path(recorded).name
                else: p=None
                check(filename+":"+key,expected,p,recorded)
    mvrows=[]; mvaudits=[]
    for p in sorted((OUT/"inputs/mvbench").glob("*.jsonl")):
        counters=defaultdict(Counter); keys=set(); questions=set(); media=set(); bad=[]; allviews=defaultdict(set)
        for line_number,line in enumerate(p.open(encoding="utf-8"),1):
            r=json.loads(line); probs=np.asarray(r["candidate_probabilities"],float)
            key=(r["model_id"],r["question_id"],r["view_id"])
            if key in keys or r["status"]!="ok" or not np.isfinite(probs).all() or np.any(probs<0) or not np.isclose(probs.sum(),1):
                bad.append(line_number)
            keys.add(key); questions.add(r["question_id"]); media.add(r["media_path"])
            allviews[r["question_id"]].add(r["view_id"])
            counters[(r["model_id"],r["view_id"],r["frame_budget"],len(r["frame_indices"]),r["input_visual_frames"],r["scoring_method"],r["videollm_calls"],r["decoding"]["generated"])]["rows"]+=1
        for k,v in counters.items():
            mvrows.append(dict(zip(["model_id","view","frame_budget","frame_indices_count","input_visual_frames","scoring_method","forward_calls","generated"],k),rows=v["rows"]))
        mvaudits.append({"path":rel(p),"sha256":sha(p),"rows":line_number,"questions":len(questions),
                         "unique_media_paths":len(media),"bad_rows":bad,"question_view_counts":dict(Counter(len(v) for v in allviews.values()))})
        assert not bad and all(len(v)==4 for v in allviews.values())

    # The stable input-row tie rule is material; retain it in all new evaluations.
    metric_tests=[]
    for y,s in (([1,0],[.5,.5]),([0,1],[.5,.5]),([1,0,1],[.9,.8,.7]),([1,1],[1,0])):
        metric_tests.append({"correct":y,"scores":s,"prefix":prefix_aurc(np.array(y),np.array(s)),
                             "trapezoid":grouped_aurc(np.array(y),np.array(s))})
    assert metric_tests[0]["prefix"]==.25 and metric_tests[1]["prefix"]==.75
    assert metric_tests[0]["trapezoid"]==metric_tests[1]["trapezoid"]==.5
    writej("metric_contract_checks.json",{"cases":metric_tests,"tie_rule":"descending score; ties retain current input row order",
        "bootstrap_tie_order":"concatenate full video blocks in RNG sampled order, preserving original row order inside each block",
        "canonical_function":"src/evaluate_probe_escalation.py:aurc","archived_helper":"src/evaluate_v2_target_pilot.py:aurc",
        "future_policy":"P2+ uses canonical_metric.aurc throughout selection, bootstrap and reporting; split 0 retains archived selections"})
    writecsv("checksum_checks.csv",checks)
    writecsv("split_inventory.csv",splitrows)
    writecsv("herbench_source_balance.csv",balance)
    writecsv("pilot_candidate_audit.csv",pilotrows)
    writecsv("decision_comparison.csv",decisions)
    writecsv("test_metrics_both_definitions.csv",testrows)
    writecsv("test_macro_gains.csv",macrorows)
    writecsv("mvbench_views.csv",mvrows)
    writej("mvbench_raw_audit.json",mvaudits)
    writej("bundle_inspection.json",bundle_info)
    unchanged={p:sha(ROOT/p)==h for p,h in before.items()}
    assert all(unchanged.values())
    writej("archive_preservation.json",{"tracked_files_checked":len(unchanged),"all_byte_identical_to_start":True,
                                        "changed_files":[],"input_sha256_at_start":before})
    summary={"phase":"P0","status":"S1_partial_missing_target_raw", "commit":commit,
             "checkpoint":"S1; do not start P1 or P2 without review",
             "checksum_status_counts":dict(Counter(r["status"] for r in checks)),
             "test_macro_gains":macrorows,"pilot_decision_recomputation":"blocked_missing_target_raw_and_pilot_scores",
             "mvbench":"found_and_copied_from_Katana","gpu_jobs_submitted":0,
             "tests":"split counts and joins, no video overlap, fixed selections, prefix archived gains, MVBench raw integrity, stable tie examples, archive byte preservation"}
    writej("audit_summary.json",summary)
    versions={p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","scipy","joblib")}
    writej("environment.json",{"executable":sys.executable,"python":sys.version,"platform":platform.platform(),"packages":versions,
         "bundle_sklearn_version":"1.7.2 from estimator warnings; bundles inspected only",
         "gpu":{"used":False,"gpu_hours":0,"historical_gpu":"H200 from sealed PBS configuration; not a timing measurement"}})
    # Report and final checksums are emitted by finalize_p0.py after code review.
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
