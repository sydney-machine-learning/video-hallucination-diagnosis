"""Two-phase TRACE-RC analysis for sealed HERBench Lite-v2 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
for candidate in (
    HERE,
    WORKSPACE / "probe_route" / "src",
    Path.home() / "probe_route" / "src",
):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import analyze_trace_videomme_confirmation as common  # noqa: E402
import evaluate_response_graph_features as trace  # noqa: E402
from evaluate_v2_target_pilot import (  # noqa: E402
    SOURCE_CANDIDATES,
    make_logistic,
    promote,
    relative_gain,
)


TARGET_DATASET = "HERBench-Lite-v2"
SEED = 20260802
PRIMARY_MINIMUM_GAIN = 1.0
PRIMARY_WORST_GAIN = -1.0
PROMOTION_SCHEMA_VERSION = "trace-rc-herbench-promotion-lock-1.0"
CONFIRMATION_SCHEMA_VERSION = "trace-rc-herbench-confirmation-gate-1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen" in lowered:
        return "qwen"
    if "internvl" in lowered:
        return "internvl"
    if "llava" in lowered:
        return "llava"
    raise ValueError(f"unrecognized model family: {model_id}")


def validate_family_alignment(source_models: list[str], target_models: list[str]) -> None:
    source = [family(model) for model in source_models]
    target = [family(model) for model in target_models]
    if source != target or sorted(source) != ["internvl", "llava", "qwen"]:
        raise ValueError(f"source/target family order mismatch: {source} vs {target}")


def load_manifest(path: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    frame = pd.DataFrame(rows)
    if len(frame) != 1971 or frame["video_id"].nunique() != 68:
        raise ValueError("unexpected HERBench manifest inventory")
    if any(column in frame for column in ("answer", "answer_index", "gold")):
        raise ValueError("label leakage in inference manifest")
    return frame


def target_metadata(manifest: pd.DataFrame, labels_path: Path, role: str) -> pd.DataFrame:
    selected = manifest.loc[manifest["role"].eq(role)].copy()
    labels = pd.read_csv(labels_path)
    merged = selected.merge(
        labels[["question_id", "video_id", "gold_candidate_index"]],
        on=["question_id", "video_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(selected) or len(labels) != len(selected):
        raise ValueError(f"{role} label inventory mismatch")
    return pd.DataFrame(
        {
            "dataset": TARGET_DATASET,
            "qid": merged["question_id"].astype(str),
            "group_id": merged["video_id"].astype(str),
            "gold": merged["gold_candidate_index"].astype(int),
            "candidate_count": merged["candidates"].map(len).astype(int),
            "task": merged["task"].astype(str),
            "role": role,
        }
    )


def build_tables(args: argparse.Namespace, labels: Path, role: str):
    source_values, source_models = trace.read_results(args.development_result)
    source_metadata = trace.load_metadata(args.feature, args.vhd_scored, args.temp_scored)
    source, source_base_columns = trace.graph_table(
        source_metadata, source_values, source_models
    )
    target_values, target_models = trace.read_results(args.target_result)
    manifest = load_manifest(args.manifest)
    target_meta = target_metadata(manifest, labels, role)
    graph_meta = target_meta[
        ["dataset", "qid", "group_id", "gold", "candidate_count"]
    ]
    target, target_base_columns = trace.graph_table(
        graph_meta, target_values, target_models
    )
    validate_family_alignment(source_models, target_models)
    if source_base_columns != target_base_columns:
        raise RuntimeError("source/target base graph schema mismatch")
    target = target.merge(
        target_meta[["qid", "task", "role"]], on="qid", how="left", validate="many_to_one"
    )
    feature_columns = [
        column
        for column in source.columns
        if column not in {"dataset", "qid", "group_id", "model_id", "correct"}
    ]
    target_features = [
        column
        for column in target.columns
        if column
        not in {"dataset", "qid", "group_id", "model_id", "correct", "task", "role"}
    ]
    if feature_columns != target_features:
        raise RuntimeError("source/target response graph feature mismatch")
    return source, target, source_models, target_models, feature_columns, source_base_columns


def fit_base_model(frame: pd.DataFrame, columns: list[str]):
    model = make_logistic()
    model.fit(frame[columns], frame["correct"].astype(int))
    return model


def pilot_phase(args: argparse.Namespace) -> int:
    source, pilot, source_models, target_models, columns, base_columns = build_tables(
        args, args.pilot_labels, "calibration"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_models_fit = {
        "source_trace_logistic": trace.fit_graph_logistic(source, columns),
        "source_trace_hgb": common.fit_hgb(source, columns),
    }
    # Promotion helper uses these historical column names.
    pilot["score__full_graph_logistic"] = source_models_fit[
        "source_trace_logistic"
    ].predict_proba(pilot[columns])[:, 1]
    pilot["score__full_graph_gradient_boosting"] = source_models_fit[
        "source_trace_hgb"
    ].predict_proba(pilot[columns])[:, 1]
    pilot["score__raw_confidence"] = pilot["own_base_confidence"].to_numpy(float)

    scalar_columns = [
        name
        for name in columns
        if name
        in {
            "own_base_confidence",
            "own_base_entropy",
            "own_base_margin",
            "candidate_count_scaled",
        }
    ]
    decisions: dict[str, dict[str, Any]] = {}
    target_models_fit: dict[str, Any] = {}
    base_models_fit: dict[str, Any] = {}
    scalar_models_fit: dict[str, Any] = {}
    pilot_rows = []
    for model_index, model_id in enumerate(target_models):
        group = pilot.loc[pilot["model_id"].eq(model_id)].reset_index(drop=True)
        decision = promote(
            group,
            columns,
            args.bootstrap_iterations,
            args.alpha,
            SEED + 1009 * model_index,
            "robust_safe",
        )
        target_models_fit[model_id] = decision.target_model
        base_models_fit[model_id] = fit_base_model(group, base_columns)
        scalar_models_fit[model_id] = fit_base_model(group, scalar_columns)
        decisions[model_id] = {
            "model_family": family(model_id),
            "selected": decision.selected,
            "pilot_gain_percent": decision.pilot_gain,
            "multiplicity_adjusted_lower_bound_percent": decision.lower_bound,
            "pilot_questions": int(len(group)),
            "pilot_videos": int(group["group_id"].nunique()),
        }
        raw = group["score__raw_confidence"].to_numpy(float)
        for candidate_name, candidate_score in decision.candidate_scores.items():
            pilot_rows.append(
                {
                    "model_id": model_id,
                    "candidate": candidate_name,
                    "pilot_gain_percent": relative_gain(
                        group["correct"].to_numpy(int), raw, candidate_score
                    ),
                    "promoted": candidate_name == decision.selected,
                }
            )
        if decision.selected == "raw_fallback":
            if decision.lower_bound != 0.0:
                raise RuntimeError("fallback decision lower bound must be zero")

    model_bundle = {
        "source_models": source_models_fit,
        "target_models": target_models_fit,
        "target_base_models": base_models_fit,
        "target_scalar_models": scalar_models_fit,
        "feature_columns": columns,
        "base_columns": base_columns,
        "scalar_columns": scalar_columns,
        "source_model_ids": source_models,
        "target_model_ids": target_models,
    }
    bundle_path = args.output_dir / "pilot_model_bundle.joblib"
    joblib.dump(model_bundle, bundle_path, compress=3)
    pd.DataFrame(pilot_rows).to_csv(args.output_dir / "pilot_candidate_metrics.csv", index=False)
    prelock = json.loads(args.preinference_lock.read_text(encoding="utf-8"))
    lock = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "phase": "pilot_complete_before_test_label_opening",
        "target_dataset": TARGET_DATASET,
        "familywise_alpha": args.alpha,
        "bootstrap_iterations": args.bootstrap_iterations,
        "candidate_library": [
            "source_trace_hgb",
            "target_logistic",
        ],
        "candidate_lower_bound_margins_percent": {
            "source_trace_hgb": 0.5,
            "target_logistic": 0.0,
        },
        "decisions": decisions,
        "manifest_sha256": sha256_file(args.manifest),
        "pilot_labels_sha256": sha256_file(args.pilot_labels),
        "sealed_test_labels_sha256_from_prelock": prelock[
            "sealed_test_labels_sha256"
        ],
        "model_bundle_sha256": sha256_file(bundle_path),
    }
    lock_path = args.output_dir / "promotion_decision_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "promotion_decision_lock.sha256").write_text(
        f"{sha256_file(lock_path)}  {lock_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


def method_metrics(frame: pd.DataFrame, score_columns: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for method, scores in score_columns.items():
        for model_id, positions in frame.groupby("model_id", sort=True).indices.items():
            index = np.asarray(positions, dtype=int)
            group = frame.iloc[index]
            labels = group["correct"].to_numpy(int)
            raw = group["own_base_confidence"].to_numpy(float)
            score = np.asarray(scores, dtype=float)[index]
            raw_aurc = trace.aurc(labels, raw)
            value = trace.aurc(labels, score)
            rows.append(
                {
                    "method": method,
                    "model_id": model_id,
                    "n": len(group),
                    "videos": int(group["group_id"].nunique()),
                    "accuracy": float(labels.mean()),
                    "raw_aurc": raw_aurc,
                    "aurc": value,
                    "relative_aurc_reduction_percent": 100.0
                    * (raw_aurc - value)
                    / raw_aurc,
                    "auroc": float(roc_auc_score(labels, score)),
                    "ece15": common.ece(labels, np.clip(score, 0.0, 1.0)),
                    "brier": float(np.mean(np.square(np.clip(score, 0.0, 1.0) - labels))),
                    "risk_at_50pct": common.risk_at_coverage(labels, score, 0.5),
                }
            )
    return pd.DataFrame(rows)


def primary_bootstrap(
    frame: pd.DataFrame, raw: np.ndarray, selected: np.ndarray, iterations: int
) -> np.ndarray:
    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    positions = {
        group: np.flatnonzero(frame["group_id"].astype(str).to_numpy() == group)
        for group in groups
    }
    rng = np.random.default_rng(SEED)
    output = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        take = np.concatenate([positions[group] for group in sampled])
        sample = frame.iloc[take].reset_index(drop=True)
        sample_raw = raw[take]
        sample_selected = selected[take]
        gains = []
        for _, model_positions in sample.groupby("model_id", sort=True).indices.items():
            idx = np.asarray(model_positions, dtype=int)
            gains.append(
                relative_gain(
                    sample.iloc[idx]["correct"].to_numpy(int),
                    sample_raw[idx],
                    sample_selected[idx],
                )
            )
        output[iteration] = float(np.mean(gains))
    return output


def final_phase(args: argparse.Namespace) -> int:
    lock_path = args.output_dir / "promotion_decision_lock.json"
    lock_digest_path = args.output_dir / "promotion_decision_lock.sha256"
    bundle_path = args.output_dir / "pilot_model_bundle.joblib"
    expected_lock_digest, expected_lock_name = (
        lock_digest_path.read_text(encoding="utf-8").strip().split(maxsplit=1)
    )
    if expected_lock_name != lock_path.name:
        raise RuntimeError("promotion lock digest names an unexpected file")
    if sha256_file(lock_path) != expected_lock_digest:
        raise RuntimeError("promotion decision lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    prelock = json.loads(args.preinference_lock.read_text(encoding="utf-8"))
    if sha256_file(args.test_labels) != prelock["sealed_test_labels_sha256"]:
        raise RuntimeError("sealed test labels differ from pre-inference lock")
    if sha256_file(bundle_path) != lock["model_bundle_sha256"]:
        raise RuntimeError("pilot model bundle hash mismatch")

    # Only after the pilot decision and its fitted model have passed their
    # digest checks may the final phase parse the sealed answer file.
    source, test, source_ids, target_ids, columns, base_columns = build_tables(
        args, args.test_labels, "test"
    )
    del source, source_ids, base_columns
    bundle = joblib.load(bundle_path)
    if columns != bundle["feature_columns"] or target_ids != bundle["target_model_ids"]:
        raise RuntimeError("final feature/model schema differs from pilot")

    raw = test["own_base_confidence"].to_numpy(float)
    source_logistic = bundle["source_models"]["source_trace_logistic"].predict_proba(
        test[columns]
    )[:, 1]
    source_hgb = bundle["source_models"]["source_trace_hgb"].predict_proba(
        test[columns]
    )[:, 1]
    target_score = np.empty(len(test), dtype=float)
    target_base = np.empty(len(test), dtype=float)
    target_scalar = np.empty(len(test), dtype=float)
    selected = raw.copy()
    selected_names = np.empty(len(test), dtype=object)
    for model_id, positions in test.groupby("model_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=int)
        target_score[idx] = bundle["target_models"][model_id].predict_proba(
            test.iloc[idx][columns]
        )[:, 1]
        target_base[idx] = bundle["target_base_models"][model_id].predict_proba(
            test.iloc[idx][bundle["base_columns"]]
        )[:, 1]
        target_scalar[idx] = bundle["target_scalar_models"][model_id].predict_proba(
            test.iloc[idx][bundle["scalar_columns"]]
        )[:, 1]
        decision = lock["decisions"][model_id]["selected"]
        if decision == "source_trace_logistic":
            selected[idx] = source_logistic[idx]
        elif decision == "source_trace_hgb":
            selected[idx] = source_hgb[idx]
        elif decision == "target_logistic":
            selected[idx] = target_score[idx]
        elif decision == "raw_fallback":
            if not np.array_equal(selected[idx], raw[idx]):
                raise RuntimeError(f"{model_id}: exact fallback failed")
        else:
            raise RuntimeError(f"unknown promotion decision {decision}")
        selected_names[idx] = decision

    methods = {
        "raw_confidence": raw,
        "target_scalar_logistic": target_scalar,
        "target_base_graph_logistic": target_base,
        "source_trace_logistic": source_logistic,
        "source_trace_hgb": source_hgb,
        "target_full_graph_logistic": target_score,
        "TRACE-RC": selected,
    }
    metrics = method_metrics(test, methods)
    trace_metrics = metrics.loc[metrics["method"].eq("TRACE-RC")]
    mean_gain = float(trace_metrics["relative_aurc_reduction_percent"].mean())
    worst_gain = float(trace_metrics["relative_aurc_reduction_percent"].min())
    distribution = primary_bootstrap(test, raw, selected, args.final_iterations)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    accepted = sum(
        decision["selected"] != "raw_fallback"
        for decision in lock["decisions"].values()
    )
    checks = {
        "macro_gain_at_least_one_percent": mean_gain >= PRIMARY_MINIMUM_GAIN,
        "grouped_ci_lower_above_zero": float(lower) > 0.0,
        "worst_model_gain_at_least_minus_one_percent": worst_gain >= PRIMARY_WORST_GAIN,
        "at_least_one_environment_promoted": accepted >= 1,
        "all_rejections_exact_fallback": all(
            decision["selected"] != "raw_fallback"
            or np.array_equal(
                selected[
                    test["model_id"].eq(model_id).to_numpy()
                ],
                raw[test["model_id"].eq(model_id).to_numpy()],
            )
            for model_id, decision in lock["decisions"].items()
        ),
    }
    gate = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "target_dataset": TARGET_DATASET,
        "test_questions": int(test["qid"].nunique()),
        "test_videos": int(test["group_id"].nunique()),
        "models": len(target_ids),
        "promoted_environments": accepted,
        "macro_mean_relative_aurc_reduction_percent": mean_gain,
        "video_grouped_bootstrap_95ci_percent": [float(lower), float(upper)],
        "worst_model_gain_percent": worst_gain,
        "checks": checks,
        "primary_gate_pass": all(checks.values()),
        "bootstrap_iterations": args.final_iterations,
        "bootstrap_seed": SEED,
    }
    prediction = test[
        ["dataset", "qid", "group_id", "task", "model_id", "correct"]
    ].copy()
    prediction["raw_confidence"] = raw
    prediction["trace_rc_score"] = selected
    prediction["promotion_decision"] = selected_names.tolist()
    for name, values in methods.items():
        prediction[f"score__{name}"] = values
    task_rows = []
    for (model_id, task), group in prediction.groupby(["model_id", "task"], sort=True):
        labels = group["correct"].to_numpy(int)
        base_score = group["raw_confidence"].to_numpy(float)
        method_score = group["trace_rc_score"].to_numpy(float)
        task_rows.append(
            {
                "model_id": model_id,
                "task": task,
                "n": len(group),
                "accuracy": float(labels.mean()),
                "relative_aurc_reduction_percent": relative_gain(
                    labels, base_score, method_score
                ),
            }
        )
    metrics.to_csv(args.output_dir / "test_model_metrics.csv", index=False)
    prediction.to_csv(args.output_dir / "test_scored_predictions.csv", index=False)
    pd.DataFrame(task_rows).to_csv(args.output_dir / "test_task_metrics.csv", index=False)
    pd.DataFrame({"macro_gain_percent": distribution}).to_csv(
        args.output_dir / "test_bootstrap_distribution.csv", index=False
    )
    (args.output_dir / "confirmation_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    archive_files = [
        path
        for path in args.output_dir.rglob("*")
        if path.is_file() and path.name != "archive.sha256"
    ]
    with (args.output_dir / "archive.sha256").open("w", encoding="utf-8") as handle:
        for path in sorted(archive_files):
            handle.write(f"{sha256_file(path)}  {path.relative_to(args.output_dir).as_posix()}\n")
    print(json.dumps(gate, indent=2, sort_keys=True))
    # A scientifically negative gate is a completed experiment, not an
    # execution failure.  The JSON records the outcome and the batch remains
    # successful so that the single queued allocation always archives it.
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature", type=Path, action="append", required=True)
    parser.add_argument("--development-result", type=Path, action="append", required=True)
    parser.add_argument("--target-result", type=Path, action="append", required=True)
    parser.add_argument("--vhd-scored", type=Path, required=True)
    parser.add_argument("--temp-scored", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preinference-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    pilot = subparsers.add_parser("pilot")
    add_common_arguments(pilot)
    pilot.add_argument("--pilot-labels", type=Path, required=True)
    pilot.add_argument("--bootstrap-iterations", type=int, default=2000)
    pilot.add_argument("--alpha", type=float, default=0.05)
    final = subparsers.add_parser("final")
    add_common_arguments(final)
    final.add_argument("--test-labels", type=Path, required=True)
    final.add_argument("--final-iterations", type=int, default=10000)
    args = parser.parse_args()
    if args.phase == "pilot":
        return pilot_phase(args)
    return final_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
