#!/usr/bin/env python3
"""Evaluate the locked pilot-gated TRACE-Eco V2 development amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_response_graph_features as trace
import evaluate_trace_budgeted_contracts as v1


PILOT_SEED = "trace-eco-pilot-v2-20260801"
PILOT_GROUPS = 64


def pilot_group_ids(target: pd.DataFrame, dataset: str) -> list[str]:
    groups = sorted({str(value) for value in target["group_id"]})
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{PILOT_SEED}|{dataset}|{group}".encode()).hexdigest(),
            group,
        ),
    )
    return ranked[: min(PILOT_GROUPS, len(ranked))]


def source_contract_audit(
    table: pd.DataFrame, held_out: str
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    source = table.loc[table["dataset"].ne(held_out)].reset_index(drop=True)
    target = table.loc[table["dataset"].eq(held_out)].reset_index(drop=True)
    models = sorted(target["model_id"].unique())
    audit_rows: list[dict[str, object]] = []
    target_scores: dict[str, np.ndarray] = {}
    for contract in v1.CONTRACTS:
        columns = v1.columns_for_contract(table, contract)
        transport = v1.transport_columns(columns)
        fitted = trace.fit_graph_logistic(source, columns)
        target_scores[contract.name] = fitted.predict_proba(target[columns])[:, 1]
        rehearsals = {(contract.name, model): [] for model in models}
        for validation_domain in sorted(source["dataset"].unique()):
            inner = source.loc[source["dataset"].ne(validation_domain)].reset_index(drop=True)
            validation_all = source.loc[source["dataset"].eq(validation_domain)].reset_index(drop=True)
            inner_model = trace.fit_graph_logistic(inner, columns)
            validation_score = inner_model.predict_proba(validation_all[columns])[:, 1]
            for model_id, positions in validation_all.groupby("model_id", sort=True).indices.items():
                idx = np.asarray(positions, dtype=int)
                validation = validation_all.iloc[idx]
                reference = inner.loc[inner["model_id"].eq(model_id)]
                radius = trace.support_threshold(inner, model_id, transport)
                shift = trace.distribution_shift(reference, validation, transport)
                if shift <= radius:
                    labels = validation["correct"].to_numpy(int)
                    raw = validation["own_base_confidence"].to_numpy(float)
                    rehearsals[(contract.name, model_id)].append(
                        v1.gain(labels, raw, validation_score[idx])
                    )
        for model_id in models:
            gains = rehearsals[(contract.name, model_id)]
            audit_rows.append(
                {
                    "held_out_dataset": held_out,
                    "model_id": model_id,
                    "contract": contract.name,
                    "calls": contract.calls,
                    "frames": contract.frames,
                    "supported_rehearsals": len(gains),
                    "mean_rehearsal_gain_percent": float(np.mean(gains)) if gains else np.nan,
                    "worst_rehearsal_gain_percent": float(min(gains)) if gains else np.nan,
                    "eligible": v1.eligible(gains),
                    "rehearsal_gains_json": json.dumps(gains, separators=(",", ":")),
                }
            )
    return pd.DataFrame(audit_rows), target_scores


def evaluate_fold(
    table: pd.DataFrame, held_out: str
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    pd.DataFrame,
]:
    source = table.loc[table["dataset"].ne(held_out)].reset_index(drop=True)
    target = table.loc[table["dataset"].eq(held_out)].reset_index(drop=True)
    contract_audit, scores = source_contract_audit(table, held_out)
    selected_groups = pilot_group_ids(target, held_out)
    full = next(contract for contract in v1.CONTRACTS if contract.name == "full_panel")
    full_columns = v1.columns_for_contract(table, full)
    full_transport = v1.transport_columns(full_columns)
    metrics: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = contract_audit.to_dict("records")
    scored_rows: list[pd.DataFrame] = []

    for model_id, positions in target.groupby("model_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=int)
        target_model = target.iloc[idx]
        raw = target_model["own_base_confidence"].to_numpy(float)
        candidates = contract_audit.loc[contract_audit["model_id"].eq(model_id)].to_dict("records")
        selected = v1.choose_contract(candidates)
        score = raw.copy()
        used_graph = False
        pilot_supported = False
        pilot_shift = np.nan
        pilot_radius = np.nan
        pilot_questions = 0
        pilot_groups_used = 0

        if selected is None:
            selected_name = "raw_fallback_without_contract"
            mean_calls = 1.0
            mean_frames = 8.0
        else:
            selected_name = str(selected["contract"])
            contract = next(item for item in v1.CONTRACTS if item.name == selected_name)
            pilot = target_model.loc[target_model["group_id"].astype(str).isin(selected_groups)]
            pilot_questions = len(pilot)
            pilot_groups_used = pilot["group_id"].astype(str).nunique()
            reference = source.loc[source["model_id"].eq(model_id)]
            pilot_radius = trace.support_threshold(source, model_id, full_transport)
            pilot_shift = trace.distribution_shift(reference, pilot, full_transport)
            pilot_supported = bool(pilot_shift <= pilot_radius)
            if pilot_supported:
                score = scores[selected_name][idx]
                used_graph = True
                remainder_calls, remainder_frames = contract.calls, contract.frames
            else:
                remainder_calls, remainder_frames = 1, 8
            n = len(target_model)
            mean_calls = (pilot_questions * full.calls + (n - pilot_questions) * remainder_calls) / n
            mean_frames = (pilot_questions * full.frames + (n - pilot_questions) * remainder_frames) / n

        if not used_graph and not np.array_equal(score, raw):
            raise RuntimeError("pilot-gated exact fallback failed")
        labels = target_model["correct"].to_numpy(int)
        metrics.append(
            {
                "held_out_dataset": held_out,
                "model_id": model_id,
                "method": "TRACE-Eco-Pilot",
                "n": len(target_model),
                "relative_aurc_reduction_percent": v1.gain(labels, raw, score),
                "calls": mean_calls,
                "frames": mean_frames,
                "used_graph": used_graph,
            }
        )
        scored = target_model[
            ["dataset", "qid", "group_id", "model_id", "correct"]
        ].copy()
        scored["raw_confidence"] = raw
        scored["eco_score"] = score
        scored["calls"] = mean_calls
        scored["frames"] = mean_frames
        scored["used_graph"] = used_graph
        scored_rows.append(scored)
        decisions.append(
            {
                "held_out_dataset": held_out,
                "model_id": model_id,
                "contract": "TRACE-Eco-Pilot decision",
                "selected_contract": selected_name,
                "pilot_groups_planned": min(PILOT_GROUPS, target["group_id"].astype(str).nunique()),
                "pilot_groups_used": pilot_groups_used,
                "pilot_questions": pilot_questions,
                "pilot_shift": pilot_shift,
                "pilot_radius": pilot_radius,
                "pilot_supported": pilot_supported,
                "used_graph": used_graph,
                "calls": mean_calls,
                "frames": mean_frames,
                "pilot_group_ids_sha256": hashlib.sha256(
                    "\n".join(selected_groups).encode()
                ).hexdigest(),
            }
        )
    return metrics, decisions, pd.concat(scored_rows, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if v1.sha256(args.table) != lock["input_feature_table_sha256"]:
        raise RuntimeError("feature table hash does not match pilot amendment lock")
    if lock["pilot"]["semantic_groups"] != PILOT_GROUPS:
        raise RuntimeError("pilot group count does not match lock")
    table = pd.read_csv(args.table)
    metrics: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for held_out in sorted(table["dataset"].unique()):
        fold_metrics, fold_audits, _ = evaluate_fold(table, held_out)
        metrics.extend(fold_metrics)
        audits.extend(fold_audits)
    metric_frame = pd.DataFrame(metrics)
    audit_frame = pd.DataFrame(audits)
    summary = metric_frame.groupby("method", as_index=False).agg(
        environments=("relative_aurc_reduction_percent", "size"),
        mean_gain_percent=("relative_aurc_reduction_percent", "mean"),
        worst_gain_percent=("relative_aurc_reduction_percent", "min"),
        positive_environments=("relative_aurc_reduction_percent", lambda x: int((x > 0).sum())),
        negative_environments=("relative_aurc_reduction_percent", lambda x: int((x < 0).sum())),
        mean_calls=("calls", "mean"),
        mean_frames=("frames", "mean"),
    )
    row = summary.iloc[0]
    gate = lock["promotion_gate"]
    conditions = {
        "mean_gain": bool(row.mean_gain_percent >= gate["minimum_mean_gain_percent"]),
        "worst_environment": bool(row.worst_gain_percent >= gate["minimum_worst_environment_gain_percent"]),
        "mean_calls": bool(row.mean_calls <= gate["maximum_mean_calls"]),
        "mean_frames": bool(row.mean_frames <= gate["maximum_mean_model_input_frames"]),
        "exact_fallback": bool(
            metric_frame.loc[~metric_frame["used_graph"], "relative_aurc_reduction_percent"].eq(0.0).all()
        ),
    }
    decision = {
        "analysis_role": lock["analysis_role"],
        "target_labels_used_for_contract_pilot_or_support_decisions": False,
        "model_inference_performed": False,
        "environments": int(row.environments),
        "mean_gain_percent": float(row.mean_gain_percent),
        "worst_gain_percent": float(row.worst_gain_percent),
        "mean_calls": float(row.mean_calls),
        "mean_frames": float(row.mean_frames),
        "graph_used_environments": int(metric_frame["used_graph"].sum()),
        "promotion_conditions": conditions,
        "promotion_gate_pass": bool(all(conditions.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(args.output_dir / "environment_metrics.csv", index=False)
    audit_frame.to_csv(args.output_dir / "pilot_audit.csv", index=False)
    summary.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    (args.output_dir / "promotion_gate.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
