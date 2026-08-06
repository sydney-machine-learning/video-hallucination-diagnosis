#!/usr/bin/env python3
"""Evaluate locked TRACE-Eco acquisition contracts on development datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_response_graph_features as trace


META = {"dataset", "qid", "group_id", "model_id", "correct"}
MIN_REHEARSALS = 2
MIN_WORST_GAIN = 0.0
MIN_MEAN_GAIN = 5.0


@dataclass(frozen=True)
class Contract:
    name: str
    roles: tuple[str, ...]
    calls: int
    frames: int


CONTRACTS = (
    Contract("base_panel", (), 3, 24),
    Contract("base_plus_invariant", ("invariant",), 6, 48),
    Contract("base_plus_removal", ("removal",), 6, 48),
    Contract("base_plus_addition", ("addition",), 6, 72),
    Contract("base_plus_invariant_removal", ("invariant", "removal"), 9, 72),
    Contract("base_plus_invariant_addition", ("invariant", "addition"), 9, 96),
    Contract("base_plus_addition_removal", ("addition", "removal"), 9, 96),
    Contract("full_panel", ("invariant", "addition", "removal"), 12, 120),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in META and not column.startswith("rank__")
    ]


def columns_for_contract(frame: pd.DataFrame, contract: Contract) -> list[str]:
    columns = all_feature_columns(frame)
    typed_prefixes = tuple(f"{role}_" for role in trace.ROLES[1:])
    selected_prefixes = tuple(f"{role}_" for role in contract.roles)
    return [
        column
        for column in columns
        if not column.startswith(typed_prefixes)
        or column.startswith(selected_prefixes)
    ]


def transport_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if not column.startswith("model_identity_") and "_is_self" not in column
    ]


def gain(labels: np.ndarray, raw: np.ndarray, score: np.ndarray) -> float:
    raw_aurc = trace.aurc(labels, raw)
    return 100.0 * (raw_aurc - trace.aurc(labels, score)) / raw_aurc


def eligible(rehearsal_gains: list[float]) -> bool:
    return bool(
        len(rehearsal_gains) >= MIN_REHEARSALS
        and min(rehearsal_gains) >= MIN_WORST_GAIN
        and float(np.mean(rehearsal_gains)) >= MIN_MEAN_GAIN
    )


def choose_contract(rows: list[dict[str, object]]) -> dict[str, object] | None:
    candidates = [row for row in rows if bool(row["eligible"])]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (int(row["frames"]), int(row["calls"]), str(row["contract"])),
    )


def evaluate_outer_fold(
    table: pd.DataFrame, held_out: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source = table.loc[table["dataset"].ne(held_out)].reset_index(drop=True)
    target = table.loc[table["dataset"].eq(held_out)].reset_index(drop=True)
    models = sorted(target["model_id"].unique())
    contract_scores: dict[str, np.ndarray] = {}
    contract_rows: list[dict[str, object]] = []
    rehearsal_by_contract_model: dict[tuple[str, str], list[float]] = {}

    for contract in CONTRACTS:
        columns = columns_for_contract(table, contract)
        transport = transport_columns(columns)
        fitted = trace.fit_graph_logistic(source, columns)
        contract_scores[contract.name] = fitted.predict_proba(target[columns])[:, 1]
        rehearsal_by_contract_model.update(
            {(contract.name, model_id): [] for model_id in models}
        )
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
                    rehearsal_by_contract_model[(contract.name, model_id)].append(
                        gain(labels, raw, validation_score[idx])
                    )

        for model_id in models:
            rehearsals = rehearsal_by_contract_model[(contract.name, model_id)]
            contract_rows.append(
                {
                    "held_out_dataset": held_out,
                    "model_id": model_id,
                    "contract": contract.name,
                    "calls": contract.calls,
                    "frames": contract.frames,
                    "supported_rehearsals": len(rehearsals),
                    "mean_rehearsal_gain_percent": float(np.mean(rehearsals)) if rehearsals else np.nan,
                    "worst_rehearsal_gain_percent": float(min(rehearsals)) if rehearsals else np.nan,
                    "eligible": eligible(rehearsals),
                    "rehearsal_gains_json": json.dumps(rehearsals, separators=(",", ":")),
                }
            )

    metric_rows: list[dict[str, object]] = []
    raw_all = target["own_base_confidence"].to_numpy(float)
    for model_id, positions in target.groupby("model_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=int)
        target_model = target.iloc[idx]
        labels = target_model["correct"].to_numpy(int)
        raw = raw_all[idx]

        for contract in CONTRACTS:
            score = contract_scores[contract.name][idx]
            metric_rows.append(
                {
                    "held_out_dataset": held_out,
                    "model_id": model_id,
                    "method": f"static::{contract.name}",
                    "n": len(idx),
                    "relative_aurc_reduction_percent": gain(labels, raw, score),
                    "calls": contract.calls,
                    "frames": contract.frames,
                    "used_graph": True,
                }
            )

        candidates = [
            row
            for row in contract_rows
            if row["held_out_dataset"] == held_out and row["model_id"] == model_id
        ]
        selected = choose_contract(candidates)
        eco = raw.copy()
        used_graph = False
        target_supported = False
        target_shift = np.nan
        target_radius = np.nan
        if selected is None:
            selected_name = "raw_fallback_without_contract"
            calls, frames = 1, 8
        else:
            selected_name = str(selected["contract"])
            contract = next(item for item in CONTRACTS if item.name == selected_name)
            columns = columns_for_contract(table, contract)
            transport = transport_columns(columns)
            reference = source.loc[source["model_id"].eq(model_id)]
            target_radius = trace.support_threshold(source, model_id, transport)
            target_shift = trace.distribution_shift(reference, target_model, transport)
            target_supported = bool(target_shift <= target_radius)
            calls, frames = contract.calls, contract.frames
            if target_supported:
                eco = contract_scores[selected_name][idx]
                used_graph = True
        if not used_graph and not np.array_equal(eco, raw):
            raise RuntimeError("TRACE-Eco exact fallback failed")
        metric_rows.append(
            {
                "held_out_dataset": held_out,
                "model_id": model_id,
                "method": "TRACE-Eco",
                "n": len(idx),
                "relative_aurc_reduction_percent": gain(labels, raw, eco),
                "calls": calls,
                "frames": frames,
                "used_graph": used_graph,
            }
        )
        contract_rows.append(
            {
                "held_out_dataset": held_out,
                "model_id": model_id,
                "contract": "TRACE-Eco decision",
                "calls": calls,
                "frames": frames,
                "supported_rehearsals": 0 if selected is None else selected["supported_rehearsals"],
                "mean_rehearsal_gain_percent": np.nan if selected is None else selected["mean_rehearsal_gain_percent"],
                "worst_rehearsal_gain_percent": np.nan if selected is None else selected["worst_rehearsal_gain_percent"],
                "eligible": selected is not None,
                "rehearsal_gains_json": "[]" if selected is None else selected["rehearsal_gains_json"],
                "selected_contract": selected_name,
                "target_shift": target_shift,
                "target_radius": target_radius,
                "target_supported": target_supported,
                "used_graph": used_graph,
            }
        )
    return metric_rows, contract_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if sha256(args.table) != lock["input_feature_table_sha256"]:
        raise RuntimeError("feature table hash does not match TRACE-Eco lock")
    if [contract.name for contract in CONTRACTS] != lock["contracts"]:
        raise RuntimeError("contract inventory does not match TRACE-Eco lock")
    table = pd.read_csv(args.table)
    metrics: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for held_out in sorted(table["dataset"].unique()):
        fold_metrics, fold_audits = evaluate_outer_fold(table, held_out)
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
    ).sort_values(["mean_gain_percent", "mean_calls"], ascending=[False, True])

    eco = summary.loc[summary["method"].eq("TRACE-Eco")].iloc[0]
    gate = lock["promotion_gate"]
    conditions = {
        "mean_gain": bool(eco.mean_gain_percent >= gate["minimum_mean_gain_percent"]),
        "worst_environment": bool(eco.worst_gain_percent >= gate["minimum_worst_environment_gain_percent"]),
        "mean_calls": bool(eco.mean_calls <= gate["maximum_mean_calls"]),
        "mean_frames": bool(eco.mean_frames <= gate["maximum_mean_model_input_frames"]),
        "exact_fallback": bool(
            metric_frame.loc[
                metric_frame["method"].eq("TRACE-Eco") & ~metric_frame["used_graph"],
                "relative_aurc_reduction_percent",
            ].eq(0.0).all()
        ),
    }
    decision = {
        "analysis_role": lock["analysis_role"],
        "target_labels_used_for_contract_or_support_decisions": False,
        "model_inference_performed": False,
        "environments": int(eco.environments),
        "mean_gain_percent": float(eco.mean_gain_percent),
        "worst_gain_percent": float(eco.worst_gain_percent),
        "mean_calls": float(eco.mean_calls),
        "mean_frames": float(eco.mean_frames),
        "graph_used_environments": int(
            metric_frame.loc[metric_frame["method"].eq("TRACE-Eco"), "used_graph"].sum()
        ),
        "promotion_conditions": conditions,
        "promotion_gate_pass": bool(all(conditions.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(args.output_dir / "environment_metrics.csv", index=False)
    audit_frame.to_csv(args.output_dir / "contract_selection_audit.csv", index=False)
    summary.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    (args.output_dir / "promotion_gate.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
