#!/usr/bin/env python3
"""Apply frozen TRACE to Video-MME and run the locked grouped analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

import evaluate_response_graph_features as trace
import evaluate_trace_eco_pilot as eco


SEED = 20260801


def target_metadata(
    manifest_path: Path,
    labels_path: Path,
    dataset_name: str = "Video-MME",
) -> pd.DataFrame:
    manifest = pd.DataFrame(
        [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    labels = pd.read_csv(labels_path)
    merged = manifest.merge(labels[["question_id", "gold_candidate_index"]], on="question_id", validate="one_to_one")
    return pd.DataFrame(
        {
            "dataset": dataset_name,
            "qid": merged["question_id"].astype(str),
            "group_id": merged["video_id"].astype(str),
            "gold": merged["gold_candidate_index"].astype(int),
            "candidate_count": merged["candidates"].map(len).astype(int),
        }
    )


def ece(labels: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            mask = (scores >= edges[index]) & (scores < edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
    return float(value)


def risk_at_coverage(labels: np.ndarray, scores: np.ndarray, coverage: float) -> float:
    count = max(1, int(np.ceil(coverage * len(labels))))
    selected = np.argsort(-scores, kind="stable")[:count]
    return float(1.0 - labels[selected].mean())


def source_temperature_scores(
    source: pd.DataFrame,
    target: pd.DataFrame,
    values: dict[tuple[str, str, str, str], np.ndarray],
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit standard multiclass temperature scaling on source domains only."""
    gold = {
        (str(row.dataset), str(row.qid)): int(row.gold)
        for row in metadata.itertuples(index=False)
    }
    output = np.empty(len(target), dtype=float)
    temperatures: dict[str, float] = {}
    for model_id in sorted(source["model_id"].unique()):
        model_source = source.loc[source["model_id"].eq(model_id)].reset_index(drop=True)
        weights = trace.equal_domain_weights(model_source)
        records = []
        for row in model_source.itertuples(index=False):
            probability = values[(str(row.dataset), str(model_id), str(row.qid), "base")]
            records.append((np.log(np.clip(probability, 1e-12, 1.0)), gold[(str(row.dataset), str(row.qid))]))

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            losses = []
            for log_probability, answer in records:
                scaled = log_probability / temperature
                scaled -= scaled.max()
                normalized = scaled - np.log(np.exp(scaled).sum())
                losses.append(-normalized[answer])
            return float(np.average(losses, weights=weights))

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        if not result.success:
            raise RuntimeError(f"temperature optimization failed for {model_id}")
        temperature = float(np.exp(result.x))
        temperatures[str(model_id)] = temperature
        positions = target.index[target["model_id"].eq(model_id)].to_numpy(int)
        for index in positions:
            row = target.iloc[index]
            probability = values[(str(row["dataset"]), str(model_id), str(row["qid"]), "base")]
            scaled = np.log(np.clip(probability, 1e-12, 1.0)) / temperature
            scaled -= scaled.max()
            calibrated = np.exp(scaled) / np.exp(scaled).sum()
            output[index] = calibrated[int(np.argmax(probability))]
    return output, temperatures


def paired_group_bootstrap(
    frame: pd.DataFrame,
    iterations: int,
    score_column: str = "trace_score",
) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    groups = np.asarray(sorted(frame["group_id"].unique()))
    models = sorted(frame["model_id"].unique())
    positions = {
        (group, model): frame.index[
            frame["group_id"].eq(group) & frame["model_id"].eq(model)
        ].to_numpy(int)
        for group in groups
        for model in models
    }
    gains = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        model_gains = []
        for model in models:
            idx = np.concatenate([positions[(group, model)] for group in sampled])
            subset = frame.loc[idx]
            labels = subset["correct"].to_numpy(int)
            raw = trace.aurc(labels, subset["raw_confidence"].to_numpy(float))
            safe = trace.aurc(labels, subset[score_column].to_numpy(float))
            model_gains.append(100.0 * (raw - safe) / raw)
        gains[iteration] = np.mean(model_gains)
    return gains


def permute_typed_roles(frame: pd.DataFrame) -> pd.DataFrame:
    """Destroy intervention semantics while preserving every typed node value."""
    result = frame.copy()
    roles = list(trace.ROLES[1:])
    suffixes = sorted(
        set(column[len(roles[0]) + 1 :] for column in frame.columns if column.startswith(f"{roles[0]}_"))
        & set(column[len(roles[1]) + 1 :] for column in frame.columns if column.startswith(f"{roles[1]}_"))
        & set(column[len(roles[2]) + 1 :] for column in frame.columns if column.startswith(f"{roles[2]}_"))
    )
    original = frame.copy()
    for index, row in frame.iterrows():
        key = f"{row['dataset']}::{row['qid']}::{row['model_id']}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "little") ^ SEED
        permutation = np.random.default_rng(seed).permutation(len(roles))
        for destination_index, source_index in enumerate(permutation):
            destination, source = roles[destination_index], roles[source_index]
            for suffix in suffixes:
                result.at[index, f"{destination}_{suffix}"] = original.at[index, f"{source}_{suffix}"]
    return result


def replace_panel_with_self(frame: pd.DataFrame) -> pd.DataFrame:
    """Ablate cross-family information by duplicating each reference model node."""

    result = frame.copy()
    result["base_vote_fraction"] = 1.0
    result["base_selected_support_mean"] = result["own_base_confidence"]
    result["base_selected_support_std"] = 0.0
    result["base_consensus_entropy"] = result["own_base_entropy"]
    peer_indices = sorted(
        {
            int(column.split("_")[2])
            for column in frame.columns
            if column.startswith("base_peer_") and column.endswith("_agree")
        }
    )
    for peer_index in peer_indices:
        result[f"base_peer_{peer_index}_agree"] = 1.0
        result[f"base_peer_{peer_index}_support"] = result["own_base_confidence"]
        result[f"base_peer_{peer_index}_js"] = 0.0

    for role in trace.ROLES[1:]:
        present = result[f"{role}_own_present"].astype(float)
        flip = result[f"{role}_own_flip"].astype(float)
        result[f"{role}_panel_fraction"] = present
        result[f"{role}_vote_fraction"] = present * (1.0 - flip)
        result[f"{role}_selected_support_mean"] = present * (
            result["own_base_confidence"] + result[f"{role}_own_selected_delta"]
        )
        result[f"{role}_selected_support_std"] = 0.0
        result[f"{role}_consensus_entropy"] = present * (
            result["own_base_entropy"] + result[f"{role}_own_entropy_delta"]
        )
    return result


def fit_hgb(source: pd.DataFrame, columns: list[str]) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=SEED,
    )
    model.fit(
        source[columns],
        source["correct"],
        sample_weight=trace.equal_domain_weights(source),
    )
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=Path, action="append", required=True)
    parser.add_argument("--development-result", type=Path, action="append", required=True)
    parser.add_argument("--confirmation-result", type=Path, action="append", required=True)
    parser.add_argument("--vhd-scored", type=Path, required=True)
    parser.add_argument("--temp-scored", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sealed-labels", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--target-dataset", default="Video-MME")
    parser.add_argument("--eco-pilot-lock", type=Path)
    parser.add_argument("--full-confirmation-minimum-gain", type=float, default=0.0)
    parser.add_argument("--eco-confirmation-minimum-gain", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_results = args.development_result + args.confirmation_result
    values, models = trace.read_results(all_results)
    development_metadata = trace.load_metadata(args.feature, args.vhd_scored, args.temp_scored)
    confirmation_metadata = target_metadata(
        args.manifest, args.sealed_labels, args.target_dataset
    )
    metadata = pd.concat([development_metadata, confirmation_metadata], ignore_index=True)
    table, base_columns = trace.graph_table(metadata, values, models)
    source = table.loc[table["dataset"].ne(args.target_dataset)].reset_index(drop=True)
    target = table.loc[table["dataset"].eq(args.target_dataset)].reset_index(drop=True)
    feature_columns = [
        column for column in table.columns
        if column not in {"dataset", "qid", "group_id", "model_id", "correct"}
    ]
    transport_columns = [
        column for column in feature_columns
        if not column.startswith("model_identity_") and "_is_self" not in column
    ]
    scalar_columns = [
        column for column in feature_columns
        if column in {
            "own_base_confidence", "own_base_entropy", "own_base_margin",
            "candidate_count_scaled",
        }
        or column.startswith("model_identity_")
    ]
    typed_own_columns = [
        column for column in feature_columns
        if column in scalar_columns
        or any(column.startswith(f"{role}_own_") for role in trace.ROLES[1:])
    ]
    scalar_model = trace.fit_graph_logistic(source, scalar_columns)
    scalar_hgb = fit_hgb(source, scalar_columns)
    cross_model = trace.fit_graph_logistic(source, base_columns)
    typed_own_model = trace.fit_graph_logistic(source, typed_own_columns)
    model = trace.fit_graph_logistic(source, feature_columns)
    graph_hgb = fit_hgb(source, feature_columns)
    permuted_source = permute_typed_roles(source)
    permuted_target = permute_typed_roles(target)
    permuted_model = trace.fit_graph_logistic(permuted_source, feature_columns)
    self_panel_source = replace_panel_with_self(source)
    self_panel_target = replace_panel_with_self(target)
    self_panel_model = trace.fit_graph_logistic(self_panel_source, feature_columns)
    graph_score = model.predict_proba(target[feature_columns])[:, 1]
    target["raw_confidence"] = target["own_base_confidence"].astype(float)
    target["graph_score"] = graph_score
    target["trace_score"] = target["raw_confidence"]

    present = sum(target[f"{role}_own_present"] for role in trace.ROLES[1:])
    flips = sum(target[f"{role}_own_flip"] for role in trace.ROLES[1:])
    consistency = 1.0 - flips / np.maximum(1.0, present)
    temperature_score, temperatures = source_temperature_scores(
        source, target, values, metadata
    )
    method_scores = {
        "raw_confidence": target["raw_confidence"].to_numpy(float),
        "source_temperature_scaling": temperature_score,
        "normalized_negative_entropy": 1.0 - target["own_base_entropy"].to_numpy(float),
        "probability_margin": target["own_base_margin"].to_numpy(float),
        "panel_vote_fraction": np.clip(
            target["base_vote_fraction"].to_numpy(float)
            + 1e-7 * target["raw_confidence"].to_numpy(float), 0.0, 1.0
        ),
        "panel_selected_support": target["base_selected_support_mean"].to_numpy(float),
        "same_model_view_consistency": np.clip(
            consistency.to_numpy(float)
            + 1e-7 * target["raw_confidence"].to_numpy(float), 0.0, 1.0
        ),
        "scalar_logistic": scalar_model.predict_proba(target[scalar_columns])[:, 1],
        "scalar_gradient_boosting": scalar_hgb.predict_proba(target[scalar_columns])[:, 1],
        "cross_model_base_logistic": cross_model.predict_proba(target[base_columns])[:, 1],
        "typed_own_view_logistic": typed_own_model.predict_proba(target[typed_own_columns])[:, 1],
        "full_graph_logistic": graph_score,
        "full_graph_gradient_boosting": graph_hgb.predict_proba(
            target[feature_columns]
        )[:, 1],
        "random_typed_edge_permutation": permuted_model.predict_proba(
            permuted_target[feature_columns]
        )[:, 1],
        "within_model_duplicate_graph_control": self_panel_model.predict_proba(
            self_panel_target[feature_columns]
        )[:, 1],
    }

    gate_rows = []
    support_only_score = target["raw_confidence"].to_numpy(float).copy()
    gain_only_score = target["raw_confidence"].to_numpy(float).copy()
    for model_id, positions in target.groupby("model_id", sort=True).indices.items():
        idx = np.asarray(positions, dtype=int)
        reference = source.loc[source["model_id"].eq(model_id)]
        target_model = target.iloc[idx]
        threshold = trace.support_threshold(source, model_id, transport_columns)
        shift = trace.distribution_shift(reference, target_model, transport_columns)
        supported = bool(shift <= threshold)
        gain_enabled, worst_source_gain, rehearsals = trace.source_gain_gate(
            source, model_id, feature_columns, transport_columns
        )
        used_graph = supported and gain_enabled
        if supported:
            support_only_score[idx] = graph_score[idx]
        if gain_enabled:
            gain_only_score[idx] = graph_score[idx]
        if used_graph:
            target.loc[idx, "trace_score"] = graph_score[idx]
        else:
            if not np.array_equal(
                target.loc[idx, "trace_score"].to_numpy(float),
                target.loc[idx, "raw_confidence"].to_numpy(float),
            ):
                raise RuntimeError(f"{model_id}: exact fallback audit failed")
        gate_rows.append(
            {
                "model_id": model_id,
                "target_shift": shift,
                "support_threshold": threshold,
                "supported": supported,
                "worst_supported_source_gain_percent": worst_source_gain,
                "supported_source_rehearsals": rehearsals,
                "gain_gate_enabled": gain_enabled,
                "used_graph": used_graph,
            }
        )

    method_scores["support_gate_only"] = support_only_score
    method_scores["source_gain_gate_only"] = gain_only_score
    method_scores["TRACE"] = target["trace_score"].to_numpy(float)
    method_cost = {
        "raw_confidence": (1, 8),
        "source_temperature_scaling": (1, 8),
        "normalized_negative_entropy": (1, 8),
        "probability_margin": (1, 8),
        "scalar_logistic": (1, 8),
        "scalar_gradient_boosting": (1, 8),
        "panel_vote_fraction": (3, 24),
        "panel_selected_support": (3, 24),
        "cross_model_base_logistic": (3, 24),
        "same_model_view_consistency": (4, 40),
        "typed_own_view_logistic": (4, 40),
        "full_graph_logistic": (12, 120),
        "full_graph_gradient_boosting": (12, 120),
        "random_typed_edge_permutation": (12, 120),
        "within_model_duplicate_graph_control": (4, 40),
        "support_gate_only": (12, 120),
        "source_gain_gate_only": (12, 120),
        "TRACE": (12, 120),
    }
    for name, score in method_scores.items():
        target[f"score__{name}"] = score

    eco_scored: pd.DataFrame | None = None
    eco_audit_rows: list[dict[str, object]] = []
    if args.eco_pilot_lock is not None:
        eco_lock = json.loads(args.eco_pilot_lock.read_text(encoding="utf-8"))
        if eco_lock.get("schema_version") != "trace-eco-pilot-amendment-lock-1.0":
            raise RuntimeError("unexpected TRACE-Eco-Pilot lock schema")
        if eco_lock["pilot"]["semantic_groups"] != eco.PILOT_GROUPS:
            raise RuntimeError("TRACE-Eco-Pilot lock/implementation mismatch")
        expected_sources = {"MVBench", "NExT-GQA", "STAR", "TempCompass", "VHD"}
        if set(source["dataset"].unique()) != expected_sources:
            raise RuntimeError("TRACE-Eco-Pilot source dataset inventory mismatch")
        _, eco_audit_rows, eco_scored = eco.evaluate_fold(table, args.target_dataset)
        key_columns = ["dataset", "qid", "group_id", "model_id"]
        eco_join = target[key_columns].merge(
            eco_scored[key_columns + ["eco_score", "calls", "frames", "used_graph"]],
            on=key_columns,
            how="left",
            validate="one_to_one",
        )
        if eco_join["eco_score"].isna().any():
            raise RuntimeError("TRACE-Eco-Pilot target join is incomplete")
        target["eco_score"] = eco_join["eco_score"].to_numpy(float)
        target["eco_calls"] = eco_join["calls"].to_numpy(float)
        target["eco_frames"] = eco_join["frames"].to_numpy(float)
        target["eco_used_graph"] = eco_join["used_graph"].to_numpy(bool)
        target["score__TRACE-Eco-Pilot"] = target["eco_score"]

    metric_rows = []
    for method_name, all_scores in method_scores.items():
        for model_id, positions in target.groupby("model_id", sort=True).indices.items():
            idx = np.asarray(positions, dtype=int)
            group = target.iloc[idx]
            labels = group["correct"].to_numpy(int)
            raw = group["raw_confidence"].to_numpy(float)
            score = np.asarray(all_scores)[idx]
            raw_aurc, method_aurc = trace.aurc(labels, raw), trace.aurc(labels, score)
            metric_rows.append(
                {
                    "method": method_name,
                    "model_id": model_id,
                    "n": len(group),
                    "videos": group["group_id"].nunique(),
                    "accuracy": labels.mean(),
                    "aurc": method_aurc,
                    "raw_aurc": raw_aurc,
                    "relative_aurc_reduction_percent": 100.0 * (raw_aurc - method_aurc) / raw_aurc,
                    "auroc": roc_auc_score(labels, score),
                    "ece15": ece(labels, np.clip(score, 0.0, 1.0)),
                    "brier": float(np.mean(np.square(np.clip(score, 0.0, 1.0) - labels))),
                    "risk_at_50pct": risk_at_coverage(labels, score, 0.5),
                    "panel_videollm_calls_per_question": method_cost[method_name][0],
                    "panel_visual_frames_per_question": method_cost[method_name][1],
                }
            )
    if eco_scored is not None:
        for model_id, positions in target.groupby("model_id", sort=True).indices.items():
            idx = np.asarray(positions, dtype=int)
            group = target.iloc[idx]
            labels = group["correct"].to_numpy(int)
            raw = group["raw_confidence"].to_numpy(float)
            score = group["eco_score"].to_numpy(float)
            raw_aurc, method_aurc = trace.aurc(labels, raw), trace.aurc(labels, score)
            metric_rows.append(
                {
                    "method": "TRACE-Eco-Pilot",
                    "model_id": model_id,
                    "n": len(group),
                    "videos": group["group_id"].nunique(),
                    "accuracy": labels.mean(),
                    "aurc": method_aurc,
                    "raw_aurc": raw_aurc,
                    "relative_aurc_reduction_percent": 100.0 * (raw_aurc - method_aurc) / raw_aurc,
                    "auroc": roc_auc_score(labels, score),
                    "ece15": ece(labels, np.clip(score, 0.0, 1.0)),
                    "brier": float(np.mean(np.square(np.clip(score, 0.0, 1.0) - labels))),
                    "risk_at_50pct": risk_at_coverage(labels, score, 0.5),
                    "panel_videollm_calls_per_question": float(group["eco_calls"].iloc[0]),
                    "panel_visual_frames_per_question": float(group["eco_frames"].iloc[0]),
                }
            )
    metrics = pd.DataFrame(metric_rows)

    mechanism_rows = []
    available_add_remove = target["addition_own_present"].eq(1) & target["removal_own_present"].eq(1)
    subsets = {
        "all": np.ones(len(target), dtype=bool),
        "unanimous": target["base_vote_fraction"].eq(1).to_numpy(),
        "consistent_yet_wrong": (
            target["base_vote_fraction"].eq(1) & target["correct"].eq(0)
        ).to_numpy(),
        "unanimous_correct": (
            target["base_vote_fraction"].eq(1) & target["correct"].eq(1)
        ).to_numpy(),
        "typed_evidence_sensitive": (
            available_add_remove
            & target["addition_own_selected_delta"].gt(0)
            & target["removal_own_selected_delta"].lt(0)
        ).to_numpy(),
    }
    for subset_name, mask in subsets.items():
        for model_id, positions in target.loc[mask].groupby("model_id", sort=True).indices.items():
            idx = np.asarray(positions, dtype=int)
            group = target.loc[mask].iloc[idx]
            mechanism_rows.append(
                {
                    "subset": subset_name,
                    "model_id": model_id,
                    "n": len(group),
                    "accuracy": group["correct"].mean(),
                    "mean_raw_confidence": group["raw_confidence"].mean(),
                    "mean_full_graph_score": group["graph_score"].mean(),
                    "mean_trace_score": group["trace_score"].mean(),
                }
            )
    bootstrap = paired_group_bootstrap(target, args.iterations)
    primary_metrics = metrics.loc[metrics["method"].eq("TRACE")]
    observed = float(primary_metrics["relative_aurc_reduction_percent"].mean())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    worst = float(primary_metrics["relative_aurc_reduction_percent"].min())
    gate = {
        "schema_version": (
            "trace-videomme-confirmation-gate-1.0"
            if args.target_dataset == "Video-MME"
            else f"trace-{args.target_dataset.lower()}-confirmation-gate-1.0"
        ),
        "dataset": args.target_dataset,
        "questions": int(target["qid"].nunique()),
        "videos": int(target["group_id"].nunique()),
        "models": len(models),
        "macro_mean_relative_aurc_reduction_percent": observed,
        "video_grouped_bootstrap_95ci_percent": [float(lower), float(upper)],
        "worst_model_gain_percent": worst,
        "minimum_macro_gain_percent": args.full_confirmation_minimum_gain,
        "primary_gate_pass": bool(
            observed >= args.full_confirmation_minimum_gain
            and lower > 0
            and worst >= 0
        ),
        "bootstrap_iterations": args.iterations,
        "bootstrap_seed": SEED,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target.to_csv(args.output_dir / "scored_predictions.csv", index=False)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    metrics.groupby("method", as_index=False).agg(
        environments=("model_id", "size"),
        mean_relative_aurc_reduction_percent=("relative_aurc_reduction_percent", "mean"),
        worst_relative_aurc_reduction_percent=("relative_aurc_reduction_percent", "min"),
        mean_auroc=("auroc", "mean"),
        mean_ece15=("ece15", "mean"),
        mean_brier=("brier", "mean"),
        mean_risk_at_50pct=("risk_at_50pct", "mean"),
    ).sort_values("mean_relative_aurc_reduction_percent", ascending=False).to_csv(
        args.output_dir / "baseline_summary.csv", index=False
    )
    pd.DataFrame(gate_rows).to_csv(args.output_dir / "transport_gate_audit.csv", index=False)
    if eco_scored is not None:
        eco_bootstrap = paired_group_bootstrap(
            target, args.iterations, score_column="eco_score"
        )
        eco_metrics = metrics.loc[metrics["method"].eq("TRACE-Eco-Pilot")]
        eco_observed = float(
            eco_metrics["relative_aurc_reduction_percent"].mean()
        )
        eco_lower, eco_upper = np.quantile(eco_bootstrap, [0.025, 0.975])
        eco_worst = float(
            eco_metrics["relative_aurc_reduction_percent"].min()
        )
        eco_mean_calls = float(
            eco_metrics["panel_videollm_calls_per_question"].mean()
        )
        eco_mean_frames = float(
            eco_metrics["panel_visual_frames_per_question"].mean()
        )
        fallback = target.loc[~target["eco_used_graph"]]
        exact_fallback = bool(
            np.array_equal(
                fallback["eco_score"].to_numpy(float),
                fallback["raw_confidence"].to_numpy(float),
            )
        )
        eco_gate = {
            "schema_version": f"trace-eco-{args.target_dataset.lower()}-confirmation-gate-1.0",
            "dataset": args.target_dataset,
            "questions": int(target["qid"].nunique()),
            "groups": int(target["group_id"].nunique()),
            "models": len(models),
            "macro_mean_relative_aurc_reduction_percent": eco_observed,
            "grouped_bootstrap_95ci_percent": [float(eco_lower), float(eco_upper)],
            "worst_model_gain_percent": eco_worst,
            "mean_calls": eco_mean_calls,
            "mean_frames": eco_mean_frames,
            "exact_fallback": exact_fallback,
            "minimum_macro_gain_percent": args.eco_confirmation_minimum_gain,
            "confirmation_gate_pass": bool(
                eco_observed >= args.eco_confirmation_minimum_gain
                and eco_lower >= 0
                and eco_worst >= 0
                and eco_mean_calls <= 6.0
                and eco_mean_frames <= 60.0
                and exact_fallback
            ),
            "bootstrap_iterations": args.iterations,
            "bootstrap_seed": SEED,
        }
        pd.DataFrame(eco_audit_rows).to_csv(
            args.output_dir / "eco_pilot_audit.csv", index=False
        )
        pd.DataFrame({"macro_gain_percent": eco_bootstrap}).to_csv(
            args.output_dir / "eco_bootstrap_distribution.csv", index=False
        )
        (args.output_dir / "eco_confirmation_gate.json").write_text(
            json.dumps(eco_gate, indent=2) + "\n", encoding="utf-8"
        )
    pd.DataFrame(mechanism_rows).to_csv(args.output_dir / "mechanism_subsets.csv", index=False)
    (args.output_dir / "source_temperatures.json").write_text(
        json.dumps(temperatures, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame({"macro_gain_percent": bootstrap}).to_csv(args.output_dir / "bootstrap_distribution.csv", index=False)
    (args.output_dir / "confirmation_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(gate, indent=2))
    if eco_scored is not None:
        print(json.dumps(eco_gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
