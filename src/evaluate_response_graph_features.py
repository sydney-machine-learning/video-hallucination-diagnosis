"""Cross-domain reliability prediction from typed model-view response graphs.

Each question forms a graph whose nodes are model-view probability vectors.
Edges connect (a) the same model across typed counterfactual views and (b)
different models under the same view.  This exploratory evaluator materializes
relation-aware graph statistics and tests them under whole-dataset holdout.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_probe_escalation import aurc


SEED = 20260801
ROLES = ("base", "invariant", "addition", "removal")
VIEW_ROLE = {
    "base": "base",
    "invariant_nuisance": "invariant",
    "temporal_evidence_sufficiency": "addition",
    "dense_acquisition": "addition",
    "evidence_addition": "addition",
    "evidence_removal": "removal",
    "temporal_order_destruction": "removal",
}


def entropy(probability: np.ndarray) -> float:
    value = np.clip(np.asarray(probability, float), 1e-12, 1.0)
    return float(-np.sum(value * np.log(value)) / math.log(len(value)))


def js(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    midpoint = (a + b) / 2.0
    value = 0.5 * np.sum(a * np.log(np.clip(a, 1e-12, 1.0) / midpoint))
    value += 0.5 * np.sum(b * np.log(np.clip(b, 1e-12, 1.0) / midpoint))
    return float(value / math.log(2.0))


def read_results(paths: list[Path]) -> tuple[dict[tuple[str, str, str, str], np.ndarray], list[str]]:
    values: dict[tuple[str, str, str, str], np.ndarray] = {}
    models: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "ok":
                    raise ValueError(f"{path}:{line_number}: non-ok row")
                view = str(row.get("view_id"))
                if view not in VIEW_ROLE:
                    continue
                dataset = str(row["dataset"])
                model = str(row["model_id"])
                qid = str(row.get("pilot_id") or row.get("question_id"))
                key = (dataset, model, qid, VIEW_ROLE[view])
                probability = np.asarray(row["candidate_probabilities"], dtype=float)
                if key in values or not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1.0):
                    raise ValueError(f"{path}:{line_number}: invalid/duplicate response node")
                values[key] = probability / probability.sum()
                models.add(model)
    observed_models = sorted(models)
    if len(observed_models) != 3:
        raise ValueError(f"expected exactly three models, got {observed_models}")
    return values, observed_models


def load_metadata(feature_paths: list[Path], vhd: Path, temp: Path) -> pd.DataFrame:
    public = pd.concat([pd.read_csv(path) for path in feature_paths], ignore_index=True, sort=False)
    public = public.drop_duplicates(["dataset", "pilot_id"])[
        ["dataset", "pilot_id", "video_id", "gold_candidate_index", "candidate_count"]
    ].rename(columns={"pilot_id": "qid", "video_id": "group_id", "gold_candidate_index": "gold"})
    vhd_frame = pd.read_csv(vhd).drop_duplicates("question_id")
    vhd_frame["gold"] = vhd_frame["final_correct_answer"].map({"A": 0, "B": 1})
    vhd_meta = vhd_frame[["question_id", "clip_id", "gold"]].rename(
        columns={"question_id": "qid", "clip_id": "group_id"}
    )
    vhd_meta.insert(0, "dataset", "VHD")
    vhd_meta["candidate_count"] = 2
    temp_frame = pd.read_csv(temp).drop_duplicates("question_id")
    temp_frame["gold"] = np.where(temp_frame["gold_answer"].str.lower().eq("yes"), 0, 1)
    temp_meta = temp_frame[["question_id", "media_id_y", "gold"]].rename(
        columns={"question_id": "qid", "media_id_y": "group_id"}
    )
    temp_meta.insert(0, "dataset", "TempCompass")
    temp_meta["candidate_count"] = 2
    metadata = pd.concat([public, vhd_meta, temp_meta], ignore_index=True, sort=False)
    metadata["gold"] = metadata["gold"].astype(int)
    metadata["candidate_count"] = metadata["candidate_count"].astype(int)
    return metadata


def vector(values: dict[tuple[str, str, str, str], np.ndarray], dataset: str, model: str, qid: str, role: str, count: int) -> np.ndarray | None:
    result = values.get((dataset, model, qid, role))
    if result is None:
        return None
    if len(result) != count:
        raise ValueError(f"{dataset}/{model}/{qid}/{role}: candidate mismatch")
    return result


def graph_table(metadata: pd.DataFrame, values: dict[tuple[str, str, str, str], np.ndarray], models: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    base_feature_names: list[str] | None = None
    for item in metadata.to_dict("records"):
        dataset, qid = str(item["dataset"]), str(item["qid"])
        count = int(item["candidate_count"])
        bases = [vector(values, dataset, model, qid, "base", count) for model in models]
        if any(value is None for value in bases):
            raise ValueError(f"{dataset}/{qid}: incomplete base model panel")
        base = [np.asarray(value) for value in bases]
        base_answers = np.asarray([value.argmax() for value in base])
        mean_base = np.mean(base, axis=0)
        for reference_index, reference_model in enumerate(models):
            selected = int(base_answers[reference_index])
            names: list[str] = []
            features: list[float] = []

            def add(name: str, value: float) -> None:
                names.append(name)
                features.append(float(value))

            own = base[reference_index]
            sorted_own = np.sort(own)
            add("own_base_confidence", own[selected])
            add("own_base_entropy", entropy(own))
            add("own_base_margin", sorted_own[-1] - sorted_own[-2] if count > 1 else 1.0)
            add("candidate_count_scaled", count / 5.0)
            for model_index, model in enumerate(models):
                add(f"model_identity_{model_index}", reference_index == model_index)
            add("base_vote_fraction", np.mean(base_answers == selected))
            selected_support = np.asarray([value[selected] for value in base])
            add("base_selected_support_mean", selected_support.mean())
            add("base_selected_support_std", selected_support.std())
            add("base_consensus_entropy", entropy(mean_base))
            for peer_index, peer in enumerate(base):
                add(f"base_peer_{peer_index}_is_self", peer_index == reference_index)
                add(f"base_peer_{peer_index}_agree", peer.argmax() == selected)
                add(f"base_peer_{peer_index}_support", peer[selected])
                add(f"base_peer_{peer_index}_js", 0.0 if peer_index == reference_index else js(own, peer))
            current_base_names = list(names)

            for role in ROLES[1:]:
                own_view = vector(values, dataset, reference_model, qid, role, count)
                present_views = [vector(values, dataset, model, qid, role, count) for model in models]
                add(f"{role}_own_present", own_view is not None)
                if own_view is None:
                    for suffix in ("js", "entropy_delta", "selected_delta", "flip"):
                        add(f"{role}_own_{suffix}", 0.0)
                else:
                    add(f"{role}_own_js", js(own, own_view))
                    add(f"{role}_own_entropy_delta", entropy(own_view) - entropy(own))
                    add(f"{role}_own_selected_delta", own_view[selected] - own[selected])
                    add(f"{role}_own_flip", own_view.argmax() != selected)
                present = [value for value in present_views if value is not None]
                add(f"{role}_panel_fraction", len(present) / len(models))
                if present:
                    view_answers = np.asarray([value.argmax() for value in present])
                    support = np.asarray([value[selected] for value in present])
                    add(f"{role}_vote_fraction", np.mean(view_answers == selected))
                    add(f"{role}_selected_support_mean", support.mean())
                    add(f"{role}_selected_support_std", support.std())
                    add(f"{role}_consensus_entropy", entropy(np.mean(present, axis=0)))
                else:
                    for suffix in ("vote_fraction", "selected_support_mean", "selected_support_std", "consensus_entropy"):
                        add(f"{role}_{suffix}", 0.0)
            if feature_names is None:
                feature_names = names
                base_feature_names = current_base_names
            elif names != feature_names:
                raise RuntimeError("response graph feature schema drift")
            row = {
                "dataset": dataset,
                "qid": qid,
                "group_id": str(item["group_id"]),
                "model_id": reference_model,
                "correct": int(selected == int(item["gold"])),
            }
            row.update(dict(zip(names, features)))
            rows.append(row)
    assert feature_names is not None and base_feature_names is not None
    return pd.DataFrame(rows), base_feature_names


def equal_domain_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("dataset")["dataset"].transform("size").to_numpy(float)
    return 1.0 / counts


def add_environment_ranks(table: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = table.copy()
    fixed_markers = ("model_identity_", "_is_self", "_present", "_panel_fraction")
    fixed = [
        column
        for column in feature_columns
        if column == "candidate_count_scaled" or any(marker in column for marker in fixed_markers)
    ]
    dynamic = [column for column in feature_columns if column not in fixed]
    rank_columns = []
    for column in dynamic:
        name = f"rank__{column}"
        result[name] = result.groupby(["dataset", "model_id"], sort=False)[column].rank(
            method="average", pct=True
        )
        rank_columns.append(name)
    return result, fixed + rank_columns


def distribution_shift(reference: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> float:
    """Label-free RMS standardized mean shift from reference to target."""
    reference_values = reference[columns].to_numpy(float)
    target_values = target[columns].to_numpy(float)
    scale = reference_values.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    standardized = (target_values.mean(axis=0) - reference_values.mean(axis=0)) / scale
    return float(np.sqrt(np.mean(np.square(standardized))))


def support_threshold(source: pd.DataFrame, model_id: str, columns: list[str]) -> float:
    """Calibrate a support radius using only leave-one-source-domain shifts."""
    model_source = source.loc[source["model_id"].eq(model_id)]
    shifts = []
    for validation_domain in sorted(model_source["dataset"].unique()):
        reference = model_source.loc[model_source["dataset"].ne(validation_domain)]
        validation = model_source.loc[model_source["dataset"].eq(validation_domain)]
        shifts.append(distribution_shift(reference, validation, columns))
    return float(max(shifts))


def fit_graph_logistic(source: pd.DataFrame, columns: list[str]):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.2, max_iter=3000, random_state=SEED),
    )
    model.fit(
        source[columns],
        source["correct"],
        logisticregression__sample_weight=equal_domain_weights(source),
    )
    return model


def source_gain_gate(
    source: pd.DataFrame,
    model_id: str,
    feature_columns: list[str],
    transport_columns: list[str],
) -> tuple[bool, float, int]:
    """Require non-negative gain in every supported source-domain rehearsal."""
    gains = []
    for validation_domain in sorted(source["dataset"].unique()):
        inner_source = source.loc[source["dataset"].ne(validation_domain)].reset_index(drop=True)
        validation = source.loc[
            source["dataset"].eq(validation_domain) & source["model_id"].eq(model_id)
        ].reset_index(drop=True)
        reference = inner_source.loc[inner_source["model_id"].eq(model_id)]
        threshold = support_threshold(inner_source, model_id, transport_columns)
        shift = distribution_shift(reference, validation, transport_columns)
        if shift > threshold:
            continue
        model = fit_graph_logistic(inner_source, feature_columns)
        labels = validation["correct"].to_numpy(int)
        raw = aurc(labels, validation["own_base_confidence"].to_numpy(float))
        graph = aurc(labels, model.predict_proba(validation[feature_columns])[:, 1])
        gains.append(100.0 * (raw - graph) / raw)
    if not gains:
        return False, float("nan"), 0
    return bool(min(gains) >= 0.0), float(min(gains)), len(gains)


def evaluate(args: argparse.Namespace) -> None:
    values, models = read_results(args.result)
    metadata = load_metadata(args.feature, args.vhd_scored, args.temp_scored)
    table, base_columns = graph_table(metadata, values, models)
    feature_columns = [column for column in table.columns if column not in {"dataset", "qid", "group_id", "model_id", "correct"}]
    table, rank_feature_columns = add_environment_ranks(table, feature_columns)
    transport_columns = [
        column for column in feature_columns
        if not column.startswith("model_identity_") and "_is_self" not in column
    ]
    methods = (
        "raw_confidence",
        "base_logistic",
        "graph_logistic",
        "graph_hgb",
        "graph_rank_logistic",
        "graph_rank_hgb",
        "transport_safe_graph",
    )
    prediction_rows = []
    metric_rows = []
    gate_rows = []
    for held_out in sorted(table["dataset"].unique()):
        source = table.loc[table["dataset"].ne(held_out)].reset_index(drop=True)
        target = table.loc[table["dataset"].eq(held_out)].reset_index(drop=True)
        weight = equal_domain_weights(source)
        base_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=3000, random_state=SEED))
        graph_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=3000, random_state=SEED))
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=SEED,
        )
        rank_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.2, max_iter=3000, random_state=SEED))
        rank_hgb = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=SEED,
        )
        base_model.fit(source[base_columns], source["correct"], logisticregression__sample_weight=weight)
        graph_model.fit(source[feature_columns], source["correct"], logisticregression__sample_weight=weight)
        hgb.fit(source[feature_columns], source["correct"], sample_weight=weight)
        rank_model.fit(source[rank_feature_columns], source["correct"], logisticregression__sample_weight=weight)
        rank_hgb.fit(source[rank_feature_columns], source["correct"], sample_weight=weight)
        scores = {
            "raw_confidence": target["own_base_confidence"].to_numpy(float),
            "base_logistic": base_model.predict_proba(target[base_columns])[:, 1],
            "graph_logistic": graph_model.predict_proba(target[feature_columns])[:, 1],
            "graph_hgb": hgb.predict_proba(target[feature_columns])[:, 1],
            "graph_rank_logistic": rank_model.predict_proba(target[rank_feature_columns])[:, 1],
            "graph_rank_hgb": rank_hgb.predict_proba(target[rank_feature_columns])[:, 1],
        }
        safe_score = scores["raw_confidence"].copy()
        for model_id, positions in target.groupby("model_id", sort=True).indices.items():
            idx = np.asarray(positions, dtype=int)
            reference = source.loc[source["model_id"].eq(model_id)]
            target_model = target.iloc[idx]
            threshold = support_threshold(source, model_id, transport_columns)
            shift = distribution_shift(reference, target_model, transport_columns)
            supported = bool(shift <= threshold)
            gain_enabled, worst_source_gain, supported_rehearsals = source_gain_gate(
                source, model_id, feature_columns, transport_columns
            )
            use_graph = supported and gain_enabled
            if use_graph:
                safe_score[idx] = scores["graph_logistic"][idx]
            gate_rows.append(
                {
                    "held_out_dataset": held_out,
                    "model_id": model_id,
                    "target_shift": shift,
                    "support_threshold": threshold,
                    "supported": supported,
                    "worst_supported_source_gain_percent": worst_source_gain,
                    "supported_source_rehearsals": supported_rehearsals,
                    "gain_gate_enabled": gain_enabled,
                    "used_graph": use_graph,
                }
            )
        scores["transport_safe_graph"] = safe_score
        for method in methods:
            for model, positions in target.groupby("model_id", sort=True).indices.items():
                idx = np.asarray(positions, dtype=int)
                labels = target.iloc[idx]["correct"].to_numpy(int)
                raw_aurc = aurc(labels, scores["raw_confidence"][idx])
                method_aurc = aurc(labels, scores[method][idx])
                metric_rows.append(
                    {
                        "held_out_dataset": held_out,
                        "model_id": model,
                        "method": method,
                        "n": len(idx),
                        "accuracy": float(labels.mean()),
                        "aurc": method_aurc,
                        "raw_confidence_aurc": raw_aurc,
                        "relative_aurc_reduction_percent": 100.0 * (raw_aurc - method_aurc) / raw_aurc,
                    }
                )
            for index, row in target.iterrows():
                prediction_rows.append(
                    {
                        "held_out_dataset": held_out,
                        "qid": row["qid"],
                        "group_id": row["group_id"],
                        "model_id": row["model_id"],
                        "correct": int(row["correct"]),
                        "method": method,
                        "score": float(scores[method][index]),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    aggregate = metrics.groupby("method", as_index=False).agg(
        environments=("aurc", "size"),
        mean_gain_percent=("relative_aurc_reduction_percent", "mean"),
        worst_gain_percent=("relative_aurc_reduction_percent", "min"),
        positive_environments=("relative_aurc_reduction_percent", lambda value: int((value > 0).sum())),
    ).sort_values("mean_gain_percent", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "response_graph_features.csv", index=False)
    metrics.to_csv(args.output_dir / "environment_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(args.output_dir / "transport_gate_audit.csv", index=False)
    print(aggregate.to_string(index=False))
    for method in ("graph_logistic", "graph_hgb", "graph_rank_logistic", "graph_rank_hgb", "transport_safe_graph"):
        print("\n", method)
        print(metrics.loc[metrics["method"].eq(method), ["held_out_dataset", "model_id", "relative_aurc_reduction_percent"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=Path, action="append", required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--vhd-scored", type=Path, required=True)
    parser.add_argument("--temp-scored", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
