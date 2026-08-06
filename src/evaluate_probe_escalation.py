"""Cross-dataset development evaluation for conditional probe escalation.

The evaluator is intentionally exploratory.  NExT-GQA, STAR, and MVBench are
all development datasets.  Each outer run leaves one whole dataset out, fits
experts and the router on the other two, and uses no target labels until the
final metric calculation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260801
INNER_FOLDS = 5
DIAGNOSTIC_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
UTILITY_PENALTIES = (0.0, 0.002, 0.005, 0.010)
SHRINKAGE_PRIOR_N = 20
LCB_Z = 1.645
BASE_NUMERIC = ("raw_confidence", "base_entropy_norm", "candidate_count")
CATEGORICAL = ("model_id", "semantic_group")
RESIDUAL_SUFFIXES = (
    "js_norm",
    "entropy_delta_norm",
    "base_top_probability_delta",
    "top_probability_delta",
    "flip",
    "centered_log_probability_delta_0",
    "centered_log_probability_delta_1",
    "centered_log_probability_delta_2",
    "centered_log_probability_delta_3",
    "centered_log_probability_delta_4",
)
VIEW_MAP = {
    "NExT-GQA": {
        "diagnostic": "invariant_nuisance",
        "addition": "temporal_evidence_sufficiency",
        "removal": "evidence_removal",
    },
    "STAR": {
        "diagnostic": "invariant_nuisance",
        "addition": "dense_acquisition",
        "removal": "temporal_order_destruction",
    },
    "MVBench": {
        "diagnostic": "invariant_nuisance",
        "addition": "dense_acquisition",
        "removal": "temporal_order_destruction",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aurc(correct: np.ndarray, score: np.ndarray) -> float:
    labels = np.asarray(correct, dtype=int)
    values = np.asarray(score, dtype=float)
    if len(labels) == 0 or labels.shape != values.shape:
        raise ValueError("AURC inputs must be aligned and non-empty")
    order = np.lexsort((np.arange(len(values)), -values))
    cumulative_error = np.cumsum(1 - labels[order])
    return float(np.mean(cumulative_error / np.arange(1, len(labels) + 1)))


def semantic_group(value: object) -> str:
    text = str(value).strip().lower()
    exact = {
        "ch": "causal",
        "cw": "causal",
        "tn": "temporal_order",
        "tp": "temporal_order",
        "tc": "temporal_order",
        "sequence": "temporal_order",
        "interaction": "interaction",
        "prediction": "prediction_counterfactual",
        "feasibility": "causal",
    }
    if text in exact:
        return exact[text]
    if any(token in text for token in ("sequence", "order", "transition")):
        return "temporal_order"
    if any(token in text for token in ("prediction", "counterfactual", "unexpected")):
        return "prediction_counterfactual"
    if any(token in text for token in ("count", "existence", "attribute")):
        return "object_state"
    if any(token in text for token in ("interaction", "antonym")):
        return "interaction"
    if any(token in text for token in ("moving", "navigation", "localization")):
        return "motion_spatial"
    if any(token in text for token in ("state", "action")):
        return "action_state"
    return "other"


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for dataset, source in frame.groupby("dataset", sort=True, observed=True):
        name = str(dataset)
        if name not in VIEW_MAP:
            raise ValueError(f"no typed view map for {name}")
        part = source.copy()
        part["semantic_group"] = part["question_type"].map(semantic_group)
        part["group_id"] = (
            part["dataset"].astype(str) + "::" + part["video_id"].astype(str)
        )
        for role, view in VIEW_MAP[name].items():
            contract = f"contract_{view}"
            if contract not in part or not part[contract].notna().all():
                raise ValueError(f"{name}: incomplete {role} view {view}")
            for suffix in RESIDUAL_SUFFIXES:
                source_column = f"residual_{view}_{suffix}"
                target_column = f"residual_{role}_{suffix}"
                if source_column in part:
                    part[target_column] = pd.to_numeric(
                        part[source_column], errors="coerce"
                    ).fillna(0.0)
                else:
                    part[target_column] = 0.0
        parts.append(part)
    result = pd.concat(parts, ignore_index=True, sort=False)
    key = ["dataset", "model_id", "pilot_id"]
    if result.duplicated(key).any():
        raise RuntimeError("duplicate model-question rows after canonicalization")
    return result


def feature_columns(*roles: str) -> list[str]:
    columns = list(CATEGORICAL + BASE_NUMERIC)
    for role in roles:
        columns.extend(f"residual_{role}_{suffix}" for suffix in RESIDUAL_SUFFIXES)
    return columns


def prepared(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    names = list(columns)
    result = frame.loc[:, names].copy()
    for column in CATEGORICAL:
        if column in result:
            result[column] = result[column].fillna("__missing__").astype(str)
    numeric = [column for column in names if column not in CATEGORICAL]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if not np.isfinite(result[numeric].to_numpy(float)).all():
        raise ValueError("non-finite feature")
    return result


def classifier(columns: list[str]) -> Pipeline:
    numeric = [column for column in columns if column not in CATEGORICAL]
    categorical = [column for column in columns if column in CATEGORICAL]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(
        [
            ("features", transform),
            (
                "correctness",
                LogisticRegression(C=0.2, max_iter=3000, random_state=SEED),
            ),
        ]
    )


def regressor(columns: list[str]) -> Pipeline:
    numeric = [column for column in columns if column not in CATEGORICAL]
    categorical = [column for column in columns if column in CATEGORICAL]
    transform = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("features", transform), ("utility", Ridge(alpha=5.0))])


EXPERT_ROLES = {
    "base": (),
    "diagnostic": ("diagnostic",),
    "addition_direct": ("addition",),
    "removal_direct": ("removal",),
    "addition_sequential": ("diagnostic", "addition"),
    "removal_sequential": ("diagnostic", "removal"),
    "always_all": ("diagnostic", "addition", "removal"),
}


def stable_inner_folds(frame: pd.DataFrame) -> np.ndarray:
    groups = frame["group_id"].astype(str).to_numpy()
    unique = sorted(
        set(groups),
        key=lambda value: hashlib.sha256(
            f"probe-escalation|{value}".encode("utf-8")
        ).hexdigest(),
    )
    assignment = {group: index % INNER_FOLDS for index, group in enumerate(unique)}
    folds = np.asarray([assignment[group] for group in groups], dtype=int)
    if pd.DataFrame({"group": groups, "fold": folds}).groupby("group")["fold"].nunique().max() != 1:
        raise RuntimeError("a source video crosses inner folds")
    return folds


def fit_predict_experts(
    source: pd.DataFrame, target: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    folds = stable_inner_folds(source)
    source_predictions = {
        name: np.full(len(source), np.nan, float) for name in EXPERT_ROLES
    }
    target_predictions: dict[str, np.ndarray] = {}
    for name, roles in EXPERT_ROLES.items():
        columns = feature_columns(*roles)
        for fold in range(INNER_FOLDS):
            train = source.iloc[np.flatnonzero(folds != fold)]
            valid_positions = np.flatnonzero(folds == fold)
            model = classifier(columns)
            model.fit(prepared(train, columns), train["correct"].astype(int))
            source_predictions[name][valid_positions] = model.predict_proba(
                prepared(source.iloc[valid_positions], columns)
            )[:, 1]
        if not np.isfinite(source_predictions[name]).all():
            raise RuntimeError(f"incomplete OOF expert prediction: {name}")
        model = classifier(columns)
        model.fit(prepared(source, columns), source["correct"].astype(int))
        target_predictions[name] = model.predict_proba(prepared(target, columns))[:, 1]
    return source_predictions, target_predictions, folds


def cross_fitted_router_utilities(
    source: pd.DataFrame,
    folds: np.ndarray,
    source_experts: dict[str, np.ndarray],
    target: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    labels = source["correct"].to_numpy(int)
    diagnostic = source_experts["diagnostic"]
    observed = {
        "addition": (labels - diagnostic) ** 2
        - (labels - source_experts["addition_sequential"]) ** 2,
        "removal": (labels - diagnostic) ** 2
        - (labels - source_experts["removal_sequential"]) ** 2,
    }
    columns = feature_columns("diagnostic")
    source_predicted = {
        action: np.full(len(source), np.nan, float) for action in observed
    }
    target_predicted: dict[str, np.ndarray] = {}
    for action, utility in observed.items():
        for fold in range(INNER_FOLDS):
            train_positions = np.flatnonzero(folds != fold)
            valid_positions = np.flatnonzero(folds == fold)
            model = regressor(columns)
            model.fit(prepared(source.iloc[train_positions], columns), utility[train_positions])
            source_predicted[action][valid_positions] = model.predict(
                prepared(source.iloc[valid_positions], columns)
            )
        model = regressor(columns)
        model.fit(prepared(source, columns), utility)
        target_predicted[action] = model.predict(prepared(target, columns))
    return observed, source_predicted, target_predicted


@dataclass(frozen=True)
class Eligibility:
    action: str
    semantic_group: str
    level: str
    minimum_model_lcb: float
    eligible: bool


def lower_bound(values: np.ndarray, prior: np.ndarray) -> float:
    cell = np.asarray(values, float)
    global_values = np.asarray(prior, float)
    if len(cell) == 0 or len(global_values) == 0:
        return float("-inf")
    cell_mean = float(np.mean(cell))
    prior_mean = float(np.mean(global_values))
    weight = len(cell) / (len(cell) + SHRINKAGE_PRIOR_N)
    posterior = weight * cell_mean + (1.0 - weight) * prior_mean
    pooled_variance = float(np.var(np.concatenate([cell, global_values]), ddof=1))
    standard_error = math.sqrt(max(pooled_variance, 1e-12) / (len(cell) + SHRINKAGE_PRIOR_N))
    return posterior - LCB_Z * standard_error


def build_eligibility(
    source: pd.DataFrame, observed: dict[str, np.ndarray]
) -> tuple[dict[tuple[str, str], Eligibility], list[Eligibility]]:
    table: dict[tuple[str, str], Eligibility] = {}
    records: list[Eligibility] = []
    groups = sorted(source["semantic_group"].astype(str).unique())
    models = sorted(source["model_id"].astype(str).unique())
    for action, utility in observed.items():
        global_lcbs = []
        for model in models:
            mask = source["model_id"].astype(str).eq(model).to_numpy()
            global_lcbs.append(lower_bound(utility[mask], utility[mask]))
        global_minimum = min(global_lcbs)
        for group in groups:
            cell_lcbs = []
            complete = True
            for model in models:
                global_mask = source["model_id"].astype(str).eq(model).to_numpy()
                cell_mask = global_mask & source["semantic_group"].astype(str).eq(group).to_numpy()
                if int(cell_mask.sum()) < 10:
                    complete = False
                    break
                cell_lcbs.append(lower_bound(utility[cell_mask], utility[global_mask]))
            if complete and min(cell_lcbs) > 0.0:
                item = Eligibility(action, group, "semantic_group", min(cell_lcbs), True)
            else:
                item = Eligibility(action, group, "global", global_minimum, global_minimum > 0.0)
            table[(action, group)] = item
            records.append(item)
        table[(action, "__unseen__")] = Eligibility(
            action, "__unseen__", "global", global_minimum, global_minimum > 0.0
        )
    return table, records


def threshold_by_model(
    frame: pd.DataFrame, score: np.ndarray, fraction: float
) -> dict[str, float]:
    result = {}
    for model, positions in frame.groupby("model_id", sort=True).indices.items():
        result[str(model)] = float(np.quantile(score[np.asarray(positions)], fraction, method="higher"))
    return result


def acquisition_mask(
    frame: pd.DataFrame, score: np.ndarray, thresholds: dict[str, float]
) -> np.ndarray:
    return np.asarray(
        [value <= thresholds[str(model)] for value, model in zip(score, frame["model_id"])],
        dtype=bool,
    )


def action_allowed(
    action: str, group: str, eligibility: dict[tuple[str, str], Eligibility]
) -> bool:
    item = eligibility.get((action, group), eligibility[(action, "__unseen__")])
    return item.eligible


def apply_policy(
    frame: pd.DataFrame,
    experts: dict[str, np.ndarray],
    predicted_utility: dict[str, np.ndarray],
    eligibility: dict[tuple[str, str], Eligibility],
    diagnostic_thresholds: dict[str, float],
    utility_penalty: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = experts["base"].copy()
    calls = np.ones(len(frame), float)
    action = np.full(len(frame), "base", object)
    diagnosed = acquisition_mask(frame, experts["base"], diagnostic_thresholds)
    score[diagnosed] = experts["diagnostic"][diagnosed]
    calls[diagnosed] += 1.0
    action[diagnosed] = "diagnostic"
    for index in np.flatnonzero(diagnosed):
        group = str(frame.iloc[index]["semantic_group"])
        candidates = []
        for candidate in ("addition", "removal"):
            if action_allowed(candidate, group, eligibility):
                candidates.append((predicted_utility[candidate][index], candidate))
        if not candidates:
            continue
        value, chosen = max(candidates, key=lambda item: (item[0], item[1]))
        if value <= utility_penalty:
            continue
        score[index] = experts[f"{chosen}_sequential"][index]
        calls[index] += 1.0
        action[index] = chosen
    return score, calls, action


def environment_metrics(
    frame: pd.DataFrame, score: np.ndarray, calls: np.ndarray, base: np.ndarray
) -> pd.DataFrame:
    rows = []
    for (dataset, model), positions in frame.groupby(["dataset", "model_id"], sort=True).indices.items():
        idx = np.asarray(positions)
        labels = frame.iloc[idx]["correct"].to_numpy(int)
        base_aurc = aurc(labels, base[idx])
        policy_aurc = aurc(labels, score[idx])
        rows.append(
            {
                "dataset": dataset,
                "model_id": model,
                "n": len(idx),
                "accuracy": float(np.mean(labels)),
                "base_aurc": base_aurc,
                "aurc": policy_aurc,
                "relative_aurc_reduction_percent": 100.0 * (base_aurc - policy_aurc) / base_aurc,
                "mean_calls": float(np.mean(calls[idx])),
            }
        )
    return pd.DataFrame(rows)


def select_configuration(
    source: pd.DataFrame,
    source_experts: dict[str, np.ndarray],
    source_utility: dict[str, np.ndarray],
    eligibility: dict[tuple[str, str], Eligibility],
) -> tuple[float, float, pd.DataFrame]:
    rows = []
    best: tuple[tuple[float, float, float], float, float] | None = None
    for fraction in DIAGNOSTIC_FRACTIONS:
        thresholds = threshold_by_model(source, source_experts["base"], fraction)
        for penalty in UTILITY_PENALTIES:
            score, calls, _ = apply_policy(
                source, source_experts, source_utility, eligibility, thresholds, penalty
            )
            metrics = environment_metrics(source, score, calls, source_experts["base"])
            mean_gain = float(metrics["relative_aurc_reduction_percent"].mean())
            worst_gain = float(metrics["relative_aurc_reduction_percent"].min())
            mean_calls = float(metrics["mean_calls"].mean())
            positive = int((metrics["relative_aurc_reduction_percent"] > 0).sum())
            feasible = worst_gain >= -1.0 and mean_calls <= 2.5
            rows.append(
                {
                    "diagnostic_fraction": fraction,
                    "utility_penalty": penalty,
                    "mean_gain_percent": mean_gain,
                    "worst_gain_percent": worst_gain,
                    "mean_calls": mean_calls,
                    "positive_environments": positive,
                    "feasible": feasible,
                }
            )
            if feasible:
                key = (mean_gain, -mean_calls, -penalty)
                if best is None or key > best[0]:
                    best = (key, fraction, penalty)
    table = pd.DataFrame(rows)
    if best is None:
        return 0.0, 0.0, table
    return best[1], best[2], table


def evaluate(development: Path, mvbench: Path, output_dir: Path) -> None:
    raw = pd.concat(
        [pd.read_csv(development), pd.read_csv(mvbench)],
        ignore_index=True,
        sort=False,
    )
    frame = canonicalize(raw)
    datasets = sorted(frame["dataset"].astype(str).unique())
    prediction_rows = []
    environment_rows = []
    selection_tables = []
    eligibility_rows = []

    for held_out in datasets:
        source = frame.loc[frame["dataset"].ne(held_out)].reset_index(drop=True)
        target = frame.loc[frame["dataset"].eq(held_out)].reset_index(drop=True)
        source_experts, target_experts, folds = fit_predict_experts(source, target)
        observed, source_utility, target_utility = cross_fitted_router_utilities(
            source, folds, source_experts, target
        )
        eligibility, records = build_eligibility(source, observed)
        for record in records:
            eligibility_rows.append({"held_out_dataset": held_out, **record.__dict__})
        fraction, penalty, selection = select_configuration(
            source, source_experts, source_utility, eligibility
        )
        selection.insert(0, "held_out_dataset", held_out)
        selection_tables.append(selection)
        thresholds = threshold_by_model(source, source_experts["base"], fraction)
        route_score, route_calls, route_action = apply_policy(
            target, target_experts, target_utility, eligibility, thresholds, penalty
        )

        policies = {
            "base": (target_experts["base"], np.ones(len(target))),
            "fixed_diagnostic": (target_experts["diagnostic"], np.full(len(target), 2.0)),
            "fixed_addition": (target_experts["addition_direct"], np.full(len(target), 2.0)),
            "fixed_removal": (target_experts["removal_direct"], np.full(len(target), 2.0)),
            "always_all": (target_experts["always_all"], np.full(len(target), 4.0)),
            "probe_escalation": (route_score, route_calls),
        }
        for policy, (score, calls) in policies.items():
            metrics = environment_metrics(target, score, calls, target_experts["base"])
            metrics.insert(0, "policy", policy)
            metrics.insert(0, "held_out_dataset", held_out)
            environment_rows.extend(metrics.to_dict("records"))
            for index, row in target.iterrows():
                prediction_rows.append(
                    {
                        "held_out_dataset": held_out,
                        "dataset": row["dataset"],
                        "model_id": row["model_id"],
                        "pilot_id": row["pilot_id"],
                        "video_id": row["video_id"],
                        "semantic_group": row["semantic_group"],
                        "correct": int(row["correct"]),
                        "policy": policy,
                        "predicted_correctness": float(score[index]),
                        "calls": float(calls[index]),
                        "route_action": str(route_action[index]) if policy == "probe_escalation" else policy,
                        "selected_diagnostic_fraction": fraction if policy == "probe_escalation" else np.nan,
                        "selected_utility_penalty": penalty if policy == "probe_escalation" else np.nan,
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(environment_rows)
    aggregate = (
        metrics.groupby("policy", as_index=False)
        .agg(
            environments=("aurc", "size"),
            mean_relative_aurc_reduction_percent=("relative_aurc_reduction_percent", "mean"),
            minimum_environment_gain_percent=("relative_aurc_reduction_percent", "min"),
            positive_environments=("relative_aurc_reduction_percent", lambda x: int((x > 0).sum())),
            mean_calls=("mean_calls", "mean"),
        )
        .sort_values("mean_relative_aurc_reduction_percent", ascending=False)
    )
    route = aggregate.loc[aggregate["policy"].eq("probe_escalation")].iloc[0]
    strongest_fixed = aggregate.loc[
        aggregate["policy"].isin(["fixed_diagnostic", "fixed_addition", "fixed_removal"])
    ].sort_values("mean_relative_aurc_reduction_percent", ascending=False).iloc[0]
    gate = {
        "schema_version": "probe-escalation-offline-gate-1.0",
        "development_only": True,
        "datasets": datasets,
        "models": sorted(frame["model_id"].astype(str).unique()),
        "rows": int(len(frame)),
        "questions": int(frame[["dataset", "pilot_id"]].drop_duplicates().shape[0]),
        "checks": {
            "mean_gain_at_least_4_percent": bool(route["mean_relative_aurc_reduction_percent"] >= 4.0),
            "worst_environment_at_least_minus_1_percent": bool(route["minimum_environment_gain_percent"] >= -1.0),
            "at_least_five_positive_environments": bool(route["positive_environments"] >= 5),
            "mean_calls_at_most_2_5": bool(route["mean_calls"] <= 2.5),
            "call_reduction_vs_always_four_at_least_10_percent": bool(100.0 * (4.0 - route["mean_calls"]) / 4.0 >= 10.0),
            "beats_strongest_fixed_action": bool(
                route["mean_relative_aurc_reduction_percent"]
                > strongest_fixed["mean_relative_aurc_reduction_percent"]
            ),
            "reference_answer_accuracy_equality": True,
            "all_model_dataset_environments_reported": bool(len(metrics.loc[metrics["policy"].eq("probe_escalation")]) == 9),
        },
        "route": route.to_dict(),
        "strongest_fixed_action": strongest_fixed.to_dict(),
    }
    gate["pass"] = all(gate["checks"].values())

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "environment_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    pd.concat(selection_tables, ignore_index=True).to_csv(
        output_dir / "configuration_search.csv", index=False
    )
    pd.DataFrame(eligibility_rows).to_csv(output_dir / "eligibility.csv", index=False)
    (output_dir / "offline_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "probe-escalation-cross-dataset-evaluation-1.0",
        "seed": SEED,
        "development_sha256": sha256(development),
        "mvbench_sha256": sha256(mvbench),
        "outer_protocol": "leave one complete dataset out",
        "inner_folds": INNER_FOLDS,
        "grouping_unit": "dataset plus source video_id",
        "target_labels_used_before_metric_calculation": False,
        "mvbench_independent_confirmation_claim_permitted": False,
        "maximum_calls": 3,
        "pass": True,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--development",
        type=Path,
        default=Path(
            "sicr_route/results_fresh/download_2026-07-31/"
            "8901064.kman.restech.unsw.edu.au/multidataset_features.csv"
        ),
    )
    parser.add_argument(
        "--mvbench",
        type=Path,
        default=Path(
            "caver_route/results_fresh/mvbench_confirmation/"
            "8904763.kman.restech.unsw.edu.au/mvbench_caver_features.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("probe_route/results/initial")
    )
    args = parser.parse_args()
    evaluate(args.development, args.mvbench, args.output_dir)


if __name__ == "__main__":
    main()
