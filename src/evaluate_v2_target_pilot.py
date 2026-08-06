"""Development-only TRACE-RC target-pilot stress test on Video-MME-v2.

Video-MME-v2 labels have already been opened, so this program can select a
future protocol but cannot create a new confirmation claim.  Every split is
grouped by source video.  The target-logistic candidate is evaluated on the
pilot with inner group-cross-fitted predictions and is refit on the complete
pilot only after promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ID_COLUMNS = {"dataset", "qid", "group_id", "model_id", "correct"}
DERIVED_PREFIXES = ("score__",)
DERIVED_COLUMNS = {
    "raw_confidence",
    "graph_score",
    "trace_score",
    "eco_score",
    "eco_calls",
    "eco_frames",
    "eco_used_graph",
}
SOURCE_CANDIDATES = {
    "source_trace_logistic": "score__full_graph_logistic",
    "source_trace_hgb": "score__full_graph_gradient_boosting",
}
OWN_INTERVENTION_SUFFIXES = (
    "_own_present",
    "_own_js",
    "_own_entropy_delta",
    "_own_selected_delta",
    "_own_flip",
)
BASE_GRAPH_COLUMNS = {
    "own_base_confidence",
    "own_base_entropy",
    "own_base_margin",
    "candidate_count_scaled",
    "base_vote_fraction",
    "base_selected_support_mean",
    "base_selected_support_std",
    "base_consensus_entropy",
}


def aurc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware area under the risk-coverage curve."""

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    risks: list[float] = []
    coverages: list[float] = []
    errors = 0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and s[end] == s[start]:
            end += 1
        errors += int((1 - y[start:end]).sum())
        coverages.append(end / len(y))
        risks.append(errors / end)
        start = end
    x = np.asarray([0.0, *coverages], dtype=float)
    r = np.asarray([risks[0], *risks], dtype=float)
    return float(np.trapz(r, x))


def relative_gain(labels: np.ndarray, raw: np.ndarray, candidate: np.ndarray) -> float:
    base = aurc(labels, raw)
    if base <= 0:
        return 0.0
    return 100.0 * (base - aurc(labels, candidate)) / base


def stable_uniform(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def feature_columns(frame: pd.DataFrame, *, own_only: bool = False) -> list[str]:
    columns = []
    for name in frame.columns:
        if name in ID_COLUMNS or name in DERIVED_COLUMNS:
            continue
        if name.startswith(DERIVED_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            if own_only and not (
                name in BASE_GRAPH_COLUMNS
                or name.startswith("base_peer_")
                or name.endswith(OWN_INTERVENTION_SUFFIXES)
            ):
                continue
            columns.append(name)
    if not columns:
        raise ValueError("no numeric pre-score features found")
    return columns


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=20260802,
                ),
            ),
        ]
    )


def cross_fitted_logistic(
    pilot: pd.DataFrame, columns: list[str], folds: int
) -> tuple[np.ndarray, Pipeline]:
    groups = pilot["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("target pilot needs at least two video groups")
    prediction = np.full(len(pilot), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=n_splits)
    for train, valid in splitter.split(pilot, groups=groups):
        model = make_logistic()
        model.fit(pilot.iloc[train][columns], pilot.iloc[train]["correct"].astype(int))
        prediction[valid] = model.predict_proba(pilot.iloc[valid][columns])[:, 1]
    if not np.isfinite(prediction).all():
        raise RuntimeError("incomplete target-logistic OOF predictions")
    final = make_logistic()
    final.fit(pilot[columns], pilot["correct"].astype(int))
    return prediction, final


def grouped_bootstrap_gains(
    pilot: pd.DataFrame,
    raw: np.ndarray,
    candidate: np.ndarray,
    iterations: int,
    seed: int,
) -> np.ndarray:
    groups = pilot["group_id"].astype(str).to_numpy()
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    result = np.empty(iterations, dtype=float)
    labels = pilot["correct"].to_numpy(int)
    for index in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        take = np.concatenate([positions[group] for group in sampled])
        result[index] = relative_gain(labels[take], raw[take], candidate[take])
    return result


@dataclass(frozen=True)
class Promotion:
    selected: str
    lower_bound: float
    pilot_gain: float
    candidate_scores: dict[str, np.ndarray]
    target_model: Pipeline


def promote(
    pilot: pd.DataFrame,
    columns: list[str],
    iterations: int,
    alpha: float,
    seed: int,
    candidate_library: str,
) -> Promotion:
    raw = pilot["score__raw_confidence"].to_numpy(float)
    target_oof, target_model = cross_fitted_logistic(pilot, columns, folds=5)
    scores = {}
    if candidate_library in {"full", "robust", "robust_safe"}:
        source_names = (
            SOURCE_CANDIDATES
            if candidate_library == "full"
            else {"source_trace_hgb": SOURCE_CANDIDATES["source_trace_hgb"]}
        )
        scores.update(
            {
                name: pilot[column].to_numpy(float)
                for name, column in source_names.items()
            }
        )
    scores["target_logistic"] = target_oof
    rows = []
    adjusted_quantile = alpha / len(scores)
    for offset, (name, values) in enumerate(sorted(scores.items())):
        distribution = grouped_bootstrap_gains(
            pilot, raw, values, iterations, seed + 7919 * offset
        )
        rows.append(
            (
                name,
                float(np.quantile(distribution, adjusted_quantile)),
                relative_gain(pilot["correct"].to_numpy(int), raw, values),
            )
        )
    lower_bound_margins = {
        "source_trace_hgb": 0.5 if candidate_library == "robust_safe" else 0.0,
        "source_trace_logistic": 0.0,
        "target_logistic": 0.0,
    }
    eligible = [row for row in rows if row[1] > lower_bound_margins[row[0]]]
    if not eligible:
        return Promotion("raw_fallback", 0.0, 0.0, scores, target_model)
    selected = sorted(eligible, key=lambda row: (-row[1], row[0]))[0]
    return Promotion(selected[0], selected[1], selected[2], scores, target_model)


def split_environment(
    environment: pd.DataFrame, fraction: float, salt: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_values = sorted(environment["group_id"].astype(str).unique())
    ranked = sorted(group_values, key=lambda value: (stable_uniform(value, salt), value))
    pilot_count = max(2, min(len(ranked) - 2, round(fraction * len(ranked))))
    pilot_groups = set(ranked[:pilot_count])
    mask = environment["group_id"].astype(str).isin(pilot_groups)
    return (
        environment.loc[mask].reset_index(drop=True),
        environment.loc[~mask].reset_index(drop=True),
    )


def evaluate_split(
    environment: pd.DataFrame,
    fraction: float,
    salt: str,
    iterations: int,
    alpha: float,
    seed: int,
    candidate_library: str,
) -> dict[str, object]:
    pilot, test = split_environment(environment, fraction, salt)
    columns = feature_columns(
        environment, own_only=candidate_library == "target_only"
    )
    decision = promote(
        pilot, columns, iterations, alpha, seed, candidate_library
    )
    if decision.selected == "raw_fallback":
        test_score = test["score__raw_confidence"].to_numpy(float)
    elif decision.selected == "target_logistic":
        test_score = decision.target_model.predict_proba(test[columns])[:, 1]
    else:
        test_score = test[SOURCE_CANDIDATES[decision.selected]].to_numpy(float)
    raw = test["score__raw_confidence"].to_numpy(float)
    labels = test["correct"].to_numpy(int)
    exact_fallback = bool(
        decision.selected != "raw_fallback" or np.array_equal(raw, test_score)
    )
    return {
        "salt": salt,
        "pilot_fraction": fraction,
        "model_id": str(environment["model_id"].iloc[0]),
        "pilot_groups": int(pilot["group_id"].nunique()),
        "pilot_questions": int(len(pilot)),
        "test_groups": int(test["group_id"].nunique()),
        "test_questions": int(len(test)),
        "selected": decision.selected,
        "pilot_gain_percent": decision.pilot_gain,
        "multiplicity_adjusted_lower_bound_percent": decision.lower_bound,
        "test_gain_percent": relative_gain(labels, raw, test_score),
        "exact_fallback": exact_fallback,
    }


def parse_fractions(values: Iterable[str]) -> list[float]:
    result = sorted({float(value) for value in values})
    if not result or any(value <= 0 or value >= 1 for value in result):
        raise ValueError("pilot fractions must lie strictly between zero and one")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-fraction", action="append", default=[])
    parser.add_argument("--salts", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--candidate-library",
        choices=("target_only", "robust", "robust_safe", "full"),
        default="target_only",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.scored)
    required = {
        "group_id",
        "model_id",
        "correct",
        "score__raw_confidence",
        *SOURCE_CANDIDATES.values(),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    fractions = parse_fractions(
        args.pilot_fraction if args.pilot_fraction else ["0.25", "0.35", "0.50"]
    )
    rows = []
    for fraction in fractions:
        for salt_index in range(args.salts):
            salt = f"trace-rc-v2-dev-{salt_index:03d}"
            for model_id, environment in frame.groupby("model_id", sort=True):
                rows.append(
                    evaluate_split(
                        environment.reset_index(drop=True),
                        fraction,
                        salt,
                        args.bootstrap_iterations,
                        args.alpha,
                        20260802 + 100003 * salt_index,
                        args.candidate_library,
                    )
                )
    result = pd.DataFrame(rows)
    aggregate = (
        result.groupby("pilot_fraction", as_index=False)
        .agg(
            split_models=("test_gain_percent", "size"),
            mean_test_gain_percent=("test_gain_percent", "mean"),
            median_test_gain_percent=("test_gain_percent", "median"),
            worst_test_gain_percent=("test_gain_percent", "min"),
            positive_rate=("test_gain_percent", lambda x: float((x > 0).mean())),
            fallback_rate=("selected", lambda x: float((x == "raw_fallback").mean())),
            exact_fallback_rate=("exact_fallback", "mean"),
        )
        .sort_values("pilot_fraction")
    )
    eligible = aggregate.loc[
        aggregate["mean_test_gain_percent"].ge(0.5)
        & aggregate["worst_test_gain_percent"].ge(-1.0)
        & aggregate["positive_rate"].ge(0.25)
        & aggregate["exact_fallback_rate"].eq(1.0)
    ]
    if eligible.empty:
        selected_fraction = None
        gate_pass = False
    else:
        best = eligible.sort_values(
            ["mean_test_gain_percent", "pilot_fraction"], ascending=[False, True]
        ).iloc[0]
        selected_fraction = float(best["pilot_fraction"])
        gate_pass = True
    summary = {
        "schema_version": "trace-rc-v2-development-stress-1.0",
        "development_only": True,
        "fresh_confirmation_claim_permitted": False,
        "grouping_unit": "Video-MME-v2 source video",
        "candidate_library": args.candidate_library,
        "candidate_count_excluding_fallback": {
            "target_only": 1,
            "robust": 2,
            "robust_safe": 2,
            "full": 3,
        }[args.candidate_library],
        "familywise_alpha": args.alpha,
        "salts": args.salts,
        "bootstrap_iterations_per_decision": args.bootstrap_iterations,
        "development_gate_thresholds": {
            "mean_test_gain_percent_at_least": 0.5,
            "worst_test_gain_percent_at_least": -1.0,
            "positive_split_model_rate_at_least": 0.25,
            "exact_fallback_rate": 1.0,
        },
        "selected_pilot_fraction": selected_fraction,
        "gate_pass": gate_pass,
        "scored_sha256": hashlib.sha256(args.scored.read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "split_model_results.csv", index=False)
    aggregate.to_csv(args.output_dir / "pilot_fraction_summary.csv", index=False)
    (args.output_dir / "development_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(aggregate.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
