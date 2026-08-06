#!/usr/bin/env python3
"""Retrospective robustness and efficiency audits for TRACE-RC.

This program never changes the two sealed primary analyses.  It consumes the
hash-verified response archives after both confirmation labels have been
opened and produces clearly marked secondary analyses:

* paired video bootstrap comparisons against every reported baseline;
* fixed-test pilot-label subsampling within the originally registered pilot;
* model-panel and temporal-view compute ablations using cached probabilities;
* macro risk at fixed coverage levels and complete risk--coverage curves.

The full, base-graph, scalar, and source-HGB scores are reconstructed first.
The program stops unless they reproduce the archived confirmation scores.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
for candidate in (HERE, WORKSPACE / "probe_route" / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import analyze_trace_videomme_confirmation as common  # noqa: E402
import evaluate_response_graph_features as graph  # noqa: E402
from evaluate_v2_target_pilot import (  # noqa: E402
    make_logistic,
    promote,
    relative_gain,
    stable_uniform,
)


ID_COLUMNS = {
    "dataset",
    "qid",
    "group_id",
    "model_id",
    "correct",
    "task",
    "role",
}
SCALAR_COLUMNS = [
    "own_base_confidence",
    "own_base_entropy",
    "own_base_margin",
    "candidate_count_scaled",
]
ROLE_BUDGETS = {"base": 16, "invariant": 8, "addition": 24, "removal": 8}
ALL_ROLES = tuple(ROLE_BUDGETS)
PAIRWISE_COMPARATORS = [
    "raw_confidence",
    "target_scalar_logistic",
    "target_base_graph_logistic",
    "source_trace_logistic",
    "source_trace_hgb",
    "target_full_graph_logistic",
]
RECONSTRUCTION_TOLERANCE = 1e-8


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    dataset: str
    seed: int
    manifest: Path
    pilot_labels: Path
    test_labels: Path
    target_results: tuple[Path, ...]
    scored_predictions: Path
    promotion_lock: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "internvl" in lowered:
        return "InternVL"
    if "qwen" in lowered:
        return "Qwen"
    if "llava" in lowered:
        return "LLaVA"
    raise ValueError(f"unknown model family: {model_id}")


def default_specs(root: Path) -> list[DatasetSpec]:
    route = root / "trace_rc_route"
    her_archive = (
        route
        / "results"
        / "herbench_confirmation"
        / "8918264.kman.restech.unsw.edu.au"
    )
    per_archive = (
        route
        / "results"
        / "perception_confirmation"
        / "8919110.kman.restech.unsw.edu.au"
    )
    return [
        DatasetSpec(
            key="herbench",
            display="HERBench",
            dataset="HERBench-Lite-v2",
            seed=20260802,
            manifest=route / "input" / "herbench_lite_v2_inference.jsonl",
            pilot_labels=route / "input" / "pilot_labels.csv",
            test_labels=route / "input" / "sealed_test_labels.csv",
            target_results=tuple(
                route / "input" / "audited_raw" / "herbench" / f"{name}_predictions.jsonl"
                for name in ("qwen", "internvl", "llava")
            ),
            scored_predictions=her_archive / "test_scored_predictions.csv",
            promotion_lock=her_archive / "promotion_decision_lock.json",
        ),
        DatasetSpec(
            key="perception",
            display="Perception Test",
            dataset="Perception-Test-valid500",
            seed=20260803,
            manifest=(
                route
                / "input"
                / "perception_test"
                / "frozen"
                / "perception_test_valid500_inference.jsonl"
            ),
            pilot_labels=(route / "input" / "perception_test" / "frozen" / "pilot_labels.csv"),
            test_labels=(
                route / "input" / "perception_test" / "frozen" / "sealed_test_labels.csv"
            ),
            target_results=tuple(
                route / "input" / "audited_raw" / "perception" / f"{name}_predictions.jsonl"
                for name in ("qwen", "internvl", "llava")
            ),
            scored_predictions=per_archive / "test_scored_predictions.csv",
            promotion_lock=per_archive / "promotion_decision_lock.json",
        ),
    ]


def source_paths(root: Path) -> tuple[list[Path], list[Path], Path, Path]:
    features = [
        root
        / "sicr_route"
        / "results_fresh"
        / "download_2026-07-31"
        / "8901064.kman.restech.unsw.edu.au"
        / "multidataset_features.csv",
        root
        / "caver_route"
        / "results_fresh"
        / "mvbench_confirmation"
        / "8904763.kman.restech.unsw.edu.au"
        / "mvbench_caver_features.csv",
    ]
    results = [
        root / "sicr_route" / "results" / "katana_phase2" / "qwen3_vl_4b_predictions.jsonl",
        root / "sicr_route" / "results" / "katana_phase2" / "internvl3_5_4b_predictions.jsonl",
        root
        / "sicr_route"
        / "results_fresh"
        / "download_2026-07-31"
        / "8900849.kman.restech.unsw.edu.au"
        / "nextgqa_llava"
        / "llava_ov2_8b_predictions.jsonl",
    ]
    star = (
        root
        / "sicr_route"
        / "results_fresh"
        / "download_2026-07-31"
        / "8900849.kman.restech.unsw.edu.au"
        / "star_three_model"
    )
    mvbench = (
        root
        / "caver_route"
        / "results_fresh"
        / "mvbench_inference"
        / "8904762.kman.restech.unsw.edu.au"
    )
    vhd = (
        root
        / "probe_route"
        / "results"
        / "rava_vhd_confirmation"
        / "raw"
        / "8905351.kman.restech.unsw.edu.au"
    )
    temp = (
        root
        / "probe_route"
        / "results"
        / "prave_tempcompass_confirmation"
        / "raw"
        / "8905479.kman.restech.unsw.edu.au"
    )
    for directory in (star, mvbench, vhd, temp):
        results.extend(directory / f"{name}_predictions.jsonl" for name in ("qwen", "internvl", "llava"))
    vhd_scored = root / "probe_route" / "results" / "rava_vhd_confirmation" / "scored_predictions.csv"
    temp_scored = (
        root / "probe_route" / "results" / "prave_tempcompass_confirmation" / "scored_predictions.csv"
    )
    return features, results, vhd_scored, temp_scored


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("required files are missing:\n" + "\n".join(missing))


def load_manifest(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def target_metadata(spec: DatasetSpec) -> pd.DataFrame:
    manifest = load_manifest(spec.manifest)
    pieces = []
    for role, path in (("calibration", spec.pilot_labels), ("test", spec.test_labels)):
        selected = manifest.loc[manifest["role"].eq(role)].copy()
        labels = pd.read_csv(path)
        merged = selected.merge(
            labels[["question_id", "video_id", "gold_candidate_index"]],
            on=["question_id", "video_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(selected) or len(labels) != len(selected):
            raise ValueError(f"{spec.key}/{role}: label inventory mismatch")
        pieces.append(
            pd.DataFrame(
                {
                    "dataset": spec.dataset,
                    "qid": merged["question_id"].astype(str),
                    "group_id": merged["video_id"].astype(str),
                    "gold": merged["gold_candidate_index"].astype(int),
                    "candidate_count": merged["candidates"].map(len).astype(int),
                    "task": merged["task"].astype(str),
                    "role": role,
                }
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    if result["qid"].duplicated().any():
        raise ValueError(f"{spec.key}: duplicate question identifiers")
    return result


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in ID_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def reported_relative_gain(
    labels: np.ndarray, raw: np.ndarray, candidate: np.ndarray
) -> float:
    """Use the AURC definition used by the archived confirmation tables."""

    baseline = graph.aurc(np.asarray(labels, int), np.asarray(raw, float))
    if baseline <= 0:
        return 0.0
    value = graph.aurc(np.asarray(labels, int), np.asarray(candidate, float))
    return 100.0 * (baseline - value) / baseline


def build_graph_table(
    metadata: pd.DataFrame,
    values: dict[tuple[str, str, str, str], np.ndarray],
    models: list[str],
    roles: Iterable[str] = ALL_ROLES,
) -> tuple[pd.DataFrame, list[str]]:
    allowed = set(roles)
    panel = set(models)
    filtered = {
        key: value
        for key, value in values.items()
        if key[1] in panel and key[3] in allowed
    }
    base_meta = metadata[["dataset", "qid", "group_id", "gold", "candidate_count"]]
    table, base_columns = graph.graph_table(base_meta, filtered, models)
    table = table.merge(
        metadata[["qid", "task", "role"]],
        on="qid",
        how="left",
        validate="many_to_one",
    )
    return table, base_columns


def fit_target_scores(
    table: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    fitted = {}
    for model_id, environment in table.groupby("model_id", sort=True):
        pilot = environment.loc[environment["role"].eq("calibration")].copy()
        test = environment.loc[environment["role"].eq("test")].copy()
        model = make_logistic()
        model.fit(pilot[columns], pilot["correct"].astype(int))
        score = model.predict_proba(test[columns])[:, 1]
        raw = test["own_base_confidence"].to_numpy(float)
        labels = test["correct"].to_numpy(int)
        for row, value in zip(test.itertuples(index=False), score):
            rows.append({"qid": row.qid, "model_id": model_id, "score": float(value)})
        fitted[model_id] = model
        if graph.aurc(labels, raw) <= 0:
            raise ValueError(f"{model_id}: raw AURC is zero")
    return pd.DataFrame(rows), fitted


def score_gain(table: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    test = table.loc[table["role"].eq("test")].merge(
        scores, on=["qid", "model_id"], how="inner", validate="one_to_one"
    )
    rows = []
    for model_id, group_frame in test.groupby("model_id", sort=True):
        labels = group_frame["correct"].to_numpy(int)
        raw = group_frame["own_base_confidence"].to_numpy(float)
        candidate = group_frame["score"].to_numpy(float)
        rows.append(
            {
                "model_id": model_id,
                "family": model_family(model_id),
                "questions": len(group_frame),
                "videos": int(group_frame["group_id"].nunique()),
                "raw_aurc": graph.aurc(labels, raw),
                "candidate_aurc": graph.aurc(labels, candidate),
                "relative_aurc_reduction_percent": reported_relative_gain(
                    labels, raw, candidate
                ),
            }
        )
    return pd.DataFrame(rows)


def archived_score_frame(spec: DatasetSpec) -> pd.DataFrame:
    frame = pd.read_csv(spec.scored_predictions)
    required = {"qid", "group_id", "model_id", "correct", "score__TRACE-RC"}
    required.update(f"score__{name}" for name in PAIRWISE_COMPARATORS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{spec.key}: scored archive missing {missing}")
    return frame


def assert_reconstruction(
    spec: DatasetSpec,
    table: pd.DataFrame,
    source_hgb,
    full_columns: list[str],
    base_columns: list[str],
) -> dict[str, float]:
    archived = archived_score_frame(spec)
    comparisons: list[tuple[str, list[str], str]] = [
        ("target_scalar_logistic", SCALAR_COLUMNS, "score__target_scalar_logistic"),
        ("target_base_graph_logistic", base_columns, "score__target_base_graph_logistic"),
        ("target_full_graph_logistic", full_columns, "score__target_full_graph_logistic"),
    ]
    maximums: dict[str, float] = {}
    for name, columns, archive_column in comparisons:
        reconstructed, _ = fit_target_scores(table, columns)
        merged = reconstructed.merge(
            archived[["qid", "model_id", archive_column]],
            on=["qid", "model_id"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(archived):
            raise RuntimeError(f"{spec.key}/{name}: archived row mismatch")
        error = float(np.max(np.abs(merged["score"] - merged[archive_column])))
        maximums[name] = error
        if error > RECONSTRUCTION_TOLERANCE:
            raise RuntimeError(f"{spec.key}/{name}: reconstruction error {error:.3e}")

    test = table.loc[table["role"].eq("test")].copy()
    source_score = source_hgb.predict_proba(test[full_columns])[:, 1]
    source_rebuilt = test[["qid", "model_id"]].copy()
    source_rebuilt["score"] = source_score
    source_merged = source_rebuilt.merge(
        archived[["qid", "model_id", "score__source_trace_hgb"]],
        on=["qid", "model_id"],
        how="inner",
        validate="one_to_one",
    )
    source_error = float(
        np.max(np.abs(source_merged["score"] - source_merged["score__source_trace_hgb"]))
    )
    maximums["source_trace_hgb"] = source_error
    if source_error > RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(f"{spec.key}/source-HGB: reconstruction error {source_error:.3e}")
    return maximums


def paired_bootstrap(
    spec: DatasetSpec, iterations: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = archived_score_frame(spec)
    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    rng = np.random.default_rng(spec.seed + 4409)
    methods = ["TRACE-RC", *PAIRWISE_COMPARATORS]
    payloads = []
    for model_id, environment in frame.groupby("model_id", sort=True):
        environment = environment.reset_index(drop=True)
        group_array = environment["group_id"].astype(str).to_numpy()
        positions = {
            group_id: np.flatnonzero(group_array == group_id) for group_id in groups
        }
        if any(len(value) == 0 for value in positions.values()):
            raise RuntimeError(f"{spec.key}/{model_id}: incomplete video panel")
        payloads.append(
            {
                "positions": positions,
                "labels": environment["correct"].to_numpy(int),
                "scores": {
                    method: environment[f"score__{method}"].to_numpy(float)
                    for method in methods
                },
            }
        )
    point = {}
    for method in methods:
        gains = []
        for payload in payloads:
            gains.append(
                reported_relative_gain(
                    payload["labels"],
                    payload["scores"]["raw_confidence"],
                    payload["scores"][method],
                )
            )
        point[method] = float(np.mean(gains))

    distribution_values = {
        comparator: np.empty(iterations, dtype=float)
        for comparator in PAIRWISE_COMPARATORS
    }
    for iteration in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        gains = {method: [] for method in methods}
        for payload in payloads:
            take = np.concatenate(
                [payload["positions"][group_id] for group_id in sampled]
            )
            labels = payload["labels"][take]
            raw = payload["scores"]["raw_confidence"][take]
            for method in methods:
                gains[method].append(
                    reported_relative_gain(
                        labels, raw, payload["scores"][method][take]
                    )
                )
        macro = {method: float(np.mean(values)) for method, values in gains.items()}
        for comparator in PAIRWISE_COMPARATORS:
            distribution_values[comparator][iteration] = (
                macro["TRACE-RC"] - macro[comparator]
            )
    distributions = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": spec.display,
                    "iteration": np.arange(iterations, dtype=int),
                    "comparator": comparator,
                    "delta_gain_percent_points": values,
                }
            )
            for comparator, values in distribution_values.items()
        ],
        ignore_index=True,
    )
    rows = []
    for comparator, values in distributions.groupby("comparator", sort=False):
        delta = values["delta_gain_percent_points"].to_numpy(float)
        lower, upper = np.quantile(delta, [0.025, 0.975])
        rows.append(
            {
                "dataset": spec.display,
                "comparator": comparator,
                "trace_rc_gain_percent": point["TRACE-RC"],
                "comparator_gain_percent": point[comparator],
                "delta_gain_percent_points": point["TRACE-RC"] - point[comparator],
                "paired_video_bootstrap_ci95_low": float(lower),
                "paired_video_bootstrap_ci95_high": float(upper),
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(rows), distributions


def pilot_subset_groups(
    pilot_groups: list[str], total_groups: int, fraction: float, salt: str
) -> set[str]:
    requested = round(total_groups * fraction)
    count = max(2, min(len(pilot_groups), requested))
    ranked = sorted(pilot_groups, key=lambda value: (stable_uniform(value, salt), value))
    return set(ranked[:count])


def pilot_label_efficiency(
    spec: DatasetSpec,
    table: pd.DataFrame,
    source_hgb,
    columns: list[str],
    fractions: list[float],
    salts: int,
    bootstrap_iterations: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = table.copy()
    frame["score__raw_confidence"] = frame["own_base_confidence"].to_numpy(float)
    frame["score__full_graph_gradient_boosting"] = source_hgb.predict_proba(frame[columns])[:, 1]
    registered = json.loads(spec.promotion_lock.read_text(encoding="utf-8"))["decisions"]
    all_groups = sorted(frame["group_id"].astype(str).unique())
    original_pilot = frame.loc[frame["role"].eq("calibration")]
    pilot_groups = sorted(original_pilot["group_id"].astype(str).unique())
    test = frame.loc[frame["role"].eq("test")].copy()
    tasks = []
    for fraction in fractions:
        salt_values = ["registered-full-pilot"] if np.isclose(fraction, 0.5) else [
            f"{spec.key}-pilot-subsample-{index:03d}" for index in range(salts)
        ]
        for salt in salt_values:
            chosen = (
                set(pilot_groups)
                if np.isclose(fraction, 0.5)
                else pilot_subset_groups(pilot_groups, len(all_groups), fraction, salt)
            )
            for model_index, (model_id, environment) in enumerate(
                frame.groupby("model_id", sort=True)
            ):
                tasks.append(
                    (
                        spec.display,
                        fraction,
                        salt,
                        model_id,
                        model_index,
                        environment,
                        test.loc[test["model_id"].eq(model_id)],
                        chosen,
                        columns,
                        spec.seed,
                        bootstrap_iterations,
                        registered[model_id]["selected"],
                    )
                )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(run_pilot_task, tasks))
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["dataset", "pilot_fraction_of_target"], as_index=False)
        .agg(
            split_models=("test_gain_percent", "size"),
            mean_pilot_videos=("pilot_videos", "mean"),
            mean_pilot_questions=("pilot_questions", "mean"),
            promotion_rate=("selected", lambda values: float((values != "raw_fallback").mean())),
            fallback_rate=("selected", lambda values: float((values == "raw_fallback").mean())),
            mean_test_gain_percent=("test_gain_percent", "mean"),
            median_test_gain_percent=("test_gain_percent", "median"),
            tenth_percentile_test_gain=("test_gain_percent", lambda values: float(np.quantile(values, 0.10))),
            worst_test_gain_percent=("test_gain_percent", "min"),
            harmful_promotion_rate=("harmful_promotion", "mean"),
            exact_fallback_rate=("exact_fallback", "mean"),
            registered_decision_match_rate=("matches_registered_decision", "mean"),
        )
        .sort_values(["dataset", "pilot_fraction_of_target"])
    )
    return detail, summary


def evaluate_pilot_task(
    dataset: str,
    fraction: float,
    salt: str,
    model_id: str,
    model_index: int,
    environment: pd.DataFrame,
    model_test: pd.DataFrame,
    chosen: set[str],
    columns: list[str],
    seed: int,
    bootstrap_iterations: int,
    registered_decision: str,
) -> dict[str, object]:
    """Evaluate one independent model/pilot-subsample task."""

    pilot = environment.loc[
        environment["role"].eq("calibration")
        & environment["group_id"].astype(str).isin(chosen)
    ].reset_index(drop=True)
    model_test = model_test.reset_index(drop=True)
    iterations = 2000 if np.isclose(fraction, 0.5) else bootstrap_iterations
    decision = promote(
        pilot,
        columns,
        iterations,
        0.05,
        seed + 1009 * model_index,
        "robust_safe",
    )
    raw = model_test["score__raw_confidence"].to_numpy(float)
    if decision.selected == "raw_fallback":
        score = raw.copy()
    elif decision.selected == "target_logistic":
        score = decision.target_model.predict_proba(model_test[columns])[:, 1]
    elif decision.selected == "source_trace_hgb":
        score = model_test["score__full_graph_gradient_boosting"].to_numpy(float)
    else:
        raise RuntimeError(f"unexpected candidate {decision.selected}")
    labels = model_test["correct"].to_numpy(int)
    gain = reported_relative_gain(labels, raw, score)
    return {
        "dataset": dataset,
        "pilot_fraction_of_target": fraction,
        "salt": salt,
        "model_id": model_id,
        "family": model_family(model_id),
        "pilot_videos": int(pilot["group_id"].nunique()),
        "pilot_questions": int(len(pilot)),
        "selected": decision.selected,
        "pilot_gain_percent": decision.pilot_gain,
        "adjusted_lower_bound_percent": decision.lower_bound,
        "test_gain_percent": gain,
        "harmful_promotion": bool(decision.selected != "raw_fallback" and gain < 0.0),
        "exact_fallback": bool(
            decision.selected != "raw_fallback" or np.array_equal(raw, score)
        ),
        "matches_registered_decision": bool(
            not np.isclose(fraction, 0.5) or decision.selected == registered_decision
        ),
    }


def run_pilot_task(arguments: tuple[object, ...]) -> dict[str, object]:
    return evaluate_pilot_task(*arguments)


def budget_audit(
    spec: DatasetSpec,
    metadata: pd.DataFrame,
    values: dict[tuple[str, str, str, str], np.ndarray],
    models: list[str],
) -> pd.DataFrame:
    rows = []

    def append_config(
        family: str,
        config: str,
        panel: list[str],
        roles: tuple[str, ...],
        columns_mode: str = "full",
    ) -> None:
        table, base_columns = build_graph_table(metadata, values, panel, roles)
        if columns_mode == "scalar":
            columns = SCALAR_COLUMNS
        elif columns_mode == "base":
            columns = base_columns
        else:
            columns = feature_columns(table)
        scores, _ = fit_target_scores(table, columns)
        gains = score_gain(table, scores)
        calls = 1 if columns_mode == "scalar" else len(panel) * len(roles)
        frames = 16 if columns_mode == "scalar" else len(panel) * sum(ROLE_BUDGETS[r] for r in roles)
        for item in gains.itertuples(index=False):
            rows.append(
                {
                    "dataset": spec.display,
                    "audit_family": family,
                    "configuration": config,
                    "panel_models": "+".join(model_family(value) for value in panel),
                    "reference_model": item.model_id,
                    "reference_family": item.family,
                    "roles": "+".join(roles),
                    "model_calls_per_question": calls,
                    "model_input_frames_per_question": frames,
                    "relative_aurc_reduction_percent": item.relative_aurc_reduction_percent,
                }
            )

    full_panel = list(models)
    append_config("model_panel", "Own scalar", full_panel, ("base",), "scalar")
    append_config("model_panel", "Three-family base", full_panel, ("base",), "base")
    for model_id in models:
        append_config("model_panel", "Own four views", [model_id], ALL_ROLES)
    for pair in itertools.combinations(models, 2):
        append_config("model_panel", "Two-family four views", list(pair), ALL_ROLES)
    append_config("model_panel", "Three-family four views", full_panel, ALL_ROLES)

    append_config("view_panel", "Base only", full_panel, ("base",), "base")
    for role in ALL_ROLES[1:]:
        append_config("view_panel", f"Base + {role}", full_panel, ("base", role))
    for pair in itertools.combinations(ALL_ROLES[1:], 2):
        append_config("view_panel", "Base + " + " + ".join(pair), full_panel, ("base", *pair))
    append_config("view_panel", "All temporal relations", full_panel, ALL_ROLES)
    return pd.DataFrame(rows)


def risk_coverage(spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = archived_score_frame(spec)
    methods = {
        "Raw confidence": "score__raw_confidence",
        "Target base graph": "score__target_base_graph_logistic",
        "TRACE-RC": "score__TRACE-RC",
    }
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 2)
    rows = []
    for label, column in methods.items():
        for coverage in coverages:
            risks = []
            for _, environment in frame.groupby("model_id", sort=True):
                risks.append(
                    common.risk_at_coverage(
                        environment["correct"].to_numpy(int),
                        environment[column].to_numpy(float),
                        float(coverage),
                    )
                )
            rows.append(
                {
                    "dataset": spec.display,
                    "method": label,
                    "coverage": float(coverage),
                    "macro_risk": float(np.mean(risks)),
                }
            )
    curve = pd.DataFrame(rows)
    fixed = curve.loc[curve["coverage"].isin([0.25, 0.50, 0.75, 1.00])].copy()
    return curve, fixed


def latex_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def display_two_decimals(value: float) -> str:
    numeric = 0.0 if abs(float(value)) < 0.005 else float(value)
    return f"{numeric:.2f}"


def write_latex_assets(
    output_dir: Path,
    pairwise: pd.DataFrame,
    pilot: pd.DataFrame,
    budget: pd.DataFrame,
    fixed_risk: pd.DataFrame,
) -> None:
    asset_dir = output_dir / "paper_artifacts"
    asset_dir.mkdir(parents=True, exist_ok=True)

    comparator_labels = {
        "raw_confidence": "Raw confidence",
        "target_scalar_logistic": "Target scalar",
        "target_base_graph_logistic": "Target base graph",
        "source_trace_logistic": "Source graph logistic",
        "source_trace_hgb": "Source graph HGB",
        "target_full_graph_logistic": "Unguarded target full graph",
    }
    pair_rows = []
    for row in pairwise.itertuples(index=False):
        pair_rows.append(
            f"{latex_escape(row.dataset)} & {latex_escape(comparator_labels[row.comparator])} & "
            f"{row.delta_gain_percent_points:.2f} & "
            f"[{row.paired_video_bootstrap_ci95_low:.2f}, "
            f"{row.paired_video_bootstrap_ci95_high:.2f}] \\\\"
        )
    (asset_dir / "trace_rc_posthoc_pairwise_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}YYrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Target & Baseline & \multicolumn{1}{c}{Gain (pp)} & "
                r"\multicolumn{1}{c}{95\% CI} \\",
                r"\midrule",
                *pair_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    pilot_rows = []
    for row in pilot.itertuples(index=False):
        pilot_rows.append(
            f"{latex_escape(row.dataset)} & {100*row.pilot_fraction_of_target:.0f}\% & "
            f"{row.mean_pilot_videos:.0f} & {100*row.promotion_rate:.1f}\% & "
            f"{row.mean_test_gain_percent:.2f}\% & {100*row.harmful_promotion_rate:.1f}\% \\\\"
        )
    (asset_dir / "trace_rc_posthoc_pilot_efficiency_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}Yrrrrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Target & \multicolumn{1}{c}{Pilot} & \multicolumn{1}{c}{Videos} & "
                r"\multicolumn{1}{c}{Use} & \multicolumn{1}{c}{Gain} & "
                r"\multicolumn{1}{c}{Harmful} \\",
                r"\midrule",
                *pilot_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    model_budget = budget.loc[budget["audit_family"].eq("model_panel")]
    aggregate = (
        model_budget.groupby(
            ["dataset", "configuration", "model_calls_per_question", "model_input_frames_per_question"],
            as_index=False,
        )
        .agg(
            mean_gain=("relative_aurc_reduction_percent", "mean"),
            worst_gain=("relative_aurc_reduction_percent", "min"),
        )
        .sort_values(["dataset", "model_calls_per_question"])
    )
    budget_rows = []
    for row in aggregate.itertuples(index=False):
        budget_rows.append(
            f"{latex_escape(row.dataset)} & {latex_escape(row.configuration)} & "
            f"{row.model_calls_per_question} & {row.model_input_frames_per_question} & "
            f"{display_two_decimals(row.mean_gain)}\% & "
            f"{display_two_decimals(row.worst_gain)}\% \\\\"
        )
    (asset_dir / "trace_rc_posthoc_budget_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}YYrrrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Target & Panel & \multicolumn{1}{c}{Calls} & \multicolumn{1}{c}{Frames} & "
                r"\multicolumn{1}{c}{Mean} & \multicolumn{1}{c}{Min.} \\",
                r"\midrule",
                *budget_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    risk_rows = []
    for row in fixed_risk.itertuples(index=False):
        risk_rows.append(
            f"{latex_escape(row.dataset)} & {latex_escape(row.method)} & "
            f"{100*row.coverage:.0f}\% & {100*row.macro_risk:.2f}\% \\\\"
        )
    (asset_dir / "trace_rc_posthoc_risk_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrr}",
                r"\toprule",
                r"Target & Method & Coverage & Average risk \\",
                r"\midrule",
                *risk_rows,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "trace_rc_route" / "results" / "posthoc_robustness",
    )
    parser.add_argument("--paired-bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--pilot-bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--pilot-salts", type=int, default=20)
    parser.add_argument("--pilot-workers", type=int, default=4)
    args = parser.parse_args()

    root = args.workspace.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = default_specs(root)
    feature_paths, source_result_paths, vhd_scored, temp_scored = source_paths(root)
    all_inputs = [*feature_paths, *source_result_paths, vhd_scored, temp_scored]
    for spec in specs:
        all_inputs.extend(
            [
                spec.manifest,
                spec.pilot_labels,
                spec.test_labels,
                spec.scored_predictions,
                spec.promotion_lock,
                *spec.target_results,
            ]
        )
    require_files(all_inputs)

    source_values, source_models = graph.read_results(source_result_paths)
    source_meta = graph.load_metadata(feature_paths, vhd_scored, temp_scored)
    source_table, _ = graph.graph_table(source_meta, source_values, source_models)
    source_columns = feature_columns(source_table)
    source_hgb = common.fit_hgb(source_table, source_columns)

    reconstruction_records = []
    pairwise_summaries = []
    pairwise_distributions = []
    pilot_details = []
    pilot_summaries = []
    budget_records = []
    risk_curves = []
    risk_fixed = []

    for spec in specs:
        metadata = target_metadata(spec)
        values, models = graph.read_results(list(spec.target_results))
        if [model_family(value) for value in source_models] != [model_family(value) for value in models]:
            raise RuntimeError(f"{spec.key}: source and target model-family order differs")
        full_table, base_columns = build_graph_table(metadata, values, models, ALL_ROLES)
        columns = feature_columns(full_table)
        if source_columns != columns:
            raise RuntimeError(f"{spec.key}: source and target feature schemas differ")
        reconstruction = assert_reconstruction(
            spec, full_table, source_hgb, columns, base_columns
        )
        reconstruction_records.extend(
            {"dataset": spec.display, "score": name, "maximum_absolute_error": error}
            for name, error in reconstruction.items()
        )

        pair_summary, pair_distribution = paired_bootstrap(
            spec, args.paired_bootstrap_iterations
        )
        pairwise_summaries.append(pair_summary)
        pairwise_distributions.append(pair_distribution)

        pilot_detail, pilot_summary = pilot_label_efficiency(
            spec,
            full_table,
            source_hgb,
            columns,
            fractions=[0.10, 0.20, 0.30, 0.40, 0.50],
            salts=args.pilot_salts,
            bootstrap_iterations=args.pilot_bootstrap_iterations,
            workers=args.pilot_workers,
        )
        if not bool(
            pilot_detail.loc[
                pilot_detail["pilot_fraction_of_target"].eq(0.5),
                "matches_registered_decision",
            ].all()
        ):
            raise RuntimeError(f"{spec.key}: full-pilot decision was not reproduced")
        pilot_details.append(pilot_detail)
        pilot_summaries.append(pilot_summary)

        budget_records.append(budget_audit(spec, metadata, values, models))
        curve, fixed = risk_coverage(spec)
        risk_curves.append(curve)
        risk_fixed.append(fixed)

    reconstruction_frame = pd.DataFrame(reconstruction_records)
    pairwise_frame = pd.concat(pairwise_summaries, ignore_index=True)
    pairwise_distribution_frame = pd.concat(pairwise_distributions, ignore_index=True)
    pilot_detail_frame = pd.concat(pilot_details, ignore_index=True)
    pilot_summary_frame = pd.concat(pilot_summaries, ignore_index=True)
    budget_frame = pd.concat(budget_records, ignore_index=True)
    risk_curve_frame = pd.concat(risk_curves, ignore_index=True)
    risk_fixed_frame = pd.concat(risk_fixed, ignore_index=True)

    outputs = {
        "reconstruction_checks.csv": reconstruction_frame,
        "paired_bootstrap_summary.csv": pairwise_frame,
        "paired_bootstrap_distributions.csv": pairwise_distribution_frame,
        "pilot_label_efficiency_detail.csv": pilot_detail_frame,
        "pilot_label_efficiency_summary.csv": pilot_summary_frame,
        "compute_budget_ablation.csv": budget_frame,
        "risk_coverage_curves.csv": risk_curve_frame,
        "risk_at_fixed_coverage.csv": risk_fixed_frame,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    write_latex_assets(
        output, pairwise_frame, pilot_summary_frame, budget_frame, risk_fixed_frame
    )

    manifest = {
        "schema_version": "trace-rc-posthoc-robustness-1.0",
        "retrospective_secondary_analysis": True,
        "changes_sealed_primary_results": False,
        "fixed_test_policy": (
            "pilot fractions subsample only the registered calibration videos; "
            "the original sealed test split remains fixed"
        ),
        "parameters": {
            "paired_bootstrap_iterations": args.paired_bootstrap_iterations,
            "pilot_bootstrap_iterations": args.pilot_bootstrap_iterations,
            "pilot_salts": args.pilot_salts,
            "pilot_workers": args.pilot_workers,
            "pilot_fractions": [0.10, 0.20, 0.30, 0.40, 0.50],
        },
        "input_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(set(all_inputs))
        },
        "output_sha256": {
            path.name: sha256_file(path)
            for path in sorted(output.glob("*.csv"))
        },
        "reconstruction_maximum_absolute_error": float(
            reconstruction_frame["maximum_absolute_error"].max()
        ),
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(reconstruction_frame.to_string(index=False))
    print("\nPilot-label efficiency")
    print(pilot_summary_frame.to_string(index=False))
    print("\nPaired comparisons")
    print(pairwise_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
