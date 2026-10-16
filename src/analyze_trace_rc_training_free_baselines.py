#!/usr/bin/env python3
"""Training-free reliability baselines for the held-out evaluations.

This retrospective analysis uses only cached option probabilities.  It does
not change either target-check decision.  The baselines are deliberately
transparent: margin, entropy, cross-model agreement or support, temporal
stability, and the corresponding full-panel summaries.  Paired uncertainty is
estimated by resampling complete videos and recomputing the macro AURC gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_trace_rc_posthoc_robustness as audit
import evaluate_response_graph_features as graph


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BASELINE_ORDER = [
    "confidence_margin",
    "negative_entropy",
    "base_vote_then_confidence",
    "base_support",
    "own_temporal_stability",
    "all_view_vote_then_confidence",
    "all_view_support",
]
BASELINE_LABELS = {
    "confidence_margin": "Top-two margin",
    "negative_entropy": "Negative entropy",
    "base_vote_then_confidence": "Base cross-model vote",
    "base_support": "Base cross-model support",
    "own_temporal_stability": "Own-view stability",
    "all_view_vote_then_confidence": "All-view panel vote",
    "all_view_support": "All-view panel support",
}
VIEW_PREFIXES = ("base", "invariant", "addition", "removal")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def primary_then_confidence(primary: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Rank first by a discrete score and use confidence only for exact ties."""

    primary = np.asarray(primary, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if primary.shape != confidence.shape:
        raise ValueError("primary and confidence scores must have the same shape")
    return primary + 1e-9 * confidence


def training_free_score_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Create one row per fixed test answer with all training-free scores."""

    test = table.loc[table["role"].eq("test")].copy()
    required = {
        "qid",
        "group_id",
        "model_id",
        "correct",
        "own_base_confidence",
        "own_base_margin",
        "own_base_entropy",
        "base_vote_fraction",
        "base_selected_support_mean",
        "invariant_own_js",
        "addition_own_js",
        "removal_own_js",
    }
    required.update(f"{prefix}_vote_fraction" for prefix in VIEW_PREFIXES)
    required.update(f"{prefix}_selected_support_mean" for prefix in VIEW_PREFIXES)
    missing = sorted(required.difference(test.columns))
    if missing:
        raise ValueError(f"response graph table is missing columns: {missing}")

    result = test[["qid", "group_id", "model_id", "correct"]].copy()
    raw = test["own_base_confidence"].to_numpy(float)
    result["score__raw_confidence"] = raw
    result["score__confidence_margin"] = test["own_base_margin"].to_numpy(float)
    result["score__negative_entropy"] = -test["own_base_entropy"].to_numpy(float)
    result["score__base_vote_then_confidence"] = primary_then_confidence(
        test["base_vote_fraction"].to_numpy(float), raw
    )
    result["score__base_support"] = test["base_selected_support_mean"].to_numpy(float)

    temporal_js = test[
        ["invariant_own_js", "addition_own_js", "removal_own_js"]
    ].to_numpy(float)
    result["score__own_temporal_stability"] = primary_then_confidence(
        -temporal_js.mean(axis=1), raw
    )

    panel_vote = test[
        [f"{prefix}_vote_fraction" for prefix in VIEW_PREFIXES]
    ].to_numpy(float)
    result["score__all_view_vote_then_confidence"] = primary_then_confidence(
        panel_vote.mean(axis=1), raw
    )
    panel_support = test[
        [f"{prefix}_selected_support_mean" for prefix in VIEW_PREFIXES]
    ].to_numpy(float)
    result["score__all_view_support"] = panel_support.mean(axis=1)

    score_columns = [column for column in result if column.startswith("score__")]
    if not np.isfinite(result[score_columns].to_numpy(float)).all():
        raise ValueError("training-free score table contains a non-finite value")
    return result


def model_metrics(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for baseline in BASELINE_ORDER:
        for model_id, environment in frame.groupby("model_id", sort=True):
            labels = environment["correct"].to_numpy(int)
            raw = environment["score__raw_confidence"].to_numpy(float)
            score = environment[f"score__{baseline}"].to_numpy(float)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "baseline_label": BASELINE_LABELS[baseline],
                    "model_id": model_id,
                    "family": audit.model_family(model_id),
                    "questions": len(environment),
                    "videos": int(environment["group_id"].nunique()),
                    "raw_aurc": graph.aurc(labels, raw),
                    "baseline_aurc": graph.aurc(labels, score),
                    "relative_aurc_reduction_percent": audit.reported_relative_gain(
                        labels, raw, score
                    ),
                }
            )
    return pd.DataFrame(rows)


def paired_video_bootstrap(
    spec: audit.DatasetSpec,
    scores: pd.DataFrame,
    archived: pd.DataFrame,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = scores.merge(
        archived[["qid", "model_id", "score__TRACE-RC"]],
        on=["qid", "model_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(scores) or len(frame) != len(archived):
        raise RuntimeError(f"{spec.key}: cached and archived test rows do not match")

    archived_raw = archived[["qid", "model_id", "score__raw_confidence"]]
    raw_check = frame.merge(
        archived_raw,
        on=["qid", "model_id"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "__archived"),
    )
    error = float(
        np.max(
            np.abs(
                raw_check["score__raw_confidence"].to_numpy(float)
                - raw_check["score__raw_confidence__archived"].to_numpy(float)
            )
        )
    )
    if error > audit.RECONSTRUCTION_TOLERANCE:
        raise RuntimeError(f"{spec.key}: raw-confidence reconstruction error {error:.3e}")

    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    payloads: list[dict[str, object]] = []
    for _, environment in frame.groupby("model_id", sort=True):
        environment = environment.reset_index(drop=True)
        group_array = environment["group_id"].astype(str).to_numpy()
        positions = {group_id: np.flatnonzero(group_array == group_id) for group_id in groups}
        if any(len(value) == 0 for value in positions.values()):
            raise RuntimeError(f"{spec.key}: a model is missing at least one test video")
        payloads.append(
            {
                "positions": positions,
                "labels": environment["correct"].to_numpy(int),
                "raw": environment["score__raw_confidence"].to_numpy(float),
                "trace": environment["score__TRACE-RC"].to_numpy(float),
                "baselines": {
                    name: environment[f"score__{name}"].to_numpy(float)
                    for name in BASELINE_ORDER
                },
            }
        )

    trace_point = []
    baseline_points = {name: [] for name in BASELINE_ORDER}
    for payload in payloads:
        labels = payload["labels"]
        raw = payload["raw"]
        trace_point.append(audit.reported_relative_gain(labels, raw, payload["trace"]))
        for name in BASELINE_ORDER:
            baseline_points[name].append(
                audit.reported_relative_gain(labels, raw, payload["baselines"][name])
            )
    trace_macro = float(np.mean(trace_point))
    baseline_macro = {name: float(np.mean(values)) for name, values in baseline_points.items()}

    rng = np.random.default_rng(spec.seed + 8123)
    distributions = {name: np.empty(iterations, dtype=float) for name in BASELINE_ORDER}
    for iteration in range(iterations):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        trace_gains: list[float] = []
        comparator_gains = {name: [] for name in BASELINE_ORDER}
        for payload in payloads:
            take = np.concatenate([payload["positions"][group_id] for group_id in sampled])
            labels = payload["labels"][take]
            raw = payload["raw"][take]
            trace_gains.append(audit.reported_relative_gain(labels, raw, payload["trace"][take]))
            for name in BASELINE_ORDER:
                comparator_gains[name].append(
                    audit.reported_relative_gain(
                        labels, raw, payload["baselines"][name][take]
                    )
                )
        trace_gain = float(np.mean(trace_gains))
        for name in BASELINE_ORDER:
            distributions[name][iteration] = trace_gain - float(
                np.mean(comparator_gains[name])
            )

    distribution_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "dataset": spec.display,
                    "iteration": np.arange(iterations, dtype=int),
                    "baseline": name,
                    "trace_minus_baseline_gain_percent_points": values,
                }
            )
            for name, values in distributions.items()
        ],
        ignore_index=True,
    )
    summary_rows = []
    for name in BASELINE_ORDER:
        values = distributions[name]
        lower, upper = np.quantile(values, [0.025, 0.975])
        summary_rows.append(
            {
                "dataset": spec.display,
                "baseline": name,
                "baseline_label": BASELINE_LABELS[name],
                "trace_rc_gain_percent": trace_macro,
                "baseline_gain_percent": baseline_macro[name],
                "trace_minus_baseline_gain_percent_points": trace_macro
                - baseline_macro[name],
                "paired_video_bootstrap_ci95_low": float(lower),
                "paired_video_bootstrap_ci95_high": float(upper),
                "bootstrap_iterations": iterations,
            }
        )
    return pd.DataFrame(summary_rows), distribution_frame


def latex_escape(value: object) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_latex_table(output_dir: Path, summary: pd.DataFrame) -> None:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            f"{latex_escape(row.dataset)} & {latex_escape(row.baseline_label)} & "
            f"{row.baseline_gain_percent:.2f}\\% & "
            f"{row.trace_minus_baseline_gain_percent_points:.2f} & "
            f"[{row.paired_video_bootstrap_ci95_low:.2f}, "
            f"{row.paired_video_bootstrap_ci95_high:.2f}] \\\\"
        )
    asset_dir = output_dir / "paper_artifacts"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "trace_rc_posthoc_training_free_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}YYrrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Target & Score & \multicolumn{1}{c}{Score gain} & "
                r"\multicolumn{1}{c}{$\Delta$} & \multicolumn{1}{c}{95\% CI} \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabularx}",
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
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    root = args.workspace.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_frames = []
    summary_frames = []
    distribution_frames = []
    inputs: list[Path] = []
    for spec in audit.default_specs(root):
        required = [
            spec.manifest,
            spec.pilot_labels,
            spec.test_labels,
            spec.scored_predictions,
            *spec.target_results,
        ]
        audit.require_files(required)
        inputs.extend(required)
        metadata = audit.target_metadata(spec)
        values, models = graph.read_results(list(spec.target_results))
        table, _ = audit.build_graph_table(metadata, values, models, audit.ALL_ROLES)
        scores = training_free_score_frame(table)
        archived = audit.archived_score_frame(spec)
        metric_frames.append(model_metrics(spec.display, scores))
        summary, distributions = paired_video_bootstrap(
            spec, scores, archived, args.bootstrap_iterations
        )
        summary_frames.append(summary)
        distribution_frames.append(distributions)

    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    distributions = pd.concat(distribution_frames, ignore_index=True)
    output_files = {
        "training_free_model_metrics.csv": metrics,
        "training_free_paired_bootstrap_summary.csv": summary,
        "training_free_paired_bootstrap_distributions.csv": distributions,
    }
    for name, frame in output_files.items():
        frame.to_csv(output / name, index=False)
    write_latex_table(output, summary)

    manifest = {
        "schema_version": "trace-rc-training-free-baselines-1.0",
        "retrospective_secondary_analysis": True,
        "changes_sealed_primary_results": False,
        "baseline_definitions": {
            name: BASELINE_LABELS[name] for name in BASELINE_ORDER
        },
        "bootstrap_iterations": args.bootstrap_iterations,
        "input_sha256": {
            str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
            for path in sorted(set(inputs))
        },
        "output_sha256": {
            name: sha256_file(output / name) for name in output_files
        },
    }
    (output / "training_free_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
