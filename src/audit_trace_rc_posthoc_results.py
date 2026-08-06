#!/usr/bin/env python3
"""Independent consistency checks for TRACE-RC post-hoc result assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    result = args.result_dir.resolve()
    workspace = args.workspace.resolve()
    manifest = json.loads((result / "analysis_manifest.json").read_text(encoding="utf-8"))

    if not manifest["retrospective_secondary_analysis"]:
        raise RuntimeError("analysis is not marked retrospective")
    if manifest["changes_sealed_primary_results"]:
        raise RuntimeError("post-hoc manifest claims to change a sealed result")
    for relative, expected in manifest["input_sha256"].items():
        path = workspace / relative
        if digest(path) != expected:
            raise RuntimeError(f"input digest mismatch: {relative}")
    for name, expected in manifest["output_sha256"].items():
        if digest(result / name) != expected:
            raise RuntimeError(f"output digest mismatch: {name}")

    reconstruction = pd.read_csv(result / "reconstruction_checks.csv")
    if len(reconstruction) != 8 or reconstruction["dataset"].nunique() != 2:
        raise RuntimeError("unexpected reconstruction inventory")
    if float(reconstruction["maximum_absolute_error"].max()) >= 1e-8:
        raise RuntimeError("archived score reconstruction exceeds tolerance")

    detail = pd.read_csv(result / "pilot_label_efficiency_detail.csv")
    summary = pd.read_csv(result / "pilot_label_efficiency_summary.csv")
    if len(detail) != 486 or len(summary) != 10:
        raise RuntimeError("unexpected pilot-efficiency inventory")
    if detail.duplicated(
        ["dataset", "pilot_fraction_of_target", "salt", "model_id"]
    ).any():
        raise RuntimeError("duplicate pilot-efficiency task")
    if not bool(detail["exact_fallback"].all()):
        raise RuntimeError("a fallback score is not exactly raw confidence")
    full = detail.loc[detail["pilot_fraction_of_target"].eq(0.5)]
    if len(full) != 6 or not bool(full["matches_registered_decision"].all()):
        raise RuntimeError("registered full-pilot decisions were not reproduced")

    recalculated = (
        detail.groupby(["dataset", "pilot_fraction_of_target"], as_index=False)
        .agg(
            split_models=("test_gain_percent", "size"),
            mean_gain=("test_gain_percent", "mean"),
            promotion_rate=("selected", lambda values: float((values != "raw_fallback").mean())),
            harmful_rate=("harmful_promotion", "mean"),
        )
        .sort_values(["dataset", "pilot_fraction_of_target"])
        .reset_index(drop=True)
    )
    ordered_summary = summary.sort_values(
        ["dataset", "pilot_fraction_of_target"]
    ).reset_index(drop=True)
    for index in range(len(recalculated)):
        left, right = recalculated.iloc[index], ordered_summary.iloc[index]
        if int(left["split_models"]) != int(right["split_models"]):
            raise RuntimeError("pilot split-model count differs from detail")
        for source, target in (
            ("mean_gain", "mean_test_gain_percent"),
            ("promotion_rate", "promotion_rate"),
            ("harmful_rate", "harmful_promotion_rate"),
        ):
            if not close(float(left[source]), float(right[target])):
                raise RuntimeError(f"pilot summary mismatch: {target}")

    pairwise = pd.read_csv(result / "paired_bootstrap_summary.csv")
    distributions = pd.read_csv(result / "paired_bootstrap_distributions.csv")
    if len(pairwise) != 12 or len(distributions) != 120000:
        raise RuntimeError("unexpected paired-bootstrap inventory")
    for row in pairwise.itertuples(index=False):
        values = distributions.loc[
            distributions["dataset"].eq(row.dataset)
            & distributions["comparator"].eq(row.comparator),
            "delta_gain_percent_points",
        ].to_numpy(float)
        if len(values) != 10000:
            raise RuntimeError("paired-bootstrap comparator has wrong iterations")
        lower, upper = np.quantile(values, [0.025, 0.975])
        if not close(lower, row.paired_video_bootstrap_ci95_low):
            raise RuntimeError("paired-bootstrap lower endpoint mismatch")
        if not close(upper, row.paired_video_bootstrap_ci95_high):
            raise RuntimeError("paired-bootstrap upper endpoint mismatch")

    expected_primary = {"HERBench": 16.64463892977936, "Perception Test": 18.8680277762828}
    for target, expected in expected_primary.items():
        observed = float(
            pairwise.loc[
                pairwise["dataset"].eq(target)
                & pairwise["comparator"].eq("raw_confidence"),
                "trace_rc_gain_percent",
            ].iloc[0]
        )
        if not close(observed, expected, tolerance=1e-9):
            raise RuntimeError(f"{target}: primary gain changed")

    budget = pd.read_csv(result / "compute_budget_ablation.csv")
    key = budget.loc[
        budget["audit_family"].eq("model_panel")
        & budget["configuration"].isin(
            ["Own scalar", "Three-family base", "Three-family four views"]
        )
    ]
    if set(key["model_calls_per_question"].astype(int)) != {1, 3, 12}:
        raise RuntimeError("canonical cost anchors are missing")
    if (budget["model_input_frames_per_question"].astype(int) <= 0).any():
        raise RuntimeError("invalid model-input frame count")

    curves = pd.read_csv(result / "risk_coverage_curves.csv")
    fixed = pd.read_csv(result / "risk_at_fixed_coverage.csv")
    if len(curves) != 114 or len(fixed) != 24:
        raise RuntimeError("unexpected risk-coverage inventory")
    full_coverage = fixed.loc[fixed["coverage"].eq(1.0)]
    for _, target in full_coverage.groupby("dataset"):
        if float(target["macro_risk"].max() - target["macro_risk"].min()) > 1e-12:
            raise RuntimeError("full-coverage risk differs despite unchanged answers")

    payload = {
        "status": "PASS",
        "inputs_verified": len(manifest["input_sha256"]),
        "outputs_verified": len(manifest["output_sha256"]),
        "pilot_tasks": len(detail),
        "paired_bootstrap_rows": len(distributions),
        "maximum_reconstruction_error": float(
            reconstruction["maximum_absolute_error"].max()
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
