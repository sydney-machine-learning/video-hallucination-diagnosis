#!/usr/bin/env python3
"""Independent integrity checks for the training-free reliability baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_TRACE_GAINS = {
    "HERBench": 16.64463892977936,
    "Perception Test": 18.8680277762828,
}


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
    manifest = json.loads(
        (result / "training_free_analysis_manifest.json").read_text(encoding="utf-8")
    )
    if not manifest["retrospective_secondary_analysis"]:
        raise RuntimeError("analysis is not marked retrospective")
    if manifest["changes_sealed_primary_results"]:
        raise RuntimeError("analysis claims to change a fixed primary result")
    if int(manifest["bootstrap_iterations"]) != 10000:
        raise RuntimeError("unexpected bootstrap iteration count")
    for relative, expected in manifest["input_sha256"].items():
        if digest(workspace / relative) != expected:
            raise RuntimeError(f"input digest mismatch: {relative}")
    for name, expected in manifest["output_sha256"].items():
        if digest(result / name) != expected:
            raise RuntimeError(f"output digest mismatch: {name}")

    metrics = pd.read_csv(result / "training_free_model_metrics.csv")
    summary = pd.read_csv(result / "training_free_paired_bootstrap_summary.csv")
    distributions = pd.read_csv(
        result / "training_free_paired_bootstrap_distributions.csv"
    )
    if len(metrics) != 42 or len(summary) != 14 or len(distributions) != 140000:
        raise RuntimeError("unexpected training-free result inventory")
    if metrics.duplicated(["dataset", "baseline", "model_id"]).any():
        raise RuntimeError("duplicate model-level baseline row")
    if summary.duplicated(["dataset", "baseline"]).any():
        raise RuntimeError("duplicate baseline summary row")

    macro = (
        metrics.groupby(["dataset", "baseline"], as_index=False)[
            "relative_aurc_reduction_percent"
        ]
        .mean()
        .rename(columns={"relative_aurc_reduction_percent": "recalculated_gain"})
    )
    checked = summary.merge(
        macro, on=["dataset", "baseline"], how="inner", validate="one_to_one"
    )
    if len(checked) != len(summary):
        raise RuntimeError("summary and model metrics do not align")
    for row in checked.itertuples(index=False):
        if not close(row.baseline_gain_percent, row.recalculated_gain):
            raise RuntimeError("macro baseline gain does not match model metrics")
        expected_delta = row.trace_rc_gain_percent - row.baseline_gain_percent
        if not close(expected_delta, row.trace_minus_baseline_gain_percent_points):
            raise RuntimeError("reported target-checked difference is inconsistent")
        values = distributions.loc[
            distributions["dataset"].eq(row.dataset)
            & distributions["baseline"].eq(row.baseline),
            "trace_minus_baseline_gain_percent_points",
        ].to_numpy(float)
        if len(values) != 10000:
            raise RuntimeError("baseline has the wrong number of bootstrap draws")
        lower, upper = np.quantile(values, [0.025, 0.975])
        if not close(lower, row.paired_video_bootstrap_ci95_low) or not close(
            upper, row.paired_video_bootstrap_ci95_high
        ):
            raise RuntimeError("bootstrap interval does not match its distribution")

    for target, expected in EXPECTED_TRACE_GAINS.items():
        observed = summary.loc[summary["dataset"].eq(target), "trace_rc_gain_percent"]
        if len(observed) != 7 or not np.allclose(
            observed.to_numpy(float), expected, atol=1e-9, rtol=0.0
        ):
            raise RuntimeError(f"{target}: archived target-checked gain changed")

    print(
        json.dumps(
            {
                "status": "PASS",
                "inputs_verified": len(manifest["input_sha256"]),
                "outputs_verified": len(manifest["output_sha256"]),
                "model_metric_rows": len(metrics),
                "bootstrap_rows": len(distributions),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
