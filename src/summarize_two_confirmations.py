"""Validate and render a non-pooled summary of two held-out evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def latex(value: object) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def verify_archive_manifest(path: Path) -> set[str]:
    seen: set[str] = set()
    for number, line in enumerate((path / "archive.sha256").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        expected, recorded = line.split(maxsplit=1)
        name = Path(recorded.strip()).name
        local = path / name
        if not local.is_file():
            raise FileNotFoundError(f"archive entry missing locally: {name}")
        if digest(local) != expected:
            raise RuntimeError(f"archive digest mismatch at line {number}: {name}")
        seen.add(name)
    return seen


def load_archive(path: Path, display_name: str) -> dict:
    gate_path = path / "confirmation_gate.json"
    metrics_path = path / "test_model_metrics.csv"
    predictions_path = path / "test_scored_predictions.csv"
    for required in (gate_path, metrics_path, predictions_path, path / "archive.sha256"):
        if not required.is_file():
            raise FileNotFoundError(required)
    archived = verify_archive_manifest(path)
    if not {gate_path.name, metrics_path.name, predictions_path.name} <= archived:
        raise RuntimeError(f"{display_name}: archive manifest omits a required result")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    trace = metrics.loc[metrics["method"].eq("TRACE-RC")].copy()
    if len(trace) != 3 or trace["model_id"].nunique() != 3:
        raise RuntimeError(f"{display_name}: expected three target-checked model rows")
    if predictions["model_id"].nunique() != 3:
        raise RuntimeError(f"{display_name}: expected three prediction environments")
    if predictions["qid"].nunique() != int(gate["test_questions"]):
        raise RuntimeError(f"{display_name}: question count differs from gate")
    if predictions["group_id"].nunique() != int(gate["test_videos"]):
        raise RuntimeError(f"{display_name}: video count differs from gate")
    calculated_mean = float(trace["relative_aurc_reduction_percent"].mean())
    calculated_worst = float(trace["relative_aurc_reduction_percent"].min())
    if abs(calculated_mean - float(gate["macro_mean_relative_aurc_reduction_percent"])) > 1e-9:
        raise RuntimeError(f"{display_name}: macro gain differs from model metrics")
    if abs(calculated_worst - float(gate["worst_model_gain_percent"])) > 1e-9:
        raise RuntimeError(f"{display_name}: worst gain differs from model metrics")
    required_checks = {
        "macro_gain_at_least_one_percent",
        "grouped_ci_lower_above_zero",
        "worst_model_gain_at_least_minus_one_percent",
        "at_least_one_environment_promoted",
        "all_rejections_exact_fallback",
    }
    if set(gate["checks"]) != required_checks:
        raise RuntimeError(f"{display_name}: unexpected confirmation checks")
    return {
        "name": display_name,
        "target_dataset": gate["target_dataset"],
        "questions": int(gate["test_questions"]),
        "videos": int(gate["test_videos"]),
        "models": int(gate["models"]),
        "promoted": int(gate["promoted_environments"]),
        "macro_gain_percent": calculated_mean,
        "ci95_percent": [float(value) for value in gate["video_grouped_bootstrap_95ci_percent"]],
        "worst_gain_percent": calculated_worst,
        "gate_pass": bool(gate["primary_gate_pass"]),
        "gate_sha256": digest(gate_path),
        "model_metrics_sha256": digest(metrics_path),
        "prediction_sha256": digest(predictions_path),
    }


def write(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herbench-dir", type=Path, required=True)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archives = [
        load_archive(args.herbench_dir.resolve(), "HERBench Lite-v2"),
        load_archive(args.perception_dir.resolve(), "Perception Test valid500"),
    ]
    if archives[0]["target_dataset"] == archives[1]["target_dataset"]:
        raise RuntimeError("the two confirmation archives identify the same target dataset")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    passes = sum(item["gate_pass"] for item in archives)
    payload = {
        "schema_version": "trace-rc-two-confirmation-summary-1.0",
        "aggregation_policy": "report benchmark-specific effects; do not pool AURC gains",
        "confirmations": archives,
        "passed_confirmations": passes,
        "all_confirmation_gates_pass": passes == len(archives),
        "total_final_test_questions_descriptive": sum(item["questions"] for item in archives),
        "total_final_test_videos_descriptive": sum(item["videos"] for item in archives),
        "total_model_benchmark_environments": sum(item["models"] for item in archives),
        "minimum_model_gain_across_confirmations_percent": min(
            item["worst_gain_percent"] for item in archives
        ),
    }
    summary_path = output / "two_confirmation_summary.json"
    write(summary_path, json.dumps(payload, indent=2, sort_keys=True))

    rows = []
    for item in archives:
        lower, upper = item["ci95_percent"]
        rows.append(
            f"{latex(item['name'].replace(' Lite-v2', '').replace(' valid500', ''))} & "
            f"{item['videos']}/{item['questions']} & {item['macro_gain_percent']:.2f}\\% & "
            f"[{lower:.2f}, {upper:.2f}]\\% & {item['worst_gain_percent']:.2f}\\% \\\\"
        )
    table = "\n".join(
        [
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Target & Test V/Q & Gain & Video-grouped 95\% CI & Worst \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    write(output / "trace_rc_two_confirmation_table.tex", table)
    if payload["all_confirmation_gates_pass"]:
        sentence = (
            "Both held-out targets meet the target-check success criteria.  HERBench "
            "uses a learned score for all three models; Perception Test uses one for two "
            "models and retains the original score for the third.  Effects are reported separately "
            "because the benchmarks have different video and question distributions."
        )
    else:
        sentence = (
            f"Only {passes} of the two held-out public targets passes its complete "
            "target-check criteria; the evidence therefore does not support a two-target "
            "confirmation claim."
        )
    write(output / "trace_rc_two_confirmation_sentence.tex", sentence)
    manifest = {
        path.name: digest(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "two_confirmation_assets.sha256.json"
    }
    write(output / "two_confirmation_assets.sha256.json", json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
