#!/usr/bin/env python3
"""Render paper assets from an audited fixed-protocol result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def latex(value: object) -> str:
    text = str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def short_model(model_id: str) -> str:
    if "InternVL" in model_id:
        return "InternVL3.5-14B"
    if "Qwen" in model_id:
        return "Qwen3-VL-8B"
    if "LLaVA" in model_id:
        return "LLaVA-OV2-8B"
    return model_id.rsplit("/", 1)[-1]


def display_method(method: str) -> str:
    return {
        # Keys are historical values stored in the archived result files.
        "TRACE-RC": "Target-checked score",
        "target_full_graph_logistic": "Target pilot: all views",
        "source_trace_logistic": "Development datasets: linear",
        "source_trace_hgb": "Development datasets: tree",
        "target_base_graph_logistic": "Target pilot: base view only",
        "target_scalar_logistic": "Target pilot: confidence only",
        "raw_confidence": "Original score",
        "target_logistic": "Target pilot: all views",
    }.get(method, method.replace("_", " "))


def write(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("herbench", "perception"), default="herbench")
    args = parser.parse_args()
    source = args.analysis_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = "trace_rc_herbench" if args.target == "herbench" else "trace_rc_perception"
    target_name = "HERBench" if args.target == "herbench" else "Perception Test"

    gate = json.loads((source / "confirmation_gate.json").read_text(encoding="utf-8"))
    lock = json.loads(
        (source / "promotion_decision_lock.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(source / "test_model_metrics.csv")
    tasks = pd.read_csv(source / "test_task_metrics.csv")
    required = {
        "method",
        "model_id",
        "accuracy",
        "aurc",
        "raw_aurc",
        "relative_aurc_reduction_percent",
        "auroc",
        "ece15",
        "brier",
        "risk_at_50pct",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"test_model_metrics.csv missing {missing}")

    trace = metrics.loc[metrics["method"].eq("TRACE-RC")].copy()
    if len(trace) != gate["models"]:
        raise ValueError("TRACE-RC model count differs from confirmation gate")
    trace = trace.sort_values("model_id")
    mean_gain = float(gate["macro_mean_relative_aurc_reduction_percent"])
    lower, upper = map(float, gate["video_grouped_bootstrap_95ci_percent"])
    worst = float(gate["worst_model_gain_percent"])
    passed = bool(gate["primary_gate_pass"])

    main_rows = []
    for row in trace.itertuples(index=False):
        decision = lock["decisions"][row.model_id]["selected"]
        main_rows.append(
            f"{latex(short_model(row.model_id))} & {row.accuracy:.3f} & "
            f"{row.raw_aurc:.4f} & {row.aurc:.4f} & "
            f"{row.relative_aurc_reduction_percent:.2f}\\% & "
            f"{latex(display_method(decision))} \\\\"
        )
    main_rows.append(
        f"Macro mean & {trace['accuracy'].mean():.3f} & "
        f"{trace['raw_aurc'].mean():.4f} & {trace['aurc'].mean():.4f} & "
        f"{mean_gain:.2f}\\% & -- \\\\"
    )
    write(
        output / f"{prefix}_main_table.tex",
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}Yrrrrl@{\TraceTableEdge}}",
                r"\toprule",
                r"Model & \multicolumn{1}{c}{Acc.} & \multicolumn{1}{c}{Raw} & "
                r"\multicolumn{1}{c}{RC} & \multicolumn{1}{c}{Gain} & Use \\",
                r"\midrule",
                *main_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        ),
    )

    promotion_rows = []
    for model_id, decision in sorted(lock["decisions"].items()):
        promotion_rows.append(
            f"{latex(short_model(model_id))} & "
            f"{latex(display_method(decision['selected']))} & "
            f"{decision['pilot_videos']} & {decision['pilot_questions']} & "
            f"{decision['pilot_gain_percent']:.2f}\\% & "
            f"{decision['multiplicity_adjusted_lower_bound_percent']:.2f}\\% \\\\"
        )
    write(
        output / f"{prefix}_promotion_table.tex",
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}Ylrrrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Model & Use & \multicolumn{1}{c}{Videos} & \multicolumn{1}{c}{Qs.} & "
                r"\multicolumn{1}{c}{Gain} & \multicolumn{1}{c}{Lower} \\",
                r"\midrule",
                *promotion_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        ),
    )

    baseline = (
        metrics.groupby("method", as_index=False)
        .agg(
            mean_gain=("relative_aurc_reduction_percent", "mean"),
            worst_gain=("relative_aurc_reduction_percent", "min"),
            mean_auroc=("auroc", "mean"),
            mean_ece=("ece15", "mean"),
            mean_brier=("brier", "mean"),
        )
        .sort_values(["mean_gain", "method"], ascending=[False, True])
    )
    baseline_rows = [
        f"{latex(display_method(row.method))} & {row.mean_gain:.2f}\\% & "
        f"{row.worst_gain:.2f}\\% & "
        f"{row.mean_auroc:.3f} & {row.mean_ece:.3f} & {row.mean_brier:.3f} \\\\"
        for row in baseline.itertuples(index=False)
    ]
    write(
        output / f"{prefix}_baseline_table.tex",
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}Yrrrrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Method & \multicolumn{1}{c}{Mean} & \multicolumn{1}{c}{Min.} & "
                r"\multicolumn{1}{c}{AUROC} & \multicolumn{1}{c}{ECE} & "
                r"\multicolumn{1}{c}{Brier} \\",
                r"\midrule",
                *baseline_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        ),
    )

    task_macro = (
        tasks.groupby("task", as_index=False)
        .agg(
            questions=("n", "sum"),
            accuracy=("accuracy", "mean"),
            mean_gain=("relative_aurc_reduction_percent", "mean"),
            worst_gain=("relative_aurc_reduction_percent", "min"),
        )
        .sort_values("task")
    )
    task_rows = [
        f"{latex(row.task)} & {int(row.questions)} & {row.accuracy:.3f} & "
        f"{row.mean_gain:.2f}\\% & {row.worst_gain:.2f}\\% \\\\"
        for row in task_macro.itertuples(index=False)
    ]
    write(
        output / f"{prefix}_task_table.tex",
        "\n".join(
            [
                r"\begin{tabularx}{\columnwidth}{@{\TraceTableEdge}Yrrrr@{\TraceTableEdge}}",
                r"\toprule",
                r"Task & \multicolumn{1}{c}{Scored rows} & \multicolumn{1}{c}{Accuracy} & "
                r"\multicolumn{1}{c}{Mean gain} & \multicolumn{1}{c}{Worst} \\",
                r"\midrule",
                *task_rows,
                r"\bottomrule",
                r"\end{tabularx}",
            ]
        ),
    )

    if passed:
        if args.target == "perception":
            sentence = (
                f"On the held-out {target_name} split, the target-checked method meets its "
                f"success criteria fixed before inference. Average AURC across the three models falls by {mean_gain:.2f}\\% "
                f"(video-grouped 95\\% CI, {lower:.2f}--{upper:.2f}\\%); "
                f"exact fallback leaves the third model unchanged ({worst:.2f}\\%)."
            )
            abstract = (
                f"On a separately held-out {gate['test_questions']}-question "
                f"Perception Test split, the target-checked method reduces average AURC across the three models by "
                f"{mean_gain:.2f}\\% (95\\% CI, {lower:.2f}--{upper:.2f}\\%); "
                "exact fallback leaves the non-promoted model unchanged."
            )
        else:
            sentence = (
                f"On the held-out {target_name} split, the target-checked method meets its "
                f"success criteria fixed before inference. Average AURC across the three models falls by {mean_gain:.2f}\\% "
                f"(video-grouped 95\\% CI, {lower:.2f}--{upper:.2f}\\%), and "
                f"the smallest model-level gain is {worst:.2f}\\%."
            )
            abstract = (
                f"On a fresh {gate['test_questions']}-question {target_name} split, "
                f"the target-checked method reduces average AURC across the three models by {mean_gain:.2f}\\% "
                f"(95\\% CI, {lower:.2f}--{upper:.2f}\\%); the smallest "
                f"model-level gain is {worst:.2f}\\%."
            )
        conclusion = (
            f"On this {target_name} split, the target-checked promotion rule avoids the "
            "negative-transfer pattern that motivated the revision."
        )
    else:
        failed = ", ".join(name for name, ok in gate["checks"].items() if not ok)
        sentence = (
            f"The target-checked method does not pass its {target_name} gate fixed before inference: "
            f"macro relative AURC reduction {mean_gain:.2f}\\% "
            f"(video-grouped 95\\% CI [{lower:.2f}, {upper:.2f}]\\%), "
            f"worst-model gain {worst:.2f}\\%; failed checks: {latex(failed)}."
        )
        abstract = (
            f"The {target_name} evaluation did not meet the complete fixed-protocol "
            "gate, limiting the evidence for transfer beyond the earlier datasets."
        )
        conclusion = (
            f"{target_name} shows that the guarded promotion rule does not yet "
            "resolve every form of cross-benchmark reliability shift."
        )
    write(output / f"{prefix}_sentence.tex", sentence)
    write(output / f"{prefix}_abstract_sentence.tex", abstract)
    write(output / f"{prefix}_conclusion_sentence.tex", conclusion)

    manifest = {}
    for path in sorted(output.glob("*.tex")):
        manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(output / f"{prefix}_assets.sha256.json", json.dumps(manifest, indent=2))
    print(json.dumps({"gate_pass": passed, "output_dir": str(output), **gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
