#!/usr/bin/env python3
"""Render manuscript figures from the audited reliability-refinement results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


MODEL_ORDER = ["InternVL", "Qwen", "LLaVA"]
METHOD_ORDER = [
    "TRACE-RC",
    "target_full_graph_logistic",
    "target_base_graph_logistic",
    "target_scalar_logistic",
    "source_trace_hgb",
    "source_trace_logistic",
    "raw_confidence",
]
METHOD_LABELS = {
    # Keep the dictionary keys unchanged because they are stored in the audited
    # CSV files.  Only the reader-facing labels use the current terminology.
    "TRACE-RC": "Target-checked score",
    "target_full_graph_logistic": "Target pilot: all views",
    "target_base_graph_logistic": "Target pilot: base view only",
    "target_scalar_logistic": "Target pilot: confidence only",
    "source_trace_hgb": "Development datasets: tree",
    "source_trace_logistic": "Development datasets: linear",
    "raw_confidence": "Original score",
}


def model_family(model_id: str) -> str:
    if "InternVL" in model_id:
        return "InternVL"
    if "Qwen" in model_id:
        return "Qwen"
    if "LLaVA" in model_id:
        return "LLaVA"
    raise ValueError(f"Unknown model family: {model_id}")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_confirmation(analysis_dir: Path) -> tuple[pd.DataFrame, dict]:
    metrics = pd.read_csv(analysis_dir / "test_model_metrics.csv")
    gate = json.loads((analysis_dir / "confirmation_gate.json").read_text(encoding="utf-8"))
    return metrics, gate


def render_confirmation_overview(
    herbench_dir: Path, perception_dir: Path, output_dir: Path
) -> None:
    her_metrics, her_gate = load_confirmation(herbench_dir)
    per_metrics, per_gate = load_confirmation(perception_dir)

    fig = plt.figure(figsize=(6.65, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.08, 1.35])
    ax_gain = fig.add_subplot(grid[0, :])
    ax_key = fig.add_subplot(grid[1, :])
    ax_her = fig.add_subplot(grid[2, 0])
    ax_per = fig.add_subplot(grid[2, 1])
    ax_key.axis("off")

    target_specs = [
        ("HERBench", her_metrics, her_gate, "#3B6FB6", "//"),
        ("Perception Test", per_metrics, per_gate, "#D67A2B", "\\\\"),
    ]
    x = np.arange(4, dtype=float)
    width = 0.28
    for idx, (target, metrics, gate, color, hatch) in enumerate(target_specs):
        trace = metrics.loc[metrics["method"].eq("TRACE-RC")].copy()
        trace["family"] = trace["model_id"].map(model_family)
        family_gain = trace.set_index("family")["relative_aurc_reduction_percent"]
        values = [float(family_gain[name]) for name in MODEL_ORDER]
        values.append(float(gate["macro_mean_relative_aurc_reduction_percent"]))
        positions = x + (idx - 0.5) * width
        bars = ax_gain.bar(
            positions,
            values,
            width,
            label=target,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            zorder=3,
        )
        ci_low, ci_high = gate["video_grouped_bootstrap_95ci_percent"]
        macro = values[-1]
        ax_gain.errorbar(
            positions[-1],
            macro,
            yerr=[[macro - ci_low], [ci_high - macro]],
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            zorder=5,
        )
        label_heights = list(values)
        # Keep the two model-average labels clear of their confidence intervals.
        label_heights[-1] = float(ci_high)
        for bar, value, label_height in zip(bars, values, label_heights):
            ax_gain.text(
                bar.get_x() + bar.get_width() / 2,
                max(label_height, 0) + 0.65,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax_gain.set_xticks(x, ["InternVL3.5", "Qwen3-VL", "LLaVA-OV2", "Model average"])
    ax_gain.set_ylabel("Relative AURC reduction (%)")
    ax_gain.set_title("(a) Target-checked score by model", loc="left")
    ax_gain.set_ylim(0, 33)
    ax_gain.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)

    def baseline_panel(ax: plt.Axes, title: str, metrics: pd.DataFrame, color: str) -> None:
        grouped = (
            metrics.groupby("method", as_index=True)["relative_aurc_reduction_percent"]
            .agg(["mean", "min"])
        )
        methods = [method for method in METHOD_ORDER if method in grouped.index]
        y = np.arange(len(methods))
        mean = grouped.loc[methods, "mean"].to_numpy(float)
        worst = grouped.loc[methods, "min"].to_numpy(float)
        for yi, lo, hi in zip(y, worst, mean):
            ax.plot([lo, hi], [yi, yi], color="0.65", linewidth=1.1, zorder=1)
        ax.scatter(mean, y, s=28, marker="o", color=color, edgecolor="black", linewidth=0.5,
                   label="Average across models", zorder=3)
        ax.scatter(worst, y, s=28, marker="D", facecolor="white", edgecolor="black",
                   linewidth=0.8, label="Smallest model gain", zorder=3)
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.set_yticks(y, [METHOD_LABELS[m] for m in methods])
        ax.invert_yaxis()
        ax.set_xlabel("Relative AURC reduction (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", color="0.88", linewidth=0.6, zorder=0)
        ax.set_xlim(-7, 23)

    baseline_panel(ax_her, "(b) HERBench baselines", her_metrics, "#3B6FB6")
    baseline_panel(ax_per, "(c) Perception Test baselines", per_metrics, "#D67A2B")
    target_handles, target_labels = ax_gain.get_legend_handles_labels()
    handles = target_handles + [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.55",
               markeredgecolor="black", markersize=6, label="Average across models"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white",
               markeredgecolor="black", markersize=6, label="Smallest model gain"),
    ]
    labels = target_labels + [handle.get_label() for handle in handles[2:]]
    ax_key.legend(handles, labels, frameon=False, ncols=4, loc="center")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"trace_rc_confirmation_overview.{suffix}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_vhd_consensus(vhd_csv: Path, output_dir: Path) -> None:
    frame = pd.read_csv(vhd_csv)
    required = {
        "task",
        "unanimous_fraction",
        "false_consensus_fraction",
        "wrong_among_unanimous_fraction",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{vhd_csv} is not a VHD model-consensus table; missing columns: {missing}"
        )
    task_order = ["accuracy", "vhd_i_pdi", "vhd_ii_hce"]
    task_labels = ["Clearly visible\nevidence", "Video contradicts\nexpectation", "Key evidence\nhidden"]
    frame = frame.set_index("task").loc[task_order]

    measures = [
        ("unanimous_fraction", "All models agree", "#6C7A89", ""),
        ("false_consensus_fraction", "All agree and are wrong", "#D67A2B", "//"),
        ("wrong_among_unanimous_fraction", "Wrong when all agree", "#3B6FB6", "\\\\"),
    ]
    x = np.arange(len(frame), dtype=float)
    width = 0.19
    fig, ax = plt.subplots(figsize=(6.55, 2.55), constrained_layout=True)
    for idx, (column, label, color, hatch) in enumerate(measures):
        values = frame[column].to_numpy(float) * 100
        positions = x + (idx - 1) * width
        errors = None
        if column == "wrong_among_unanimous_fraction":
            low = frame["wrong_among_unanimous_ci_low"].to_numpy(float) * 100
            high = frame["wrong_among_unanimous_ci_high"].to_numpy(float) * 100
            errors = np.vstack([values - low, high - values])
        bars = ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3 if errors is not None else 0,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            hatch=hatch,
            zorder=3,
        )
        upper = (
            frame["wrong_among_unanimous_ci_high"].to_numpy(float) * 100
            if column == "wrong_among_unanimous_fraction"
            else values
        )
        for bar, value, label_height in zip(bars, values, upper):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_height + 1.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x, task_labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="0.88", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"trace_rc_vhd_consensus.{suffix}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def render_posthoc_pilot_efficiency(posthoc_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(posthoc_dir / "pilot_label_efficiency_summary.csv")
    targets = [("HERBench", "#3B6FB6"), ("Perception Test", "#D67A2B")]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    for ax, (target, color) in zip(axes, targets):
        data = frame.loc[frame["dataset"].eq(target)].sort_values(
            "pilot_fraction_of_target"
        )
        x = 100 * data["pilot_fraction_of_target"].to_numpy(float)
        mean = data["mean_test_gain_percent"].to_numpy(float)
        tenth = data["tenth_percentile_test_gain"].to_numpy(float)
        promote = 100 * data["promotion_rate"].to_numpy(float)
        harmful = data["harmful_promotion_rate"].to_numpy(float) > 0
        ax.plot(x, mean, color=color, marker="o", linewidth=1.6, label="Mean test gain")
        ax.plot(
            x,
            tenth,
            color=color,
            marker="s",
            linestyle=":",
            linewidth=1.1,
            label="10th-percentile gain",
        )
        if harmful.any():
            ax.scatter(
                x[harmful],
                mean[harmful],
                marker="x",
                s=55,
                linewidth=1.5,
                color="#A51C30",
                zorder=5,
                label="At least one harmful use",
            )
        ax_rate = ax.twinx()
        ax_rate.plot(
            x,
            promote,
            color="0.35",
            marker="^",
            linestyle="--",
            linewidth=1.1,
            label="Learned score used",
        )
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xlabel("Labeled target videos (%)")
        ax.set_ylabel("Relative AURC reduction (%)")
        ax_rate.set_ylabel("Learned score used (%)")
        ax_rate.set_ylim(0, 105)
        ax.set_title(target, loc="left")
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            rate_handles, rate_labels = ax_rate.get_legend_handles_labels()
            shared_handles = handles + rate_handles
            shared_labels = labels + rate_labels
    fig.legend(
        shared_handles,
        shared_labels,
        frameon=False,
        fontsize=8,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"trace_rc_posthoc_pilot_efficiency.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def render_posthoc_compute(posthoc_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(posthoc_dir / "compute_budget_ablation.csv")
    frame = frame.loc[frame["audit_family"].eq("view_panel")]
    aggregate = (
        frame.groupby(
            [
                "dataset",
                "configuration",
                "model_calls_per_question",
                "model_input_frames_per_question",
            ],
            as_index=False,
        )
        .agg(
            mean_gain=("relative_aurc_reduction_percent", "mean"),
            worst_gain=("relative_aurc_reduction_percent", "min"),
        )
    )
    targets = [("HERBench", "#3B6FB6"), ("Perception Test", "#D67A2B")]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    for ax, (target, color) in zip(axes, targets):
        data = aggregate.loc[aggregate["dataset"].eq(target)].copy()
        highlights = {
            "Base only": "Base",
            "Base + addition": "Base+dense",
            "All temporal relations": "All views",
        }
        other = data.loc[~data["configuration"].isin(highlights)]
        ax.scatter(
            other["model_calls_per_question"],
            other["mean_gain"],
            s=28,
            facecolor="white",
            edgecolor="0.45",
            linewidth=0.8,
            label="Other view subsets",
            zorder=2,
        )
        key = data.loc[data["configuration"].isin(highlights)].sort_values(
            "model_calls_per_question"
        )
        ax.plot(
            key["model_calls_per_question"],
            key["mean_gain"],
            color=color,
            marker="o",
            linewidth=1.5,
            label="Base / base+dense / all views",
            zorder=3,
        )
        for row in key.itertuples(index=False):
            is_full = row.configuration == "All temporal relations"
            ax.annotate(
                highlights[row.configuration],
                (row.model_calls_per_question, row.mean_gain),
                xytext=(-3 if is_full else 3, 5),
                textcoords="offset points",
            fontsize=8,
                ha="right" if is_full else "left",
            )
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_xlim(2.55, 12.45)
        ax.set_ylim(-0.8, float(data["mean_gain"].max()) + 2.5)
        ax.set_xticks([3, 6, 9, 12])
        ax.set_xlabel("Model calls per question")
        ax.set_ylabel("Average relative AURC reduction (%)")
        ax.set_title(target, loc="left")
        ax.grid(color="0.88", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"trace_rc_posthoc_compute_ablation.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def render_posthoc_efficiency_audits(posthoc_dir: Path, output_dir: Path) -> None:
    """Render the four efficiency panels on one aligned 2x2 canvas."""
    pilot_frame = pd.read_csv(posthoc_dir / "pilot_label_efficiency_summary.csv")
    compute_frame = pd.read_csv(posthoc_dir / "compute_budget_ablation.csv")
    compute_frame = compute_frame.loc[compute_frame["audit_family"].eq("view_panel")]
    aggregate = (
        compute_frame.groupby(
            [
                "dataset",
                "configuration",
                "model_calls_per_question",
                "model_input_frames_per_question",
            ],
            as_index=False,
        )
        .agg(
            mean_gain=("relative_aurc_reduction_percent", "mean"),
            worst_gain=("relative_aurc_reduction_percent", "min"),
        )
    )

    targets = [("HERBench", "#3B6FB6"), ("Perception Test", "#D67A2B")]
    fig = plt.figure(figsize=(7.15, 5.15))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.94,
        bottom=0.10,
        top=0.90,
        wspace=0.42,
        hspace=0.46,
    )
    top_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    bottom_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]

    shared_handles: list[object] = []
    shared_labels: list[str] = []
    for index, (ax, (target, color)) in enumerate(zip(top_axes, targets)):
        data = pilot_frame.loc[pilot_frame["dataset"].eq(target)].sort_values(
            "pilot_fraction_of_target"
        )
        x = 100 * data["pilot_fraction_of_target"].to_numpy(float)
        mean = data["mean_test_gain_percent"].to_numpy(float)
        tenth = data["tenth_percentile_test_gain"].to_numpy(float)
        promote = 100 * data["promotion_rate"].to_numpy(float)
        harmful = data["harmful_promotion_rate"].to_numpy(float) > 0

        ax.plot(x, mean, color=color, marker="o", linewidth=1.6, label="Mean test gain")
        ax.plot(
            x,
            tenth,
            color=color,
            marker="s",
            linestyle=":",
            linewidth=1.1,
            label="10th-percentile gain",
        )
        if harmful.any():
            ax.scatter(
                x[harmful],
                mean[harmful],
                marker="x",
                s=55,
                linewidth=1.5,
                color="#A51C30",
                zorder=5,
                label="At least one harmful use",
            )
        ax_rate = ax.twinx()
        ax_rate.plot(
            x,
            promote,
            color="0.35",
            marker="^",
            linestyle="--",
            linewidth=1.1,
            label="Learned score used",
        )
        ax_rate.set_ylim(0, 105)
        if index == 0:
            ax_rate.tick_params(right=False, labelright=False)
        else:
            ax_rate.set_ylabel("Learned score used (%)")

        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xlabel("Labeled target videos (%)")
        if index == 0:
            ax.set_ylabel("Relative AURC reduction (%)")
        ax.set_title(target, loc="left")
        ax.grid(axis="y", color="0.88", linewidth=0.6)

        if index == 0:
            handles, labels = ax.get_legend_handles_labels()
            rate_handles, rate_labels = ax_rate.get_legend_handles_labels()
            shared_handles = handles + rate_handles
            shared_labels = labels + rate_labels

    fig.legend(
        shared_handles,
        shared_labels,
        frameon=False,
        fontsize=8,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
    )

    highlights = {
        "Base only": "Base",
        "Base + addition": "Base+dense",
        "All temporal relations": "All views",
    }
    for index, (ax, (target, color)) in enumerate(zip(bottom_axes, targets)):
        data = aggregate.loc[aggregate["dataset"].eq(target)].copy()
        other = data.loc[~data["configuration"].isin(highlights)]
        ax.scatter(
            other["model_calls_per_question"],
            other["mean_gain"],
            s=28,
            facecolor="white",
            edgecolor="0.45",
            linewidth=0.8,
            label="Other view subsets",
            zorder=2,
        )
        key = data.loc[data["configuration"].isin(highlights)].sort_values(
            "model_calls_per_question"
        )
        ax.plot(
            key["model_calls_per_question"],
            key["mean_gain"],
            color=color,
            marker="o",
            linewidth=1.5,
            label="Base / base+dense / all views",
            zorder=3,
        )
        for row in key.itertuples(index=False):
            is_full = row.configuration == "All temporal relations"
            ax.annotate(
                highlights[row.configuration],
                (row.model_calls_per_question, row.mean_gain),
                xytext=(-3 if is_full else 3, 5),
                textcoords="offset points",
                fontsize=8,
                ha="right" if is_full else "left",
            )
        ax.axhline(0, color="0.25", linewidth=0.7)
        ax.set_xlim(2.55, 12.45)
        ax.set_ylim(-0.8, float(data["mean_gain"].max()) + 2.5)
        ax.set_xticks([3, 6, 9, 12])
        ax.set_xlabel("Model calls per question")
        if index == 0:
            ax.set_ylabel("Average relative AURC reduction (%)")
        ax.set_title(target, loc="left")
        ax.grid(color="0.88", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="lower right")

    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"trace_rc_posthoc_efficiency_audits.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def render_posthoc_risk_coverage(posthoc_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(posthoc_dir / "risk_coverage_curves.csv")
    targets = ["HERBench", "Perception Test"]
    method_specs = [
        ("Raw confidence", "Original score", "0.45", "--"),
        ("Target base graph", "Target pilot: base view only", "#6C7A89", ":"),
        ("TRACE-RC", "Target-checked score", "#3B6FB6", "-"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.22, top=0.88, wspace=0.34)
    for ax, target in zip(axes, targets):
        data = frame.loc[frame["dataset"].eq(target)]
        for method, display_label, color, style in method_specs:
            line = data.loc[data["method"].eq(method)].sort_values("coverage")
            ax.plot(
                100 * line["coverage"],
                100 * line["macro_risk"],
                color=color,
                linestyle=style,
                linewidth=1.5,
                label=display_label,
            )
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Average risk across models (%)")
        ax.set_title(target, loc="left")
        ax.grid(color="0.88", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"trace_rc_posthoc_risk_coverage.{suffix}",
            bbox_inches="tight",
            pad_inches=0.03,
        )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--herbench-dir", type=Path, required=True)
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument(
        "--vhd-csv",
        type=Path,
        help=(
            "optional VHD model-consensus table; omit this argument when the "
            "raw model-response archive is unavailable"
        ),
    )
    parser.add_argument("--posthoc-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_confirmation_overview(args.herbench_dir, args.perception_dir, args.output_dir)
    if args.vhd_csv is not None:
        render_vhd_consensus(args.vhd_csv, args.output_dir)
    if args.posthoc_dir is not None:
        render_posthoc_pilot_efficiency(args.posthoc_dir, args.output_dir)
        render_posthoc_compute(args.posthoc_dir, args.output_dir)
        render_posthoc_efficiency_audits(args.posthoc_dir, args.output_dir)
        render_posthoc_risk_coverage(args.posthoc_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
