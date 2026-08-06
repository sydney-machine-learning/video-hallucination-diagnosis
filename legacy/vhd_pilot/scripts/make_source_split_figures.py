from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
SCORED = TABLE_DIR / "vhd_item_level_scored.csv"

MODEL_ORDER = ["Qwen2.5-Omni-3B", "GPT-5.4", "GPT-5.5", "GPT-5.6"]
SPLIT_ORDER = ["real", "ai_generated"]
SPLIT_LABELS = ["Real", "AI-generated"]
COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
BOOTSTRAP_REPEATS = 1000


def expected_calibration_error(group: pd.DataFrame) -> float:
    group = group.dropna(subset=["confidence_norm", "is_correct"]).copy()
    if group.empty:
        return np.nan
    bins = np.linspace(0.0, 1.0, 6)
    group["bin"] = pd.cut(group["confidence_norm"], bins, include_lowest=True, labels=False)
    total = len(group)
    value = 0.0
    for _, bin_group in group.groupby("bin", observed=True):
        value += len(bin_group) / total * abs(
            bin_group["confidence_norm"].mean() - bin_group["is_correct"].mean()
        )
    return float(value)


def bootstrap_mean_ci(
    values: pd.Series,
    label: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float, float]:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260711)
    samples = np.empty(repeats, dtype=float)
    for i in range(repeats):
        samples[i] = rng.choice(values, size=len(values), replace=True).mean()
        if (i + 1) % 100 == 0 or i + 1 == repeats:
            print(f"{label}: bootstrap sample {i + 1}/{repeats}", flush=True)
    return float(values.mean()), *np.percentile(samples, [2.5, 97.5])


def bootstrap_hce_ci(
    group: pd.DataFrame,
    label: str,
    repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[float, float, float]:
    group = group.dropna(subset=["confidence_norm", "is_correct"]).copy()
    if group.empty:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260711)
    clip_index, clip_ids = pd.factorize(group["clip_id"], sort=False)
    n_clips = len(clip_ids)
    if n_clips == 0:
        return np.nan, np.nan, np.nan
    confidence = group["confidence_norm"].to_numpy(dtype=float)
    correct = group["is_correct"].to_numpy(dtype=float)
    bin_index = pd.cut(
        confidence,
        np.linspace(0.0, 1.0, 6),
        include_lowest=True,
        labels=False,
    )
    samples = np.empty(repeats, dtype=float)
    for i in range(repeats):
        sampled = rng.integers(0, n_clips, size=n_clips)
        clip_weights = np.bincount(sampled, minlength=n_clips)
        row_weights = clip_weights[clip_index].astype(float)
        total = row_weights.sum()
        value = 0.0
        for bin_id in range(5):
            mask = bin_index == bin_id
            weights = row_weights[mask]
            mass = weights.sum()
            if mass == 0:
                continue
            mean_confidence = np.average(confidence[mask], weights=weights)
            mean_correct = np.average(correct[mask], weights=weights)
            value += mass / total * abs(mean_confidence - mean_correct)
        samples[i] = value
        if (i + 1) % 100 == 0 or i + 1 == repeats:
            print(f"{label}: bootstrap sample {i + 1}/{repeats}", flush=True)
    return float(expected_calibration_error(group)), *np.percentile(samples, [2.5, 97.5])


def per_clip_split_scores(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, split, clip), group in scored.groupby(
        ["model_display", "video_split", "clip_id"]
    ):
        rows.append(
            {
                "model_display": model,
                "video_split": split,
                "clip_id": clip,
                "accuracy": group[group["task"] == "accuracy"]["is_correct"].mean(),
                "pdi": group[group["task"] == "vhd_i_pdi"]["is_prior_choice"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize(clip_scores: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        for model in MODEL_ORDER:
            clip_group = clip_scores[
                (clip_scores["video_split"] == split)
                & (clip_scores["model_display"] == model)
            ]
            for metric in ["accuracy", "pdi"]:
                mean, low, high = bootstrap_mean_ci(
                    clip_group[metric],
                    f"{model} {split} {metric}",
                )
                rows.append(
                    {
                        "metric": metric,
                        "video_split": split,
                        "model_display": model,
                        "mean": mean,
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            hce_items = scored[
                (scored["video_split"] == split)
                & (scored["model_display"] == model)
                & (scored["task"] == "vhd_ii_hce")
                & scored["is_valid_response"]
            ]
            mean, low, high = bootstrap_hce_ci(hce_items, f"{model} {split} hce")
            rows.append(
                {
                    "metric": "hce",
                    "video_split": split,
                    "model_display": model,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, file_name: str) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 72,
            "axes.labelsize": 72,
            "xtick.labelsize": 68,
            "ytick.labelsize": 68,
            "legend.fontsize": 64,
        }
    )
    data = summary[summary["metric"] == metric].copy()
    fig, axis = plt.subplots(figsize=(28.0, 15.5))

    x = np.arange(len(SPLIT_ORDER))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODEL_ORDER))

    for color, offset, model in zip(COLORS, offsets, MODEL_ORDER):
        model_data = data[data["model_display"] == model].set_index("video_split").loc[SPLIT_ORDER]
        values = model_data["mean"].to_numpy(dtype=float)
        low = model_data["ci_low"].to_numpy(dtype=float)
        high = model_data["ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([values - low, high - values])
        xpos = x + offset
        axis.bar(
            xpos,
            values,
            width=width,
            color=color,
            edgecolor="#222222",
            linewidth=3.2,
            label=model,
        )
        axis.errorbar(
            xpos,
            values,
            yerr=yerr,
            fmt="none",
            ecolor="#111111",
            elinewidth=6.0,
            capsize=18,
            capthick=6.0,
        )
        for xx, value, upper in zip(xpos, values, high):
            if value >= 0.18:
                label_y = value * 0.5
                label_color = "white"
                label_rotation = 90
                label_size = 58
            else:
                label_y = max(upper + 0.045, 0.055)
                label_color = "#111111"
                label_rotation = 0
                label_size = 42
            axis.text(
                xx,
                label_y,
                f"{value:.2f}",
                ha="center",
                va="center",
                rotation=label_rotation,
                fontsize=label_size,
                color=label_color,
                fontweight="bold",
            )

    axis.set_xticks(x)
    axis.set_xticklabels(SPLIT_LABELS)
    axis.set_ylim(0, 1.2)
    axis.set_ylabel("Score")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(4.8)
    axis.spines["bottom"].set_linewidth(4.8)
    axis.tick_params(axis="both", width=4.8, length=16)
    axis.grid(axis="y", color="#D0D0D0", linewidth=3.2)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.4,
    )
    fig.tight_layout()
    path = FIGURE_DIR / file_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    global BOOTSTRAP_REPEATS
    parser = argparse.ArgumentParser(description="Compute real versus AI-generated VHD results.")
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    args = parser.parse_args()
    BOOTSTRAP_REPEATS = args.bootstrap_repeats
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not SCORED.exists():
        parser.error(f"Run make_main_results_figure.py first; missing {SCORED}")

    scored = pd.read_csv(SCORED)
    clip_scores = per_clip_split_scores(scored)
    summary = summarize(clip_scores, scored)
    clip_scores.to_csv(TABLE_DIR / "vhd_source_split_clip_scores.csv", index=False)
    summary.to_csv(TABLE_DIR / "vhd_source_split_summary.csv", index=False)
    paths = [
        plot_metric(summary, "accuracy", "sourcesplitaccuracy.png"),
        plot_metric(summary, "pdi", "sourcesplitpdi.png"),
        plot_metric(summary, "hce", "sourcesplithce.png"),
    ]
    print(summary.to_string(index=False))
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
