from __future__ import annotations

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
CATEGORY_ORDER = [
    "fire_smoke_or_burning",
    "liquid_pouring_or_flow",
    "falling_or_gravity",
    "collision_or_impact",
    "glass_breaking_or_fragments",
    "deformation_or_melting",
]
CATEGORY_LABELS = {
    "fire_smoke_or_burning": "Fire",
    "liquid_pouring_or_flow": "Liquid",
    "falling_or_gravity": "Falling",
    "collision_or_impact": "Impact",
    "glass_breaking_or_fragments": "Glass",
    "deformation_or_melting": "Deform",
}


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


def summarize(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for category in CATEGORY_ORDER:
            group = scored[
                (scored["model_display"] == model)
                & (scored["vhd_category"] == category)
                & (scored["is_valid_response"])
            ]

            accuracy_group = group[group["task"] == "accuracy"]
            pdi_group = group[group["task"] == "vhd_i_pdi"]
            hce_group = group[group["task"] == "vhd_ii_hce"]

            rows.extend(
                [
                    {
                        "model_display": model,
                        "vhd_category": category,
                        "category_label": CATEGORY_LABELS[category],
                        "metric": "accuracy",
                        "score": accuracy_group["is_correct"].mean(),
                        "n_clips": accuracy_group["clip_id"].nunique(),
                    },
                    {
                        "model_display": model,
                        "vhd_category": category,
                        "category_label": CATEGORY_LABELS[category],
                        "metric": "pdi",
                        "score": pdi_group["is_prior_choice"].mean(),
                        "n_clips": pdi_group["clip_id"].nunique(),
                    },
                    {
                        "model_display": model,
                        "vhd_category": category,
                        "category_label": CATEGORY_LABELS[category],
                        "metric": "hce",
                        "score": expected_calibration_error(hce_group),
                        "n_clips": hce_group["clip_id"].nunique(),
                    },
                ]
            )
    return pd.DataFrame(rows)


def matrix_for_metric(summary: pd.DataFrame, metric: str) -> np.ndarray:
    data = summary[summary["metric"] == metric]
    matrix = (
        data.pivot(index="model_display", columns="vhd_category", values="score")
        .reindex(index=MODEL_ORDER, columns=CATEGORY_ORDER)
        .to_numpy(dtype=float)
    )
    return matrix.T


def plot_heatmap(summary: pd.DataFrame, metric: str, file_name: str, cmap: str) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 104,
            "axes.labelsize": 104,
            "xtick.labelsize": 96,
            "ytick.labelsize": 96,
        }
    )

    matrix = matrix_for_metric(summary, metric)
    fig, axis = plt.subplots(figsize=(34.0, 34.0))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")

    axis.set_xticks(np.arange(len(MODEL_ORDER)))
    axis.set_xticklabels(MODEL_ORDER)
    plt.setp(axis.get_xticklabels(), rotation=28, ha="right", rotation_mode="anchor")
    axis.set_yticks(np.arange(len(CATEGORY_ORDER)))
    axis.set_yticklabels([CATEGORY_LABELS[c] for c in CATEGORY_ORDER])
    axis.tick_params(axis="both", width=5.0, length=16)

    for spine in axis.spines.values():
        spine.set_linewidth(5.0)

    axis.set_xticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=6.0)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            label = "-" if np.isnan(value) else f"{value:.2f}"
            text_color = "white" if not np.isnan(value) and value >= 0.55 else "#111111"
            axis.text(col, row, label, ha="center", va="center", color=text_color, fontsize=94)

    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.025)
    colorbar.ax.tick_params(labelsize=88, width=4.5, length=14)
    colorbar.outline.set_linewidth(4.5)

    fig.tight_layout(pad=2.2)
    path = FIGURE_DIR / file_name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    if not SCORED.exists():
        raise FileNotFoundError(f"Run make_main_results_figure.py first; missing {SCORED}")
    scored = pd.read_csv(SCORED)
    summary = summarize(scored)
    summary.to_csv(TABLE_DIR / "categorymetricsummary.csv", index=False)
    paths = [
        plot_heatmap(summary, "accuracy", "categoryaccuracy.png", "Blues"),
        plot_heatmap(summary, "pdi", "categorypdi.png", "OrRd"),
        plot_heatmap(summary, "hce", "categoryhce.png", "PuRd"),
    ]
    print(summary.to_string(index=False))
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
