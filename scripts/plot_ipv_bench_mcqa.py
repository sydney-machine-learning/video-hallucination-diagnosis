import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
SUMMARY_PATH = ROOT / "data" / "metric_summaries" / "ipv_bench_mcqa_summary.csv"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.labelsize": 23,
        "xtick.labelsize": 16,
        "ytick.labelsize": 19,
        "legend.fontsize": 15,
    }
)

def fig_ipv_bench_mcqa():
    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    models = [row["model_label"] for row in rows]
    scores = [float(row["mcqa_accuracy_pct"]) for row in rows]
    colours = [row["colour"] for row in rows]

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=200)
    x = range(len(models))

    ax.grid(axis="y", color="#D9D9D9", linewidth=1.0, alpha=0.65)
    ax.set_axisbelow(True)

    for index, (score, colour) in enumerate(zip(scores, colours)):
        ax.scatter(
            index,
            score,
            s=330,
            facecolor=colour,
            edgecolor=colour,
            linewidth=3.0,
            zorder=3,
        )
        ax.text(
            index,
            score + 1.6,
            f"{score:.1f}",
            ha="center",
            va="bottom",
            color=colour,
            fontsize=17,
            fontweight="semibold",
        )

    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.set_ylim(50, 90)
    ax.set_ylabel("IPV-Bench MCQA Accuracy (%)")
    ax.set_xticks(list(x), models, rotation=22, ha="right")
    ax.set_yticks([50, 60, 70, 80, 90])

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.4)

    fig.subplots_adjust(left=0.11, right=0.985, top=0.96, bottom=0.19)
    fig.savefig(ROOT / "ipv_bench_mcqa_projected.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_ipv_bench_mcqa()
