"""Generate all experiment figures from the row-level CSV pipeline.

Workflow:
1. Run compute_experiment_metrics.py to derive summaries from sample-level CSV files.
2. Use this script to regenerate the paper figure PNGs from those summaries.
"""

from compute_experiment_metrics import ROOT
from plot_paper_figures import (
    fig_accuracy_drop,
    fig_inverse_scaling,
    fig_per_category,
    fig_reliability,
    fig_text_only,
)
from plot_ipv_bench_mcqa import fig_ipv_bench_mcqa

if __name__ == "__main__":
    fig_inverse_scaling()
    fig_accuracy_drop()
    fig_text_only()
    fig_per_category()
    fig_reliability()
    fig_ipv_bench_mcqa()
    print("Saved paper figure PNGs to", ROOT)
