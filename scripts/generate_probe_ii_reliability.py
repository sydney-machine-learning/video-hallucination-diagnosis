from compute_experiment_metrics import ROOT
from plot_paper_figures import fig_reliability

if __name__ == "__main__":
    fig_reliability()
    print("Saved probe_ii_a_reliability.png to", ROOT)
