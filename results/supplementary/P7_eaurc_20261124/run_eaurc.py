"""Excess AURC (E-AURC) of the original and target-checked scores on the two held-out test splits.

E-AURC = AURC - AURC of the oracle ordering that ranks every correct answer first.
AURC is the prefix-risk mean used throughout the paper; the recomputed original and
target-checked AURCs must equal the archived values in test_model_metrics.csv.
"""
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT.parents[1]
TARGETS = {'HERBench': RES / 'herbench_confirmation/8918264.kman.restech.unsw.edu.au',
           'Perception Test': RES / 'perception_confirmation/8919110.kman.restech.unsw.edu.au'}


def aurc(correct, score):
    order = np.argsort(-score, kind='mergesort'); err = 1 - correct[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def oracle(correct):
    err = np.sort(1 - correct)
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


rows = []
for target, folder in TARGETS.items():
    pred = pd.read_csv(folder / 'test_scored_predictions.csv')
    archived = pd.read_csv(folder / 'test_model_metrics.csv')
    for model, g in pred.groupby('model_id'):
        y = g.correct.to_numpy(float)
        a0, a1, ao = aurc(y, g.score__raw_confidence.to_numpy()), aurc(y, g.trace_rc_score.to_numpy()), oracle(y)
        ref = archived[(archived.model_id == model) & (archived.method == 'raw_confidence')].raw_aurc.iloc[0]
        assert abs(a0 - ref) < 1e-12, (target, model, a0, ref)
        rows.append(dict(target=target, model=model, n=len(y), accuracy=y.mean(), oracle_aurc=ao,
                         aurc_original=a0, aurc_checked=a1, eaurc_original=a0 - ao, eaurc_checked=a1 - ao,
                         gap_closed_percent=100 * (a0 - a1) / (a0 - ao)))
out = pd.DataFrame(rows)
out.to_csv(ROOT / 'eaurc_by_model.csv', index=False)
mean = out.groupby('target')[['oracle_aurc', 'eaurc_original', 'eaurc_checked', 'gap_closed_percent']].mean()
mean.to_csv(ROOT / 'eaurc_mean.csv')
print(out.round(4).to_string(index=False)); print(mean.round(4))
