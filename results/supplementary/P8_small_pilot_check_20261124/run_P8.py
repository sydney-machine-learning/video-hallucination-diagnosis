"""Post hoc: is the target check needed when the labeled pilot is small?

For the 30 additional splits of each target (supplementary response banks), the pilot is
subsampled to a fraction of all target videos while the test half stays fixed. For each
split, model and fraction we compare
  - checked: the prespecified two-candidate rule (selector.promote, robust_safe),
  - unchecked LR: the target-pilot logistic score always used,
  - unchecked HGB: the development HGB score always used,
by test relative AURC reduction and harm (gain < 0). The 50% row reuses the full pilot.
Subsets are drawn from each split's pilot videos with a fixed seed, before any test label is read.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd, joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SUP = ROOT.parent; REPO = SUP.parents[1]
sys.path.insert(0, str(REPO / 'src'))
from evaluate_probe_escalation import aurc
import evaluate_v2_target_pilot as selector
selector.aurc = aurc

MODELS = ['internvl14', 'qwen8', 'llava8']
TARGETS = {
    'Perception': dict(dir=SUP / 'cpu_round2', feat='A_features.csv', cols='A_feature_columns.json', assign='A_video_assignments.csv',
                       bundle=REPO / 'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/pilot_model_bundle.joblib', seed=20260803, videos=500),
    'HERBench': dict(dir=SUP / 'P3H_herbench_repeat_20261120', feat='H_features.csv', cols='H_feature_columns.json', assign='H_video_assignments.csv',
                     bundle=REPO / 'results/herbench_confirmation/8918264.kman.restech.unsw.edu.au/pilot_model_bundle.joblib', seed=20260802, videos=68),
}
FRACTIONS = [0.1, 0.2, 0.3, 0.5]


def gain(y, raw, s):
    b = aurc(y, raw); return 100 * (b - aurc(y, s)) / b if b > 0 else 0.


def one(target, m, split, frac):
    cfg = TARGETS[target]; dest = ROOT / 'runs' / target / m / f'split_{split:02d}_f{int(frac*100):02d}.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists(): return json.loads(dest.read_text())
    f = pd.read_csv(cfg['dir'] / cfg['feat'], float_precision='round_trip'); f = f[f.model_id.eq(m)].reset_index(drop=True)
    a = pd.read_csv(cfg['dir'] / cfg['assign']); roles = a[a.split_id.eq(split)].set_index('video_id').role
    f['role'] = f.group_id.map(roles); pilot_all = f[f.role.eq('calibration')].copy(); test = f[f.role.eq('test')].copy()
    vids = np.array(sorted(pilot_all.group_id.unique()))
    k = max(5, int(round(frac * cfg['videos']))) if frac < 0.5 else len(vids)
    rng = np.random.default_rng(20261124 + 1000 * split + int(frac * 100))
    keep = set(rng.choice(vids, size=min(k, len(vids)), replace=False)) if frac < 0.5 else set(vids)
    pilot = pilot_all[pilot_all.group_id.isin(keep)].copy()
    columns = json.loads((cfg['dir'] / cfg['cols']).read_text()); arch = joblib.load(cfg['bundle']); assert columns == arch['feature_columns']
    src = arch['source_models']['source_trace_hgb']
    pilot['score__full_graph_gradient_boosting'] = src.predict_proba(pilot[columns])[:, 1]
    # The 50% row uses the same seed as the main repeated-split analysis, so it reproduces it exactly.
    seed = cfg['seed'] + 1000003 * split + 1009 * MODELS.index(m) + (17 * int(frac * 100) if frac < 0.5 else 0)
    with threadpool_limits(limits=1):
        p = selector.promote(pilot, columns, 2000, .05, seed, 'robust_safe')
    y = test.correct.to_numpy(); raw = test.score__raw_confidence.to_numpy()
    lr = p.target_model.predict_proba(test[columns])[:, 1]; hgb = src.predict_proba(test[columns])[:, 1]
    checked = lr if p.selected == 'target_logistic' else hgb if p.selected == 'source_trace_hgb' else raw
    out = dict(target=target, model=m, split=split, fraction=frac, pilot_videos=len(keep), pilot_questions=len(pilot),
               selected=p.selected, checked_gain=gain(y, raw, checked), unchecked_lr_gain=gain(y, raw, lr), unchecked_hgb_gain=gain(y, raw, hgb))
    dest.write_text(json.dumps(out) + '\n'); return out


def summarize():
    d = pd.DataFrame([json.loads(p.read_text()) for p in (ROOT / 'runs').glob('*/*/*.json')])
    d.to_csv(ROOT / 'P8_runs.csv', index=False)
    rows = []
    for (t, fr), g in d.groupby(['target', 'fraction']):
        row = dict(target=t, fraction=fr, evaluations=len(g), pilot_videos=g.pilot_videos.median(), retained_original=int(g.selected.eq('raw_fallback').sum()))
        for c in ['checked', 'unchecked_lr', 'unchecked_hgb']:
            v = g[f'{c}_gain']; row[f'{c}_mean'] = v.mean(); row[f'{c}_harm'] = int((v < 0).sum()); row[f'{c}_worst'] = v.min()
        rows.append(row)
    s = pd.DataFrame(rows); s.to_csv(ROOT / 'P8_summary.csv', index=False)
    (ROOT / 'P8_REPORT.md').write_text('# Small pilots: checked vs unchecked (post hoc, splits 1-30)\n\n' + s.round(2).to_string(index=False) + '\n', encoding='utf-8')
    print(s.round(2).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=14); a = ap.parse_args()
    jobs = [(t, m, s, fr) for t in TARGETS for m in MODELS for s in range(1, 31) for fr in FRACTIONS]
    with ProcessPoolExecutor(a.workers) as pool:
        fs = [pool.submit(one, *j) for j in jobs]
        for i, fut in enumerate(as_completed(fs), 1):
            if i % 60 == 0: print(i, '/', len(jobs), flush=True)
            fut.result()
    summarize()
