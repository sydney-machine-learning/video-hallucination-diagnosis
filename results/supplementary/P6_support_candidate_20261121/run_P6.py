"""Post hoc analysis: add label-free all-view support as a third candidate in the selection check.

Uses the supplementary response banks and the same 30 additional video splits as
cpu_round2 (Perception Test) and P3H (HERBench). For every target, model and split it runs
  - rule2: the original check (pilot LR, development HGB), via selector.promote, same seeds;
  - rule3: the same check with all-view support added as a candidate, threshold 0,
    Bonferroni level 0.05/3 for every candidate.
The rule was added after the main results were known, so it is exploratory.
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
                       bundle=REPO / 'results/perception_confirmation/8919110.kman.restech.unsw.edu.au/pilot_model_bundle.joblib', seed=20260803, n=250),
    'HERBench': dict(dir=SUP / 'P3H_herbench_repeat_20261120', feat='H_features.csv', cols='H_feature_columns.json', assign='H_video_assignments.csv',
                     bundle=REPO / 'results/herbench_confirmation/8918264.kman.restech.unsw.edu.au/pilot_model_bundle.joblib', seed=20260802, n=34),
}
MARGIN = {'source_trace_hgb': 0.5, 'target_logistic': 0.0, 'all_view_support': 0.0}


def gain(y, raw, s):
    b = aurc(y, raw); return 100 * (b - aurc(y, s)) / b if b > 0 else 0.


def promote3(pilot, columns, iterations, alpha, seed):
    raw = pilot.score__raw_confidence.to_numpy(float)
    oof, model = selector.cross_fitted_logistic(pilot, columns, folds=5)
    scores = {'source_trace_hgb': pilot.score__full_graph_gradient_boosting.to_numpy(float), 'target_logistic': oof,
              'all_view_support': pilot.all_view_support.to_numpy(float)}
    q = alpha / len(scores); rows = []
    for off, (name, v) in enumerate(sorted(scores.items())):
        d = selector.grouped_bootstrap_gains(pilot, raw, v, iterations, seed + 7919 * off)
        rows.append((name, float(np.quantile(d, q))))
    ok = [r for r in rows if r[1] > MARGIN[r[0]]]
    sel = sorted(ok, key=lambda r: (-r[1], r[0]))[0][0] if ok else 'raw_fallback'
    return sel, dict(rows), model


def one(target, m, split):
    cfg = TARGETS[target]; dest = ROOT / 'runs' / target / m / f'split_{split:02d}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'DONE.json').exists(): return json.loads((dest / 'DONE.json').read_text())
    f = pd.read_csv(cfg['dir'] / cfg['feat'], float_precision='round_trip'); f = f[f.model_id.eq(m)].reset_index(drop=True)
    a = pd.read_csv(cfg['dir'] / cfg['assign']); roles = a[a.split_id.eq(split)].set_index('video_id').role
    f['role'] = f.group_id.map(roles); pilot = f[f.role.eq('calibration')].copy(); test = f[f.role.eq('test')].copy()
    assert pilot.group_id.nunique() == test.group_id.nunique() == cfg['n'] and not set(pilot.group_id) & set(test.group_id)
    columns = json.loads((cfg['dir'] / cfg['cols']).read_text()); arch = joblib.load(cfg['bundle']); assert columns == arch['feature_columns']
    src = arch['source_models']['source_trace_hgb']
    pilot['score__full_graph_gradient_boosting'] = src.predict_proba(pilot[columns])[:, 1]
    seed = cfg['seed'] + 1000003 * split + 1009 * MODELS.index(m)
    with threadpool_limits(limits=1):
        p2 = selector.promote(pilot, columns, 2000, .05, seed, 'robust_safe')
        sel3, lb3, model3 = promote3(pilot, columns, 2000, .05, seed)
    y = test.correct.to_numpy(); raw = test.score__raw_confidence.to_numpy()
    sc = {'original': raw, 'development_hgb': src.predict_proba(test[columns])[:, 1], 'all_view_support': test.all_view_support.to_numpy()}
    lr2 = p2.target_model.predict_proba(test[columns])[:, 1]; lr3 = model3.predict_proba(test[columns])[:, 1]
    key = {'raw_fallback': 'original', 'source_trace_hgb': 'development_hgb', 'all_view_support': 'all_view_support'}
    s2 = lr2 if p2.selected == 'target_logistic' else sc[key[p2.selected]]
    s3 = lr3 if sel3 == 'target_logistic' else sc[key[sel3]]
    done = dict(target=target, model=m, split=split, rule2_selected=p2.selected, rule3_selected=sel3,
                rule2_gain=gain(y, raw, s2), rule3_gain=gain(y, raw, s3), support_gain=gain(y, raw, sc['all_view_support']),
                pilot_lr_gain=gain(y, raw, lr2), rule3_lower_bounds=lb3)
    (dest / 'DONE.json').write_text(json.dumps(done, indent=2) + '\n'); return done


def summarize():
    d = pd.DataFrame([json.loads(p.read_text()) for p in (ROOT / 'runs').glob('*/*/split_*/DONE.json')])
    d.drop(columns='rule3_lower_bounds').to_csv(ROOT / 'P6_splits.csv', index=False)
    rep = d[d.split.ge(1)]; lines = ['# Support as a third candidate (post hoc, splits 1-30)', '']
    for t, g in rep.groupby('target'):
        lines += [f'## {t}', '']
        per = g.groupby('model')[['rule2_gain', 'rule3_gain', 'support_gain', 'pilot_lr_gain']].mean()
        lines += [per.round(2).to_string(), '']
        mac = g.groupby('split')[['rule2_gain', 'rule3_gain', 'support_gain', 'pilot_lr_gain']].mean()
        lines += ['Mean across models: ' + ', '.join(f'{c} {mac[c].mean():.2f}' for c in mac), '']
        lines += ['Harm (evaluations < 0 of 90): ' + ', '.join(f'{c} {int((g[c] < 0).sum())}' for c in ['rule2_gain', 'rule3_gain', 'support_gain', 'pilot_lr_gain'])]
        lines += [f'Rule3 minus support, per split: mean {(mac.rule3_gain - mac.support_gain).mean():.2f}, rule3 ahead in {int((mac.rule3_gain > mac.support_gain).sum())}/30',
                  f'Rule3 minus rule2, per split: mean {(mac.rule3_gain - mac.rule2_gain).mean():.2f}, min split mean rule3 {mac.rule3_gain.min():.2f}', '']
        lines += ['Rule3 selections:', g.groupby('model').rule3_selected.value_counts().unstack(fill_value=0).to_string(), '']
        lines += ['Rule2 selections:', g.groupby('model').rule2_selected.value_counts().unstack(fill_value=0).to_string(), '']
    (ROOT / 'P6_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=14); a = ap.parse_args()
    jobs = [(t, m, s) for t in TARGETS for m in MODELS for s in range(31)]
    with ProcessPoolExecutor(a.workers) as pool:
        fs = [pool.submit(one, *j) for j in jobs]
        for i, f in enumerate(as_completed(fs), 1): r = f.result(); print(i, '/', len(jobs), r['target'], r['model'], r['split'], r['rule3_selected'], round(r['rule3_gain'], 2), flush=True)
    summarize()
