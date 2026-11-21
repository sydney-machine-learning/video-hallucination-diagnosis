"""HERBench repeated-split analysis on the supplementary response bank (CPU only).

Mirrors cpu_round2/run_A.py (Perception Test) with the HERBench manifest, labels and
P2 video splits. Split 0 is the original reference assignment; splits 1-30 are the
predetermined additional splits.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import sys, json, hashlib, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO / 'src'))
import evaluate_response_graph_features as graph
from evaluate_probe_escalation import aurc
import evaluate_v2_target_pilot as selector
selector.aurc = aurc  # prefix-risk mean, as in cpu_round2

MODELS = ['internvl14', 'qwen8', 'llava8']
ASSIGN = ROOT.parent / 'P2/video_assignments.csv'
BUNDLE = REPO / 'results/herbench_confirmation/8918264.kman.restech.unsw.edu.au/pilot_model_bundle.joblib'
MANIFEST = ROOT / 'full_manifest.jsonl'
LABELS = [REPO / 'input/herbench' / n for n in ['pilot_labels.csv', 'sealed_test_labels.csv']]
ROLES = [(0, 'base', 16), (2, 'invariant', 8), (4, 'addition', 24), (6, 'removal', 8)]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, o): Path(p).write_text(json.dumps(o, indent=2, default=str) + '\n', encoding='utf-8')
def readjsonl(p): return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
def gain(y, raw, s):
    base = aurc(y, raw); return 100 * (base - aurc(y, s)) / base if base > 0 else 0.


def prepare():
    manifest = readjsonl(MANIFEST); qids = [q['question_id'] for q in manifest]
    assert len(qids) == len(set(qids)) == 1971
    labels = pd.concat([pd.read_csv(p) for p in LABELS]).set_index('question_id')
    assert set(labels.index) == set(qids)
    assign = pd.read_csv(ASSIGN); assign = assign[assign.dataset.eq('HERBench')]
    for s, g in assign.groupby('split_id'):
        assert g.groupby('role').size().to_dict() == {'calibration': 34, 'test': 34}
        assert set(g.video_id) == {q['video_id'] for q in manifest}
    assign.to_csv(ROOT / 'H_video_assignments.csv', index=False)
    values = {}; audits = []
    for m in MODELS:
        rows = []
        for p in sorted((ROOT / 'shards' / m).glob('shard_*/responses.jsonl')): rows += readjsonl(p)
        bank = {(r['question_id'], r['phase']): r for r in rows}
        assert len(rows) == len(bank) == 4 * 1971, (m, len(rows), len(bank))
        mismatch = 0
        for q in manifest:
            k = len(q['candidates'])
            for phase, role, budget in ROLES:
                r = bank[(q['question_id'], phase)]; prob = np.array(r['candidate_probabilities'])
                assert r['video_id'] == q['video_id'] and r['frame_budget'] == budget
                assert prob.shape == (k,) and np.isfinite(prob).all() and np.isclose(prob.sum(), 1)
                values[('HERBench', m, q['question_id'], role)] = prob
                mismatch += r['answer'] != chr(65 + int(prob.argmax()))
        audits.append(dict(model=m, rows=len(rows), answer_vs_argmax_mismatch=mismatch,
                           model_ids=sorted({r['model_id'] for r in rows}), revisions=sorted({r['model_revision'] for r in rows})))
    meta = pd.DataFrame([dict(dataset='HERBench', qid=q['question_id'], group_id=q['video_id'],
                              gold=int(labels.loc[q['question_id'], 'gold_candidate_index']), candidate_count=len(q['candidates'])) for q in manifest])
    table, _ = graph.graph_table(meta, values, MODELS)
    features = [c for c in table if c not in ['dataset', 'qid', 'group_id', 'model_id', 'correct']]
    assert features == joblib.load(BUNDLE)['feature_columns']
    table['score__raw_confidence'] = table.own_base_confidence
    table['base_support'] = table.base_selected_support_mean
    table['all_view_support'] = table[[f'{r}_selected_support_mean' for r in ['base', 'invariant', 'addition', 'removal']]].mean(axis=1)
    table.to_csv(ROOT / 'H_features.csv', index=False); dump(ROOT / 'H_feature_columns.json', features)
    acc = []
    for q in manifest:
        ps = np.stack([values[('HERBench', m, q['question_id'], 'base')] for m in MODELS]); ans = ps.argmax(axis=1)
        votes = np.bincount(ans, minlength=ps.shape[1]); tied = np.flatnonzero(votes == votes.max())
        winner = int(tied[np.argmax(ps.mean(axis=0)[tied])]); gold = int(labels.loc[q['question_id'], 'gold_candidate_index'])
        acc.append(dict(qid=q['question_id'], group_id=q['video_id'], majority_vote=int(winner == gold), **{m: int(ans[i] == gold) for i, m in enumerate(MODELS)}))
    pd.DataFrame(acc).to_csv(ROOT / 'H_answer_accuracy.csv', index=False)
    dump(ROOT / 'H_PREFLIGHT.json', dict(status='passed', models=audits, feature_columns_match_bundle=True,
                                         manifest_sha256=sha(MANIFEST), bundle_sha256=sha(BUNDLE)))
    print('PREFLIGHT PASSED', audits, flush=True)


def one(m, split):
    dest = ROOT / 'H_runs' / m / f'split_{split:02d}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'DONE.json').exists(): return json.loads((dest / 'DONE.json').read_text())
    frame = pd.read_csv(ROOT / 'H_features.csv', float_precision='round_trip'); frame = frame[frame.model_id.eq(m)].reset_index(drop=True)
    assign = pd.read_csv(ROOT / 'H_video_assignments.csv'); roles = assign[assign.split_id.eq(split)].set_index('video_id').role
    frame['role'] = frame.group_id.map(roles); pilot = frame[frame.role.eq('calibration')].copy(); test = frame[frame.role.eq('test')].copy()
    assert not set(pilot.group_id) & set(test.group_id) and pilot.group_id.nunique() == test.group_id.nunique() == 34
    columns = json.loads((ROOT / 'H_feature_columns.json').read_text()); archive = joblib.load(BUNDLE)
    source = archive['source_models']['source_trace_hgb']
    pilot['score__full_graph_gradient_boosting'] = source.predict_proba(pilot[columns])[:, 1]
    seed = 20260802 + 1000003 * split + 1009 * MODELS.index(m)
    with threadpool_limits(limits=1): promoted = selector.promote(pilot, columns, 2000, .05, seed, 'robust_safe')
    score = {'original': test.score__raw_confidence.to_numpy(), 'pilot_lr': promoted.target_model.predict_proba(test[columns])[:, 1],
             'development_hgb': source.predict_proba(test[columns])[:, 1], 'base_support': test.base_support.to_numpy(),
             'all_view_support': test.all_view_support.to_numpy()}
    score['checked'] = score[{'raw_fallback': 'original', 'target_logistic': 'pilot_lr', 'source_trace_hgb': 'development_hgb'}[promoted.selected]]
    y = test.correct.to_numpy(); out = test[['qid', 'group_id', 'model_id', 'correct']].copy(); metrics = []
    for method, s in score.items():
        out[method] = s
        metrics.append(dict(model=m, split=split, method=method, accuracy=y.mean(), aurc=aurc(y, s), gain_pct=gain(y, score['original'], s), selected=promoted.selected))
    out.to_csv(dest / 'test_predictions.csv', index=False); pd.DataFrame(metrics).to_csv(dest / 'metrics.csv', index=False)
    done = dict(model=m, split=split, selected=promoted.selected, gain=metrics[-1]['gain_pct']); dump(dest / 'DONE.json', done)
    return done


def summarize():
    allm = pd.concat([pd.read_csv(p) for p in (ROOT / 'H_runs').glob('*/split_*/metrics.csv')], ignore_index=True)
    allm.to_csv(ROOT / 'H_selection_splits.csv', index=False)
    rep = allm[allm.split.ge(1)]
    per = rep.groupby(['model', 'method']).agg(mean_gain=('gain_pct', 'mean'), min_gain=('gain_pct', 'min'),
                                               harm=('gain_pct', lambda v: int((v < 0).sum())), splits=('split', 'nunique')).reset_index()
    macro = rep.groupby(['split', 'method']).gain_pct.mean().unstack()
    per.to_csv(ROOT / 'H_selection_summary.csv', index=False); macro.to_csv(ROOT / 'H_macro_by_split.csv')
    counts = rep[rep.method.eq('checked')].groupby('model').selected.value_counts().unstack(fill_value=0)
    counts.to_csv(ROOT / 'H_selection_counts.csv')
    lines = ['# HERBench repeated splits (supplementary bank)', '', 'Splits 1-30; split 0 is the reference assignment.', '',
             per.to_string(index=False), '', '## Mean across three models', '',
             macro.agg(['mean', 'min']).T.to_string(), '', '## Checked minus label-free support (pp, splits 1-30)', '']
    for b in ['base_support', 'all_view_support']:
        d = macro['checked'] - macro[b]; lines.append(f'- vs {b}: mean {d.mean():.2f}, positive in {(d > 0).sum()}/30, min {d.min():.2f}')
    lines += ['', '## Selection counts', '', counts.to_string(), '', '## Reference split 0', '',
              allm[allm.split.eq(0)].pivot(index='model', columns='method', values='gain_pct').to_string()]
    acc = pd.read_csv(ROOT / 'H_answer_accuracy.csv'); assign = pd.read_csv(ROOT / 'H_video_assignments.csv')
    t0 = set(assign[(assign.split_id == 0) & assign.role.eq('test')].video_id)
    lines += ['', '## Accuracy on reference test split (new bank)', '', acc[acc.group_id.isin(t0)][MODELS + ['majority_vote']].mean().to_frame('accuracy').to_string()]
    (ROOT / 'H_REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=['prepare', 'run', 'summarize']); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'summarize': summarize()
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            fs = [pool.submit(one, m, s) for m in MODELS for s in range(31)]
            for i, f in enumerate(as_completed(fs), 1): print(i, '/93', f.result(), flush=True)
        summarize()
