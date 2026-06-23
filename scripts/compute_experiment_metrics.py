import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / 'data' / 'experiment_logs'
OUT_DIR = ROOT / 'data' / 'metric_summaries'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    with (RAW_DIR / name).open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, fieldnames, rows):
    with (OUT_DIR / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


meta = read_csv('model_metadata.csv')
meta_map = {r['model_key']: r for r in meta}
model_order = [r['model_key'] for r in meta]

# Probe I main summaries
probe_i = read_csv('probe_i_main_responses.csv')
rows_by_model_cond = defaultdict(list)
for r in probe_i:
    rows_by_model_cond[(r['model_key'], r['condition'])].append(r)

summary_rows = []
for key in model_order:
    m = meta_map[key]
    reversed_rows = rows_by_model_cond[(key, 'reversed')]
    original_rows = rows_by_model_cond[(key, 'original')]
    text_rows = rows_by_model_cond[(key, 'text_only')]

    pdi_simple = mean([1.0 if r['chosen_answer_type'] == 'prior' else 0.0 for r in reversed_rows])
    orig_acc = mean([float(r['is_correct']) for r in original_rows])
    rev_acc = mean([float(r['is_correct']) for r in reversed_rows])
    text_pdi = mean([1.0 if r['chosen_answer_type'] == 'prior' else 0.0 for r in text_rows])

    pdi_theory = ''
    if m['supports_logprob'] == 'yes':
        log_odds = [float(r['logprob_prior']) - float(r['logprob_visual']) for r in reversed_rows]
        pdi_theory = f"{mean(log_odds):.4f}"

    summary_rows.append({
        'model_key': key,
        'model_name': m['model_name'],
        'model_label': m['model_label'],
        'size_b': m['size_b'],
        'pdi_simple': f'{pdi_simple:.4f}',
        'pdi_theory': pdi_theory,
        'orig_accuracy': f'{orig_acc:.4f}',
        'rev_accuracy': f'{rev_acc:.4f}',
        'text_only_pdi': f'{text_pdi:.4f}',
        'accuracy_gap': f'{(orig_acc - rev_acc):.4f}',
        'visual_override_ratio': f'{(1.0 - pdi_simple / text_pdi):.4f}',
    })
write_csv('probe_i_model_summary.csv', list(summary_rows[0].keys()), summary_rows)

# Probe I category summary
cat_rows = read_csv('probe_i_category_responses.csv')
cat_group = defaultdict(list)
for r in cat_rows:
    cat_group[(r['category_label'], r['model_key'])].append(r)
per_category_rows = []
for category_label, _ in sorted({(r['category_label'], r['category_key']) for r in cat_rows}, key=lambda x: x[1]):
    for key in model_order:
        rows = cat_group[(category_label, key)]
        pdi = mean([1.0 if r['chosen_answer_type'] == 'prior' else 0.0 for r in rows])
        per_category_rows.append({'category_label': category_label, 'model_key': key, 'pdi_simple': f'{pdi:.4f}'})
write_csv('probe_i_per_category_summary.csv', ['category_label','model_key','pdi_simple'], per_category_rows)

# Probe II-A masking summary + reliability bins
mask_rows = read_csv('probe_ii_masking_responses.csv')
mask_group = defaultdict(list)
reliability_group = defaultdict(list)
for r in mask_rows:
    mask_group[(r['model_key'], int(r['mask_level_pct']))].append(r)
    if int(r['mask_level_pct']) == 100:
        reliability_group[(r['model_key'], float(r['confidence']))].append(r)

mask_summary = []
for key in model_order:
    for level in [0,25,50,75,100]:
        rows = mask_group[(key, level)]
        conf = [float(r['confidence']) for r in rows]
        acc = [float(r['uncertainty_correct']) for r in rows]
        hce = 0.0
        bin_group = defaultdict(list)
        for r in rows:
            bin_group[float(r['confidence'])].append(float(r['uncertainty_correct']))
        n = len(rows)
        for cbin, vals in sorted(bin_group.items()):
            hce += (len(vals) / n) * abs(cbin - mean(vals))
        mask_summary.append({'model_key': key, 'mask_level_pct': level, 'hce': f'{hce:.4f}'})
write_csv('probe_ii_masking_summary.csv', ['model_key','mask_level_pct','hce'], mask_summary)

reliability_rows = []
for key in model_order:
    for conf in sorted({float(r['confidence']) for r in mask_rows if int(r['mask_level_pct']) == 100}):
        rows = reliability_group[(key, conf)]
        reliability_rows.append({
            'model_key': key,
            'confidence': f'{conf:.1f}',
            'fraction_correct_uncertainty': f"{mean([float(r['uncertainty_correct']) for r in rows]):.4f}"
        })
write_csv('probe_ii_reliability_summary.csv', ['model_key','confidence','fraction_correct_uncertainty'], reliability_rows)

# Probe II-B object permanence summary
op_rows = read_csv('probe_ii_object_permanence_responses.csv')
op_group = defaultdict(list)
for r in op_rows:
    op_group[r['model_key']].append(r)

oh_rows = []
for key in model_order:
    rows = op_group[key]
    acc = mean([float(r['is_correct']) for r in rows])
    conf = mean([float(r['confidence']) for r in rows])
    oh_rows.append({'model_key': key, 'accuracy': f'{acc:.4f}', 'avg_confidence': f'{conf:.4f}', 'excess_confidence': f'{(conf - acc):.4f}'})
write_csv('probe_ii_object_permanence_summary.csv', ['model_key','accuracy','avg_confidence','excess_confidence'], oh_rows)

print('Wrote metric summaries to', OUT_DIR)
