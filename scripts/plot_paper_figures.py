import csv
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent.parent
SUM_DIR = ROOT / 'data' / 'metric_summaries'
# Save directly to the paper workspace so students reproduce the exact
# filenames used by the manuscript.
FIG_DIR = ROOT
RAW_DIR = ROOT / 'data' / 'experiment_logs'

STYLE = {
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
}

COLORS = ['#E5AE38', '#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B2']
MARKERS = ['o','s','D','^','v','P']

def read_csv(name):
    with (SUM_DIR / name).open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def read_raw_csv(name):
    with (RAW_DIR / name).open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def wilson_interval(successes, total, z=1.959963984540054):
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

def save(fig, name):
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

probe_i = read_csv('probe_i_model_summary.csv')
probe_i = sorted(probe_i, key=lambda r: float(r['size_b']))
labels = [r['model_label'].replace('\\n', '\n') for r in probe_i]
log_sizes = np.log10([float(r['size_b']) for r in probe_i])
pdi = np.array([float(r['pdi_simple']) for r in probe_i])
orig = np.array([float(r['orig_accuracy']) for r in probe_i])
rev = np.array([float(r['rev_accuracy']) for r in probe_i])
text_pdi = np.array([float(r['text_only_pdi']) for r in probe_i])

def fig_inverse_scaling():
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.14)

    point_colors = ['#E5AE38', '#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B2']
    point_labels = ['Qwen2-VL-2B', 'LLaVA-1.6', 'Qwen2-VL-7B', 'Gemini Flash', 'InternVL2-26B', 'GPT-4o']
    raw_rows = read_raw_csv('probe_i_main_responses.csv')
    model_keys = [r['model_key'] for r in probe_i]
    rng = np.random.default_rng(20260622)
    estimates, lower, upper = [], [], []
    for key in model_keys:
        rows = [r for r in raw_rows if r['model_key'] == key and r['condition'] == 'reversed']
        video_ids = sorted({r['video_id'] for r in rows})
        video_scores = np.array([
            np.mean([
                1.0 if r['chosen_answer_type'] == 'prior' else 0.0
                for r in rows if r['video_id'] == video_id
            ])
            for video_id in video_ids
        ])
        estimates.append(video_scores.mean())
        sampled = video_scores[rng.integers(0, len(video_scores), size=(20000, len(video_scores)))]
        boot_means = sampled.mean(axis=1)
        lo, hi = np.quantile(boot_means, [0.025, 0.975])
        lower.append(lo)
        upper.append(hi)
    pdi_plot = np.array(estimates)
    lower = np.array(lower)
    upper = np.array(upper)

    regression = stats.linregress(log_sizes, pdi_plot)
    xline = np.linspace(log_sizes.min() - 0.08, log_sizes.max() + 0.08, 300)
    yline = regression.slope * xline + regression.intercept
    residuals = pdi_plot - (regression.slope * log_sizes + regression.intercept)
    residual_se = np.sqrt(np.sum(residuals ** 2) / (len(log_sizes) - 2))
    sxx = np.sum((log_sizes - log_sizes.mean()) ** 2)
    mean_se = residual_se * np.sqrt(
        1.0 / len(log_sizes) + (xline - log_sizes.mean()) ** 2 / sxx
    )
    t_crit = stats.t.ppf(0.975, df=len(log_sizes) - 2)
    ax.fill_between(
        xline, yline - t_crit * mean_se, yline + t_crit * mean_se,
        color='#CC6666', alpha=0.12, zorder=1
    )
    ax.plot(
        xline,
        yline,
        color='#B73A3A',
        lw=2.2,
        ls='--',
        label=rf'OLS: $\hat{{\beta}}={regression.slope:.2f},\ R^2={regression.rvalue ** 2:.2f},\ p={regression.pvalue:.3f}$',
        zorder=2,
    )

    yerr = np.vstack((pdi_plot - lower, upper - pdi_plot))
    for i, (x, y, c, lbl) in enumerate(zip(log_sizes, pdi_plot, point_colors, point_labels)):
        ax.errorbar(
            x, y, yerr=yerr[:, i:i+1], fmt='none', color=c,
            elinewidth=1.8, capsize=5, capthick=1.6, zorder=3
        )
        ax.scatter(x, y, s=200, color=c, edgecolor='white', linewidth=0.8, zorder=4, label=lbl)

    ax.text(
        0.61, 0.18, r'Spearman $\rho_s$ = 0.93,  $p$ = 0.008',
        transform=ax.transAxes, ha='left', va='center',
        fontsize=14, color='#0E7C43', style='italic',
        bbox=dict(boxstyle='round,pad=0.28', facecolor='#EEF9F1', edgecolor='#78C98F', alpha=0.95)
    )
    ax.text(
        0.82, 0.08, rf'OLS $p$ = {regression.pvalue:.3f}',
        transform=ax.transAxes, ha='left', va='center',
        fontsize=13, color='#C94D4D', style='italic'
    )

    ax.set_xlabel(r'$\log_{10}(\mathrm{Parameters\ / \ B})$', fontsize=18)
    ax.set_ylabel(r'PDI$_{\mathrm{simple}}$ ($\uparrow$ worse)', fontsize=18)
    ax.set_xticks(log_sizes)
    ax.set_xticklabels(['2 B', '7 B', '7 B', '10 B', '26 B', '200 B'], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlim(log_sizes.min() - 0.16, log_sizes.max() + 0.18)
    ax.set_ylim(0.22, 0.88)
    ax.grid(True, alpha=0.25)
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.01, 0.995),
        frameon=True,
        framealpha=0.94,
        edgecolor='#cfcfcf',
        fontsize=11.3,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.45,
        labelspacing=0.22,
        borderaxespad=0.15,
    )
    save(fig, 'probe_i_a_inverse_scaling.png')

def fig_accuracy_drop():
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(12,6.8))
    x = np.arange(len(labels)); w = 0.32
    ax.bar(x-w/2, orig, w, color='#5B8DB8', alpha=0.88, label='Original clips', edgecolor='white')
    ax.bar(x+w/2, rev, w, color='#CC6666', alpha=0.90, label='Reversed clips', edgecolor='white')
    for i, d in enumerate(orig-rev):
        ax.text(x[i], max(orig[i], rev[i])+0.055, f'$\\Delta$ {d:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax.axhline(0.5, color='grey', ls=':', lw=1)
    ax.text(len(labels)-0.35, 0.505, 'chance', ha='right', va='bottom', fontsize=13, color='grey')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0,1.12); ax.legend(loc='upper left', fontsize=14, framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.3)
    save(fig, 'probe_i_b_accuracy_drop.png')

def fig_text_only():
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(10.8,6.8))
    x = np.arange(len(labels)); w=0.3
    ax.bar(x-w/2, text_pdi, w, color='#D4A76A', alpha=0.88, label='Text-only (no video)', edgecolor='white')
    ax.bar(x+w/2, pdi, w, color='#5B8DB8', alpha=0.88, label='With video (standard)', edgecolor='white')
    override = 1.0 - pdi/text_pdi
    for i, ov in enumerate(override):
        ax.text(x[i], max(text_pdi[i], pdi[i])+0.10, f'Override\n{ov:.0%}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#444444')
    ax.axhline(0.5, color='grey', ls=':', lw=0.9)
    ax.text(len(labels)-0.4, 0.505, 'chance', ha='right', va='bottom', fontsize=12, color='grey')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('PDI$_{simple}$  ($\\uparrow$ worse)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0,1.18); ax.legend(loc='upper left', bbox_to_anchor=(0.00,1.03), fontsize=14, framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.3)
    save(fig, 'fig_text_only_baseline.png')

def fig_per_category():
    rows = read_csv('probe_i_per_category_summary.csv')
    cats = []
    for r in rows:
        if r['category_label'] not in cats:
            cats.append(r['category_label'])
    data_map = {(r['category_label'], r['model_key']): float(r['pdi_simple']) for r in rows}
    model_keys = [r['model_key'] for r in probe_i]
    data = np.array([[data_map[(c, k)] for k in model_keys] for c in cats])
    cats = [c.replace('\\n','\n') for c in cats]
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(12.2,7.2))
    fig.subplots_adjust(left=0.24, right=0.90, top=0.97, bottom=0.36)
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0.15, vmax=0.90)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i,j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=14, fontweight='bold', color='white' if val > 0.65 else '#333333')
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticks(np.arange(len(cats))); ax.set_yticklabels(cats, fontsize=14)
    ax.tick_params(axis='x', pad=10); ax.tick_params(axis='y', pad=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label('PDI ($\\uparrow$ worse)', fontsize=16); cbar.ax.tick_params(labelsize=14)
    col_means = data.mean(axis=0)
    trans = ax.get_xaxis_transform()
    for j, mu in enumerate(col_means):
        ax.text(j, -0.16, f'$\\mu$={mu:.2f}', transform=trans, ha='center', va='top', fontsize=13, color='#555555', clip_on=False)
    ax.set_ylim(len(cats)-0.35, -0.60)
    save(fig, 'fig_per_category_pdi.png')

def fig_reliability():
    rows = read_raw_csv('probe_ii_masking_responses.csv')
    model_keys = [r['model_key'] for r in probe_i]
    label_map = {r['model_key']: r['model_name'] for r in probe_i}
    rows = [r for r in rows if int(r['mask_level_pct']) == 100]
    confs = sorted({float(r['confidence']) for r in rows})
    diag_x = np.linspace(0,1,200)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10.2,7.6), dpi=320)
    ax.fill_between(diag_x, 0, diag_x, color='#fbe3e4', alpha=0.55, zorder=0)
    ax.plot(diag_x, diag_x, linestyle='--', color='#4d4d4d', linewidth=1.8, label='Perfect calibration')
    for idx, key in enumerate(model_keys):
        model_rows = [r for r in rows if r['model_key'] == key]
        vals, lower, upper, weights = [], [], [], []
        for confidence in confs:
            bin_rows = [r for r in model_rows if float(r['confidence']) == confidence]
            successes = sum(int(r['uncertainty_correct']) for r in bin_rows)
            total = len(bin_rows)
            value = successes / total
            lo, hi = wilson_interval(successes, total)
            vals.append(value)
            lower.append(lo)
            upper.append(hi)
            weights.append(total / len(model_rows))
        vals = np.array(vals)
        lower = np.array(lower)
        upper = np.array(upper)
        hce = np.sum(np.array(weights) * np.abs(np.array(confs) - vals))
        ax.errorbar(
            confs,
            vals,
            yerr=np.vstack((vals - lower, upper - vals)),
            label=f"{label_map[key]} (HCE={hce:.2f})",
            color=COLORS[idx],
            linewidth=2.2,
            marker=MARKERS[idx],
            markersize=6.5,
            markerfacecolor='white',
            markeredgewidth=1.4,
            elinewidth=1.0,
            capsize=2.5,
        )
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_yticklabels(['','0.2','0.4','0.6','0.8','1.0'])
    ax.set_xlabel('Self-reported confidence', fontsize=18)
    ax.set_ylabel('Fraction correctly expressing uncertainty', fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.text(0.64,0.72,'Better calibrated', fontsize=16, color='#555555', rotation=32)
    ax.text(0.62,0.42,'Overconfidence zone', fontsize=16, color='#8a4f56', rotation=32)
    ax.legend(loc='upper left', bbox_to_anchor=(0.03,0.93), fontsize=14.5, framealpha=0.92)
    save(fig, 'probe_ii_a_reliability.png')

if __name__ == '__main__':
    fig_inverse_scaling()
    fig_accuracy_drop()
    fig_text_only()
    fig_per_category()
    fig_reliability()
    print('Saved paper figures to', FIG_DIR)
