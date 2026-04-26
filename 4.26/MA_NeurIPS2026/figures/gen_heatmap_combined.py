"""
Combined heatmap: 6 RQ panels in one row, shared y-axis (model names left),
RQ name labelled at the bottom of each panel. Models sorted globally by
total normalised PASS strength across the 6 RQs.

Output: figures/summary/heatmaps/all_rq.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from _style import apply, REG
from _per_model_data import PER_MODEL

apply()
ROOT = os.path.dirname(os.path.abspath(__file__))


# ---- column specs per RQ ----
def cols_rq1(d):
    ρ = d['rq1_residual']
    raws = [f'{ρ:.1f}%', f'{abs(ρ - 100):.1f}%', d['rq1_sign']]
    str0 = float(np.clip(50 + (ρ - 100) / 20, 0, 100)) if ρ > 100 \
           else float(np.clip(100 - min(ρ, 100), 0, 100))
    s = [str0,
         float(np.clip(abs(ρ - 100), 0, 100)),
         80 if d['rq1_sign'] == 'Gen' else 60]
    return s, raws

def cols_rq2(d):
    τ = d['rq2_tau']
    return ([float(np.clip(100 - τ, 0, 100)),
             float(np.clip(100 - τ, 0, 100)),
             float(np.clip(50 + (10 - τ) * 5, 0, 100))],
            [f'{τ:.1f}%', f'{100 - τ:.1f}%', f'{10 - τ:+.1f}%'])

def cols_rq3(d):
    π = d['rq3_pi_F']
    tok = d['rq3_top1']
    is_ft = tok in {"'\\n\\n'", "' @'", "''", "' '", "' .'",
                    "'ed'", "'ky'", "' the'", "'in'", "'_'",
                    "'\\n\\n\\n'", "'The'"}
    return ([π * 100, (1 - π) * 100, 90 if is_ft else 30],
            [f'{π:.2f}', f'{1 - π:.2f}', tok])

def cols_rq4(d):
    η, R2 = d['rq4_eta'], d['rq4_R2']
    gap = R2 - 0.95
    return ([float(np.clip((η - 1) / 2.5 * 100, 0, 100)),
             R2 * 100,
             float(np.clip(50 + gap * 200, 0, 100))],
            [f'{η:.2f}', f'{R2:.2f}', f'{gap:+.2f}'])

def cols_rq5(d):
    s, m = d['rq5_single'], d['rq5_macro']
    s_abs = abs(s) if s is not None else None
    m_abs = abs(m) if m is not None else None
    vals = [v for v in (s_abs, m_abs) if v is not None]
    best = max(vals) if vals else None
    return (
        [np.nan if s_abs is None else float(np.clip(s_abs, 0, 100)),
         np.nan if m_abs is None else float(np.clip(m_abs, 0, 100)),
         np.nan if best is None else float(np.clip(best, 0, 100))],
        [None if s_abs is None else f'{s_abs:.0f}%',
         None if m_abs is None else f'{m_abs:.0f}%',
         None if best is None else f'{best:.0f}%'])

def cols_rq6(d):
    r = d['rq6_recovery']
    if r is None:
        return [np.nan, np.nan], [None, None]
    gap = r - 30
    return ([float(np.clip(r, 0, 100)),
             float(np.clip(50 + gap * 1.5, 0, 100))],
            [f'{r:.0f}%', f'{gap:+.0f}%'])


SPECS = [
    ('RQ1', 'attention',     [r'$\rho$', r'$|\Delta\mathrm{MA}|$', 'sign'],     cols_rq1),
    ('RQ2', 'MLP retain',    [r'$\tau$', r'$100-\tau$', 'gap'],                 cols_rq2),
    ('RQ3', r'$\pi(F)$',     [r'$\pi(F)$', r'$1-\pi(F)$', r'$u_{1}$'],          cols_rq3),
    ('RQ4', 'SVD',           [r'$\eta$', r'$R^{2}$', 'gap'],                    cols_rq4),
    ('RQ5', 'V-ablation',    [r'$|\Delta_{V}^{\rm sgl}|$',
                              r'$|\Delta_{V}^{\rm mac}|$', 'best'],             cols_rq5),
    ('RQ6', 'recovery',      [r'$r^{*}$', 'gap'],                               cols_rq6),
]

CMAP = LinearSegmentedColormap.from_list(
    'PassStrength',
    ['#E58E73', '#F4D86F', '#A8D9A0', '#5FAE73'], N=256).copy()
CMAP.set_bad('#E0E0E0')


# ---- gather data and pick global model order ----
models = list(PER_MODEL.keys())
panel_strength = {}   # rq -> (n_models, n_cols)
panel_raw = {}
for rq, _, _, fn in SPECS:
    S = []
    R = []
    for m in models:
        s, r = fn(PER_MODEL[m])
        S.append(s); R.append(r)
    panel_strength[rq] = np.array(S, dtype=float)
    panel_raw[rq] = R

# Total score: nanmean across all sub-cells of all panels
all_cells = np.concatenate([panel_strength[rq[0]] for rq in SPECS], axis=1)
mean_score = np.nanmean(all_cells, axis=1)
# tie-break: prefer fewer NaN
n_nan = np.isnan(all_cells).sum(axis=1)
order = np.lexsort((n_nan, -mean_score))
models = [models[i] for i in order]
for rq in panel_strength:
    panel_strength[rq] = panel_strength[rq][order]
    panel_raw[rq] = [panel_raw[rq][i] for i in order]


# ---- figure layout ----
n_models = len(models)
col_counts = [s.shape[1] for s in panel_strength.values()]
total_cols = sum(col_counts)

# 1 row × 6 panels, widths proportional to col count, plus a legend gap on right
fig = plt.figure(figsize=(0.65 * total_cols + 2.4, 0.31 * n_models + 1.4),
                 facecolor='white')
gs = GridSpec(1, len(SPECS), figure=fig,
              width_ratios=col_counts,
              left=0.13, right=0.93, top=0.93, bottom=0.10,
              wspace=0.10)

axes = []
im = None
for k, (rq, rq_lab, col_labels, _) in enumerate(SPECS):
    ax = fig.add_subplot(gs[0, k])
    axes.append(ax)
    M = panel_strength[rq]
    R = panel_raw[rq]
    masked = np.ma.masked_invalid(M)
    im = ax.imshow(masked, cmap=CMAP, vmin=0, vmax=100,
                   aspect='auto', interpolation='nearest')

    # Cell text
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            txt = R[i][j]
            if txt is None: continue
            v = M[i, j]
            color = 'white' if (np.isfinite(v) and (v < 28 or v > 78)) else '#222'
            if not np.isfinite(v): color = '#666'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold')

    # Top: column sub-labels
    ax.set_xticks(np.arange(M.shape[1]))
    ax.set_xticklabels(col_labels, fontsize=7.8)
    ax.tick_params(axis='x', length=0, pad=2)
    ax.xaxis.tick_top()

    # Bottom: RQ banner via xlabel
    ax.set_xlabel(f'$\\mathbf{{{rq}}}$  {rq_lab}', fontsize=10, labelpad=10)

    # Y axis: only first panel shows model names
    if k == 0:
        ax.set_yticks(np.arange(n_models))
        ax.set_yticklabels(models, fontsize=8)
        ax.tick_params(axis='y', length=0)
        # regime swatch on the very left
        for i, m in enumerate(models):
            ax.add_patch(plt.Rectangle((-0.74, i - 0.40), 0.18, 0.80,
                                        color=REG[PER_MODEL[m]['regime']],
                                        ec='none', clip_on=False))
        ax.set_xlim(-0.85, M.shape[1] - 0.5)
    else:
        ax.set_yticks([])

    # White grid separators
    for x in np.arange(M.shape[1] + 1) - 0.5:
        ax.axvline(x, color='white', linewidth=1.2)
    for y in np.arange(n_models + 1) - 0.5:
        ax.axhline(y, color='white', linewidth=0.8)
    ax.set_ylim(n_models - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

# Shared colorbar on the right
cbar_ax = fig.add_axes([0.945, 0.18, 0.012, 0.55])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Strength (0–100)', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Regime legend at the bottom
handles = [mpatches.Patch(color=REG[r], label=r)
           for r in ['CONC', 'FS', 'DISP', 'ANOM']]
fig.legend(handles=handles, loc='lower center',
           bbox_to_anchor=(0.5, 0.005), ncol=4,
           frameon=False, fontsize=8.5, handletextpad=0.4,
           columnspacing=1.6)

out_dir = os.path.join(ROOT, 'summary', 'heatmaps')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'all_rq.png')
fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'Saved: {out}')
