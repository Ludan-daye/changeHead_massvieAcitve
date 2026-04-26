"""
Per-RQ 26-model heatmap. One figure per RQ.

Each heatmap: 26 rows (models, sorted by primary RQ metric) × N columns
(sub-metrics specific to that RQ). Cell text = raw metric, cell color =
normalised strength (0..100). PASS line marked on the relevant column.
No titles (per user spec).

Output:  heatmaps/<RQ_dir>/heatmap/26models.png
        (mirrors figures/<RQ_dir>/<panel_type>/<file>.png layout)
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Pull style/data from the sibling figures/ module
SIBLING = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'figures'))
if SIBLING not in sys.path:
    sys.path.insert(0, SIBLING)
from _style import apply, REG
from _per_model_data import PER_MODEL

apply()
ROOT = os.path.dirname(os.path.abspath(__file__))
RQ_DIRS = {
    'RQ1': 'RQ1_attention',
    'RQ2': 'RQ2_mlp_source',
    'RQ3': 'RQ3_function_words',
    'RQ4': 'RQ4_svd_alignment',
    'RQ5': 'RQ5_v_ablation',
    'RQ6': 'RQ6_topk_scan',
}

CMAP = LinearSegmentedColormap.from_list(
    'PassStrength',
    ['#E58E73', '#F4D86F', '#A8D9A0', '#5FAE73'], N=256)


def draw_heatmap(rows, col_labels, raw_text, strength,
                 sort_key, save_path, fig_h=8.6,
                 pass_col_idx=None, pass_threshold_label=None):
    """
    rows: list of (model_name, dict) sorted by sort_key already
    col_labels: list of column header LaTeX strings
    raw_text: 2D list[len(rows)][len(cols)] of cell strings (None → blank cell)
    strength: 2D ndarray[len(rows), len(cols)] in [0,100], NaN → no-data
    """
    n_rows, n_cols = strength.shape
    fig_h_use = max(2.0, 0.30 * n_rows + 0.6)
    fig, ax = plt.subplots(figsize=(0.95 + 1.05 * n_cols, fig_h_use),
                           facecolor='white')
    masked = np.ma.masked_invalid(strength)
    cmap = CMAP.copy() if hasattr(CMAP, 'copy') else CMAP
    cmap.set_bad('#E0E0E0')
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100,
                   aspect='auto', interpolation='nearest')

    # Cell annotations (skip blanks)
    for i in range(n_rows):
        for j in range(n_cols):
            txt = raw_text[i][j]
            if txt is None:
                continue
            v = strength[i, j]
            if np.isnan(v):
                txt_color = '#666'
            else:
                txt_color = 'white' if (v < 28 or v > 78) else '#222'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color=txt_color, fontweight='bold')

    # Column headers (top)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.xaxis.tick_top()

    # Row labels (model names)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.tick_params(axis='y', length=0)

    # Regime swatch on the left
    for i, (m, d) in enumerate(rows):
        ax.add_patch(plt.Rectangle((-0.62, i - 0.40), 0.18, 0.80,
                                    color=REG[d['regime']], ec='none',
                                    clip_on=False))
    handles = [mpatches.Patch(color=REG[r], label=r)
               for r in ['CONC', 'FS', 'DISP', 'ANOM']]
    ax.legend(handles=handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.015), ncol=4,
              frameon=False, fontsize=8.5,
              handletextpad=0.4, columnspacing=1.4)

    # White grid separators
    for x in np.arange(n_cols + 1) - 0.5:
        ax.axvline(x, color='white', linewidth=1.4)
    for y in np.arange(n_rows + 1) - 0.5:
        ax.axhline(y, color='white', linewidth=1.0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.045, shrink=0.55)
    cbar.set_label('Strength (0–100)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(-0.7, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def clamp(v, lo=0, hi=100):
    return float(np.clip(v, lo, hi))


# ============= RQ1: Attention ablation =============
def make_rq1():
    cols_label = [r'$\rho$ (residual%)',
                  r'$|\Delta\mathrm{MA}|$ (%)',
                  r'sign']
    rows = sorted(PER_MODEL.items(), key=lambda kv: kv[1]['rq1_residual'])
    n = len(rows)
    strength = np.zeros((n, 3))
    raw = []
    for i, (m, d) in enumerate(rows):
        ρ = d['rq1_residual']
        absΔ = abs(ρ - 100)
        sign = d['rq1_sign']
        # strength: smaller residual = stronger H_0 falsification
        strength[i, 0] = clamp(100 - min(ρ, 100), 0, 100) if ρ <= 100 else 0
        # if Sup (ρ > 100): residual is "boosted", treat amplitude as effect strength
        if ρ > 100:
            strength[i, 0] = clamp(50 + (ρ - 100) / 20, 0, 100)
        strength[i, 1] = clamp(absΔ, 0, 100)
        strength[i, 2] = 80 if sign == 'Gen' else 60  # both informative; just hue
        raw.append([f'{ρ:.1f}%', f'{absΔ:.1f}%', sign])
    save = os.path.join(ROOT, RQ_DIRS['RQ1'], 'heatmap', '26models.png')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    draw_heatmap(rows, cols_label, raw, strength, None, save)
    print(f'Saved: {save}')


# ============= RQ2: MLP retain =============
def make_rq2():
    cols_label = [r'$\tau$ retain (%)',
                  r'effect 100$-\tau$ (%)',
                  r'PASS gap (10$-\tau$)']
    rows = sorted(PER_MODEL.items(), key=lambda kv: kv[1]['rq2_tau'])
    n = len(rows)
    strength = np.zeros((n, 3))
    raw = []
    for i, (m, d) in enumerate(rows):
        τ = d['rq2_tau']
        eff = 100 - τ
        gap = 10 - τ
        strength[i, 0] = clamp(100 - τ, 0, 100)
        strength[i, 1] = clamp(eff, 0, 100)
        strength[i, 2] = clamp(50 + gap * 5, 0, 100)
        raw.append([f'{τ:.1f}%', f'{eff:.1f}%', f'{gap:+.1f}%'])
    save = os.path.join(ROOT, RQ_DIRS['RQ2'], 'heatmap', '26models.png')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    draw_heatmap(rows, cols_label, raw, strength, None, save)
    print(f'Saved: {save}')


# ============= RQ3: Function-token trigger =============
def make_rq3():
    cols_label = [r'$\pi(F)$',
                  r'$1-\pi(F)$',
                  r'$u_{1}$ Top-1']
    rows = sorted(PER_MODEL.items(), key=lambda kv: -kv[1]['rq3_pi_F'])
    n = len(rows)
    strength = np.zeros((n, 3))
    raw = []
    for i, (m, d) in enumerate(rows):
        π = d['rq3_pi_F']
        strength[i, 0] = π * 100
        strength[i, 1] = (1 - π) * 100
        # column 3: token category — FT/structure → high, else low
        tok = d['rq3_top1']
        is_ft = tok in {"'\\n\\n'", "' @'", "''", "' '", "' .'",
                        "'ed'", "'ky'", "' the'", "'in'", "'_'",
                        "'\\n\\n\\n'", "'The'"}
        strength[i, 2] = 90 if is_ft else 30
        raw.append([f'{π:.2f}', f'{1 - π:.2f}', tok])
    save = os.path.join(ROOT, RQ_DIRS['RQ3'], 'heatmap', '26models.png')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    draw_heatmap(rows, cols_label, raw, strength, None, save)
    print(f'Saved: {save}')


# ============= RQ4: SVD alignment =============
def make_rq4():
    cols_label = [r'$\eta=\sigma_{1}/\sigma_{2}$',
                  r'$R^{2}$ rank-1',
                  r'PASS gap']
    rows = sorted(PER_MODEL.items(), key=lambda kv: -kv[1]['rq4_R2'])
    n = len(rows)
    strength = np.zeros((n, 3))
    raw = []
    for i, (m, d) in enumerate(rows):
        η = d['rq4_eta']
        R2 = d['rq4_R2']
        gap = R2 - 0.95
        strength[i, 0] = clamp((η - 1) / 2.5 * 100, 0, 100)
        strength[i, 1] = R2 * 100
        strength[i, 2] = clamp(50 + gap * 200, 0, 100)
        raw.append([f'{η:.2f}', f'{R2:.2f}', f'{gap:+.2f}'])
    save = os.path.join(ROOT, RQ_DIRS['RQ4'], 'heatmap', '26models.png')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    draw_heatmap(rows, cols_label, raw, strength, None, save)
    print(f'Saved: {save}')


# ============= RQ5: V-ablation causality =============
def make_rq5():
    cols_label = [r'$|\Delta_{V}^{\rm sgl}|$ (%)',
                  r'$|\Delta_{V}^{\rm mac}|$ (%)',
                  r'best (%)']
    def best(d):
        s, m = d['rq5_single'], d['rq5_macro']
        vals = [abs(v) for v in (s, m) if v is not None]
        return max(vals) if vals else None
    # Drop rows where the model has neither single nor macro measurement
    pool = [(m, d) for m, d in PER_MODEL.items() if best(d) is not None]
    rows = sorted(pool, key=lambda kv: -best(kv[1]))
    n = len(rows)
    strength = np.full((n, 3), np.nan)
    raw = []
    for i, (m, d) in enumerate(rows):
        s = d['rq5_single']; mac = d['rq5_macro']
        s_abs = abs(s) if s is not None else None
        m_abs = abs(mac) if mac is not None else None
        b = best(d)
        if s_abs is not None: strength[i, 0] = clamp(s_abs, 0, 100)
        if m_abs is not None: strength[i, 1] = clamp(m_abs, 0, 100)
        strength[i, 2] = clamp(b, 0, 100)
        raw.append([
            None if s_abs is None else f'{s_abs:.0f}%',
            None if m_abs is None else f'{m_abs:.0f}%',
            f'{b:.0f}%',
        ])
    save = os.path.join(ROOT, RQ_DIRS['RQ5'], 'heatmap', '26models.png')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    draw_heatmap(rows, cols_label, raw, strength, None, save)
    print(f'Saved: {save}')


# ============= RQ6: Recovery rate =============
def make_rq6():
    cols_label = [r'$r^{*}$ (%)',
                  r'PASS gap ($r^{*}-30$)']
    pool = [(m, d) for m, d in PER_MODEL.items() if d['rq6_recovery'] is not None]
    rows = sorted(pool, key=lambda kv: -kv[1]['rq6_recovery'])
    n = len(rows)
    strength = np.zeros((n, 2))
    raw = []
    for i, (m, d) in enumerate(rows):
        r = d['rq6_recovery']
        gap = r - 30
        strength[i, 0] = clamp(r, 0, 100)
        strength[i, 1] = clamp(50 + gap * 1.5, 0, 100)
        raw.append([f'{r:.0f}%', f'{gap:+.0f}%'])
    save = os.path.join(ROOT, RQ_DIRS['RQ6'], 'heatmap', '26models.png')
    os.makedirs(os.path.dirname(save), exist_ok=True)
    draw_heatmap(rows, cols_label, raw, strength, None, save)
    print(f'Saved: {save}')


def write_missing_data_md():
    lines = ['# Heatmap missing-data report', '',
             'For each RQ, models that lack data are dropped from the heatmap.',
             'Within RQ5 a sub-metric column may be blank even when the row is shown.',
             '']

    # RQ1-4: all 26 models present (no row drops)
    for rq, key, label in [
        ('RQ1', 'rq1_residual', 'attention residual ρ'),
        ('RQ2', 'rq2_tau',      'MLP retain τ'),
        ('RQ3', 'rq3_pi_F',     'function-token trigger π(F)'),
        ('RQ4', 'rq4_R2',       'rank-1 fit R²'),
    ]:
        miss = [m for m, d in PER_MODEL.items() if d[key] is None]
        lines.append(f'## {rq} — {label}')
        lines.append(f'Rows shown: **{len(PER_MODEL) - len(miss)}/26**.')
        lines.append('Missing: ' + ('none ✓' if not miss else ', '.join(miss)))
        lines.append('')

    # RQ5: row drop iff both single AND macro None; sub-cell N/A listed too
    miss_row5, miss_sgl, miss_mac = [], [], []
    for m, d in PER_MODEL.items():
        s, mac = d['rq5_single'], d['rq5_macro']
        if s is None and mac is None: miss_row5.append(m)
        if s is None: miss_sgl.append(m)
        if mac is None: miss_mac.append(m)
    lines.append('## RQ5 — V-ablation (single + macro)')
    lines.append(f'Rows shown: **{len(PER_MODEL) - len(miss_row5)}/26**.')
    lines.append('Missing entire row (no single & no macro): ' +
                 ('none ✓' if not miss_row5 else ', '.join(miss_row5)))
    lines.append(f'Single-V N/A ({len(miss_sgl)} models): ' + ', '.join(miss_sgl))
    lines.append(f'Macro-V N/A ({len(miss_mac)} models): ' + ', '.join(miss_mac))
    lines.append('')

    # RQ6: row drop iff recovery is None
    miss6 = [m for m, d in PER_MODEL.items() if d['rq6_recovery'] is None]
    lines.append('## RQ6 — single-layer recovery r*')
    lines.append(f'Rows shown: **{len(PER_MODEL) - len(miss6)}/26**.')
    lines.append(f'Missing ({len(miss6)} models): ' + ', '.join(miss6))
    lines.append('')

    out = os.path.join(ROOT, 'MISSING_DATA.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Saved: {out}')


def main():
    # Drop the deprecated combined 6×26 figure
    legacy = os.path.join(ROOT, 'summary')
    if os.path.isdir(legacy):
        import shutil
        shutil.rmtree(legacy)
        print(f'Removed: {legacy}')

    make_rq1()
    make_rq2()
    make_rq3()
    make_rq4()
    make_rq5()
    make_rq6()
    write_missing_data_md()


if __name__ == '__main__':
    main()
