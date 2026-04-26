"""
Generate 156 = 6 RQ × 26 models per-model × per-RQ panels.
Each panel: a self-contained card showing the key metric, a small visualisation,
and a PASS/FAIL banner. Saved to:
    final_experiments/<RQ_dir>/figures/<model_folder>/<RQ>_<model>.png

Run:
    python gen_per_model_per_rq.py
"""
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from _style import (apply, REG, P_BLUE, P_TEAL, P_CORAL,
                    P_AMBER, P_PURPLE, P_DARK, P_RED)
from _per_model_data import PER_MODEL, NAME_TO_FOLDER, PASS_FN

apply()

ROOT = '/Users/a1-6/importantfile/Research/ma/4.26/MA_NeurIPS2026/figures'
RQ_DIRS = {
    'RQ1': 'RQ1_attention',
    'RQ2': 'RQ2_mlp_source',
    'RQ3': 'RQ3_function_words',
    'RQ4': 'RQ4_svd_alignment',
    'RQ5': 'RQ5_v_ablation',
    'RQ6': 'RQ6_topk_scan',
}
# Per-RQ panel type folder name (the kind of plot stored inside)
RQ_PANEL_TYPE = {
    'RQ1': 'attention_ablation',
    'RQ2': 'mlp_retain',
    'RQ3': 'function_token_pi',
    'RQ4': 'svd_eta_R2',
    'RQ5': 'v_ablation',
    'RQ6': 'recovery_rate',
}
RQ_TITLES = {
    'RQ1': 'attention is not the MA generator',
    'RQ2': 'MLP is the MA generator',
    'RQ3': 'function tokens trigger MA',
    'RQ4': 'SVD expansion mechanism',
    'RQ5': '$v_{1}$ causality',
    'RQ6': 'single-layer sufficiency',
}


def header(ax, model, rq, passed):
    """PASS/FAIL pill only — no title (per user spec)."""
    color = '#3FA467' if passed else '#C0392B'
    label = 'PASS' if passed else 'FAIL'
    ax.text(0.985, 1.04, label, transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', color='white',
            ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.30', fc=color, ec='none'),
            clip_on=False)


def fig_rq1(d, model, save_path):
    """Residual MA after attn off + ΔMA direction."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ρ = d['rq1_residual']
    sign = d['rq1_sign']
    delta = ρ - 100  # ΔMA% relative to baseline

    bars = ax.barh([0, 1], [100, ρ],
                    color=['#CCCCCC', P_BLUE if sign == 'Gen' else P_CORAL],
                    edgecolor='white', linewidth=1.0, height=0.55)
    ax.text(100, 0, ' baseline (100%)', va='center', fontsize=8, color='#666')
    ax.text(ρ + 5, 1, f' ρ={ρ:.1f}%  (Δ={delta:+.1f}%, {sign})',
            va='center', fontsize=8.5, color='#222', fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['baseline', 'attn off'], fontsize=8)
    ax.set_xscale('symlog')
    ax.set_xlim(0, max(150, ρ * 1.5))
    ax.set_xlabel(r'MA top-1 magnitude (% of baseline)')
    header(ax, model, 'RQ1', PASS_FN['RQ1'](d))
    fig.savefig(save_path)
    plt.close(fig)


def fig_rq2(d, model, save_path):
    """Retain τ% after MLP global ablation; PASS if τ ≤ 10%."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    τ = d['rq2_tau']
    color = P_TEAL if τ <= 10 else P_CORAL
    ax.barh([0], [100], color='#EAEAEA', edgecolor='white', height=0.5)
    ax.barh([0], [τ], color=color, edgecolor='white', height=0.5)
    ax.axvline(10, color=P_RED, linewidth=1.2, linestyle='--', alpha=0.85)
    ax.text(10, 0.4, ' PASS  τ ≤ 10%', fontsize=8, color=P_RED,
            ha='left', style='italic', fontweight='bold')
    ax.text(τ / 2 if τ > 8 else τ + 3, 0,
            f'τ = {τ:.1f}%', ha='center', va='center',
            fontsize=14, fontweight='bold',
            color='white' if τ > 8 else '#222')
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel(r'Retain $\tau = \mathrm{MA}_{\rm MLP{-}off}/\mathrm{MA}_{\rm baseline}$ (%)')
    header(ax, model, 'RQ2', PASS_FN['RQ2'](d))
    fig.savefig(save_path)
    plt.close(fig)


def fig_rq3(d, model, save_path):
    """π(F) = function-token trigger probability + Top-1 token."""
    fig = plt.figure(figsize=(5.0, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.35,
                           left=0.10, right=0.96, top=0.83, bottom=0.18)

    # left: π(F) gauge
    axL = fig.add_subplot(gs[0, 0])
    π = d['rq3_pi_F']
    color = P_BLUE if π >= 0.5 else P_CORAL
    axL.barh([0], [1.0], color='#EAEAEA', edgecolor='white', height=0.5)
    axL.barh([0], [π], color=color, edgecolor='white', height=0.5)
    axL.axvline(0.5, color=P_RED, linewidth=1.2, linestyle='--', alpha=0.85)
    axL.text(0.51, -0.45, 'PASS  $\\pi(F)\\geq 0.5$', fontsize=7.5,
             color=P_RED, style='italic', va='top')
    axL.text(π / 2 if π > 0.18 else π + 0.04, 0,
             f'π(F)\n{π:.2f}', ha='center', va='center',
             fontsize=11, fontweight='bold',
             color='white' if π > 0.18 else '#222')
    axL.set_xlim(0, 1.05)
    axL.set_ylim(-0.6, 0.4)
    axL.set_yticks([])
    axL.set_xlabel(r'$\pi(F)$')

    # right: Top-1 token card
    axR = fig.add_subplot(gs[0, 1])
    axR.axis('off')
    bg = mpatches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                  boxstyle='round,pad=0.02',
                                  fc='#F8F8F8', ec='#DDDDDD', linewidth=0.8,
                                  transform=axR.transAxes)
    axR.add_patch(bg)
    axR.text(0.5, 0.78, '$u_{1}$ Top-1', ha='center', fontsize=8.5,
             color='#666', transform=axR.transAxes)
    axR.text(0.5, 0.42, d['rq3_top1'], ha='center', fontsize=14,
             fontweight='bold', color=P_DARK, family='monospace',
             transform=axR.transAxes)
    axR.text(0.5, 0.15, 'function token / structure', ha='center',
             fontsize=7, color='#888', style='italic',
             transform=axR.transAxes)

    passed = PASS_FN['RQ3'](d)
    fig.text(0.96, 0.94,
             'PASS' if passed else 'FAIL',
             ha='right', va='center', fontsize=9.5, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.30',
                       fc='#3FA467' if passed else '#C0392B', ec='none'))
    fig.savefig(save_path)
    plt.close(fig)


def fig_rq4(d, model, save_path):
    """η + R² dual gauge."""
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 3.0))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.78, bottom=0.18, wspace=0.40)

    eta = d['rq4_eta']; R2 = d['rq4_R2']

    # η bar
    axes[0].barh([0], [4.0], color='#EAEAEA', edgecolor='white', height=0.5)
    axes[0].barh([0], [min(eta, 4.0)], color=P_PURPLE, edgecolor='white', height=0.5)
    axes[0].axvline(2.0, color='#999', linewidth=0.8, linestyle=':', alpha=0.7)
    axes[0].text(eta / 2, 0, f'η\n{eta:.2f}', ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white' if eta > 0.7 else '#222')
    axes[0].set_xlim(0, 4.0)
    axes[0].set_ylim(-0.6, 0.4)
    axes[0].set_yticks([])
    axes[0].set_xlabel(r'$\eta = \sigma_{1}/\sigma_{2}$')

    # R² bar
    color = P_TEAL if R2 >= 0.95 else (P_AMBER if R2 >= 0.6 else P_CORAL)
    axes[1].barh([0], [1.0], color='#EAEAEA', edgecolor='white', height=0.5)
    axes[1].barh([0], [R2], color=color, edgecolor='white', height=0.5)
    axes[1].axvline(0.95, color=P_RED, linewidth=1.2, linestyle='--', alpha=0.85)
    axes[1].text(0.95, -0.45, 'PASS\n0.95', fontsize=7.5, color=P_RED,
                 style='italic', ha='center', va='top')
    axes[1].text(R2 / 2 if R2 > 0.18 else R2 + 0.04, 0, f'$R^{{2}}$\n{R2:.2f}',
                 ha='center', va='center',
                 fontsize=11, fontweight='bold',
                 color='white' if R2 > 0.18 else '#222')
    axes[1].set_xlim(0, 1.05)
    axes[1].set_ylim(-0.6, 0.4)
    axes[1].set_yticks([])
    axes[1].set_xlabel(r'Rank-1 fit $R^{2}$')

    passed = PASS_FN['RQ4'](d)
    fig.text(0.96, 0.94,
             'PASS' if passed else 'FAIL',
             ha='right', va='center', fontsize=9.5, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.30',
                       fc='#3FA467' if passed else '#C0392B', ec='none'))
    fig.savefig(save_path)
    plt.close(fig)


def fig_rq5(d, model, save_path):
    """V-ablation single + macro ΔMA% bars."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    labels = ['Single-V', 'Macro-V']
    vals = [d['rq5_single'], d['rq5_macro']]
    show_v = [v if v is not None else 0 for v in vals]
    colors = []
    for v in vals:
        if v is None: colors.append('#CCCCCC')
        elif v <= -80: colors.append(P_PURPLE)
        else: colors.append(P_AMBER)

    x = np.arange(len(labels))
    ax.bar(x, show_v, color=colors, edgecolor='white', linewidth=1.0, width=0.55)
    for i, v in enumerate(vals):
        if v is None:
            ax.text(i, 5, 'N/A', ha='center', fontsize=9, color='#666', fontstyle='italic')
        else:
            ax.text(i, v - 4 if v < -10 else v + 3, f'{v:+.1f}%',
                    ha='center',
                    va='top' if v < -10 else 'bottom',
                    fontsize=9.5, fontweight='bold', color='#222')
    ax.axhline(-80, color=P_RED, linewidth=1.2, linestyle='--', alpha=0.85)
    ax.text(1.4, -80 + 3, 'PASS  Δ ≤ -80%', fontsize=7.5, color=P_RED,
            ha='right', style='italic')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r'$\Delta_{V}\,\mathrm{MA}$ (%)')
    ax.set_ylim(-110, 15)
    header(ax, model, 'RQ5', PASS_FN['RQ5'](d))
    fig.savefig(save_path)
    plt.close(fig)


def fig_rq6(d, model, save_path):
    """Recovery rate r* gauge."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    r = d['rq6_recovery']
    ax.barh([0], [100], color='#EAEAEA', edgecolor='white', height=0.5)
    if r is None:
        ax.text(50, 0, 'not measured', ha='center', va='center',
                fontsize=12, color='#888', fontstyle='italic')
    else:
        color = P_TEAL if r >= 30 else P_CORAL
        ax.barh([0], [r], color=color, edgecolor='white', height=0.5)
        ax.text(r / 2 if r >= 12 else 8, 0,
                f'$r^{{*}}$ = {r:.1f}%', ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white' if r >= 12 else '#222')
    ax.axvline(30, color=P_RED, linewidth=1.2, linestyle='--', alpha=0.85)
    ax.text(31, -0.45, 'PASS  $r^{*}\\geq 30\\%$', fontsize=7.5, color=P_RED,
            style='italic', fontweight='bold', va='top')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(r'Recovery rate $r^{*}$ (Eq. 10)')
    header(ax, model, 'RQ6', PASS_FN['RQ6'](d))
    fig.savefig(save_path)
    plt.close(fig)


GENERATORS = {
    'RQ1': fig_rq1, 'RQ2': fig_rq2, 'RQ3': fig_rq3,
    'RQ4': fig_rq4, 'RQ5': fig_rq5, 'RQ6': fig_rq6,
}


def main():
    n = 0
    for model_name, d in PER_MODEL.items():
        slug = NAME_TO_FOLDER[model_name]
        for rq, gen in GENERATORS.items():
            outdir = os.path.join(ROOT, RQ_DIRS[rq], RQ_PANEL_TYPE[rq])
            os.makedirs(outdir, exist_ok=True)
            out = os.path.join(outdir, f'{slug}.png')
            gen(d, model_name, out)
            n += 1
    print(f'Generated {n} figures across 6 RQs × 26 models')


if __name__ == '__main__':
    main()
