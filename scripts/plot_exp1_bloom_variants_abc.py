#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _load_layerwise_json(path: Path) -> dict[int, dict]:
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    out: dict[int, dict] = {}
    for k, v in raw.items():
        try:
            layer_id = int(k)
        except ValueError:
            continue
        out[layer_id] = v
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _setup_academic_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.color': '#CCCCCC',
        'grid.alpha': 0.5,
    })

    sns.set_palette('deep')
    sns.set_style('whitegrid', {
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _legend_ul(ax):
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=False,
        shadow=False,
        framealpha=0.95,
        edgecolor='#BBBBBB',
        borderpad=0.25,
        labelspacing=0.25,
        handletextpad=0.5,
        handlelength=1.8,
        markerscale=0.9,
        prop={'size': 9},
    )


def _default_xticks(layers: list[int]) -> list[int]:
    if not layers:
        return []
    max_l = max(layers)
    if max_l <= 8:
        return layers
    ticks = [0, 5, 10, 15, 20, 25, max_l]
    ticks = [t for t in ticks if t in layers]
    if ticks and ticks[-1] != max_l:
        ticks.append(max_l)
    return sorted(set(ticks))


def _plot_overlay(ax, layers, baseline_y, suppressed_y, baseline_std, suppressed_std, baseline_label, suppressed_label):
    baseline_color = '#377eb8'
    suppressed_color = '#e41a1c'

    ax.plot(layers, baseline_y, marker='o', markersize=4.5, markerfacecolor='white',
            label=baseline_label, color=baseline_color)
    ax.plot(layers, suppressed_y, marker='s', markersize=4.5, markerfacecolor='white',
            label=suppressed_label, color=suppressed_color)

    if baseline_std is not None and np.isfinite(baseline_std).any():
        ax.fill_between(layers, baseline_y - baseline_std, baseline_y + baseline_std,
                        color=baseline_color, alpha=0.15, linewidth=0)
    if suppressed_std is not None and np.isfinite(suppressed_std).any():
        ax.fill_between(layers, suppressed_y - suppressed_std, suppressed_y + suppressed_std,
                        color=suppressed_color, alpha=0.15, linewidth=0)


def plot_a_log_overlay(layers, baseline_y, suppressed_y, baseline_std, suppressed_std, xlabel, ylabel, baseline_label, suppressed_label, outdir: Path, stem: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)
    _plot_overlay(ax, layers, baseline_y, suppressed_y, baseline_std, suppressed_std, baseline_label, suppressed_label)

    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    xt = _default_xticks(layers)
    ax.set_xticks(xt)
    ax.margins(x=0.01)
    _legend_ul(ax)

    plt.tight_layout()
    png_path = outdir / f"{stem}_A_logy.png"
    pdf_path = outdir / f"{stem}_A_logy.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def plot_b_dual_panel(layers, baseline_y, suppressed_y, baseline_std, suppressed_std, xlabel, ylabel, baseline_label, suppressed_label, outdir: Path, stem: str):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.2), dpi=300, sharex=True)

    ax0, ax1 = axes
    _plot_overlay(ax0, layers, baseline_y, baseline_y, baseline_std, None, baseline_label, None)
    ax0.lines[-1].remove()
    ax0.set_ylabel(ylabel, fontweight='bold')
    ax0.set_title('Baseline', fontweight='bold')

    _plot_overlay(ax1, layers, suppressed_y, suppressed_y, suppressed_std, None, suppressed_label, None)
    ax1.lines[-1].remove()
    ax1.set_xlabel(xlabel, fontweight='bold')
    ax1.set_ylabel(ylabel, fontweight='bold')
    ax1.set_title('Head Suppression', fontweight='bold')

    xt = _default_xticks(layers)
    ax1.set_xticks(xt)
    ax0.margins(x=0.01)
    ax1.margins(x=0.01)

    for ax in axes:
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.95,
            edgecolor='#BBBBBB',
            borderpad=0.25,
            labelspacing=0.25,
            handletextpad=0.5,
            handlelength=1.8,
            markerscale=0.9,
            prop={'size': 9},
        )

    plt.tight_layout()
    png_path = outdir / f"{stem}_B_dualpanel.png"
    pdf_path = outdir / f"{stem}_B_dualpanel.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def plot_c_ratio(layers, baseline_y, suppressed_y, xlabel, outdir: Path, stem: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)

    baseline_color = '#377eb8'
    suppressed_color = '#e41a1c'

    eps = 1e-12
    ratio = suppressed_y / np.maximum(baseline_y, eps)

    ax.plot(layers, ratio, marker='o', markersize=4.0, markerfacecolor='white',
            color=suppressed_color, label='Suppressed / Baseline')
    ax.axhline(1.0, color=baseline_color, linestyle='--', linewidth=1.2, alpha=0.8, label='No change (1.0)')

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Ratio', fontweight='bold')

    xt = _default_xticks(layers)
    ax.set_xticks(xt)
    ax.margins(x=0.01)
    _legend_ul(ax)

    plt.tight_layout()
    png_path = outdir / f"{stem}_C_ratio.png"
    pdf_path = outdir / f"{stem}_C_ratio.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return png_path, pdf_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--suppressed', required=True)
    parser.add_argument('--metric', default='top1_mean')
    parser.add_argument('--metric_std', default=None)
    parser.add_argument('--xlabel', default='Layer')
    parser.add_argument('--ylabel', default=None)
    parser.add_argument('--baseline_label', default='Baseline')
    parser.add_argument('--suppressed_label', default='All Heads Disabled')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--outfile_stem', default='bloom_7b1_exp1_baseline_vs_head_suppression')
    args = parser.parse_args()

    baseline = _load_layerwise_json(Path(args.baseline))
    suppressed = _load_layerwise_json(Path(args.suppressed))

    layers = sorted(set(baseline.keys()) & set(suppressed.keys()))
    if not layers:
        raise RuntimeError('No overlapping layer ids found between baseline and suppressed results')

    baseline_y = np.array([baseline[l][args.metric] for l in layers], dtype=float)
    suppressed_y = np.array([suppressed[l][args.metric] for l in layers], dtype=float)

    baseline_std = None
    suppressed_std = None
    if args.metric_std is not None:
        baseline_std = np.array([baseline[l].get(args.metric_std, np.nan) for l in layers], dtype=float)
        suppressed_std = np.array([suppressed[l].get(args.metric_std, np.nan) for l in layers], dtype=float)

    _setup_academic_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ylabel = args.ylabel or args.metric

    a_png, a_pdf = plot_a_log_overlay(
        layers, baseline_y, suppressed_y, baseline_std, suppressed_std,
        args.xlabel, ylabel, args.baseline_label, args.suppressed_label,
        outdir, args.outfile_stem,
    )
    b_png, b_pdf = plot_b_dual_panel(
        layers, baseline_y, suppressed_y, baseline_std, suppressed_std,
        args.xlabel, ylabel, args.baseline_label, args.suppressed_label,
        outdir, args.outfile_stem,
    )
    c_png, c_pdf = plot_c_ratio(
        layers, baseline_y, suppressed_y,
        args.xlabel, outdir, args.outfile_stem,
    )

    print(str(a_png))
    print(str(a_pdf))
    print(str(b_png))
    print(str(b_pdf))
    print(str(c_png))
    print(str(c_pdf))


if __name__ == '__main__':
    main()
