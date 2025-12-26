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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--suppressed', required=True)
    parser.add_argument('--metric', default='top1_mean')
    parser.add_argument('--metric_std', default=None)
    parser.add_argument('--xlabel', default='Layer')
    parser.add_argument('--ylabel', default=None)
    parser.add_argument('--title', default=None)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--outfile_stem', default=None)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    suppressed_path = Path(args.suppressed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = _load_layerwise_json(baseline_path)
    suppressed = _load_layerwise_json(suppressed_path)

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

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)

    baseline_color = '#377eb8'
    suppressed_color = '#e41a1c'

    ax.plot(layers, baseline_y, marker='o', markersize=4.5, markerfacecolor='white',
            label='Baseline', color=baseline_color)
    ax.plot(layers, suppressed_y, marker='s', markersize=4.5, markerfacecolor='white',
            label='Head Suppression', color=suppressed_color)

    if baseline_std is not None and np.isfinite(baseline_std).any():
        ax.fill_between(layers, baseline_y - baseline_std, baseline_y + baseline_std,
                        color=baseline_color, alpha=0.15, linewidth=0)
    if suppressed_std is not None and np.isfinite(suppressed_std).any():
        ax.fill_between(layers, suppressed_y - suppressed_std, suppressed_y + suppressed_std,
                        color=suppressed_color, alpha=0.15, linewidth=0)

    ax.set_xlabel(args.xlabel, fontweight='bold')
    ax.set_ylabel(args.ylabel or args.metric, fontweight='bold')

    if args.title is not None and args.title != "":
        ax.set_title(args.title, fontweight='bold')

    ax.set_xticks(layers)
    ax.margins(x=0.01)
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
        prop={'size': 9}
    )

    plt.tight_layout()

    if args.outfile_stem:
        stem = args.outfile_stem
    else:
        stem = f"bloom_7b1_exp1_baseline_vs_head_suppression_{args.metric}"

    png_path = outdir / f"{stem}.png"
    pdf_path = outdir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(str(png_path))
    print(str(pdf_path))


if __name__ == '__main__':
    main()
