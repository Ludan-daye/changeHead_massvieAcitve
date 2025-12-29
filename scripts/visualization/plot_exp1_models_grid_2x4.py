#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'axes.edgecolor': '#777777',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'grid.color': '#CCCCCC',
        'grid.alpha': 0.25,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.pad_inches': 0.02,
    })

    sns.set_palette('deep')
    sns.set_style('white', {
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


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


def _plot_single_panel(ax, layers, baseline_y, suppressed_y, baseline_std, suppressed_std, title, xlabel, ylabel, show_xlabel, show_ylabel):
    palette = sns.color_palette('Set2', 2)
    baseline_color = palette[0]
    suppressed_color = palette[1]

    lw = 2.0
    ms = 5.0

    # Keep lines dense but draw sparse markers (<=10 markers per subplot total)
    n = len(layers)
    markers_per_line = min(5, n) if n > 0 else 0
    if markers_per_line > 0:
        mark_idx = np.unique(np.round(np.linspace(0, n - 1, markers_per_line)).astype(int)).tolist()
    else:
        mark_idx = None

    ax.plot(
        layers,
        baseline_y,
        label='Baseline',
        color=baseline_color,
        linewidth=lw,
        linestyle='-',
        marker='o',
        markersize=ms,
        markevery=mark_idx,
        markerfacecolor=baseline_color,
        markeredgecolor=baseline_color,
        markeredgewidth=0.0,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        layers,
        suppressed_y,
        label='All Heads Disabled',
        color=suppressed_color,
        linewidth=lw,
        linestyle='-',
        marker='o',
        markersize=ms,
        markevery=mark_idx,
        markerfacecolor=suppressed_color,
        markeredgecolor=suppressed_color,
        markeredgewidth=0.0,
        alpha=0.85,
        zorder=2,
    )

    ax.set_title(title, fontweight='bold', pad=4)

    # X ticks every 3 layers
    if layers:
        last_layer = max(layers)
        xticks = [x for x in range(0, last_layer + 1, 3) if x in layers]
        if last_layer not in xticks:
            if xticks and (last_layer - xticks[-1] < 3):
                # Prevent overlap like "..., 27, 29" by replacing the last tick
                xticks[-1] = last_layer
            else:
                xticks.append(last_layer)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], rotation=0, fontsize=8)
    ax.tick_params(axis='x', which='major', pad=1)
    ax.tick_params(direction='out', length=3, width=1.0)

    if show_xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    else:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_models_dir', default=None)
    parser.add_argument('--metric', default='top1_mean')
    parser.add_argument('--metric_std', default='top1_std')
    parser.add_argument('--xlabel', default='Layer')
    parser.add_argument('--ylabel', default='Top-1 activation (mean)')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--outfile_stem', default='exp1_top1_mean_models_grid_2x4')
    parser.add_argument('--single_plots', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    models_dir = Path(args.results_models_dir) if args.results_models_dir else (base_dir / 'results' / 'models')

    model_specs = [
        ('gptj_6b', 'exp1', 'GPT-J-6B'),
        ('bloom_7b1', 'exp1', 'BLOOM-7B1'),
        ('qwen2.5_7b', 'exp1', 'Qwen-2.5-7B'),
        ('falcon_7b', 'exp1', 'Falcon-7B'),
        ('mistral_7b_v03', 'exp1', 'Mistral-7B'),
        ('opt_6.7b', 'exp1_opt_6.7b', 'OPT-6.7B'),
        ('gpt2', 'exp1_feasibility_test', 'GPT-2'),
        ('llama2_13b', 'exp1_llama2_13b', 'LLaMA-2-13B'),
    ]

    _setup_academic_style()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.single_plots:
        single_outdir = outdir / "exp1_figures"
        single_outdir.mkdir(parents=True, exist_ok=True)
        for (model_name, exp1_dir, title) in model_specs:
            base_path = models_dir / model_name / exp1_dir
            baseline_path = base_path / 'baseline' / 'results.json'
            suppressed_path = base_path / 'all_heads_disabled' / 'results.json'

            baseline = _load_layerwise_json(baseline_path)
            suppressed = _load_layerwise_json(suppressed_path)

            layers = sorted(set(baseline.keys()) & set(suppressed.keys()))
            baseline_y = np.array([baseline[l][args.metric] for l in layers], dtype=float)
            suppressed_y = np.array([suppressed[l][args.metric] for l in layers], dtype=float)

            baseline_std = None
            suppressed_std = None
            if args.metric_std is not None:
                baseline_std = np.array([baseline[l].get(args.metric_std, np.nan) for l in layers], dtype=float)
                suppressed_std = np.array([suppressed[l].get(args.metric_std, np.nan) for l in layers], dtype=float)

            fig, ax = plt.subplots(1, 1, figsize=(4.6, 3.4), dpi=300)
            _plot_single_panel(
                ax,
                layers,
                baseline_y,
                suppressed_y,
                baseline_std,
                suppressed_std,
                title,
                args.xlabel,
                args.ylabel,
                True,
                True,
            )

            fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.90)

            out_stem = f"{model_name}_exp1_{args.metric}_baseline_vs_all_heads_disabled"
            png_path = single_outdir / f"{out_stem}.png"
            pdf_path = single_outdir / f"{out_stem}.pdf"
            fig.savefig(png_path, dpi=400, bbox_inches='tight')
            fig.savefig(pdf_path, bbox_inches='tight')
            plt.close(fig)

            print(str(png_path))
            print(str(pdf_path))

        return

    # Preload all panels to compute a shared y-range for easier cross-model comparison
    panels = []
    global_min = np.inf
    global_max = -np.inf
    for (model_name, exp1_dir, title) in model_specs:
        base_path = models_dir / model_name / exp1_dir
        baseline_path = base_path / 'baseline' / 'results.json'
        suppressed_path = base_path / 'all_heads_disabled' / 'results.json'

        baseline = _load_layerwise_json(baseline_path)
        suppressed = _load_layerwise_json(suppressed_path)

        layers = sorted(set(baseline.keys()) & set(suppressed.keys()))
        baseline_y = np.array([baseline[l][args.metric] for l in layers], dtype=float)
        suppressed_y = np.array([suppressed[l][args.metric] for l in layers], dtype=float)

        panels.append((title, layers, baseline_y, suppressed_y, baseline, suppressed))

        vals = np.concatenate([baseline_y, suppressed_y])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            global_min = min(global_min, float(vals.min()))
            global_max = max(global_max, float(vals.max()))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        global_min, global_max = 0.0, 1.0
    yr = global_max - global_min
    pad = 0.06 * yr if yr > 0 else 1e-6
    shared_ylim = (global_min - pad, global_max + pad)

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.5), dpi=300)

    handles = None
    labels = None

    for idx, (title, layers, baseline_y, suppressed_y, baseline, suppressed) in enumerate(panels):
        r = idx // 4
        c = idx % 4
        ax = axes[r, c]

        baseline_std = None
        suppressed_std = None
        if args.metric_std is not None:
            baseline_std = np.array([baseline[l].get(args.metric_std, np.nan) for l in layers], dtype=float)
            suppressed_std = np.array([suppressed[l].get(args.metric_std, np.nan) for l in layers], dtype=float)

        show_xlabel = (r == 1)
        show_ylabel = (c == 0)

        _plot_single_panel(
            ax,
            layers,
            baseline_y,
            suppressed_y,
            baseline_std,
            suppressed_std,
            title,
            args.xlabel,
            args.ylabel,
            show_xlabel,
            show_ylabel,
        )

        ax.set_ylim(shared_ylim)

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.legend().remove()

    if handles is not None and labels is not None:
        fig.legend(
            handles,
            labels,
            loc='upper center',
            ncol=2,
            frameon=False,
            fancybox=False,
            shadow=False,
            borderpad=0.25,
            labelspacing=0.25,
            handletextpad=0.6,
            handlelength=2.0,
            prop={'size': 10},
            bbox_to_anchor=(0.5, 0.995),
        )

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.10, top=0.90, wspace=0.25, hspace=0.35)

    png_path = outdir / f"{args.outfile_stem}.png"
    pdf_path = outdir / f"{args.outfile_stem}.pdf"

    fig.savefig(png_path, dpi=400, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)

    print(str(png_path))
    print(str(pdf_path))


if __name__ == '__main__':
    main()
