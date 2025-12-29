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


def _setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#777777',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'savefig.pad_inches': 0.02,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
    })
    sns.set_style('white', {
        'axes.grid': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
    })


def _load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _load_layerwise_results(path: Path) -> dict[int, dict]:
    raw = _load_json(path)
    out: dict[int, dict] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                lid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                out[lid] = v
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _parse_layer_contribution(raw: dict) -> tuple[dict[int, float], str | None]:
    model_title = raw.get('model') if isinstance(raw, dict) else None

    if isinstance(raw, dict) and 'layer_contribution' in raw and isinstance(raw['layer_contribution'], dict):
        contrib = raw['layer_contribution']
    elif isinstance(raw, dict) and 'layer_contributions' in raw and isinstance(raw['layer_contributions'], dict):
        contrib = raw['layer_contributions']
    else:
        contrib = raw

    out: dict[int, float] = {}
    if isinstance(contrib, dict):
        for k, v in contrib.items():
            try:
                lid = int(k)
            except Exception:
                continue
            try:
                out[lid] = float(v)
            except Exception:
                continue

    return out, model_title


def _find_exp1_variant_path(model_dir: Path, variant: str) -> Path:
    p = model_dir / 'exp1' / variant / 'results.json'
    if p.exists():
        return p

    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith('exp1'):
            continue
        cand = child / variant / 'results.json'
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Could not locate exp1 {variant} results.json under {model_dir}")


def _marker_indices(n: int, max_markers: int = 10) -> list[int] | None:
    if n <= 0:
        return None
    m = min(max_markers, n)
    idx = np.unique(np.round(np.linspace(0, n - 1, m)).astype(int)).tolist()
    return idx


def plot_model(model_name: str, exp2_path: Path, exp1_baseline_path: Path, exp1_disabled_path: Path, outdir: Path, metric: str, logy: bool):
    exp2_raw = _load_json(exp2_path)
    contrib_map, title = _parse_layer_contribution(exp2_raw)
    if not contrib_map:
        raise ValueError(f"Empty layer contribution in {exp2_path}")

    baseline = _load_layerwise_results(exp1_baseline_path)
    disabled = _load_layerwise_results(exp1_disabled_path)
    if not baseline or not disabled:
        raise ValueError(f"Empty exp1 results for {model_name}")

    last_layer = max(baseline.keys())
    target_layer = max(0, last_layer - 1)

    if target_layer not in baseline or metric not in baseline[target_layer]:
        raise KeyError(f"Metric {metric} missing in baseline for layer {target_layer} ({exp1_baseline_path})")
    if target_layer not in disabled or metric not in disabled[target_layer]:
        raise KeyError(f"Metric {metric} missing in all_heads_disabled for layer {target_layer} ({exp1_disabled_path})")

    baseline_val = float(baseline[target_layer][metric])
    disabled_val = float(disabled[target_layer][metric])

    layers = sorted(contrib_map.keys())
    y = np.array([contrib_map[l] for l in layers], dtype=float)

    best_layer = int(layers[int(np.nanargmin(y))])
    best_val = float(np.nanmin(y))

    denom = (disabled_val - baseline_val)
    if abs(denom) < 1e-9:
        suppression = np.full_like(y, np.nan, dtype=float)
    else:
        suppression = (disabled_val - y) / denom * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.8), dpi=300, sharex=True, gridspec_kw={'height_ratios': [2.1, 1.0]})
    ax0, ax1 = axes

    palette = sns.color_palette('Set2', 3)
    mark_idx = _marker_indices(len(layers), max_markers=9)

    ax0.plot(layers, y, color=palette[0], linewidth=2.0, marker='o', markersize=4.6, markevery=mark_idx, alpha=0.9)

    ax0.axhline(baseline_val, color=palette[1], linewidth=1.6, linestyle='-')
    ax0.axhline(disabled_val, color=palette[2], linewidth=1.6, linestyle='-')

    ax0.scatter([best_layer], [best_val], color=palette[0], s=28, zorder=3)
    ax0.text(best_layer, best_val, f"  min@L{best_layer}", va='center', ha='left', fontsize=9)

    if logy:
        ax0.set_yscale('log')

    ax0.set_ylabel('Massive activation (Top1)')
    ax0.yaxis.set_major_locator(MaxNLocator(nbins=4))

    y0_min, y0_max = ax0.get_ylim()
    ax0.text(0.01, baseline_val, 'Baseline', va='bottom', ha='left', fontsize=9, color=palette[1], transform=ax0.get_yaxis_transform())
    ax0.text(0.01, disabled_val, 'All-heads-disabled', va='bottom', ha='left', fontsize=9, color=palette[2], transform=ax0.get_yaxis_transform())
    ax0.set_title(title or model_name, fontweight='bold', pad=4)

    ax1.plot(layers, suppression, color='#444444', linewidth=1.8, marker='o', markersize=4.2, markevery=mark_idx, alpha=0.9)
    ax1.axhline(0.0, color='#999999', linewidth=1.0)
    ax1.axhline(100.0, color='#999999', linewidth=1.0)
    ax1.set_ylabel('Suppression (%)')
    ax1.set_xlabel('Restored layer')
    ax1.set_ylim(-5, 105)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=3))

    if layers:
        last = max(layers)
        xticks = [x for x in range(0, last + 1, 3) if x in layers]
        if last not in xticks:
            if xticks and (last - xticks[-1] < 3):
                xticks[-1] = last
            else:
                xticks.append(last)
        ax1.set_xticks(xticks)

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.90, hspace=0.12)

    out_stem = f"{model_name}_exp2_layer_curve_vs_exp1_reference_{metric}"
    out_png = outdir / f"{out_stem}.png"
    out_pdf = outdir / f"{out_stem}.pdf"
    fig.savefig(out_png, dpi=400, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)

    print(str(out_png))
    print(str(out_pdf))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_models_dir', default=None)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--metric', default='top1_mean')
    parser.add_argument('--logy', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    models_dir = Path(args.results_models_dir) if args.results_models_dir else (base_dir / 'results' / 'models')

    outdir = Path(args.outdir) / 'exp2_figures'
    outdir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    exp2_files = sorted(models_dir.glob('*/exp2/layer_contribution.json'))
    if not exp2_files:
        raise FileNotFoundError(f"No exp2 layer_contribution.json found under {models_dir}")

    for exp2_path in exp2_files:
        model_dir = exp2_path.parent.parent
        model_name = model_dir.name

        try:
            exp1_baseline_path = _find_exp1_variant_path(model_dir, 'baseline')
            exp1_disabled_path = _find_exp1_variant_path(model_dir, 'all_heads_disabled')
        except FileNotFoundError:
            continue

        plot_model(
            model_name=model_name,
            exp2_path=exp2_path,
            exp1_baseline_path=exp1_baseline_path,
            exp1_disabled_path=exp1_disabled_path,
            outdir=outdir,
            metric=args.metric,
            logy=args.logy,
        )


if __name__ == '__main__':
    main()
