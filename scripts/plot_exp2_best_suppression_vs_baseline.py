#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
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


def _parse_layer_contribution(raw: dict) -> tuple[dict[int, float], str | None]:
    """Return mapping restore_layer_id -> activation_value, and an optional model title."""
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


def _find_exp1_baseline_path(model_dir: Path) -> Path:
    # Common case
    p = model_dir / 'exp1' / 'baseline' / 'results.json'
    if p.exists():
        return p

    # Fallback: find any exp1* directory containing baseline/results.json
    for child in sorted(model_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith('exp1'):
            continue
        cand = child / 'baseline' / 'results.json'
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Could not locate exp1 baseline results.json under {model_dir}")


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


def plot_one(model_name: str, model_title: str, exp2_path: Path, exp1_baseline_path: Path, outdir: Path, metric: str, criterion: str, logy: bool):
    exp2_raw = _load_json(exp2_path)
    contrib_map, exp2_title = _parse_layer_contribution(exp2_raw)

    if not contrib_map:
        raise ValueError(f"Empty layer contribution in {exp2_path}")

    if criterion == 'min':
        best_layer, best_val = min(contrib_map.items(), key=lambda kv: kv[1])
    elif criterion == 'max':
        best_layer, best_val = max(contrib_map.items(), key=lambda kv: kv[1])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    baseline = _load_layerwise_results(exp1_baseline_path)
    if not baseline:
        raise ValueError(f"Empty baseline results in {exp1_baseline_path}")

    last_layer = max(baseline.keys())
    target_layer = max(0, last_layer - 1)  # penultimate

    if metric not in baseline[target_layer]:
        raise KeyError(f"Metric {metric} not found in exp1 baseline for layer {target_layer} ({exp1_baseline_path})")

    baseline_val = float(baseline[target_layer][metric])

    title = exp2_title or model_title or model_name

    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2), dpi=300)

    palette = sns.color_palette('Set2', 2)
    labels = ['Baseline', f"Supp (L{best_layer})"]
    values = [baseline_val, best_val]

    x = np.arange(len(labels))
    ax.bar(x[0], values[0], color=palette[0], alpha=0.85, width=0.6)
    ax.bar(x[1], values[1], color=palette[1], alpha=0.85, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Massive activation (Top1)')
    ax.set_title(title, fontweight='bold', pad=4)

    if logy:
        ax.set_yscale('log')

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.88)

    out_stem = f"{model_name}_exp2_{criterion}_layer_vs_exp1_baseline_{metric}"
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
    parser.add_argument('--criterion', choices=['min', 'max'], default='min')
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
            exp1_baseline_path = _find_exp1_baseline_path(model_dir)
        except FileNotFoundError:
            # If a model has exp2 but no exp1 baseline results, skip
            continue

        plot_one(
            model_name=model_name,
            model_title=model_name,
            exp2_path=exp2_path,
            exp1_baseline_path=exp1_baseline_path,
            outdir=outdir,
            metric=args.metric,
            criterion=args.criterion,
            logy=args.logy,
        )


if __name__ == '__main__':
    main()
