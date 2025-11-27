#!/usr/bin/env python3
"""
Resume Experiment 1 from Phase 2 (All Heads Disabled)
Uses existing baseline results and only runs the remaining phases.
"""

import os
import sys
import argparse
import json

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import from original exp1
from exp1_feasibility_test import run_experiment, generate_visualizations, generate_summary_report

def main():
    parser = argparse.ArgumentParser(
        description='Resume Experiment 1 from Phase 2'
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='llama2_13b', help='Model name')
    parser.add_argument('--access_token', type=str, default='type in your access token here',
                        help='Hugging Face access token')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'c4', 'RedPajama'], help='Dataset name')
    parser.add_argument('--nsamples', type=int, default=30,
                        help='Number of samples to analyze')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Output arguments
    parser.add_argument('--savedir', type=str, default='results/exp1_llama2_13b/',
                        help='Directory with existing baseline results')

    args = parser.parse_args()

    # Check if baseline exists
    baseline_path = os.path.join(args.savedir, 'baseline', 'results.json')
    if not os.path.exists(baseline_path):
        print(f"❌ ERROR: Baseline results not found at {baseline_path}")
        print("Please run the full experiment first.")
        return

    print("\n" + "="*80)
    print("RESUMING EXPERIMENT 1 FROM PHASE 2")
    print("="*80)
    print(f"\nUsing existing baseline from: {baseline_path}")
    
    # Load baseline results
    print("\n📂 Loading baseline results...")
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    # Convert string keys to int
    baseline_results = {int(k): v for k, v in baseline_results.items()}
    print(f"✅ Loaded baseline for {len(baseline_results)} layers")

    # Run all heads disabled (Phase 2)
    print("\n🔴 PHASE 2: Running All Heads Disabled Experiment")
    disabled_results, disabled_stats = run_experiment(args, mode='all_disabled')

    # Save disabled results
    print("\n💾 Saving Phase 2 results...")
    with open(os.path.join(args.savedir, 'all_heads_disabled', 'results.json'), 'w') as f:
        import numpy as np
        json.dump({k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                   for k, v in disabled_results.items()}, f, indent=2)
    print("✅ Phase 2 results saved")

    # We need baseline_stats for visualization, but we don't have it
    # Create a dummy one (visualizations will still work with just results)
    baseline_stats = None

    # Generate visualizations (Phase 3)
    print("\n🎨 PHASE 3: Generating Visualizations")
    generate_visualizations(baseline_results, disabled_results,
                          baseline_stats, disabled_stats, args.savedir)

    # Generate summary report (Phase 4)
    print("\n📊 PHASE 4: Generating Summary Report")
    generate_summary_report(baseline_results, disabled_results, args.savedir)

    print("\n" + "="*80)
    print("✅ EXPERIMENT 1 COMPLETE (Resumed from Phase 2)")
    print("="*80)
    print(f"\nResults saved to: {args.savedir}")
    print("\nGenerated files:")
    print("  📁 baseline/")
    print("     └─ results.json (existing)")
    print("  📁 all_heads_disabled/")
    print("     └─ results.json (new)")
    print("  📁 comparison/")
    print("     ├─ exp1_top1_comparison.png")
    print("     ├─ exp1_percentage_change_heatmap.png")
    print("     ├─ exp1_layerwise_breakdown.png")
    print("     ├─ exp1_critical_dimensions.png")
    print("     └─ EXPERIMENT_1_SUMMARY.txt")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
