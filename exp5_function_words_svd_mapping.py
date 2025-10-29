#!/usr/bin/env python3
"""
Experiment 5: Function Words Mapping in SVD Space (Left/Right Singular Vectors)

Research Question:
  How do function words map to the left vs right singular vector spaces of W₂?
  Do function words show different alignment patterns compared to content words?

Key Analysis:
  1. SVD decomposition of W₂ (MLP down-projection matrix)
  2. Project token representations onto left singular vectors (U)
  3. Project token representations onto right singular vectors (Vt)
  4. Analyze concentration, stability, and asymmetry patterns
  5. Compare function words vs content words across all singular directions

Hypothesis:
  - Function words have low-dimensional projections (concentrated on few singular vectors)
  - Function words are more stable across contexts
  - Left/right asymmetry reveals where information is determined (Linear1 vs Linear2)
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import json
from datetime import datetime

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import lib
import monkey_patch as mp

# ============================================================================
# WORD CATEGORIES
# ============================================================================

FUNCTION_WORDS = {
    # Articles
    'the', 'a', 'an',
    # Prepositions
    'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'about', 'after', 'before', 'between', 'through', 'during',
    'under', 'over', 'against', 'within', 'without', 'among',
    # Conjunctions
    'and', 'or', 'but', 'if', 'that', 'which', 'while', 'because',
    'although', 'though', 'unless', 'until', 'since', 'when', 'where',
    # Pronouns
    'it', 'its', 'they', 'them', 'their', 'this', 'these', 'those',
    'he', 'she', 'his', 'her', 'him', 'we', 'us', 'our',
    # Auxiliary verbs
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
    # Common adverbs
    'not', 'no', 'yes', 'also', 'just', 'only', 'very', 'so', 'too',
}

# Make case-insensitive and add variations
def prepare_word_set(words):
    result = set()
    for word in words:
        result.add(word.lower())
        result.add(word.lower().lstrip('Ġ '))  # Remove BPE markers
    return result

FUNCTION_WORDS = prepare_word_set(FUNCTION_WORDS)

# ============================================================================
# TRACKER CLASS
# ============================================================================

class FunctionWordSVDTracker:
    """Track word representations and compute SVD projections"""

    def __init__(self, layer_id, tokenizer):
        self.layer_id = layer_id
        self.tokenizer = tokenizer

        # Storage: {word: [(context_id, h2_vector), ...]}
        self.word_data = defaultdict(list)
        self.context_counter = 0

    def is_function_word(self, token_text):
        """Check if token is a function word"""
        clean_token = token_text.lower().lstrip('Ġ ')
        return clean_token in FUNCTION_WORDS

    def add_token(self, token_text, h2_vector):
        """Add token representation"""
        if self.is_function_word(token_text):
            # h2_vector shape: [3072]
            self.word_data[token_text].append((self.context_counter, h2_vector.cpu().detach().numpy()))

    def next_context(self):
        """Mark end of current sequence"""
        self.context_counter += 1

    def get_word_statistics(self):
        """Get stats for each word"""
        stats_dict = {}
        for word, occurrences in self.word_data.items():
            stats_dict[word] = {
                'count': len(occurrences),
                'contexts': len(set(oc[0] for oc in occurrences)),
                'occurrences': occurrences,
            }
        return stats_dict


# ============================================================================
# SVD ANALYSIS CLASS
# ============================================================================

class SVDSpaceAnalyzer:
    """Analyze token projections in SVD spaces"""

    def __init__(self, W2, tracker, keep_top_k=100):
        """
        Args:
            W2: Down-projection matrix [3072, 768]
            tracker: FunctionWordSVDTracker with word data
            keep_top_k: Keep top k singular vectors for analysis
        """
        self.W2 = W2.float()
        self.tracker = tracker
        self.keep_top_k = min(keep_top_k, min(W2.shape))

        # SVD decomposition
        print("Computing SVD decomposition...")
        U, S, Vt = torch.svd(self.W2)

        self.U = U[:, :self.keep_top_k]  # [3072, k]
        self.S = S[:self.keep_top_k]      # [k]
        self.Vt = Vt[:self.keep_top_k, :]  # [k, 768]

        print(f"  U shape: {self.U.shape}, S shape: {self.S.shape}, Vt shape: {self.Vt.shape}")
        print(f"  σ₁/σ₂ ratio: {(S[0]/S[1]).item():.3f}")

        # Compute projections for all words
        self.projections = {}  # {word: {'left': proj, 'right': proj, ...}}
        self._compute_projections()

    def _compute_projections(self):
        """Compute projections for all tracked words"""
        print("Computing SVD projections for all words...")

        for word, stats in self.tracker.get_word_statistics().items():
            if stats['count'] < 2:  # Skip rare words
                continue

            occurrences = stats['occurrences']
            h2_stack = np.stack([oc[1] for oc in occurrences])  # [n_occur, 3072]
            h2_tensor = torch.from_numpy(h2_stack).float()

            # Left singular space: h₂ @ U
            left_proj = h2_tensor @ self.U  # [n_occur, k]

            # Right singular space: (h₂ @ U @ Σ) @ Vt = output space
            scaled = left_proj * self.S.unsqueeze(0)  # [n_occur, k]
            right_proj = scaled @ self.Vt  # [n_occur, 768]

            self.projections[word] = {
                'left': left_proj.numpy(),
                'right': right_proj.numpy(),
                'h2': h2_tensor.numpy(),
                'count': stats['count'],
                'contexts': stats['contexts'],
            }

    def analyze_concentration(self):
        """
        Analysis 1: Variance concentration

        Returns:
            concentration_scores: {word: score} where score ∈ [0, 1]
            - High score = variance concentrated on few singular vectors (low-dim)
            - Low score = variance spread across many singular vectors (high-dim)
        """
        print("\n=== Analysis 1: Concentration on Singular Vectors ===")

        results = {}
        top_k_list = [1, 3, 5, 10]

        for word, projs in self.projections.items():
            left_proj = projs['left']  # [n_occur, k_sv]

            # Variance per singular vector
            var_per_sv = left_proj.var(axis=0)
            total_var = var_per_sv.sum()

            if total_var < 1e-6:
                continue

            var_ratio = var_per_sv / total_var
            sorted_var = sorted(var_ratio, reverse=True)

            # Concentration on top-k singular vectors
            concentration = {}
            for k in top_k_list:
                concentration[f'top_{k}'] = float(np.sum(sorted_var[:k]))

            # Top singular vectors
            top_sv_indices = np.argsort(var_ratio)[-5:][::-1]

            results[word] = {
                'concentration': concentration,
                'top_singular_vectors': list(top_sv_indices),
                'max_variance_ratio': float(var_ratio.max()),
                'min_variance_ratio': float(var_ratio.min()),
            }

        return results

    def analyze_left_right_asymmetry(self):
        """
        Analysis 2: Left vs Right Singular Space Asymmetry

        Hypothesis:
            - If function words' information is determined early (Linear1),
              they should be concentrated in left space but spread in right space
            - Content words may show different patterns
        """
        print("\n=== Analysis 2: Left-Right Space Asymmetry ===")

        results = {}

        for word, projs in self.projections.items():
            left_proj = projs['left']   # [n_occur, k_sv]
            right_proj = projs['right']  # [n_occur, 768]

            # Variance concentration
            left_var_per_dim = left_proj.var(axis=0)
            right_var_per_dim = right_proj.var(axis=0)

            left_total = left_var_per_dim.sum()
            right_total = right_var_per_dim.sum()

            if left_total < 1e-6 or right_total < 1e-6:
                continue

            # Concentration score: fraction of variance in top 3 dimensions
            left_top3 = sorted(left_var_per_dim, reverse=True)[:3].sum() / left_total
            right_top3 = sorted(right_var_per_dim, reverse=True)[:3].sum() / right_total

            # Asymmetry ratio
            asymmetry_ratio = left_top3 / (right_top3 + 1e-6)

            results[word] = {
                'left_concentration_top3': float(left_top3),
                'right_concentration_top3': float(right_top3),
                'asymmetry_ratio': float(asymmetry_ratio),  # >1 = left concentrated
                'left_total_variance': float(left_total),
                'right_total_variance': float(right_total),
            }

        return results

    def analyze_stability(self):
        """
        Analysis 3: Cross-Context Stability

        Hypothesis:
            - Function words should have stable representations (high cosine similarity)
            - Content words should vary more based on context
        """
        print("\n=== Analysis 3: Cross-Context Stability ===")

        results = {}

        for word, projs in self.projections.items():
            left_proj = projs['left']  # [n_occur, k_sv]

            if len(left_proj) < 2:
                continue

            # Compute pairwise cosine similarity
            # Normalize projections
            norms = np.linalg.norm(left_proj, axis=1, keepdims=True)
            left_proj_norm = left_proj / (norms + 1e-8)

            sims = cosine_similarity(left_proj_norm)
            upper_tri = sims[np.triu_indices_from(sims, k=1)]

            results[word] = {
                'mean_stability': float(upper_tri.mean()),
                'std_stability': float(upper_tri.std()),
                'min_stability': float(upper_tri.min()),
                'max_stability': float(upper_tri.max()),
                'n_occurrences': len(left_proj),
            }

        return results

    def analyze_singular_direction_alignment(self):
        """
        Analysis 4: Alignment with principal singular directions

        Check: Do function words align more with v₁ (principal direction)?
        """
        print("\n=== Analysis 4: Principal Direction Alignment ===")

        results = {}

        # Principal left singular vector
        v1_left = self.U[:, 0]  # [3072]

        for word, projs in self.projections.items():
            h2 = projs['h2']  # [n_occur, 3072]

            # Alignment with v₁
            alignments = []
            for h2_vec in h2:
                # Cosine similarity
                h2_norm = h2_vec / (np.linalg.norm(h2_vec) + 1e-8)
                v1_norm = v1_left.numpy() / np.linalg.norm(v1_left.numpy())
                alignment = np.dot(h2_norm, v1_norm)
                alignments.append(alignment)

            alignments = np.array(alignments)

            results[word] = {
                'mean_alignment_with_v1': float(alignments.mean()),
                'std_alignment_with_v1': float(alignments.std()),
                'max_alignment': float(alignments.max()),
                'min_alignment': float(alignments.min()),
            }

        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_concentration_analysis(concentration_results, savedir):
    """Plot 1: Concentration on singular vectors"""
    print("\nGenerating concentration plot...")

    # Separate function words
    words = list(concentration_results.keys())
    top_5_concentration = [concentration_results[w]['concentration']['top_5'] for w in words]

    # Sort by concentration
    sorted_idx = np.argsort(top_5_concentration)[::-1]
    words_sorted = [words[i] for i in sorted_idx]
    top_5_sorted = [top_5_concentration[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(words_sorted)))
    bars = ax.barh(range(len(words_sorted)), top_5_sorted, color=colors)

    ax.set_yticks(range(len(words_sorted)))
    ax.set_yticklabels(words_sorted, fontsize=9)
    ax.set_xlabel('Variance Concentration (Top 5 Singular Vectors)', fontsize=11)
    ax.set_title('Function Words: Variance Concentration in Left Singular Space', fontsize=13, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{savedir}/exp5_concentration_top5.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {savedir}/exp5_concentration_top5.png")


def plot_asymmetry_analysis(asymmetry_results, savedir):
    """Plot 2: Left-Right asymmetry"""
    print("Generating asymmetry plot...")

    words = list(asymmetry_results.keys())
    asymmetry_ratios = [asymmetry_results[w]['asymmetry_ratio'] for w in words]
    left_conc = [asymmetry_results[w]['left_concentration_top3'] for w in words]
    right_conc = [asymmetry_results[w]['right_concentration_top3'] for w in words]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Asymmetry ratio
    sorted_idx = np.argsort(asymmetry_ratios)[::-1]
    words_sorted = [words[i] for i in sorted_idx]
    ratio_sorted = [asymmetry_ratios[i] for i in sorted_idx]

    colors = ['red' if r > 1 else 'blue' for r in ratio_sorted]
    axes[0].barh(range(len(words_sorted)), ratio_sorted, color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(words_sorted)))
    axes[0].set_yticklabels(words_sorted, fontsize=9)
    axes[0].set_xlabel('Asymmetry Ratio (Left / Right)', fontsize=11)
    axes[0].set_title('Left vs Right Space Concentration', fontsize=12, fontweight='bold')
    axes[0].axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    axes[0].grid(axis='x', alpha=0.3)

    # Plot 2: Scatter plot
    axes[1].scatter(left_conc, right_conc, s=100, alpha=0.6)
    axes[1].set_xlabel('Left Space Concentration (Top 3)', fontsize=11)
    axes[1].set_ylabel('Right Space Concentration (Top 3)', fontsize=11)
    axes[1].set_title('Left vs Right Concentration', fontsize=12, fontweight='bold')
    axes[1].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, label='Equal concentration')

    for i, word in enumerate(words):
        axes[1].annotate(word, (left_conc[i], right_conc[i]), fontsize=8, alpha=0.7)

    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{savedir}/exp5_asymmetry_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {savedir}/exp5_asymmetry_analysis.png")


def plot_stability_analysis(stability_results, savedir):
    """Plot 3: Cross-context stability"""
    print("Generating stability plot...")

    words = list(stability_results.keys())
    mean_stab = [stability_results[w]['mean_stability'] for w in words]
    std_stab = [stability_results[w]['std_stability'] for w in words]

    sorted_idx = np.argsort(mean_stab)[::-1]
    words_sorted = [words[i] for i in sorted_idx]
    mean_stab_sorted = [mean_stab[i] for i in sorted_idx]
    std_stab_sorted = [std_stab[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.errorbar(range(len(words_sorted)), mean_stab_sorted, yerr=std_stab_sorted,
                fmt='o', capsize=5, capthick=2, markersize=8, color='darkblue', alpha=0.7)

    ax.set_xticks(range(len(words_sorted)))
    ax.set_xticklabels(words_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax.set_title('Function Words: Cross-Context Stability (Cosine Similarity)', fontsize=13, fontweight='bold')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='High stability (0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium stability (0.5)')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{savedir}/exp5_stability_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {savedir}/exp5_stability_analysis.png")


def plot_alignment_analysis(alignment_results, savedir):
    """Plot 4: Principal direction alignment"""
    print("Generating alignment plot...")

    words = list(alignment_results.keys())
    mean_align = [alignment_results[w]['mean_alignment_with_v1'] for w in words]

    sorted_idx = np.argsort(mean_align)[::-1]
    words_sorted = [words[i] for i in sorted_idx]
    align_sorted = [mean_align[i] for i in sorted_idx]

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(words_sorted)))

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(words_sorted)), align_sorted, color=colors)

    ax.set_yticks(range(len(words_sorted)))
    ax.set_yticklabels(words_sorted, fontsize=9)
    ax.set_xlabel('Mean Alignment with v₁ (Principal Singular Direction)', fontsize=11)
    ax.set_title('Function Words: Alignment with Principal Singular Direction', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{savedir}/exp5_alignment_v1.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {savedir}/exp5_alignment_v1.png")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Exp 5: Function Words SVD Mapping Analysis')
    parser.add_argument('--model', type=str, default='gpt2', choices=['gpt2', 'llama', 'mistral'])
    parser.add_argument('--layer_id', type=int, default=2, help='Layer to analyze (0-indexed)')
    parser.add_argument('--nsamples', type=int, default=50, help='Number of text sequences')
    parser.add_argument('--savedir', type=str, default='results/exp5_svd_mapping/')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.savedir, exist_ok=True)

    print("="*80)
    print("EXPERIMENT 5: Function Words Mapping in SVD Space")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Layer: {args.layer_id}")
    print(f"Samples: {args.nsamples}")
    print(f"Save dir: {args.savedir}")
    print()

    # Load model and data
    print("Loading model and data...")
    model, tokenizer = lib.load_model(args.model, device=args.device)
    model.eval()

    # Load data
    data = lib.load_data(tokenizer, nsamples=args.nsamples, device=args.device)

    # Setup hooks
    print("Setting up hooks...")
    layer = model.transformer.h[args.layer_id]
    mp.enable_gpt2_custom_block(layer, args.layer_id)

    # Tracker for function words
    tracker = FunctionWordSVDTracker(args.layer_id, tokenizer)

    # Setup hook to capture MLP intermediate activations (h2 = output of GELU before Linear2)
    h2_list = []
    token_list = []

    def capture_h2_hook(module, input, output):
        """Capture h2 (output of GELU, before Linear2 projection)"""
        # output[0] = activation tensor [batch, seq_len, 3072]
        h2_list.append(output.cpu().clone())

    # Register hook on GELU (which is the last layer before Linear2)
    gelu_hook = layer.mlp.act.register_forward_hook(capture_h2_hook)

    # Forward pass and data collection
    print(f"Processing {len(data)} sequences...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data)):
            tokens = batch.to(args.device)
            token_ids = batch.squeeze(0).cpu().numpy()  # [seq_len]

            # Clear h2 storage for this batch
            h2_list.clear()

            # Forward pass
            _ = model(tokens)

            # Check if h2 was captured
            if len(h2_list) > 0:
                h2 = h2_list[-1]  # Take the last captured h2 (most recent batch)

                # h2 shape: [batch, seq_len, 3072] or [seq_len, 3072]
                if h2.dim() == 3:
                    h2 = h2[0]  # Take first batch if needed [seq_len, 3072]

                # Iterate through tokens
                for tok_idx in range(min(h2.shape[0], len(token_ids))):
                    try:
                        token_id = int(token_ids[tok_idx])
                        token_text = tokenizer.decode([token_id])
                        h2_vec = h2[tok_idx, :]  # [3072]
                        tracker.add_token(token_text, h2_vec)
                    except Exception as e:
                        pass

            tracker.next_context()

    # Clean up hook
    gelu_hook.remove()

    print(f"Collected {sum(s['count'] for s in tracker.get_word_statistics().values())} function word occurrences")

    # Get W2 matrix (MLP down-projection)
    W2 = layer.mlp.c_proj.weight.t()  # [3072, 768]

    # SVD Analysis
    print("\nInitializing SVD analyzer...")
    analyzer = SVDSpaceAnalyzer(W2, tracker, keep_top_k=100)

    # Run all analyses
    print("\nRunning analyses...")
    conc_results = analyzer.analyze_concentration()
    asym_results = analyzer.analyze_left_right_asymmetry()
    stab_results = analyzer.analyze_stability()
    align_results = analyzer.analyze_singular_direction_alignment()

    # Generate plots
    print("\nGenerating visualizations...")
    plot_concentration_analysis(conc_results, args.savedir)
    plot_asymmetry_analysis(asym_results, args.savedir)
    plot_stability_analysis(stab_results, args.savedir)
    plot_alignment_analysis(align_results, args.savedir)

    # Save detailed results
    print("\nSaving detailed results...")
    results = {
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'concentration': conc_results,
        'asymmetry': asym_results,
        'stability': stab_results,
        'alignment': align_results,
        'word_stats': {w: {k: v for k, v in s.items() if k != 'occurrences'}
                       for w, s in tracker.get_word_statistics().items()},
    }

    with open(f'{args.savedir}/exp5_detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Generate summary report
    print("\nGenerating summary report...")
    summary_path = f'{args.savedir}/EXP5_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT 5: Function Words SVD Space Mapping Analysis\n")
        f.write("="*80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Model: {args.model}, Layer: {args.layer_id}\n")
        f.write(f"Samples: {args.nsamples}\n\n")

        f.write("ANALYSIS 1: CONCENTRATION ON SINGULAR VECTORS\n")
        f.write("-"*80 + "\n")
        f.write("Top 10 Most Concentrated Words (Low-Dimensional):\n")
        sorted_conc = sorted(conc_results.items(),
                            key=lambda x: x[1]['concentration']['top_5'], reverse=True)[:10]
        for word, res in sorted_conc:
            f.write(f"  {word:15s}: top_5={res['concentration']['top_5']:.3f}, "
                   f"top_3={res['concentration']['top_3']:.3f}\n")

        f.write("\n\nANALYSIS 2: LEFT-RIGHT ASYMMETRY\n")
        f.write("-"*80 + "\n")
        f.write("Asymmetry Ratio (Left / Right concentration):\n")
        f.write("  Ratio > 1: Left space more concentrated (early determination)\n")
        f.write("  Ratio < 1: Right space more concentrated (late determination)\n\n")
        sorted_asym = sorted(asym_results.items(),
                            key=lambda x: x[1]['asymmetry_ratio'], reverse=True)[:10]
        for word, res in sorted_asym:
            f.write(f"  {word:15s}: ratio={res['asymmetry_ratio']:.3f} "
                   f"(left={res['left_concentration_top3']:.3f}, "
                   f"right={res['right_concentration_top3']:.3f})\n")

        f.write("\n\nANALYSIS 3: CROSS-CONTEXT STABILITY\n")
        f.write("-"*80 + "\n")
        f.write("Mean Cosine Similarity (Higher = More Stable):\n")
        sorted_stab = sorted(stab_results.items(),
                            key=lambda x: x[1]['mean_stability'], reverse=True)[:10]
        for word, res in sorted_stab:
            f.write(f"  {word:15s}: stability={res['mean_stability']:.3f} "
                   f"(±{res['std_stability']:.3f}), n={res['n_occurrences']}\n")

        f.write("\n\nANALYSIS 4: PRINCIPAL DIRECTION ALIGNMENT\n")
        f.write("-"*80 + "\n")
        f.write("Alignment with v₁ (Principal Singular Direction):\n")
        sorted_align = sorted(align_results.items(),
                             key=lambda x: x[1]['mean_alignment_with_v1'], reverse=True)[:10]
        for word, res in sorted_align:
            f.write(f"  {word:15s}: v₁_align={res['mean_alignment_with_v1']:.3f} "
                   f"(±{res['std_alignment_with_v1']:.3f})\n")

        f.write("\n\nKEY FINDINGS\n")
        f.write("="*80 + "\n")
        f.write("1. Concentration: Function words with high concentration (top_5 > 0.5)\n")
        f.write("   suggest low-dimensional representation in singular space\n\n")
        f.write("2. Asymmetry: Ratio > 1 indicates information determined in Linear1,\n")
        f.write("   ratio < 1 indicates late determination in Linear2\n\n")
        f.write("3. Stability: High stability (>0.8) suggests context-independent\n")
        f.write("   function word representations\n\n")
        f.write("4. Alignment: Strong v₁ alignment suggests function words drive\n")
        f.write("   massive activations along principal singular direction\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "="*80)
    print("EXPERIMENT 5 COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
