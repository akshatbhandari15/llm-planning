"""
Visualization module for the VOMC-QKV analysis pipeline.

Generates publication-quality plots for:
- Context sweep results (confidence, entropy, state evolution)
- VOMC structure (transition matrices, entropy by order)
- Planning detection (MI curves, planning horizons)
- Cross-context comparisons
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import seaborn as sns

logger = logging.getLogger("vomc_pipeline.viz")

# Style configuration
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = sns.color_palette("Set2", 8)
QKV_COLORS = {"Q": "#1f77b4", "K": "#ff7f0e", "V": "#2ca02c"}


def save_fig(fig, path, dpi=150):
    """Save figure and close."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Context Sweep Plots ──────────────────────────────────────────────────

def plot_confidence_curves(results, output_dir):
    """
    Plot confidence in expected token vs context length for all prompts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Individual curves
    ax = axes[0]
    for i, r in enumerate(results):
        ax.plot(r.context_lengths, r.confidence_curve,
                alpha=0.4, linewidth=1, color=COLORS[i % len(COLORS)])
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("P(expected token)")
    ax.set_title("Confidence vs Context Length (individual prompts)")
    ax.set_xscale("log", base=2)

    # Aggregated with error bands
    ax = axes[1]
    all_lengths = sorted(set(
        c for r in results for c in r.context_lengths
    ))
    means = []
    stds = []
    for c in all_lengths:
        vals = [
            r.confidence_curve[r.context_lengths.index(c)]
            for r in results if c in r.context_lengths
        ]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        else:
            means.append(0)
            stds.append(0)

    means = np.array(means)
    stds = np.array(stds)
    ax.plot(all_lengths, means, "o-", color=COLORS[0], linewidth=2)
    ax.fill_between(all_lengths, means - stds, means + stds,
                    alpha=0.2, color=COLORS[0])
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Mean P(expected token)")
    ax.set_title("Confidence vs Context Length (aggregated)")
    ax.set_xscale("log", base=2)

    fig.suptitle("Phase 2: How Confidence Grows with Context", fontsize=14)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "confidence_curves.png"))


def plot_entropy_curves(results, output_dir):
    """Plot entropy of predicted distribution vs context length."""
    fig, ax = plt.subplots(figsize=(8, 5))

    all_lengths = sorted(set(
        c for r in results for c in r.context_lengths
    ))
    means = []
    for c in all_lengths:
        vals = [
            r.entropy_curve[r.context_lengths.index(c)]
            for r in results if c in r.context_lengths
        ]
        means.append(np.mean(vals) if vals else 0)

    ax.plot(all_lengths, means, "s-", color=COLORS[1], linewidth=2)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Prediction Entropy vs Context Length")
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "entropy_curves.png"))


def plot_state_evolution(result, tensor_types, layers, output_dir, prompt_idx=0):
    """
    Plot how QKV states evolve with context for a single prompt.
    Shows cosine similarity to the full-context state.
    """
    n_types = len(tensor_types)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 5))
    if n_types == 1:
        axes = [axes]

    for ax, ttype in zip(axes, tensor_types):
        for layer in layers:
            states = result.layer_states.get(ttype, {}).get(layer, [])
            if len(states) < 2:
                continue
            final = states[-1]
            cos_sims = []
            for v in states:
                norm_v = np.linalg.norm(v)
                norm_f = np.linalg.norm(final)
                if norm_v > 1e-10 and norm_f > 1e-10:
                    cos_sims.append(float(np.dot(v, final) / (norm_v * norm_f)))
                else:
                    cos_sims.append(0.0)

            label = f"Layer {layer}"
            ax.plot(result.context_lengths[:len(cos_sims)], cos_sims,
                    "o-", alpha=0.7, label=label, markersize=3)

        ax.set_xlabel("Context Length")
        ax.set_ylabel("Cosine Sim to Full-Context State")
        ax.set_title(f"{ttype} State Evolution")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f"State Evolution with Context (Prompt {prompt_idx})", fontsize=13)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, f"state_evolution_prompt{prompt_idx}.png"))


# ── VOMC Plots ───────────────────────────────────────────────────────────

def plot_vomc_order_selection(analyses, output_dir):
    """
    Plot BIC/AIC curves across Markov orders for each context length.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ctx_len, analysis in sorted(analyses.items()):
        orders = sorted(analysis.models.keys())
        bics = [analysis.models[o].bic for o in orders]
        aics = [analysis.models[o].aic for o in orders]

        label = f"ctx={ctx_len}"
        axes[0].plot(orders, bics, "o-", label=label, alpha=0.7)
        axes[1].plot(orders, aics, "o-", label=label, alpha=0.7)

    axes[0].set_xlabel("Markov Order")
    axes[0].set_ylabel("BIC")
    axes[0].set_title("BIC vs Markov Order")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Markov Order")
    axes[1].set_ylabel("AIC")
    axes[1].set_title("AIC vs Markov Order")
    axes[1].legend(fontsize=8)

    fig.suptitle("Phase 3: Optimal Markov Order by Context Length", fontsize=14)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "vomc_order_selection.png"))


def plot_optimal_order_growth(comparison, output_dir):
    """
    Plot how the optimal Markov order grows with context length.
    This is a key result: does effective memory increase with context?
    """
    ctx_lens = sorted(comparison["optimal_orders"].keys())
    orders = [comparison["optimal_orders"][c] for c in ctx_lens]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ctx_lens, orders, "D-", color=COLORS[3], linewidth=2, markersize=8)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Optimal Markov Order")
    ax.set_title("Effective Memory Depth vs Input Context")
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "optimal_order_growth.png"))


def plot_transition_entropy(analyses, output_dir):
    """
    Plot transition entropy by Markov order across context lengths.
    Lower entropy = more deterministic transitions = stronger planning.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for ctx_len, analysis in sorted(analyses.items()):
        orders = sorted(analysis.entropy_by_order.keys())
        entropies = [analysis.entropy_by_order[o] for o in orders]
        ax.plot(orders, entropies, "o-", label=f"ctx={ctx_len}", alpha=0.7)

    ax.set_xlabel("Markov Order")
    ax.set_ylabel("Mean Transition Entropy (bits)")
    ax.set_title("Transition Entropy by Order and Context Length")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "transition_entropy.png"))


def plot_state_space_zipf(state_space, output_dir, label=""):
    """Plot Zipf distribution of cluster sizes (echoing Songhee's analysis)."""
    sizes = np.sort(state_space.cluster_sizes)[::-1]
    sizes = sizes[sizes > 0]
    ranks = np.arange(1, len(sizes) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ranks, sizes, s=10, alpha=0.6, color=COLORS[4])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cluster Size")
    ax.set_title(f"Zipf Distribution of VOMC State Sizes {label}")

    # Fit power law
    log_r = np.log10(ranks)
    log_s = np.log10(sizes)
    slope, intercept = np.polyfit(log_r, log_s, 1)
    fit_line = 10 ** (slope * log_r + intercept)
    ax.plot(ranks, fit_line, "--", color="red",
            label=f"α = {-slope:.2f}, R² = {np.corrcoef(log_r, log_s)[0,1]**2:.3f}")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, f"state_space_zipf{label}.png"))


# ── Planning Detection Plots ─────────────────────────────────────────────

def plot_mi_curves(profiles, output_dir):
    """
    Plot MI vs lookahead for different context lengths.
    The key planning result: how far ahead does the model plan?
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Separate plots for Q, K, V
    tensor_groups = {}
    for ctx_len, profile in profiles.items():
        ttype = profile.tensor_type
        if ttype not in tensor_groups:
            tensor_groups[ttype] = {}
        tensor_groups[ttype][ctx_len] = profile

    for ax_idx, (ttype, group) in enumerate(sorted(tensor_groups.items())):
        if ax_idx >= 3:
            break
        ax = axes[ax_idx]
        for ctx_len in sorted(group.keys()):
            profile = group[ctx_len]
            lookaheads = [r.lookahead for r in profile.mi_results]
            mi_vals = [r.mi_value for r in profile.mi_results]
            mi_errs = [r.mi_std for r in profile.mi_results]

            ax.errorbar(lookaheads, mi_vals, yerr=mi_errs,
                       fmt="o-", label=f"ctx={ctx_len}", alpha=0.7, capsize=3)

            # Mark significant points
            for r in profile.mi_results:
                if r.is_significant:
                    ax.scatter([r.lookahead], [r.mi_value],
                             marker="*", s=100, color="red", zorder=5)

        ax.set_xlabel("Lookahead (tokens)")
        ax.set_ylabel("Mutual Information (bits)")
        ax.set_title(f"{ttype}: MI vs Lookahead")
        ax.legend(fontsize=8)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Phase 4: Forward Planning Detection", fontsize=14)
    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "mi_curves.png"))


def plot_planning_horizon(comparison, output_dir):
    """
    Plot planning horizon (max significant lookahead) vs context length.
    This is the headline result.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ctx_lens = sorted(comparison.horizon_curve.keys())
    horizons = [comparison.horizon_curve[c] for c in ctx_lens]

    ax.plot(ctx_lens, horizons, "D-", color=COLORS[5], linewidth=2, markersize=8)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Planning Horizon (tokens ahead)")
    ax.set_title("How Far Ahead Does the Model Plan?")
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "planning_horizon.png"))


def plot_mi_heatmap(comparison, output_dir):
    """
    Heatmap of MI: context length x lookahead.
    Reveals the structure of planning across scales.
    """
    ctx_lens = sorted(comparison.mi_surfaces.keys())
    if not ctx_lens:
        return

    max_la = max(len(v) for v in comparison.mi_surfaces.values())
    matrix = np.zeros((len(ctx_lens), max_la))

    for i, c in enumerate(ctx_lens):
        curve = comparison.mi_surfaces[c]
        matrix[i, :len(curve)] = curve

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_yticks(range(len(ctx_lens)))
    ax.set_yticklabels(ctx_lens)
    ax.set_xlabel("Lookahead (tokens)")
    ax.set_ylabel("Context Length")
    ax.set_title("MI Heatmap: Context Length × Lookahead")
    plt.colorbar(im, ax=ax, label="MI (bits)")

    fig.tight_layout()
    save_fig(fig, os.path.join(output_dir, "mi_heatmap.png"))


# ── Summary Dashboard ────────────────────────────────────────────────────

def plot_summary_dashboard(sweep_results, vomc_comparison, planning_comparison, output_dir):
    """
    Create a single summary dashboard combining key results.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Confidence vs Context
    ax1 = fig.add_subplot(gs[0, 0])
    all_lengths = sorted(set(
        c for r in sweep_results for c in r.context_lengths
    ))
    means = []
    for c in all_lengths:
        vals = [
            r.confidence_curve[r.context_lengths.index(c)]
            for r in sweep_results if c in r.context_lengths
        ]
        means.append(np.mean(vals) if vals else 0)
    ax1.plot(all_lengths, means, "o-", color=COLORS[0], linewidth=2)
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("P(expected)")
    ax1.set_title("Confidence Growth")
    ax1.set_xscale("log", base=2)

    # 2. Optimal Order vs Context
    ax2 = fig.add_subplot(gs[0, 1])
    if vomc_comparison:
        ctx = sorted(vomc_comparison["optimal_orders"].keys())
        orders = [vomc_comparison["optimal_orders"][c] for c in ctx]
        ax2.plot(ctx, orders, "D-", color=COLORS[3], linewidth=2)
    ax2.set_xlabel("Context Length")
    ax2.set_ylabel("Optimal Order")
    ax2.set_title("Effective Memory Depth")
    ax2.set_xscale("log", base=2)

    # 3. Planning Horizon vs Context
    ax3 = fig.add_subplot(gs[0, 2])
    if planning_comparison:
        ctx = sorted(planning_comparison.horizon_curve.keys())
        horizons = [planning_comparison.horizon_curve[c] for c in ctx]
        ax3.plot(ctx, horizons, "D-", color=COLORS[5], linewidth=2)
    ax3.set_xlabel("Context Length")
    ax3.set_ylabel("Planning Horizon")
    ax3.set_title("Forward Planning Range")
    ax3.set_xscale("log", base=2)

    # 4. MI Heatmap
    ax4 = fig.add_subplot(gs[1, :])
    if planning_comparison and planning_comparison.mi_surfaces:
        ctx_lens = sorted(planning_comparison.mi_surfaces.keys())
        max_la = max(len(v) for v in planning_comparison.mi_surfaces.values())
        matrix = np.zeros((len(ctx_lens), max_la))
        for i, c in enumerate(ctx_lens):
            curve = planning_comparison.mi_surfaces[c]
            matrix[i, :len(curve)] = curve
        im = ax4.imshow(matrix, aspect="auto", cmap="YlOrRd", origin="lower")
        ax4.set_yticks(range(len(ctx_lens)))
        ax4.set_yticklabels(ctx_lens)
        ax4.set_xlabel("Lookahead (tokens)")
        ax4.set_ylabel("Context Length")
        ax4.set_title("MI Surface: Context × Lookahead")
        plt.colorbar(im, ax=ax4, label="MI (bits)")

    fig.suptitle("VOMC-QKV Planning Analysis: Summary Dashboard", fontsize=16, y=0.98)
    save_fig(fig, os.path.join(output_dir, "summary_dashboard.png"))