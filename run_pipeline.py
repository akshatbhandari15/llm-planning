#!/usr/bin/env python3
"""
VOMC-QKV Analysis Pipeline: Main Entry Point

Orchestrates the full experimental protocol:
  Phase 1: State extraction (Q, K, V from transformer layers)
  Phase 2: Context window sweep (vary context length)
  Phase 3: VOMC construction (build variable-order Markov chains)
  Phase 4: Planning detection (mutual information analysis)

Usage:
    python run_pipeline.py                          # Full pipeline, defaults
    python run_pipeline.py --model gpt2-medium      # Larger model
    python run_pipeline.py --phase 2                # Run only Phase 2
    python run_pipeline.py --max-context 64         # Cap context at 64 tokens
    python run_pipeline.py --device cuda             # GPU acceleration
"""

import argparse
import os
import sys
import json
import logging
from typing import Dict, List

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.state_extractor import QKVExtractor
from src.context_sweep import ContextSweeper, ContextSweepCollection
from src.vomc import VOMCBuilder, VOMCStateSpace, sequences_from_trajectories
from src.planning_detector import PlanningDetector
from src import visualization as viz
from src.utils import (
    set_seed, setup_logging, save_results,
    FACTUAL_PROMPTS, NARRATIVE_PROMPTS,
)
from configs.default_config import PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="VOMC-QKV Analysis Pipeline")
    parser.add_argument("--model", default="gpt2", help="Model name (default: gpt2)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", type=int, default=0,
                       help="Run specific phase (1-4), 0=all")
    parser.add_argument("--max-context", type=int, default=128,
                       help="Maximum context length to sweep")
    parser.add_argument("--n-trajectories", type=int, default=30,
                       help="Trajectories per context length for VOMC")
    parser.add_argument("--generation-length", type=int, default=20,
                       help="Tokens to generate per trajectory")
    parser.add_argument("--max-order", type=int, default=6,
                       help="Maximum Markov order to test")
    parser.add_argument("--n-clusters", type=int, default=32,
                       help="Number of VOMC states (clusters)")
    parser.add_argument("--max-lookahead", type=int, default=8,
                       help="Maximum planning lookahead")
    parser.add_argument("--n-permutations", type=int, default=100,
                       help="Permutations for MI significance test")
    parser.add_argument("--target-layers", type=str, default=None,
                       help="Comma-separated layer indices (default: all)")
    parser.add_argument("--tensor-types", type=str, default="Q,K,V",
                       help="Tensor types to analyze (comma-separated)")
    return parser.parse_args()


def run_phase1(extractor, config, output_dir, logger):
    """
    Phase 1: Validate state extraction on a single example.
    Quick sanity check that hooks work and QKV tensors look reasonable.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: State Extraction Validation")
    logger.info("=" * 60)

    test_prompt = "The capital of France is"
    input_ids = extractor.tokenizer.encode(test_prompt, return_tensors="pt")

    profile = extractor.extract(input_ids, return_logits=True)

    logger.info(f"Prompt: '{test_prompt}'")
    logger.info(f"Predicted: '{profile.predicted_token}' (p={profile.predicted_prob:.4f})")
    logger.info(f"Layers extracted: {sorted(profile.states.keys())}")

    # Validate shapes
    for layer, states in profile.states.items():
        for state in states:
            logger.info(
                f"  Layer {layer}, Pos {state.position} ('{state.token_str}'): "
                f"Q={state.Q.shape}, K={state.K.shape}, V={state.V.shape}"
            )
        break  # Just show first layer

    # Compare Q, K, V norms across layers
    qkv_norms = {"Q": [], "K": [], "V": []}
    for layer in sorted(profile.states.keys()):
        last_state = profile.states[layer][-1]  # Last position
        qkv_norms["Q"].append(float(np.linalg.norm(last_state.Q)))
        qkv_norms["K"].append(float(np.linalg.norm(last_state.K)))
        qkv_norms["V"].append(float(np.linalg.norm(last_state.V)))

    save_results(
        {"prompt": test_prompt, "predicted": profile.predicted_token,
         "predicted_prob": profile.predicted_prob, "qkv_norms": qkv_norms},
        os.path.join(output_dir, "phase1_validation.json"),
    )
    logger.info("Phase 1 complete.\n")
    return profile


def run_phase2(extractor, config, output_dir, logger, max_context=128):
    """
    Phase 2: Context window sweep.
    For each prompt, truncate to increasing lengths, extract QKV states,
    and track how predictions evolve.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Context Window Sweep")
    logger.info("=" * 60)

    # Determine context lengths
    context_lengths = [c for c in [1, 2, 4, 8, 16, 32, 64, 128, 256]
                       if c <= max_context]

    sweeper = ContextSweeper(
        extractor=extractor,
        context_lengths=context_lengths,
        tensor_types=config.sweep.tensor_types,
    )

    collection = sweeper.sweep_all(FACTUAL_PROMPTS)

    # Log summary
    for i, result in enumerate(collection.results):
        logger.info(
            f"  Prompt {i}: '{result.prompt_text[:50]}...' -> "
            f"confidence range: [{min(result.confidence_curve):.3f}, "
            f"{max(result.confidence_curve):.3f}]"
        )

    # Generate plots
    viz.plot_confidence_curves(collection.results, output_dir)
    viz.plot_entropy_curves(collection.results, output_dir)

    # State evolution for first few prompts
    layers_to_show = [0, extractor.n_layers // 2, extractor.n_layers - 1]
    for idx in range(min(3, len(collection.results))):
        viz.plot_state_evolution(
            collection.results[idx],
            tensor_types=config.sweep.tensor_types,
            layers=layers_to_show,
            output_dir=output_dir,
            prompt_idx=idx,
        )

    # Save numerical results
    sweep_data = {
        "context_lengths": context_lengths,
        "prompts": [r.prompt_text for r in collection.results],
        "confidence_curves": [r.confidence_curve for r in collection.results],
        "entropy_curves": [r.entropy_curve for r in collection.results],
        "rank_curves": [r.rank_curve for r in collection.results],
    }
    save_results(sweep_data, os.path.join(output_dir, "phase2_sweep.json"))

    logger.info("Phase 2 complete.\n")
    return collection


def run_phase3(extractor, collection, config, output_dir, logger,
               n_trajectories=30, generation_length=20, max_order=6,
               n_clusters=32, max_context=128):
    """
    Phase 3: VOMC construction.
    Generate trajectories at each context length, discretize states,
    build Markov chains, and find optimal order.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: VOMC Construction")
    logger.info("=" * 60)

    builder = VOMCBuilder(
        n_clusters=n_clusters,
        max_order=max_order,
        reduce_dim=True,
        reduced_dim=min(32, extractor.hidden_dim),
        model_selection="bic",
        min_transitions=3,
    )

    context_lengths = [c for c in [4, 8, 16, 32, 64, 128]
                       if c <= max_context]

    # Use a subset of narrative prompts as seeds
    seed_prompts = NARRATIVE_PROMPTS[:5]
    analyses = {}

    for ctx_len in context_lengths:
        logger.info(f"\n--- Context length: {ctx_len} ---")

        # Generate trajectories at this context length
        all_trajectories = []
        all_state_vectors = []

        for prompt in seed_prompts:
            tokens = extractor.tokenizer.encode(prompt)
            if len(tokens) < ctx_len:
                continue

            for trial in range(n_trajectories // len(seed_prompts) + 1):
                if len(all_trajectories) >= n_trajectories:
                    break

                input_ids = torch.tensor(tokens[:ctx_len])
                temp = np.random.choice([0.5, 0.8, 1.0])

                trajectory = extractor.extract_generation_trajectory(
                    input_ids,
                    n_steps=generation_length,
                    temperature=temp,
                    tensor_type="V",
                )
                all_trajectories.append(trajectory)

                # Collect state vectors for clustering
                last_layer = max(trajectory[0]["states"].keys())
                for step in trajectory:
                    vec = step["states"].get(last_layer)
                    if vec is not None:
                        all_state_vectors.append(vec)

        if len(all_state_vectors) < n_clusters:
            logger.warning(
                f"Only {len(all_state_vectors)} vectors for ctx={ctx_len}, "
                f"need at least {n_clusters}. Skipping."
            )
            continue

        # Build state space
        X = np.array(all_state_vectors)
        actual_k = min(n_clusters, len(X) // 3)
        builder.n_clusters = actual_k
        state_space = builder.build_state_space(X)

        viz.plot_state_space_zipf(state_space, output_dir, label=f"_ctx{ctx_len}")

        # Convert trajectories to discrete sequences
        discrete_seqs = sequences_from_trajectories(
            all_trajectories, state_space, tensor_type="V", layer=-1
        )

        # Run VOMC analysis
        analysis = builder.analyze_sequences(
            discrete_seqs, state_space, context_length=ctx_len
        )
        analyses[ctx_len] = analysis

        logger.info(
            f"  Optimal order: {analysis.optimal_order}, "
            f"Entropy@1: {analysis.entropy_by_order.get(1, 0):.3f}"
        )

    # Cross-context comparison
    if len(analyses) >= 2:
        comparison = builder.compare_across_contexts(analyses)
        viz.plot_vomc_order_selection(analyses, output_dir)
        viz.plot_optimal_order_growth(comparison, output_dir)
        viz.plot_transition_entropy(analyses, output_dir)
        save_results(comparison, os.path.join(output_dir, "phase3_comparison.json"))
    else:
        comparison = None
        logger.warning("Not enough context lengths for comparison.")

    logger.info("Phase 3 complete.\n")
    return analyses, comparison


def run_phase4(extractor, config, output_dir, logger,
               n_trajectories=30, generation_length=20,
               max_lookahead=8, n_permutations=100, max_context=128):
    """
    Phase 4: Planning detection via mutual information.
    For each context length, measure MI between current QKV state
    and future tokens at varying lookaheads.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4: Planning Detection")
    logger.info("=" * 60)

    detector = PlanningDetector(
        max_lookahead=max_lookahead,
        mi_method="ksg",
        mi_n_neighbors=5,
        n_permutations=n_permutations,
        alpha=0.05,
    )

    context_lengths = [c for c in [4, 16, 64, 128]
                       if c <= max_context]

    seed_prompts = NARRATIVE_PROMPTS[:5]
    all_profiles = {}

    for ctx_len in context_lengths:
        logger.info(f"\n--- Context length: {ctx_len} ---")

        # Generate trajectories
        trajectories = []
        for prompt in seed_prompts:
            tokens = extractor.tokenizer.encode(prompt)
            if len(tokens) < ctx_len:
                continue
            for _ in range(n_trajectories // len(seed_prompts) + 1):
                if len(trajectories) >= n_trajectories:
                    break
                input_ids = torch.tensor(tokens[:ctx_len])
                traj = extractor.extract_generation_trajectory(
                    input_ids, n_steps=generation_length,
                    temperature=0.8, tensor_type="V",
                )
                trajectories.append(traj)

        if len(trajectories) < 5:
            logger.warning(f"Too few trajectories for ctx={ctx_len}. Skipping.")
            continue

        # Analyze planning for V tensor at the last layer
        profile = detector.analyze_trajectory_planning(
            trajectories, layer=-1, tensor_type="V",
            context_length=ctx_len,
        )
        all_profiles[ctx_len] = profile

        logger.info(
            f"  Planning horizon: {profile.planning_horizon} tokens ahead"
        )
        for r in profile.mi_results[:5]:
            logger.info(
                f"    k={r.lookahead}: MI={r.mi_value:.4f}, "
                f"p={r.p_value:.4f}, sig={r.is_significant}"
            )

    # Compare across contexts
    if len(all_profiles) >= 2:
        comparison = detector.compare_across_contexts(all_profiles)
        viz.plot_mi_curves(all_profiles, output_dir)
        viz.plot_planning_horizon(comparison, output_dir)
        viz.plot_mi_heatmap(comparison, output_dir)
    else:
        comparison = None

    # Save results
    planning_data = {}
    for ctx_len, profile in all_profiles.items():
        planning_data[str(ctx_len)] = {
            "planning_horizon": profile.planning_horizon,
            "mi_curve": profile.mi_curve,
            "mi_details": [
                {"lookahead": r.lookahead, "mi": r.mi_value,
                 "p_value": r.p_value, "significant": r.is_significant,
                 "n_samples": r.n_samples}
                for r in profile.mi_results
            ],
        }
    save_results(planning_data, os.path.join(output_dir, "phase4_planning.json"))

    logger.info("Phase 4 complete.\n")
    return all_profiles, comparison


def main():
    args = parse_args()
    config = PipelineConfig()
    config.model.model_name = args.model
    config.model.device = args.device
    config.seed = args.seed

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging("INFO", args.output_dir)
    set_seed(args.seed)

    # Parse target layers
    target_layers = None
    if args.target_layers:
        target_layers = [int(x) for x in args.target_layers.split(",")]

    # Parse tensor types
    config.sweep.tensor_types = args.tensor_types.split(",")

    logger.info(f"Model: {args.model}, Device: {args.device}")
    logger.info(f"Output: {args.output_dir}")

    # Initialize extractor
    extractor = QKVExtractor(
        model_name=args.model,
        device=args.device,
        target_layers=target_layers,
    )

    run_all = args.phase == 0
    sweep_collection = None
    vomc_analyses = None
    vomc_comparison = None
    planning_profiles = None
    planning_comparison = None

    # Phase 1: Validation
    if run_all or args.phase == 1:
        run_phase1(extractor, config, args.output_dir, logger)

    # Phase 2: Context Sweep
    if run_all or args.phase == 2:
        sweep_collection = run_phase2(
            extractor, config, args.output_dir, logger,
            max_context=args.max_context,
        )

    # Phase 3: VOMC Construction
    if run_all or args.phase == 3:
        vomc_analyses, vomc_comparison = run_phase3(
            extractor, sweep_collection, config, args.output_dir, logger,
            n_trajectories=args.n_trajectories,
            generation_length=args.generation_length,
            max_order=args.max_order,
            n_clusters=args.n_clusters,
            max_context=args.max_context,
        )

    # Phase 4: Planning Detection
    if run_all or args.phase == 4:
        planning_profiles, planning_comparison = run_phase4(
            extractor, config, args.output_dir, logger,
            n_trajectories=args.n_trajectories,
            generation_length=args.generation_length,
            max_lookahead=args.max_lookahead,
            n_permutations=args.n_permutations,
            max_context=args.max_context,
        )

    # Summary dashboard
    if run_all:
        sweep_results = sweep_collection.results if sweep_collection else []
        viz.plot_summary_dashboard(
            sweep_results, vomc_comparison, planning_comparison, args.output_dir
        )

    logger.info("=" * 60)
    logger.info("Pipeline complete! Results saved to: " + args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()