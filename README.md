# VOMC-QKV: Variable-Order Markov Chain Analysis of LLM Planning

**Investigating how Q, K, V representations change under varying context lengths to detect forward planning in Large Language Models.**

## Overview

This project models the internal attention representations (Query, Key, Value) of transformer-based language models as a Variable-Order Markov Chain (VOMC). By systematically increasing the input context length and observing how the state transition structure changes, we can identify:

1. **Whether the model plans ahead** — Do current QKV states contain information about tokens multiple steps in the future?
2. **How planning scales with context** — Does the planning horizon grow as the model receives more context?
3. **Which components carry planning signals** — Is planning encoded in Q (what to attend to), K (what to offer), or V (what to propagate)?

## Background

This work builds on prior lab findings:

- **Kshitig Seth**: V tensors show strongest separation between hallucinated and good outputs (TinyLlama)
- **Radhika Patel**: V representations are most sensitive to temperature perturbation; decoding is path-dependent (GPT-2)
- **Songhee Beck**: Token embedding neighborhoods follow Zipf's law; hidden state clustering peaks mid-training then collapses (Pythia-70m, TransNormerLLM3-15B)
- **Luke Estrada**: Longer sequences are more resilient to perturbation; attention layers perform self-correction (GPT-2)

## Installation

```bash
pip install torch transformers numpy scipy scikit-learn matplotlib seaborn tqdm
```

## Quick Start

```bash
# Full pipeline with GPT-2 (CPU, ~30 min)
python run_pipeline.py

# Specific phase only
python run_pipeline.py --phase 2

# Faster run with smaller parameters
python run_pipeline.py --max-context 32 --n-trajectories 10 --generation-length 10

# Larger model on GPU
python run_pipeline.py --model gpt2-medium --device cuda --max-context 256
```

## Pipeline Architecture

```
Phase 1: State Extraction Validation
  └─ Verify QKV hooks work, inspect tensor shapes and norms

Phase 2: Context Window Sweep
  └─ For each prompt × context length:
     ├─ Extract Q, K, V at last position across all layers
     ├─ Record predicted distribution, confidence, entropy
     └─ Track state evolution (cosine similarity to full-context state)

Phase 3: VOMC Construction
  └─ For each context length:
     ├─ Generate N trajectories (autoregressive, varying temperature)
     ├─ Discretize V states into clusters (state alphabet)
     ├─ Build transition matrices at orders 1..max_order
     ├─ Select optimal order via BIC
     └─ Compare transition structure across context lengths

Phase 4: Planning Detection
  └─ For each context length × lookahead k:
     ├─ Collect (state_at_t, token_at_t+k) pairs across trajectories
     ├─ Estimate MI(state; future_token) via KSG estimator
     ├─ Test significance via permutation test
     └─ Determine planning horizon (max significant k)
```

## Project Structure

```
vomc_project/
├── run_pipeline.py              # Main entry point
├── configs/
│   └── default_config.py        # Configuration dataclasses
├── src/
│   ├── __init__.py
│   ├── state_extractor.py       # QKV extraction with forward hooks
│   ├── context_sweep.py         # Context length sweep experiments
│   ├── vomc.py                  # VOMC construction and analysis
│   ├── planning_detector.py     # MI-based planning detection
│   ├── visualization.py         # Publication-quality plots
│   └── utils.py                 # Shared utilities
└── results/                     # Generated outputs
    ├── phase1_validation.json
    ├── phase2_sweep.json
    ├── phase3_comparison.json
    ├── phase4_planning.json
    ├── confidence_curves.png
    ├── entropy_curves.png
    ├── vomc_order_selection.png
    ├── optimal_order_growth.png
    ├── mi_curves.png
    ├── planning_horizon.png
    ├── mi_heatmap.png
    └── summary_dashboard.png
```

## Key Outputs

| File | Description |
|------|-------------|
| `confidence_curves.png` | How prediction confidence grows with context |
| `optimal_order_growth.png` | Effective Markov memory depth vs context length |
| `mi_curves.png` | Mutual information vs lookahead for Q, K, V |
| `planning_horizon.png` | How far ahead the model plans vs context |
| `mi_heatmap.png` | Full MI surface (context × lookahead) |
| `summary_dashboard.png` | Combined dashboard of all key results |

## Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt2` | HuggingFace model name |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--phase` | `0` | Run specific phase (1-4), 0=all |
| `--max-context` | `128` | Maximum context length |
| `--n-trajectories` | `30` | Trajectories per context length |
| `--generation-length` | `20` | Tokens generated per trajectory |
| `--max-order` | `6` | Maximum Markov order |
| `--n-clusters` | `32` | VOMC state count |
| `--max-lookahead` | `8` | Planning detection lookahead |
| `--n-permutations` | `100` | MI significance permutations |
| `--target-layers` | all | Comma-separated layer indices |
| `--tensor-types` | `Q,K,V` | Tensor types to extract |

## Extending the Analysis

### Adding a new model
Just pass `--model <name>` with any HuggingFace causal LM. The extractor auto-detects GPT-2 and LLaMA-style architectures.

### Custom prompts
Edit `FACTUAL_PROMPTS` and `NARRATIVE_PROMPTS` in `src/utils.py`, or create your own prompt list and modify `run_pipeline.py`.

### Ising model connection (future work)
Luke's proposed Ising model analogy can be tested by treating the VOMC transition matrix as a coupling matrix and analyzing its spectral properties under varying "field strength" (context length).