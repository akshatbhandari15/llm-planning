"""
Utility functions shared across the VOMC-QKV pipeline.
"""

import os
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch


def setup_logging(level: str = "INFO", log_dir: str = "./results") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("vomc_pipeline")
    logger.setLevel(getattr(logging, level))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"run_{ts}.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity between rows of A and B.
    A: (n, d), B: (m, d) -> returns (n, m) similarity matrix.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


def cosine_similarity_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Jensen-Shannon divergence between two probability distributions.
    Symmetric and bounded in [0, ln(2)].
    """
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)


def entropy(p: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon entropy of a probability distribution."""
    p = np.asarray(p, dtype=np.float64) + eps
    p /= p.sum()
    return -np.sum(p * np.log2(p))


def transition_entropy(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-state entropy of a transition matrix.
    transition_matrix: (n_states, n_states), rows sum to 1.
    Returns: (n_states,) array of entropies.
    """
    entropies = np.zeros(transition_matrix.shape[0])
    for i in range(transition_matrix.shape[0]):
        row = transition_matrix[i]
        if row.sum() > 0:
            entropies[i] = entropy(row)
    return entropies


def save_results(data: dict, filepath: str):
    """Save results as JSON (converting numpy arrays to lists)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, "w") as f:
        json.dump(convert(data), f, indent=2)


def load_results(filepath: str) -> dict:
    """Load results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


# ── Prompt datasets ──────────────────────────────────────────────────────

# Factual prompts with known continuations (for controlled experiments)
FACTUAL_PROMPTS = [
    {"prompt": "The capital of France is", "expected": " Paris"},
    {"prompt": "Water boils at a temperature of", "expected": " 100"},
    {"prompt": "The largest planet in our solar system is", "expected": " Jupiter"},
    {"prompt": "The chemical symbol for gold is", "expected": " Au"},
    {"prompt": "The speed of light is approximately", "expected": " 300"},
    {"prompt": "DNA stands for deoxyribonucle", "expected": "ic"},
    {"prompt": "The author of Romeo and Juliet is William", "expected": " Shakespeare"},
    {"prompt": "Photosynthesis converts sunlight into", "expected": " energy"},
    {"prompt": "The Great Wall of China was built to protect against", "expected": " inv"},
    {"prompt": "In mathematics, pi is approximately equal to", "expected": " 3"},
    {"prompt": "The human heart has four", "expected": " cham"},
    {"prompt": "The Amazon River flows through South", "expected": " America"},
    {"prompt": "Albert Einstein developed the theory of", "expected": " relat"},
    {"prompt": "Oxygen makes up approximately 21 percent of Earth's", "expected": " atmosphere"},
    {"prompt": "The Declaration of Independence was signed in", "expected": " 17"},
]

# Narrative prompts for multi-token generation analysis
NARRATIVE_PROMPTS = [
    "Once upon a time in a kingdom far away, there lived a brave knight who",
    "The scientist carefully examined the results of the experiment and noticed that",
    "After years of research, the team finally discovered that the ancient artifact was",
    "The bird soared through the clear blue sky, its wings catching the warm updraft as it",
    "In the depths of the ocean, a creature unlike anything ever seen before slowly",
    "The professor stood before the class and began to explain the fundamental principles of",
    "As the sun set over the mountains, the travelers realized they had been walking toward",
    "The algorithm processed millions of data points and determined that the optimal solution was",
    "During the final moments of the game, the player made an unexpected decision to",
    "The ancient manuscript contained a message that, when translated, revealed the location of",
]


def get_incremental_contexts(text: str, tokenizer, max_lengths: List[int]) -> List[dict]:
    """
    Create incrementally longer contexts from a text.
    Returns list of {context_ids, context_length, full_text} dicts.
    """
    tokens = tokenizer.encode(text)
    contexts = []
    for length in max_lengths:
        if length > len(tokens):
            continue
        ctx_ids = tokens[:length]
        contexts.append({
            "context_ids": ctx_ids,
            "context_length": length,
            "context_text": tokenizer.decode(ctx_ids),
        })
    return contexts