"""
Context Sweep: Systematically vary input context length and collect
QKV state profiles to study how planning changes with context.

This module implements Phase 2 of the protocol:
- Take prompts with known continuations
- Truncate to increasing context lengths
- Extract QKV states at each length
- Track how predicted distributions change
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import tqdm

from .state_extractor import QKVExtractor, SequenceQKVProfile
from .utils import cosine_similarity_vectors, entropy, js_divergence

logger = logging.getLogger("vomc_pipeline.sweep")


@dataclass
class ContextSweepResult:
    """Results for a single prompt across all context lengths."""
    prompt_text: str
    expected_token: str
    context_lengths: List[int]
    # Per context length: QKV profile at the last position
    profiles: List[SequenceQKVProfile]
    # Per context length: top-k predicted tokens and probs
    predictions: List[Dict]
    # Per context length: per-layer state vectors (for VOMC input)
    layer_states: Dict[str, Dict[int, List[np.ndarray]]]
    # Derived metrics
    confidence_curve: List[float]       # P(expected_token) vs context length
    entropy_curve: List[float]          # Entropy vs context length
    rank_curve: List[int]               # Rank of expected token vs context length


@dataclass
class ContextSweepCollection:
    """Collection of sweep results across all prompts."""
    results: List[ContextSweepResult]
    context_lengths: List[int]
    # Aggregated state matrices for VOMC construction
    # Key: tensor_type -> layer -> list of state vectors (across all prompts & lengths)
    aggregated_states: Dict[str, Dict[int, List[np.ndarray]]] = field(
        default_factory=dict
    )
    # Metadata for each aggregated state: (prompt_idx, context_length, position)
    state_metadata: List[Tuple[int, int, int]] = field(default_factory=list)


class ContextSweeper:
    """
    Performs the context window sweep experiment.

    For each prompt, creates truncated versions at specified lengths,
    extracts QKV states, and tracks how predictions evolve.
    """

    def __init__(
        self,
        extractor: QKVExtractor,
        context_lengths: List[int] = None,
        tensor_types: List[str] = None,
        top_k: int = 20,
    ):
        self.extractor = extractor
        self.tokenizer = extractor.tokenizer
        self.context_lengths = context_lengths or [1, 2, 4, 8, 16, 32, 64, 128]
        self.tensor_types = tensor_types or ["Q", "K", "V"]
        self.top_k = top_k

    def sweep_prompt(
        self,
        prompt: str,
        expected_token: str = None,
    ) -> ContextSweepResult:
        """
        Run context sweep on a single prompt.

        Args:
            prompt: Full prompt text.
            expected_token: Expected next token (for tracking confidence).

        Returns:
            ContextSweepResult with all extracted data.
        """
        # Tokenize the full prompt
        full_ids = self.tokenizer.encode(prompt)

        # Determine which context lengths are feasible
        feasible_lengths = [c for c in self.context_lengths if c <= len(full_ids)]
        if len(full_ids) not in feasible_lengths:
            feasible_lengths.append(len(full_ids))
        feasible_lengths = sorted(set(feasible_lengths))

        profiles = []
        predictions = []
        confidence_curve = []
        entropy_curve = []
        rank_curve = []
        layer_states = {t: {} for t in self.tensor_types}

        # Find the expected token id
        expected_id = None
        if expected_token:
            expected_ids = self.tokenizer.encode(expected_token)
            if expected_ids:
                expected_id = expected_ids[0]

        for ctx_len in feasible_lengths:
            # Truncate to this context length
            truncated_ids = torch.tensor(full_ids[:ctx_len])

            # Extract full QKV profile
            profile = self.extractor.extract(truncated_ids, return_logits=True)
            profiles.append(profile)

            # Get top-k predictions
            if profile.logits is not None:
                logits = profile.logits
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                top_indices = np.argsort(probs)[::-1][:self.top_k]
                top_preds = {
                    "tokens": [self.tokenizer.decode([idx]) for idx in top_indices],
                    "token_ids": top_indices.tolist(),
                    "probs": probs[top_indices].tolist(),
                }
                predictions.append(top_preds)

                # Confidence in expected token
                if expected_id is not None and expected_id < len(probs):
                    confidence_curve.append(float(probs[expected_id]))
                    rank = int(np.where(np.argsort(probs)[::-1] == expected_id)[0][0]) + 1
                    rank_curve.append(rank)
                else:
                    confidence_curve.append(0.0)
                    rank_curve.append(len(probs))

                # Entropy of distribution
                entropy_curve.append(entropy(probs))
            else:
                predictions.append({})
                confidence_curve.append(0.0)
                entropy_curve.append(0.0)
                rank_curve.append(0)

            # Collect per-layer states for each tensor type at the last position
            for tensor_type in self.tensor_types:
                last_pos_states = self.extractor.extract_last_position(
                    truncated_ids, tensor_type=tensor_type
                )
                for layer, vec in last_pos_states.items():
                    if layer not in layer_states[tensor_type]:
                        layer_states[tensor_type][layer] = []
                    layer_states[tensor_type][layer].append(vec)

        return ContextSweepResult(
            prompt_text=prompt,
            expected_token=expected_token or "",
            context_lengths=feasible_lengths,
            profiles=profiles,
            predictions=predictions,
            layer_states=layer_states,
            confidence_curve=confidence_curve,
            entropy_curve=entropy_curve,
            rank_curve=rank_curve,
        )

    def sweep_all(
        self,
        prompts: List[Dict],
        show_progress: bool = True,
    ) -> ContextSweepCollection:
        """
        Run context sweep on all prompts.

        Args:
            prompts: List of {"prompt": str, "expected": str} dicts.
            show_progress: Show tqdm progress bar.

        Returns:
            ContextSweepCollection with all results and aggregated states.
        """
        results = []
        iterator = tqdm(prompts, desc="Sweeping prompts") if show_progress else prompts

        for p in iterator:
            result = self.sweep_prompt(
                prompt=p["prompt"],
                expected_token=p.get("expected"),
            )
            results.append(result)

        # Aggregate states across all prompts for VOMC construction
        collection = ContextSweepCollection(
            results=results,
            context_lengths=self.context_lengths,
        )
        self._aggregate_states(collection)

        return collection

    def _aggregate_states(self, collection: ContextSweepCollection):
        """
        Aggregate QKV states from all prompts into matrices
        suitable for VOMC state-space construction.
        """
        aggregated = {t: {} for t in self.tensor_types}
        metadata = []

        for prompt_idx, result in enumerate(collection.results):
            for tensor_type in self.tensor_types:
                for layer, vecs in result.layer_states[tensor_type].items():
                    if layer not in aggregated[tensor_type]:
                        aggregated[tensor_type][layer] = []
                    for i, vec in enumerate(vecs):
                        aggregated[tensor_type][layer].append(vec)
                        if tensor_type == self.tensor_types[0]:
                            ctx_len = result.context_lengths[i]
                            metadata.append((prompt_idx, ctx_len, -1))

        collection.aggregated_states = aggregated
        collection.state_metadata = metadata

    def compute_state_evolution(
        self,
        result: ContextSweepResult,
        tensor_type: str = "V",
        layer: int = -1,
    ) -> Dict:
        """
        Compute how QKV states evolve with context length for a single prompt.

        Returns dict with:
            - cosine_to_final: cosine similarity between each context state
              and the full-context state
            - consecutive_cosines: cosine similarity between consecutive states
            - state_norms: L2 norm of state at each context length
        """
        states = result.layer_states.get(tensor_type, {})
        if layer == -1:
            layer = max(states.keys()) if states else 0
        vectors = states.get(layer, [])

        if len(vectors) < 2:
            return {"cosine_to_final": [], "consecutive_cosines": [], "state_norms": []}

        final_state = vectors[-1]
        cosine_to_final = [
            cosine_similarity_vectors(v, final_state) for v in vectors
        ]
        consecutive_cosines = [
            cosine_similarity_vectors(vectors[i], vectors[i + 1])
            for i in range(len(vectors) - 1)
        ]
        state_norms = [float(np.linalg.norm(v)) for v in vectors]

        return {
            "cosine_to_final": cosine_to_final,
            "consecutive_cosines": consecutive_cosines,
            "state_norms": state_norms,
        }