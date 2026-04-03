"""
State Extractor: Extract Q, K, V representations from transformer models.

This module attaches forward hooks to attention layers to capture the
Query, Key, and Value projections at each layer during inference.
Supports GPT-2 family and compatible architectures.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("vomc_pipeline.extractor")


@dataclass
class QKVState:
    """Container for extracted Q, K, V states at a single position."""
    layer: int
    position: int                # Token position in the sequence
    Q: np.ndarray               # (n_heads * head_dim,) or (n_heads, head_dim)
    K: np.ndarray
    V: np.ndarray
    token_id: Optional[int] = None
    token_str: Optional[str] = None


@dataclass
class SequenceQKVProfile:
    """Complete QKV profile for a full sequence across all layers."""
    context_length: int
    context_text: str
    states: Dict[int, List[QKVState]]   # layer -> list of QKVStates per position
    logits: Optional[np.ndarray] = None # Next-token logits at the last position
    predicted_token: Optional[str] = None
    predicted_prob: Optional[float] = None


class QKVExtractor:
    """
    Extracts Q, K, V representations from a causal language model.

    Usage:
        extractor = QKVExtractor("gpt2")
        profile = extractor.extract(input_ids, target_layers=[0, 5, 11])
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "cpu",
        target_layers: Optional[List[int]] = None,
    ):
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device

        # Detect architecture and set up layer references
        self._setup_architecture(target_layers)

        # Storage for hooked activations
        self._qkv_cache: Dict[int, dict] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    def _setup_architecture(self, target_layers: Optional[List[int]]):
        """Detect model architecture and identify attention modules."""
        config = self.model.config

        # Determine number of layers and attention structure
        if hasattr(config, "n_layer"):
            # GPT-2 style
            self.n_layers = config.n_layer
            self.n_heads = config.n_head
            self.head_dim = config.n_embd // config.n_head
            self.hidden_dim = config.n_embd
            self.arch = "gpt2"
        elif hasattr(config, "num_hidden_layers"):
            # LLaMA / generic style
            self.n_layers = config.num_hidden_layers
            self.n_heads = config.num_attention_heads
            self.head_dim = config.hidden_size // config.num_attention_heads
            self.hidden_dim = config.hidden_size
            self.arch = "generic"
        else:
            raise ValueError(f"Unsupported architecture: {type(config)}")

        if target_layers is None:
            self.target_layers = list(range(self.n_layers))
        else:
            self.target_layers = [l for l in target_layers if l < self.n_layers]

        logger.info(
            f"Architecture: {self.arch}, Layers: {self.n_layers}, "
            f"Heads: {self.n_heads}, HeadDim: {self.head_dim}, "
            f"Extracting layers: {self.target_layers}"
        )

    def _get_attention_module(self, layer_idx: int) -> nn.Module:
        """Get the attention module for a specific layer."""
        if self.arch == "gpt2":
            return self.model.transformer.h[layer_idx].attn
        else:
            # Generic: try common patterns
            if hasattr(self.model, "model"):
                return self.model.model.layers[layer_idx].self_attn
            raise ValueError(f"Cannot locate attention module for arch={self.arch}")

    def _make_hook(self, layer_idx: int):
        """
        Create a forward hook that captures Q, K, V projections.

        For GPT-2, the attention module computes Q, K, V via a single
        combined projection (c_attn) followed by a split. We hook into
        the attention module's forward and manually extract Q, K, V.
        """
        def hook_fn(module, input_args, output):
            with torch.no_grad():
                hidden_states = input_args[0]  # (batch, seq_len, hidden_dim)

                if self.arch == "gpt2":
                    # GPT-2: c_attn projects to 3*hidden_dim, then splits
                    qkv = module.c_attn(hidden_states)  # (batch, seq, 3*hidden)
                    q, k, v = qkv.split(self.hidden_dim, dim=-1)
                else:
                    # Generic: separate q_proj, k_proj, v_proj
                    q = module.q_proj(hidden_states)
                    k = module.k_proj(hidden_states)
                    v = module.v_proj(hidden_states)

                self._qkv_cache[layer_idx] = {
                    "Q": q.cpu().numpy(),   # (batch, seq_len, hidden_dim)
                    "K": k.cpu().numpy(),
                    "V": v.cpu().numpy(),
                }

        return hook_fn

    def _register_hooks(self):
        """Attach forward hooks to target attention layers."""
        self._remove_hooks()
        self._qkv_cache.clear()
        for layer_idx in self.target_layers:
            attn = self._get_attention_module(layer_idx)
            hook = attn.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    @torch.no_grad()
    def extract(
        self,
        input_ids: torch.Tensor,
        return_logits: bool = True,
        positions: Optional[List[int]] = None,
    ) -> SequenceQKVProfile:
        """
        Extract QKV states for a given input sequence.

        Args:
            input_ids: (1, seq_len) tensor of token IDs.
            return_logits: Whether to capture next-token logits.
            positions: Specific token positions to extract (None = all).

        Returns:
            SequenceQKVProfile with layer-wise QKV states.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        # Register hooks and run forward pass
        self._register_hooks()
        outputs = self.model(input_ids)
        self._remove_hooks()

        # Process logits
        logits_np = None
        predicted_token = None
        predicted_prob = None
        if return_logits:
            logits = outputs.logits[0, -1, :]  # Last position logits
            probs = torch.softmax(logits, dim=-1)
            top_id = torch.argmax(probs).item()
            logits_np = logits.cpu().numpy()
            predicted_token = self.tokenizer.decode([top_id])
            predicted_prob = probs[top_id].item()

        # Build per-layer, per-position QKVState objects
        if positions is None:
            positions = list(range(seq_len))

        token_ids = input_ids[0].cpu().tolist()
        states: Dict[int, List[QKVState]] = {}

        for layer_idx in self.target_layers:
            cache = self._qkv_cache.get(layer_idx)
            if cache is None:
                continue
            layer_states = []
            for pos in positions:
                if pos >= seq_len:
                    continue
                state = QKVState(
                    layer=layer_idx,
                    position=pos,
                    Q=cache["Q"][0, pos, :],    # (hidden_dim,)
                    K=cache["K"][0, pos, :],
                    V=cache["V"][0, pos, :],
                    token_id=token_ids[pos],
                    token_str=self.tokenizer.decode([token_ids[pos]]),
                )
                layer_states.append(state)
            states[layer_idx] = layer_states

        context_text = self.tokenizer.decode(token_ids)

        return SequenceQKVProfile(
            context_length=seq_len,
            context_text=context_text,
            states=states,
            logits=logits_np,
            predicted_token=predicted_token,
            predicted_prob=predicted_prob,
        )

    @torch.no_grad()
    def extract_last_position(
        self,
        input_ids: torch.Tensor,
        tensor_type: str = "V",
    ) -> Dict[int, np.ndarray]:
        """
        Quick extraction of a single tensor type at the last position
        across all target layers. Returns {layer: vector} dict.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        self._register_hooks()
        self.model(input_ids)
        self._remove_hooks()

        result = {}
        for layer_idx in self.target_layers:
            cache = self._qkv_cache.get(layer_idx)
            if cache is not None:
                result[layer_idx] = cache[tensor_type][0, -1, :]
        return result

    @torch.no_grad()
    def extract_generation_trajectory(
        self,
        input_ids: torch.Tensor,
        n_steps: int = 20,
        temperature: float = 1.0,
        tensor_type: str = "V",
    ) -> List[Dict]:
        """
        Generate tokens autoregressively and extract QKV state at each step.

        Returns list of dicts, one per generation step:
            {
                "step": int,
                "token_id": int,
                "token_str": str,
                "prob": float,
                "surprisal": float,
                "states": {layer: vector},
            }
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        trajectory = []

        for step in range(n_steps):
            # Register hooks and forward
            self._register_hooks()
            outputs = self.model(input_ids)
            self._remove_hooks()

            # Sample next token
            logits = outputs.logits[0, -1, :]
            scaled_logits = logits / max(temperature, 1e-8)
            probs = torch.softmax(scaled_logits, dim=-1)

            if temperature < 0.01:
                next_id = torch.argmax(probs).item()
            else:
                next_id = torch.multinomial(probs, 1).item()

            token_prob = probs[next_id].item()
            surprisal = -np.log2(max(token_prob, 1e-15))

            # Collect states at the last position
            step_states = {}
            for layer_idx in self.target_layers:
                cache = self._qkv_cache.get(layer_idx)
                if cache is not None:
                    step_states[layer_idx] = cache[tensor_type][0, -1, :]

            trajectory.append({
                "step": step,
                "token_id": next_id,
                "token_str": self.tokenizer.decode([next_id]),
                "prob": token_prob,
                "surprisal": surprisal,
                "entropy": entropy_from_probs(probs.cpu().numpy()),
                "states": step_states,
            })

            # Append token and continue
            next_tensor = torch.tensor([[next_id]], device=self.device)
            input_ids = torch.cat([input_ids, next_tensor], dim=1)

        return trajectory


def entropy_from_probs(probs: np.ndarray, top_k: int = 100) -> float:
    """Compute entropy from a probability distribution (top-k approximation)."""
    sorted_p = np.sort(probs)[::-1][:top_k]
    sorted_p = sorted_p[sorted_p > 1e-15]
    sorted_p /= sorted_p.sum()
    return -np.sum(sorted_p * np.log2(sorted_p))