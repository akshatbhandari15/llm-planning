from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ProbeResult:
    context_text: str
    token_ids: List[int]
    probs: np.ndarray
    logits: np.ndarray
    entropy_bits: float
    attention_to_anchors: float
    top_tokens: List[str]
    top_probs: List[float]


class ModelProbe:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def _entropy_bits(self, probs: np.ndarray) -> float:
        p = probs[probs > 1e-15]
        return float(-np.sum(p * np.log2(p)))

    def _anchor_attention(
        self,
        attentions: List[torch.Tensor],
        input_ids: List[int],
        anchor_tokens: List[str],
    ) -> float:
        if not attentions:
            return 0.0

        anchor_ids = set()
        for token in anchor_tokens:
            token_piece_ids = self.tokenizer.encode(
                token,
                add_special_tokens=False,
            )
            for token_id in token_piece_ids:
                anchor_ids.add(token_id)

        if not anchor_ids:
            return 0.0

        anchor_positions = [
            idx for idx, tid in enumerate(input_ids) if tid in anchor_ids
        ]
        if not anchor_positions:
            return 0.0

        last_layer = attentions[-1][0]
        mean_head = last_layer.mean(dim=0)
        last_query = mean_head[-1]
        scores = [float(last_query[pos].item()) for pos in anchor_positions]
        return float(np.mean(scores)) if scores else 0.0

    @torch.no_grad()
    def probe(
        self,
        context_ids: List[int],
        anchor_tokens: List[str],
        temperature: float = 1.0,
        top_k: int = 5,
    ) -> ProbeResult:
        input_tensor = torch.tensor([context_ids], device=self.device)
        outputs = self.model(input_tensor, output_attentions=True)

        logits = outputs.logits[0, -1, :].float().cpu().numpy()
        scaled = logits / max(temperature, 1e-8)
        scaled = scaled - scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()

        top_idx = np.argsort(probs)[::-1][:top_k]
        top_tokens = [self.tokenizer.decode([int(i)]) for i in top_idx]
        top_probs = [float(probs[int(i)]) for i in top_idx]

        return ProbeResult(
            context_text=self.tokenizer.decode(context_ids),
            token_ids=context_ids,
            probs=probs,
            logits=logits,
            entropy_bits=self._entropy_bits(probs),
            attention_to_anchors=self._anchor_attention(
                outputs.attentions,
                context_ids,
                anchor_tokens,
            ),
            top_tokens=top_tokens,
            top_probs=top_probs,
        )
