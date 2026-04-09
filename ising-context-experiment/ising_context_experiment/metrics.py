from typing import Dict, List

import numpy as np


def token_probability(probs: np.ndarray, token_id: int) -> float:
    if token_id < 0 or token_id >= len(probs):
        return 0.0
    return float(probs[token_id])


def logit_margin(
    logits: np.ndarray,
    target_id: int,
    competitor_id: int,
) -> float:
    if target_id >= len(logits) or competitor_id >= len(logits):
        return 0.0
    return float(logits[target_id] - logits[competitor_id])


def finite_difference(values: List[float], contexts: List[int]) -> List[float]:
    if len(values) < 2:
        return [0.0 for _ in values]

    diffs = [0.0]
    for idx in range(1, len(values)):
        dc = max(contexts[idx] - contexts[idx - 1], 1)
        diffs.append((values[idx] - values[idx - 1]) / dc)
    return diffs


def hysteresis_area(
    contexts: List[int],
    forward_values: List[float],
    backward_values: List[float],
) -> float:
    if not contexts:
        return 0.0

    aligned = []
    backward_map = {c: v for c, v in zip(contexts, backward_values)}
    for c, fv in zip(contexts, forward_values):
        bv = backward_map.get(c)
        if bv is not None:
            aligned.append((c, fv, bv))

    if len(aligned) < 2:
        return 0.0

    area = 0.0
    for i in range(1, len(aligned)):
        c0, f0, b0 = aligned[i - 1]
        c1, f1, b1 = aligned[i]
        delta0 = abs(f0 - b0)
        delta1 = abs(f1 - b1)
        width = c1 - c0
        area += 0.5 * (delta0 + delta1) * width
    return float(area)


def summarize_prompt_curves(records: List[Dict]) -> Dict:
    if not records:
        return {
            "hysteresis_area_margin": 0.0,
            "max_abs_susceptibility": 0.0,
            "critical_context": None,
        }

    contexts = [
        int(r["context_length"])
        for r in records
        if r["direction"] == "forward"
    ]
    margins_f = [
        float(r["magnetization_margin"])
        for r in records
        if r["direction"] == "forward"
    ]
    margins_b = [
        float(r["magnetization_margin"])
        for r in records
        if r["direction"] == "backward"
    ]
    susc_f = [
        float(r["susceptibility_margin"])
        for r in records
        if r["direction"] == "forward"
    ]

    if not contexts:
        return {
            "hysteresis_area_margin": 0.0,
            "max_abs_susceptibility": 0.0,
            "critical_context": None,
        }

    area = hysteresis_area(contexts, margins_f, margins_b)
    max_abs_susc = float(max((abs(v) for v in susc_f), default=0.0))
    crit_idx = int(np.argmax(np.abs(susc_f))) if susc_f else 0
    critical_context = int(contexts[crit_idx]) if contexts else None

    return {
        "hysteresis_area_margin": area,
        "max_abs_susceptibility": max_abs_susc,
        "critical_context": critical_context,
    }
