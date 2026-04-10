import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .metrics import (
    finite_difference,
    logit_margin,
    summarize_prompt_curves,
    token_probability,
)
from .model_probe import ModelProbe
from .plots import build_summary_json, generate_plots
from .prompts import PromptSpec, default_prompt_bank


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _context_schedule(
    min_context: int,
    max_context: int,
    n_steps: int,
) -> List[int]:
    if n_steps <= 1:
        return [max(min_context, max_context)]

    log_points = np.linspace(
        np.log2(max(min_context, 1)),
        np.log2(max_context),
        n_steps,
    )
    lengths = sorted({int(round(2 ** point)) for point in log_points})
    lengths = [x for x in lengths if x >= min_context and x <= max_context]
    if max_context not in lengths:
        lengths.append(max_context)
    return sorted(set(lengths))


def _token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return int(ids[0]) if ids else -1


def _preview_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.replace("\n", " ")
    return text[: max_chars - 3].replace("\n", " ") + "..."


def _print_step_io(row: Dict, max_preview_chars: int) -> None:
    print(
        "[IO]"
        f" prompt={row['prompt_id']}"
        f" dir={row['direction']}"
        f" ctx={row['context_length']}"
    )
    print(f"  input : {_preview_text(row['context_text'], max_preview_chars)}")
    print(f"  output: {row['top_tokens']}")
    print(
        "  stats :"
        f" p_target={row['p_target']:.4f}"
        f" p_comp={row['p_competitor']:.4f}"
        f" margin={row['magnetization_margin']:.4f}"
        f" H={row['entropy_bits']:.4f}"
    )


def _run_single_direction(
    probe: ModelProbe,
    prompt_id: int,
    spec: PromptSpec,
    contexts: List[int],
    direction: str,
    temperature: float,
    top_k: int,
) -> List[Dict]:
    full_ids = probe.tokenizer.encode(spec.prompt)
    use_contexts = [c for c in contexts if c <= len(full_ids)]
    if not use_contexts:
        return []
    if direction == "backward":
        use_contexts = list(reversed(use_contexts))

    target_id = _token_id(probe.tokenizer, spec.expected)
    competitor_id = _token_id(probe.tokenizer, spec.competitor)

    rows = []
    for ctx_len in use_contexts:
        truncated = full_ids[:ctx_len]
        result = probe.probe(
            context_ids=truncated,
            anchor_tokens=spec.anchors,
            temperature=temperature,
            top_k=top_k,
        )

        row = {
            "prompt_id": prompt_id,
            "direction": direction,
            "context_length": int(ctx_len),
            "context_text": result.context_text,
            "target": spec.expected,
            "competitor": spec.competitor,
            "p_target": token_probability(result.probs, target_id),
            "p_competitor": token_probability(result.probs, competitor_id),
            "magnetization_margin": logit_margin(
                result.logits,
                target_id,
                competitor_id,
            ),
            "entropy_bits": result.entropy_bits,
            "attention_to_anchors": result.attention_to_anchors,
            "top_tokens": " | ".join(result.top_tokens),
            "top_probs": " | ".join(f"{p:.4f}" for p in result.top_probs),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["context_length"])
    margins = [float(r["magnetization_margin"]) for r in rows]
    entropies = [float(r["entropy_bits"]) for r in rows]
    contexts_sorted = [int(r["context_length"]) for r in rows]

    susc_margin = finite_difference(margins, contexts_sorted)
    susc_entropy = finite_difference(entropies, contexts_sorted)

    for row, sm, se in zip(rows, susc_margin, susc_entropy):
        row["susceptibility_margin"] = float(sm)
        row["susceptibility_entropy"] = float(se)

    if direction == "backward":
        rows = list(reversed(rows))
    return rows


def run_bidirectional_experiment(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    _set_seed(args.seed)

    probe = ModelProbe(args.model, args.device)
    prompt_bank = default_prompt_bank()[: args.max_prompts]
    contexts = _context_schedule(
        args.min_context,
        args.max_context,
        args.n_steps,
    )

    all_rows: List[Dict] = []
    per_prompt_rows: Dict[int, List[Dict]] = defaultdict(list)

    iterator = tqdm(list(enumerate(prompt_bank)), desc="Prompts")
    for prompt_id, spec in iterator:
        forward_rows = _run_single_direction(
            probe,
            prompt_id,
            spec,
            contexts,
            direction="forward",
            temperature=args.temperature,
            top_k=args.top_k,
        )
        backward_rows = _run_single_direction(
            probe,
            prompt_id,
            spec,
            contexts,
            direction="backward",
            temperature=args.temperature,
            top_k=args.top_k,
        )

        prompt_rows = forward_rows + backward_rows
        prompt_rows = sorted(
            prompt_rows,
            key=lambda r: (r["direction"], r["context_length"]),
        )

        stats = summarize_prompt_curves(prompt_rows)
        for row in prompt_rows:
            row.update(stats)
            if args.print_io:
                _print_step_io(
                    row,
                    max_preview_chars=args.max_context_preview_chars,
                )

        all_rows.extend(prompt_rows)
        per_prompt_rows[prompt_id].extend(prompt_rows)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, "observations.csv")
    df.to_csv(csv_path, index=False)

    summary = build_summary_json(df)
    summary["config"] = {
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "min_context": args.min_context,
        "max_context": args.max_context,
        "n_steps": args.n_steps,
        "max_prompts": args.max_prompts,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.no_plots:
        generate_plots(df, args.output_dir)

    print(f"Saved observations: {csv_path}")
    print(f"Saved summary: {summary_path}")
    if args.no_plots:
        print("Skipped plots (--no-plots enabled)")
    else:
        print(f"Saved plots dir: {os.path.join(args.output_dir, 'plots')}")
