#!/usr/bin/env python3
import argparse

from ising_context_experiment.experiment import run_bidirectional_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ising-style context sweep for language model "
            "logits and attention"
        )
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"]
    )
    parser.add_argument("--output-dir", type=str, default="./results_ising")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-context", type=int, default=4)
    parser.add_argument("--max-context", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=7)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bidirectional_experiment(args)


if __name__ == "__main__":
    main()
