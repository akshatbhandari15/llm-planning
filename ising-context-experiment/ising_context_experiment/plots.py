import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_plots(df: pd.DataFrame, output_dir: str) -> None:
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="context_length",
        y="magnetization_margin",
        hue="direction",
        style="direction",
        estimator="mean",
        errorbar=("sd", 1),
        marker="o",
        ax=ax,
    )
    ax.set_title("Magnetization (logit margin) vs Context")
    ax.set_ylabel("Target - Competitor Logit")
    ax.set_xscale("log", base=2)
    _save(fig, os.path.join(plot_dir, "magnetization_curves.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="context_length",
        y="entropy_bits",
        hue="direction",
        style="direction",
        estimator="mean",
        errorbar=("sd", 1),
        marker="o",
        ax=ax,
    )
    ax.set_title("Predictive Entropy vs Context")
    ax.set_ylabel("Entropy (bits)")
    ax.set_xscale("log", base=2)
    _save(fig, os.path.join(plot_dir, "entropy_curves.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="context_length",
        y="susceptibility_margin",
        hue="direction",
        style="direction",
        estimator="mean",
        errorbar=("sd", 1),
        marker="o",
        ax=ax,
    )
    ax.set_title("Susceptibility (dM/dc) vs Context")
    ax.set_ylabel("Finite-difference slope")
    ax.set_xscale("log", base=2)
    _save(fig, os.path.join(plot_dir, "susceptibility_curves.png"))

    summary = (
        df.groupby("prompt_id", as_index=False)["hysteresis_area_margin"]
        .max()
        .rename(columns={"hysteresis_area_margin": "hysteresis_area"})
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=summary,
        x="prompt_id",
        y="hysteresis_area",
        s=70,
        ax=ax,
    )
    ax.set_title("Per-prompt Hysteresis Area")
    ax.set_xlabel("Prompt ID")
    ax.set_ylabel("|Forward - Backward| Area")
    _save(fig, os.path.join(plot_dir, "hysteresis_scatter.png"))


def build_summary_json(df: pd.DataFrame) -> Dict:
    forward = df[df["direction"] == "forward"]
    backward = df[df["direction"] == "backward"]

    context_lengths = sorted(
        [int(x) for x in df["context_length"].unique().tolist()]
    )
    forward_final_margin = forward.groupby("prompt_id").tail(1)[
        "magnetization_margin"
    ].mean()
    backward_final_margin = backward.groupby("prompt_id").tail(1)[
        "magnetization_margin"
    ].mean()
    hysteresis_series = df.groupby("prompt_id")["hysteresis_area_margin"].max()
    mean_hysteresis = hysteresis_series.mean()
    max_hysteresis = hysteresis_series.max()

    return {
        "n_prompts": int(df["prompt_id"].nunique()),
        "context_lengths": context_lengths,
        "forward": {
            "mean_final_margin": float(forward_final_margin),
            "mean_entropy": float(forward["entropy_bits"].mean()),
            "mean_anchor_attention": float(
                forward["attention_to_anchors"].mean()
            ),
        },
        "backward": {
            "mean_final_margin": float(backward_final_margin),
            "mean_entropy": float(backward["entropy_bits"].mean()),
            "mean_anchor_attention": float(
                backward["attention_to_anchors"].mean()
            ),
        },
        "hysteresis": {
            "mean_area": float(mean_hysteresis),
            "max_area": float(max_hysteresis),
        },
    }
