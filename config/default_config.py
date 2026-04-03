"""
Default configuration for the VOMC-QKV analysis pipeline.

Modify these settings to adjust model, context sweep parameters,
VOMC construction, and planning detection thresholds.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Transformer model settings."""
    model_name: str = "gpt2"
    device: str = "cpu"                # "cpu" or "cuda"
    dtype: str = "float32"             # "float16" for GPU speedup
    max_seq_length: int = 512


@dataclass
class ContextSweepConfig:
    """Context window sweep parameters."""
    # Context lengths to test (tokens)
    context_lengths: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    )
    # Number of samples per context length for statistical robustness
    n_samples_per_length: int = 50
    # Temperatures to probe (parallel to Radhika/Kshitig's approach)
    temperatures: List[float] = field(
        default_factory=lambda: [0.2, 0.7, 1.0]
    )
    # Max tokens to generate after each context
    generation_length: int = 20
    # Layers to extract from (None = all layers)
    target_layers: Optional[List[int]] = None
    # Which tensor types to extract
    tensor_types: List[str] = field(
        default_factory=lambda: ["Q", "K", "V"]
    )


@dataclass
class VOMCConfig:
    """Variable-Order Markov Chain construction settings."""
    # Maximum Markov order to test
    max_order: int = 8
    # State space discretization
    n_clusters: int = 64                # Number of discrete states
    clustering_method: str = "kmeans"   # "kmeans" or "gmm"
    # Dimensionality reduction before clustering
    reduce_dim: bool = True
    reduced_dim: int = 32               # Target dimensionality (PCA)
    # Model selection criterion
    model_selection: str = "bic"        # "bic" or "aic"
    # Minimum transitions to consider a state valid
    min_transitions: int = 5


@dataclass
class PlanningConfig:
    """Planning detection parameters."""
    # How far ahead to look for planned tokens
    max_lookahead: int = 10
    # Mutual information estimation method
    mi_method: str = "ksg"             # "ksg" (Kraskov) or "binned"
    mi_n_neighbors: int = 5            # For KSG estimator
    # Significance testing
    n_permutations: int = 200          # For permutation test
    alpha: float = 0.05                # Significance level
    # Token categories to track separately
    track_categories: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sweep: ContextSweepConfig = field(default_factory=ContextSweepConfig)
    vomc: VOMCConfig = field(default_factory=VOMCConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    # Output directory
    output_dir: str = "./results"
    # Random seed for reproducibility
    seed: int = 42
    # Logging level
    log_level: str = "INFO"