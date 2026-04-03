"""
Variable-Order Markov Chain (VOMC) Construction and Analysis.

This module implements Phase 3 of the protocol:
- Discretize the continuous QKV state space into a finite set of states
- Build transition matrices at varying Markov orders
- Determine effective order via model selection (BIC/AIC)
- Analyze how transition structure changes with input context length

The key insight: by comparing VOMC structure across different context
lengths, we can identify when the model transitions from short-range
to long-range planning behavior.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("vomc_pipeline.vomc")


@dataclass
class VOMCStateSpace:
    """Discretized state space for the VOMC."""
    n_states: int
    cluster_centers: np.ndarray         # (n_states, dim)
    labels: np.ndarray                  # State assignment per sample
    pca: Optional[PCA] = None           # PCA transform if used
    scaler: Optional[StandardScaler] = None
    cluster_sizes: np.ndarray = None    # Count per cluster
    inertia: float = 0.0


@dataclass
class TransitionModel:
    """Transition model at a specific Markov order."""
    order: int
    transition_counts: Dict[Tuple, np.ndarray]  # history -> count vector
    transition_probs: Dict[Tuple, np.ndarray]   # history -> prob vector
    n_states: int
    n_valid_histories: int              # Histories with enough transitions
    log_likelihood: float
    bic: float
    aic: float
    # Per-state metrics
    mean_entropy: float                 # Mean transition entropy across states
    median_entropy: float


@dataclass
class VOMCAnalysis:
    """Complete VOMC analysis for a given context length."""
    context_length: int
    state_space: VOMCStateSpace
    models: Dict[int, TransitionModel]  # order -> TransitionModel
    optimal_order: int                  # Best order by model selection
    # Transition entropy at each order
    entropy_by_order: Dict[int, float]
    # Stationary distribution (if computable)
    stationary_dist: Optional[np.ndarray] = None


class VOMCBuilder:
    """
    Builds Variable-Order Markov Chains from sequences of QKV states.

    The process:
    1. Discretize continuous state vectors into clusters (state alphabet)
    2. Convert trajectories into sequences of discrete states
    3. Estimate transition probabilities at orders 1, 2, ..., max_order
    4. Select optimal order via BIC/AIC
    5. Analyze transition structure
    """

    def __init__(
        self,
        n_clusters: int = 64,
        max_order: int = 8,
        reduce_dim: bool = True,
        reduced_dim: int = 32,
        clustering_method: str = "kmeans",
        model_selection: str = "bic",
        min_transitions: int = 5,
    ):
        self.n_clusters = n_clusters
        self.max_order = max_order
        self.reduce_dim = reduce_dim
        self.reduced_dim = reduced_dim
        self.clustering_method = clustering_method
        self.model_selection = model_selection
        self.min_transitions = min_transitions

    def build_state_space(
        self,
        state_vectors: np.ndarray,
    ) -> VOMCStateSpace:
        """
        Discretize continuous state vectors into a finite state space.

        Args:
            state_vectors: (n_samples, dim) array of QKV states.

        Returns:
            VOMCStateSpace with cluster assignments.
        """
        n_samples, dim = state_vectors.shape
        logger.info(f"Building state space: {n_samples} samples, dim={dim}")

        # Standardize
        scaler = StandardScaler()
        X = scaler.fit_transform(state_vectors)

        # Dimensionality reduction
        pca = None
        if self.reduce_dim and dim > self.reduced_dim:
            pca = PCA(n_components=self.reduced_dim, random_state=42)
            X = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_.sum()
            logger.info(
                f"PCA: {dim}d -> {self.reduced_dim}d "
                f"(explained variance: {explained:.3f})"
            )

        # Clustering
        actual_k = min(self.n_clusters, n_samples)
        if self.clustering_method == "kmeans":
            if n_samples > 10000:
                clusterer = MiniBatchKMeans(
                    n_clusters=actual_k, random_state=42, batch_size=1024
                )
            else:
                clusterer = KMeans(
                    n_clusters=actual_k, random_state=42, n_init=10
                )
            clusterer.fit(X)
            labels = clusterer.labels_
            centers = clusterer.cluster_centers_
            inertia = clusterer.inertia_

        elif self.clustering_method == "gmm":
            gmm = GaussianMixture(
                n_components=actual_k, random_state=42, max_iter=200
            )
            gmm.fit(X)
            labels = gmm.predict(X)
            centers = gmm.means_
            inertia = -gmm.score(X) * n_samples
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        cluster_sizes = np.bincount(labels, minlength=actual_k)
        logger.info(
            f"Clustering complete: {actual_k} states, "
            f"min size={cluster_sizes.min()}, max size={cluster_sizes.max()}, "
            f"median size={np.median(cluster_sizes):.0f}"
        )

        return VOMCStateSpace(
            n_states=actual_k,
            cluster_centers=centers,
            labels=labels,
            pca=pca,
            scaler=scaler,
            cluster_sizes=cluster_sizes,
            inertia=inertia,
        )

    def assign_states(
        self,
        state_space: VOMCStateSpace,
        new_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Assign new state vectors to existing cluster states.
        """
        X = state_space.scaler.transform(new_vectors)
        if state_space.pca is not None:
            X = state_space.pca.transform(X)
        # Nearest centroid assignment
        dists = np.linalg.norm(
            X[:, None, :] - state_space.cluster_centers[None, :, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def build_transition_model(
        self,
        sequences: List[np.ndarray],
        n_states: int,
        order: int,
    ) -> TransitionModel:
        """
        Estimate transition probabilities at a given Markov order.

        Args:
            sequences: List of 1D arrays of discrete state IDs.
            n_states: Total number of states in the alphabet.
            order: Markov order (1 = first-order, 2 = second-order, etc.)

        Returns:
            TransitionModel with transition counts and probabilities.
        """
        transition_counts: Dict[Tuple, np.ndarray] = {}

        for seq in sequences:
            if len(seq) <= order:
                continue
            for t in range(order, len(seq)):
                history = tuple(seq[t - order : t])
                next_state = seq[t]
                if history not in transition_counts:
                    transition_counts[history] = np.zeros(n_states)
                transition_counts[history][next_state] += 1

        # Filter histories with too few transitions
        valid_counts = {}
        for history, counts in transition_counts.items():
            if counts.sum() >= self.min_transitions:
                valid_counts[history] = counts

        # Compute probabilities (with Laplace smoothing)
        alpha = 0.01  # Small smoothing factor
        transition_probs = {}
        for history, counts in valid_counts.items():
            smoothed = counts + alpha
            transition_probs[history] = smoothed / smoothed.sum()

        # Compute log-likelihood
        log_lik = 0.0
        n_transitions = 0
        for history, counts in valid_counts.items():
            probs = transition_probs[history]
            for state in range(n_states):
                if counts[state] > 0:
                    log_lik += counts[state] * np.log(probs[state] + 1e-15)
                    n_transitions += counts[state]

        # Compute BIC and AIC
        n_params = len(valid_counts) * (n_states - 1)
        bic = -2 * log_lik + n_params * np.log(max(n_transitions, 1))
        aic = -2 * log_lik + 2 * n_params

        # Compute transition entropy
        entropies = []
        for probs in transition_probs.values():
            p = probs[probs > 1e-15]
            p /= p.sum()
            entropies.append(-np.sum(p * np.log2(p)))

        mean_ent = np.mean(entropies) if entropies else 0.0
        median_ent = np.median(entropies) if entropies else 0.0

        return TransitionModel(
            order=order,
            transition_counts=valid_counts,
            transition_probs=transition_probs,
            n_states=n_states,
            n_valid_histories=len(valid_counts),
            log_likelihood=log_lik,
            bic=bic,
            aic=aic,
            mean_entropy=mean_ent,
            median_entropy=median_ent,
        )

    def analyze_sequences(
        self,
        sequences: List[np.ndarray],
        state_space: VOMCStateSpace,
        context_length: int = 0,
    ) -> VOMCAnalysis:
        """
        Full VOMC analysis: build models at all orders, select optimal.

        Args:
            sequences: List of discrete state sequences.
            state_space: The state space used for discretization.
            context_length: For labeling (what context length produced these).

        Returns:
            VOMCAnalysis with all models and optimal order.
        """
        n_states = state_space.n_states
        models = {}
        entropy_by_order = {}

        for order in range(1, self.max_order + 1):
            # Check if sequences are long enough
            valid_seqs = [s for s in sequences if len(s) > order]
            if len(valid_seqs) < 3:
                logger.warning(
                    f"Only {len(valid_seqs)} sequences long enough for order {order}. Stopping."
                )
                break

            model = self.build_transition_model(valid_seqs, n_states, order)
            models[order] = model
            entropy_by_order[order] = model.mean_entropy

            logger.debug(
                f"Order {order}: {model.n_valid_histories} valid histories, "
                f"BIC={model.bic:.1f}, AIC={model.aic:.1f}, "
                f"H={model.mean_entropy:.3f}"
            )

        # Select optimal order
        if not models:
            optimal_order = 1
        else:
            criterion = self.model_selection
            if criterion == "bic":
                optimal_order = min(models, key=lambda o: models[o].bic)
            elif criterion == "aic":
                optimal_order = min(models, key=lambda o: models[o].aic)
            else:
                optimal_order = 1

        logger.info(
            f"Context length {context_length}: "
            f"Optimal Markov order = {optimal_order} (by {self.model_selection})"
        )

        # Compute stationary distribution for order-1 model if available
        stationary = None
        if 1 in models:
            stationary = self._compute_stationary(models[1], n_states)

        return VOMCAnalysis(
            context_length=context_length,
            state_space=state_space,
            models=models,
            optimal_order=optimal_order,
            entropy_by_order=entropy_by_order,
            stationary_dist=stationary,
        )

    def _compute_stationary(
        self, model: TransitionModel, n_states: int
    ) -> Optional[np.ndarray]:
        """
        Compute stationary distribution for a first-order Markov chain
        via eigendecomposition of the transition matrix.
        """
        try:
            T = np.zeros((n_states, n_states))
            for (state,), probs in model.transition_probs.items():
                T[state, :] = probs

            # Ensure rows sum to 1 (fill empty rows with uniform)
            row_sums = T.sum(axis=1)
            for i in range(n_states):
                if row_sums[i] < 1e-10:
                    T[i, :] = 1.0 / n_states

            # Left eigenvector for eigenvalue 1
            eigvals, eigvecs = np.linalg.eig(T.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            stationary = np.abs(eigvecs[:, idx])
            stationary /= stationary.sum()
            return stationary

        except Exception as e:
            logger.warning(f"Could not compute stationary distribution: {e}")
            return None

    def compare_across_contexts(
        self,
        analyses: Dict[int, VOMCAnalysis],
    ) -> Dict:
        """
        Compare VOMC structure across different context lengths.

        Returns summary dict with:
            - optimal_orders: {ctx_len: optimal_order}
            - entropy_curves: {ctx_len: {order: entropy}}
            - effective_order_growth: How optimal order scales with context
            - transition_stability: JS-div between transition matrices
        """
        ctx_lengths = sorted(analyses.keys())
        optimal_orders = {c: analyses[c].optimal_order for c in ctx_lengths}
        entropy_curves = {c: analyses[c].entropy_by_order for c in ctx_lengths}

        # Compute pairwise JS-divergence between stationary distributions
        stationary_divs = {}
        for i, c1 in enumerate(ctx_lengths):
            for c2 in ctx_lengths[i + 1 :]:
                s1 = analyses[c1].stationary_dist
                s2 = analyses[c2].stationary_dist
                if s1 is not None and s2 is not None:
                    # Align dimensions
                    max_len = max(len(s1), len(s2))
                    p1 = np.zeros(max_len)
                    p2 = np.zeros(max_len)
                    p1[: len(s1)] = s1
                    p2[: len(s2)] = s2
                    p1 /= p1.sum()
                    p2 /= p2.sum()
                    m = 0.5 * (p1 + p2)
                    kl1 = np.sum(p1 * np.log((p1 + 1e-15) / (m + 1e-15)))
                    kl2 = np.sum(p2 * np.log((p2 + 1e-15) / (m + 1e-15)))
                    jsd = 0.5 * (kl1 + kl2)
                    stationary_divs[(c1, c2)] = float(jsd)

        return {
            "optimal_orders": optimal_orders,
            "entropy_curves": entropy_curves,
            "stationary_divergences": stationary_divs,
        }


def sequences_from_trajectories(
    trajectories: List[List[Dict]],
    state_space: VOMCStateSpace,
    tensor_type: str = "V",
    layer: int = -1,
) -> List[np.ndarray]:
    """
    Convert generation trajectories (from QKVExtractor) into discrete
    state sequences for VOMC analysis.

    Args:
        trajectories: Output of extract_generation_trajectory().
        state_space: VOMCStateSpace for discretization.
        tensor_type: Which tensor type to use ("Q", "K", or "V").
        layer: Which layer to use (-1 = last available).

    Returns:
        List of 1D arrays of discrete state IDs.
    """
    discrete_sequences = []

    for traj in trajectories:
        # Stack the state vectors from this trajectory
        vectors = []
        for step in traj:
            if layer == -1:
                actual_layer = max(step["states"].keys())
            else:
                actual_layer = layer
            vec = step["states"].get(actual_layer)
            if vec is not None:
                vectors.append(vec)

        if len(vectors) < 2:
            continue

        X = np.array(vectors)
        # Transform to the state space
        X_scaled = state_space.scaler.transform(X)
        if state_space.pca is not None:
            X_scaled = state_space.pca.transform(X_scaled)

        # Assign to nearest cluster
        dists = np.linalg.norm(
            X_scaled[:, None, :] - state_space.cluster_centers[None, :, :],
            axis=2,
        )
        seq = np.argmin(dists, axis=1)
        discrete_sequences.append(seq)

    return discrete_sequences