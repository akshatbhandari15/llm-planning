"""
Planning Detector: Identify tokens that are "planned ahead" by the model.

This module implements Phase 4 of the protocol:
- Compute mutual information between current QKV state and future tokens
- Test whether MI is significant via permutation tests
- Track how planning horizon changes with context length
- Identify which token types are planned furthest ahead

Key hypothesis: If the model plans ahead, the QKV state at position t
should contain information about tokens at t+2, t+3, etc., and this
predictive power should increase with longer input context.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

logger = logging.getLogger("vomc_pipeline.planning")


@dataclass
class MutualInformationResult:
    """MI between state at position t and token at position t+k."""
    lookahead: int                  # k (how far ahead)
    mi_value: float                 # Estimated MI in bits
    mi_std: float                   # Standard error of MI estimate
    p_value: float                  # Permutation test p-value
    is_significant: bool            # Below alpha threshold
    n_samples: int                  # Number of (state, token) pairs used


@dataclass
class PlanningProfile:
    """Planning profile for a specific context length and layer."""
    context_length: int
    layer: int
    tensor_type: str
    mi_results: List[MutualInformationResult]     # One per lookahead
    planning_horizon: int           # Max significant lookahead
    mi_curve: List[float]           # MI values across lookaheads
    # Token-category breakdown (if computed)
    category_horizons: Optional[Dict[str, int]] = None


@dataclass
class PlanningComparison:
    """Comparison of planning across context lengths."""
    profiles: Dict[int, PlanningProfile]    # ctx_len -> PlanningProfile
    horizon_curve: Dict[int, int]           # ctx_len -> planning horizon
    mi_surfaces: Dict[int, List[float]]     # ctx_len -> MI curve


class PlanningDetector:
    """
    Detects forward planning by measuring mutual information between
    current QKV states and future tokens.
    """

    def __init__(
        self,
        max_lookahead: int = 10,
        mi_method: str = "ksg",
        mi_n_neighbors: int = 5,
        n_permutations: int = 200,
        alpha: float = 0.05,
    ):
        self.max_lookahead = max_lookahead
        self.mi_method = mi_method
        self.mi_n_neighbors = mi_n_neighbors
        self.n_permutations = n_permutations
        self.alpha = alpha

    def compute_mi_ksg(
        self,
        continuous_X: np.ndarray,
        discrete_Y: np.ndarray,
    ) -> float:
        """
        Estimate mutual information I(X; Y) where X is continuous and
        Y is discrete, using the KSG estimator adapted for mixed variables.

        This uses the approach: I(X;Y) = H(Y) - H(Y|X), where H(Y|X) is
        estimated by looking at nearest neighbors in X-space.

        Args:
            continuous_X: (n, d) continuous variable (QKV state).
            discrete_Y: (n,) discrete variable (token ID).

        Returns:
            Estimated MI in nats (converted to bits by dividing by ln(2)).
        """
        n = len(continuous_X)
        if n < self.mi_n_neighbors + 1:
            return 0.0

        unique_labels, label_counts = np.unique(discrete_Y, return_counts=True)
        n_classes = len(unique_labels)

        if n_classes <= 1:
            return 0.0

        # H(Y) - marginal entropy of discrete variable
        p_y = label_counts / n
        h_y = -np.sum(p_y * np.log(p_y + 1e-15))

        # H(Y|X) - conditional entropy estimated via k-NN
        # For each x_i, find its k nearest neighbors and compute the
        # empirical distribution of Y among those neighbors
        k = min(self.mi_n_neighbors, n - 1)

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(continuous_X)
        _, indices = nn.kneighbors(continuous_X)

        h_y_given_x = 0.0
        for i in range(n):
            # Exclude self (first neighbor is self)
            neighbor_ids = indices[i, 1:]
            neighbor_labels = discrete_Y[neighbor_ids]
            # Empirical distribution of Y among neighbors
            vals, counts = np.unique(neighbor_labels, return_counts=True)
            p_local = counts / counts.sum()
            h_y_given_x += -np.sum(p_local * np.log(p_local + 1e-15))

        h_y_given_x /= n

        # MI = H(Y) - H(Y|X)
        mi_nats = max(0.0, h_y - h_y_given_x)
        mi_bits = mi_nats / np.log(2)
        return mi_bits

    def compute_mi_binned(
        self,
        continuous_X: np.ndarray,
        discrete_Y: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """
        Estimate MI using binning approach (faster but less accurate).

        Bins the continuous variable and uses standard discrete MI formula.
        """
        n = len(continuous_X)
        if n < 10:
            return 0.0

        # Reduce X to 1D via first PC if multi-dimensional
        if continuous_X.ndim > 1 and continuous_X.shape[1] > 1:
            # Use first principal component
            centered = continuous_X - continuous_X.mean(axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            X_1d = U[:, 0] * S[0]
        else:
            X_1d = continuous_X.ravel()

        # Bin X
        bins = np.linspace(X_1d.min() - 1e-10, X_1d.max() + 1e-10, n_bins + 1)
        X_binned = np.digitize(X_1d, bins) - 1

        # Compute MI between two discrete variables
        unique_x = np.unique(X_binned)
        unique_y = np.unique(discrete_Y)

        # Joint probability table
        p_xy = np.zeros((len(unique_x), len(unique_y)))
        x_map = {v: i for i, v in enumerate(unique_x)}
        y_map = {v: i for i, v in enumerate(unique_y)}

        for i in range(n):
            xi = x_map[X_binned[i]]
            yi = y_map[discrete_Y[i]]
            p_xy[xi, yi] += 1
        p_xy /= n

        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)

        mi = 0.0
        for i in range(len(unique_x)):
            for j in range(len(unique_y)):
                if p_xy[i, j] > 1e-15:
                    mi += p_xy[i, j] * np.log2(
                        p_xy[i, j] / (p_x[i] * p_y[j] + 1e-15) + 1e-15
                    )
        return max(0.0, mi)

    def permutation_test(
        self,
        continuous_X: np.ndarray,
        discrete_Y: np.ndarray,
        observed_mi: float,
    ) -> float:
        """
        Permutation test for MI significance.

        Shuffles Y n_permutations times and computes MI each time.
        Returns p-value = fraction of permuted MI >= observed MI.
        """
        count_ge = 0
        for _ in range(self.n_permutations):
            Y_perm = np.random.permutation(discrete_Y)
            if self.mi_method == "ksg":
                mi_perm = self.compute_mi_ksg(continuous_X, Y_perm)
            else:
                mi_perm = self.compute_mi_binned(continuous_X, Y_perm)
            if mi_perm >= observed_mi:
                count_ge += 1

        return (count_ge + 1) / (self.n_permutations + 1)

    def analyze_trajectory_planning(
        self,
        trajectories: List[List[Dict]],
        layer: int = -1,
        tensor_type: str = "V",
        context_length: int = 0,
        show_progress: bool = False,
    ) -> PlanningProfile:
        """
        Analyze planning in a set of generation trajectories.

        For each lookahead k, computes MI between state at position t
        and the token generated at position t+k.

        Args:
            trajectories: List of generation trajectories from QKVExtractor.
            layer: Which layer to analyze (-1 = last).
            tensor_type: "Q", "K", or "V".
            context_length: For labeling.

        Returns:
            PlanningProfile with MI results across lookaheads.
        """
        mi_results = []
        mi_curve = []

        lookaheads = range(1, self.max_lookahead + 1)
        iterator = tqdm(lookaheads, desc="Lookahead") if show_progress else lookaheads

        for k in iterator:
            # Collect (state_at_t, token_at_t+k) pairs from all trajectories
            X_list = []
            Y_list = []

            for traj in trajectories:
                for t in range(len(traj) - k):
                    step_t = traj[t]
                    step_tk = traj[t + k]

                    # Get state at position t
                    if layer == -1:
                        actual_layer = max(step_t["states"].keys())
                    else:
                        actual_layer = layer

                    state_vec = step_t["states"].get(actual_layer)
                    if state_vec is None:
                        continue

                    future_token_id = step_tk["token_id"]
                    X_list.append(state_vec)
                    Y_list.append(future_token_id)

            if len(X_list) < 20:
                mi_results.append(MutualInformationResult(
                    lookahead=k, mi_value=0.0, mi_std=0.0,
                    p_value=1.0, is_significant=False, n_samples=len(X_list),
                ))
                mi_curve.append(0.0)
                continue

            X = np.array(X_list)
            Y = np.array(Y_list)

            # Compute MI
            if self.mi_method == "ksg":
                mi_val = self.compute_mi_ksg(X, Y)
            else:
                mi_val = self.compute_mi_binned(X, Y)

            # Permutation test
            p_val = self.permutation_test(X, Y, mi_val)
            is_sig = p_val < self.alpha

            # Bootstrap standard error
            mi_bootstrap = []
            n_boot = min(50, self.n_permutations)
            for _ in range(n_boot):
                idx = np.random.choice(len(X), size=len(X), replace=True)
                if self.mi_method == "ksg":
                    mi_b = self.compute_mi_ksg(X[idx], Y[idx])
                else:
                    mi_b = self.compute_mi_binned(X[idx], Y[idx])
                mi_bootstrap.append(mi_b)
            mi_std = float(np.std(mi_bootstrap))

            mi_results.append(MutualInformationResult(
                lookahead=k,
                mi_value=mi_val,
                mi_std=mi_std,
                p_value=p_val,
                is_significant=is_sig,
                n_samples=len(X),
            ))
            mi_curve.append(mi_val)

            logger.debug(
                f"Lookahead {k}: MI={mi_val:.4f} ± {mi_std:.4f}, "
                f"p={p_val:.4f}, sig={is_sig}, n={len(X)}"
            )

        # Determine planning horizon (max consecutive significant lookahead)
        planning_horizon = 0
        for r in mi_results:
            if r.is_significant:
                planning_horizon = r.lookahead
            else:
                break

        return PlanningProfile(
            context_length=context_length,
            layer=layer,
            tensor_type=tensor_type,
            mi_results=mi_results,
            planning_horizon=planning_horizon,
            mi_curve=mi_curve,
        )

    def compare_across_contexts(
        self,
        profiles: Dict[int, PlanningProfile],
    ) -> PlanningComparison:
        """
        Compare planning profiles across different context lengths.
        """
        horizon_curve = {c: p.planning_horizon for c, p in profiles.items()}
        mi_surfaces = {c: p.mi_curve for c, p in profiles.items()}

        return PlanningComparison(
            profiles=profiles,
            horizon_curve=horizon_curve,
            mi_surfaces=mi_surfaces,
        )