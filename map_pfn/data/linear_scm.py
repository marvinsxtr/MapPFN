import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la


class LinearSCM:
    """Linear Structural Causal Model.

    Implements the standard linear SCM: X = AX + ε where A is the weight matrix
    and ε is additive noise. Supports both observational sampling and
    do-interventions.
    """

    def __init__(
        self,
        num_nodes: int = 10,
        edge_prob: float = 0.5,
        weight_range: tuple[float, float] | None = None,
        treatment_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize Linear SCM with adjacency matrix and weights.

        Args:
            num_nodes: Number of variables in the SCM
            edge_prob: Probability of edge existing between any two nodes
            weight_range: Range (min, max) for sampling edge weights
            treatment_range: Range for sampling treatment values
            seed: Random seed for reproducibility
        """
        if weight_range is None:
            weight_range = (0.5, 2.0)

        if treatment_range is None:
            treatment_range = (0.5, 1.5)

        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.weight_range = weight_range
        self.treatment_range = treatment_range
        self.adjacency_matrix, self.weights = self.sample_dag_and_weights()

    def sample_dag_and_weights(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Sample a random DAG and corresponding weight matrix.

        Returns:
            Tuple of (adjacency_matrix, weights) for LinearSCM initialization
        """
        perm = self.rng.permutation(self.num_nodes)
        i_idx, j_idx = np.triu_indices(self.num_nodes, k=1)
        mask = self.rng.random(len(i_idx)) < self.edge_prob

        adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int_)
        adjacency_matrix[perm[i_idx[mask]], perm[j_idx[mask]]] = 1

        weights = self.rng.uniform(self.weight_range[0], self.weight_range[1], (self.num_nodes, self.num_nodes))
        signs = self.rng.choice([-1, 1], size=(self.num_nodes, self.num_nodes))

        weights *= signs
        weights *= adjacency_matrix

        weights = self.normalize_weight_matrix(weights)

        return adjacency_matrix, weights

    @staticmethod
    def normalize_weight_matrix(weight_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize weight matrix for unit variance observations.

        This normalization follows:
        T = (I - W)^(-1)  # Transfer matrix
        D = diag(T @ T^T)  # Diagonal of T @ T^T
        W_normalized = D^(-1/2) @ W

        This ensures that the resulting observations have approximately unit variance
        and fall within the desired [-2, 2] range for 95th percentile.

        Args:
            weight_mat: Weight matrix to normalize

        Returns:
            Normalized weight matrix
        """
        I_minus_W = np.eye(weight_mat.shape[0]) - weight_mat
        T = la.inv(I_minus_W)

        diag_T_TT = np.diag(T @ T.T)

        sqrt_diag = np.sqrt(diag_T_TT)
        D_inv = np.diag(1.0 / sqrt_diag)

        return D_inv @ weight_mat

    @staticmethod
    def solve_scm(noise: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve the SCM equations for given noise terms.

        Args:
            noise: Array of shape (n_samples, num_nodes) with noise terms
            weights: Weight matrix to use for solving

        Returns:
            Array of shape (n_samples, num_nodes) with solved values
        """
        num_nodes = weights.shape[0]
        I_minus_A = np.eye(num_nodes) - weights
        I_minus_A_reg = I_minus_A + np.eye(num_nodes) * 1e-8

        try:
            observations = la.solve(I_minus_A_reg, noise.T).T
        except Exception:
            observations = np.linalg.lstsq(I_minus_A_reg, noise.T, rcond=None)[0].T

        return np.clip(observations, -1e6, 1e6)

    def sample_observations(
        self, n_samples: int, noise_std: float = 1.0, seed: int | None = None
    ) -> NDArray[np.float64]:
        """Sample observational data from the SCM.

        Solves the system (I - A)X = ε where ε ~ N(0, sigma^2 I) to generate
        samples from the observational distribution P(X).

        Args:
            n_samples: Number of samples to generate
            noise_std: Standard deviation of additive noise terms
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_samples, num_nodes) with sampled observations
        """
        rng = self.rng if seed is None else np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, (n_samples, self.num_nodes))
        return self.solve_scm(noise, self.weights)

    def sample_treatment(self, treatment_node: int, seed: int | None = None) -> float:
        """Sample a treatment node and value.

        Args:
            treatment_node: Index of node to treat
            seed: Random seed for reproducibility

        Returns:
            Tuple of (treatment_node, treatment_value)
        """
        rng = self.rng if seed is None else np.random.default_rng(seed)
        treatment_value = rng.uniform(self.treatment_range[0], self.treatment_range[1])
        return treatment_node, treatment_value

    def sample_intervention(
        self,
        n_samples: int,
        intervention_node: int,
        intervention_value: float,
        noise_std: float = 1.0,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """Sample data from SCM with do-intervention applied.

        Implements Pearl's do-intervention do(X_i = c) by:
        1. Breaking all incoming causal edges to intervention_node
        2. Setting X_i = intervention_value deterministically

        Args:
            n_samples: Number of samples to generate
            intervention_node: Index of node to intervene on
            intervention_value: Value to set intervention node to
            noise_std: Standard deviation of noise for non-intervention nodes
            seed: Random seed for reproducibility

        Returns:
            Array of shape (n_samples, num_nodes) with interventional samples
        """
        rng = self.rng if seed is None else np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, (n_samples, self.num_nodes))

        weights_intervened = self.weights.copy()
        weights_intervened[intervention_node, :] = 0.0

        noise[:, intervention_node] = intervention_value

        return self.solve_scm(noise, weights_intervened)
