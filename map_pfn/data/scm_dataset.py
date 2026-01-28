from collections.abc import Iterator
from itertools import product

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from map_pfn.data.linear_scm import LinearSCM
from map_pfn.data.utils import BatchKeys, Values


class LinearSCMDataset(IterableDataset):
    """Generates observational and interventional data from randomly generated SCMs."""

    def __init__(
        self,
        num_nodes: int,
        edge_prob: float,
        num_samples: int,
        num_contexts: int,
        weight_range: tuple[float, float] | None = None,
        treatment_range: tuple[float, float] | None = None,
        noise_std: float = 1.0,
        seed: int | None = None,
        shuffle: bool = False,
        counterfactual: bool = True,
    ) -> None:
        """Initialize the synthetic causal dataset.

        Args:
            num_nodes: Number of nodes in each causal graph
            edge_prob: Edge probability in each causal graph
            num_samples: Number of samples per SCM
            num_contexts: Number of unique context SCMs to sample from
            weight_range: Range for sampling edge weights
            treatment_range: Range for sampling treatment values
            noise_std: Standard deviation of noise terms
            seed: Random seed for reproducibility
            shuffle: Whether to shuffle the order of samples
            counterfactual: Whether to use counterfactual sampling for interventions
        """
        if weight_range is None:
            weight_range = (0.5, 2.0)

        if treatment_range is None:
            treatment_range = (0.5, 1.5)

        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.num_samples = num_samples
        self.num_contexts = num_contexts
        self.weight_range = weight_range
        self.treatment_range = treatment_range
        self.noise_std = noise_std
        self.seed = seed
        self.shuffle = shuffle
        self.counterfactual = counterfactual

        rng = np.random.default_rng(seed)
        self.context_seeds = rng.integers(0, 2**31, size=num_contexts)
        self.treatments = np.arange(num_nodes)
        self.conditions = list(product(self.context_seeds, self.treatments))

    def encode_treatment(self, treatment: tuple[int, float] | None) -> NDArray[np.float32]:
        """Encode treatment index as one-hot vector.

        Args:
            treatment: Tuple of (treatment_node, treatment_value) or None for observational data

        Returns:
            One-hot vector of shape (num_nodes,) where the treatment_node is set to treatment_value
            or all zeros for observational data
        """
        treatment_vec = np.zeros(self.num_nodes, dtype=np.float32)

        if treatment is not None:
            treatment_node, treatment_value = treatment
            treatment_vec[treatment_node] = treatment_value

        return treatment_vec

    def sample_scm(self, seed: int) -> LinearSCM:
        """Sample a new Linear SCM.

        Args:
            seed: Random seed for SCM sampling

        Returns:
            A new instance of LinearSCM
        """
        return LinearSCM(self.num_nodes, self.edge_prob, self.weight_range, self.treatment_range, seed=seed)

    def sample_observational_data(self, scm: LinearSCM, seed: int | None = None) -> NDArray[np.float32]:
        """Generate observational data from a given SCM.

        Args:
            scm: The Linear SCM to sample from
            seed: Random seed for data sampling

        Returns:
            Observational data of shape (n_samples, n_nodes)
        """
        return scm.sample_observations(self.num_samples, self.noise_std, seed=seed)

    def sample_interventional_data(
        self, scm: LinearSCM, treatment: tuple[int, float], seed: int | None = None
    ) -> NDArray[np.float32]:
        """Generate interventional data for the given treatment.

        Args:
            scm: The Linear SCM to intervene on
            treatment: Tuple of (treatment_node, treatment_value)
            seed: Random seed for data sampling

        Returns:
            Interventional data of shape (n_samples, n_nodes)
        """
        return scm.sample_intervention(self.num_samples, treatment[0], treatment[1], self.noise_std, seed=seed)

    def normalize_data(self, data: NDArray) -> NDArray:
        """Normalize data to have unit variance per variable.

        Args:
            data: Data of shape [n_samples, n_vars]

        Returns:
            Normalized data with unit variance per variable
        """
        std = data.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)

        data_norm = data / std
        data_norm = np.clip(data_norm, -10, 10)

        return data_norm

    def sample_condition(
        self,
        context_seed: int,
        treatment_node: int | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        """Sample data for a single (context, treatment) condition.

        Args:
            context_seed: Seed identifying the SCM.
            treatment_node: Index of the treatment node or None for observational data

        Returns:
            Dictionary containing:
                - treatment_id: Identifier for the treatment
                - context_id: Identifier for the context SCM
                - data: Shape [n_samples, num_nodes]
                - treatment: Shape [num_nodes] one-hot encoded treatment
        """
        scm = self.sample_scm(context_seed)

        rng = np.random.default_rng(context_seed)
        obs_seed = int(rng.integers(0, 2**31))
        int_seed = int(rng.integers(0, 2**31))
        treatment_seed = int(rng.integers(0, 2**31))

        treatment_offset = treatment_node + 1 if treatment_node is not None else 0
        int_seed = int_seed + treatment_offset if not self.counterfactual else int_seed
        treatment_seed = treatment_seed + treatment_offset

        if treatment_node is None:
            treatment = None
            data = self.sample_observational_data(scm, seed=obs_seed)
        else:
            treatment = scm.sample_treatment(treatment_node, seed=treatment_seed)
            data = self.sample_interventional_data(scm, treatment, seed=int_seed)

        data = self.normalize_data(data)

        treatment_vec = self.encode_treatment(treatment)
        treatment_id = str(treatment_node) if treatment is not None else Values.CONTROL

        return {
            BatchKeys.TREATMENT_ID: treatment_id,
            BatchKeys.CONTEXT_ID: str(context_seed),
            BatchKeys.DATA: data,
            BatchKeys.TREATMENT: treatment_vec,
        }

    def __iter__(self) -> Iterator[dict[str, NDArray[np.float32]]]:
        """Return iterator over all (context, treatment) conditions."""
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.conditions)

        for context_seed in self.context_seeds:
            yield self.sample_condition(int(context_seed), treatment_node=None)

        for context_seed, treatment_node in self.conditions:
            yield self.sample_condition(int(context_seed), treatment_node=int(treatment_node))

    def __len__(self) -> int:
        """Return the total number of unique (context, treatment) conditions."""
        return len(self.conditions) + len(self.context_seeds)
