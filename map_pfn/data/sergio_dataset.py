from collections.abc import Iterator
from itertools import product

import networkx as nx
import numpy as np
import sergio_rs
import torch
from numpy.typing import NDArray
from torch.utils.data import IterableDataset

from map_pfn.data.grn.grn import grn
from map_pfn.data.utils import BatchKeys, Values


class SergioDataset(IterableDataset):
    """Generates observational and interventional data from GRNs using SERGIO."""

    def __init__(
        self,
        num_genes: int,
        num_samples: int,
        num_contexts: int,
        num_groups_range: tuple[int, int] | None = None,
        regulators_per_gene_range: tuple[float, float] | None = None,
        delta_in_range: tuple[float, float] | None = None,
        delta_out_range: tuple[float, float] | None = None,
        modularity_range: tuple[float, float] | None = None,
        num_cell_types: int = 1,
        interaction_k_range: tuple[float, float] | None = None,
        decay_range: tuple[float, float] | None = None,
        hill_n_range: tuple[float, float] | None = None,
        noise_s_range: tuple[float, float] | None = None,
        safety_iter: int = 150,
        scale_iter: int = 10,
        dt: float = 0.01,
        mr_low_range: tuple[float, float] | None = None,
        mr_high_range: tuple[float, float] | None = None,
        outlier_mu_range: tuple[float, float] | None = None,
        library_mu_range: tuple[float, float] | None = None,
        library_sigma_range: tuple[float, float] | None = None,
        dropout_k_range: tuple[float, float] | None = None,
        dropout_q_range: tuple[float, float] | None = None,
        counterfactual: bool = True,
        seed: int | None = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize the SERGIO-based causal dataset.

        Args:
            num_genes: Number of genes in each GRN.
            num_samples: Number of cells/samples to simulate per condition.
            num_contexts: Number of unique GRNs to sample from.
            num_groups_range: Range for number of gene groups/modules k.
            regulators_per_gene_range: Range for avg regulators per gene (r = 1/p).
            delta_in_range: Range for in-degree bias (larger = less heavy tail).
            delta_out_range: Range for out-degree bias (smaller = more hub regulators).
            modularity_range: Range for within-group connectivity w (kappa).
            num_cell_types: Number of cell types for SERGIO MR profile.
            interaction_k_range: Range for SERGIO interaction strengths.
            decay_range: Range for SERGIO gene decay rates.
            hill_n_range: Range for SERGIO Hill function non-linearity.
            noise_s_range: Range for SERGIO simulation noise.
            safety_iter: SERGIO safety iterations.
            scale_iter: SERGIO scale iterations.
            dt: SERGIO time step.
            mr_low_range: SERGIO MR profile low range.
            mr_high_range: SERGIO MR profile high range.
            outlier_mu_range: Range for outlier mean parameter.
            library_mu_range: Range for library size mean parameter.
            library_sigma_range: Range for library size sigma parameter.
            dropout_k_range: Range for dropout k parameter.
            dropout_q_range: Range for dropout q parameter.
            counterfactual: Whether to use the same noise for each treatment.
            seed: Random seed for reproducibility.
            shuffle: Whether to shuffle the order of samples.
        """
        self.num_groups_range = num_groups_range or (1, 3)
        self.regulators_per_gene_range = regulators_per_gene_range or (1.5, 3.0)
        self.delta_in_range = delta_in_range or (10.0, 300.0)
        self.delta_out_range = delta_out_range or (1.0, 30.0)
        self.modularity_range = modularity_range or (1.0, 900.0)

        self.num_cell_types = num_cell_types
        self.interaction_k_range = interaction_k_range or (1.0, 5.0)
        self.decay_range = decay_range or (0.5, 1.0)
        self.hill_n_range = hill_n_range or (1.5, 2.5)
        self.noise_s_range = noise_s_range or (0.5, 1.5)
        self.safety_iter = safety_iter
        self.scale_iter = scale_iter
        self.dt = dt
        self.mr_low_range = mr_low_range or (0.5, 2.0)
        self.mr_high_range = mr_high_range or (3.0, 5.0)

        self.outlier_mu_range = outlier_mu_range or (0.8, 5.0)
        self.library_mu_range = library_mu_range or (4.5, 6.0)
        self.library_sigma_range = library_sigma_range or (0.3, 0.7)
        self.dropout_k_range = dropout_k_range or (8.0, 8.0)
        self.dropout_q_range = dropout_q_range or (45.0, 82.0)

        self.num_genes = num_genes
        self.num_samples = num_samples
        self.num_contexts = num_contexts
        self.counterfactual = counterfactual
        self.seed = seed
        self.shuffle = shuffle

        rng = np.random.default_rng(seed)
        self.context_seeds = rng.integers(0, 2**31, size=num_contexts)
        self.treatments = np.arange(num_genes)

        control_conditions = [(seed, None) for seed in self.context_seeds]
        treatment_conditions = list(product(self.context_seeds, self.treatments))
        self.conditions = control_conditions + treatment_conditions

    def _sample_grn_params(self, rng: np.random.Generator) -> dict:
        """Sample GRN generating parameters."""
        r = rng.uniform(*self.regulators_per_gene_range)
        return {
            "n": self.num_genes,
            "k": int(rng.integers(*self.num_groups_range, endpoint=True)),
            "alpha": 1e-99,
            "beta": 1 - 1 / r,
            "gamma": 1 / r,
            "delta_in": rng.uniform(*self.delta_in_range),
            "delta_out": rng.uniform(*self.delta_out_range),
            "kappa": rng.uniform(*self.modularity_range),
        }

    def sample_grn(self, seed: int) -> grn:
        """Sample a new GRN with structure and expression parameters."""
        rng = np.random.default_rng(seed)
        np.random.seed(int(rng.integers(0, 2**31)))

        params = self._sample_grn_params(rng)
        G = grn().add_structure(**params, expression_params=True)

        return G

    def _remove_cycles(self, G: grn) -> NDArray:
        """Remove cycles from GRN to create a DAG.

        Uses a greedy approach: iteratively finds cycles and removes the
        edge with the smallest absolute weight in each cycle.
        """
        beta_dag = G.beta.copy()
        G_copy = G.copy()

        while not nx.is_directed_acyclic_graph(G_copy):
            try:
                cycle = nx.find_cycle(G_copy)
            except nx.NetworkXNoCycle:
                break

            min_weight = float("inf")
            min_edge = None

            for u, v in cycle:
                w = abs(beta_dag[u, v])
                if w < min_weight:
                    min_weight = w
                    min_edge = (u, v)

            if min_edge:
                u, v = min_edge
                beta_dag[u, v] = 0
                G_copy.remove_edge(u, v)

        return beta_dag

    def _ensure_mrs(self, beta: NDArray) -> NDArray:
        """Ensure the GRN has master regulators (genes with no incoming edges)."""
        beta_mod = beta.copy()
        n_genes = beta_mod.shape[0]

        in_degrees = np.sum(beta_mod != 0, axis=0)
        out_degrees = np.sum(beta_mod != 0, axis=1)

        natural_mrs = (in_degrees == 0) & (out_degrees > 0)
        if np.any(natural_mrs):
            return beta_mod

        candidates = np.where(out_degrees > 0)[0]
        if len(candidates) == 0:
            candidates = np.arange(n_genes)

        sorted_candidates = candidates[np.argsort(in_degrees[candidates])]
        num_mrs = max(1, n_genes // 20)
        mr_candidates = sorted_candidates[:num_mrs]

        beta_mod[:, mr_candidates] = 0

        return beta_mod

    def _grn_to_sergio(
        self, G: grn, rng: np.random.Generator
    ) -> tuple[sergio_rs.GRN, sergio_rs.MrProfile, dict[int, str], set[int]]:
        """Convert grn object to sergio_rs GRN and MR profile.

        Returns:
            sergio_grn: The SERGIO GRN object.
            mr_profile: The MR expression profile.
            idx_to_name: Mapping from gene index to gene name.
            genes_in_grn: Set of gene indices that appear in the GRN.
        """
        sergio_grn = sergio_rs.GRN()

        n_genes = G.n
        idx_to_name = {i: f"GENE{i:04d}" for i in range(n_genes)}

        decays = rng.uniform(*self.decay_range, size=n_genes)
        hill_ns = rng.uniform(*self.hill_n_range, size=n_genes)

        beta_dag = self._remove_cycles(G)
        beta_dag = self._ensure_mrs(beta_dag)

        genes_in_grn: set[int] = set()

        for i in range(n_genes):
            for j in range(n_genes):
                weight = beta_dag[i, j]
                if weight != 0:
                    genes_in_grn.add(i)
                    genes_in_grn.add(j)
                    reg = sergio_rs.Gene(name=idx_to_name[i], decay=float(decays[i]))
                    tar = sergio_rs.Gene(name=idx_to_name[j], decay=float(decays[j]))
                    interaction_k = rng.uniform(*self.interaction_k_range)
                    sergio_grn.add_interaction(
                        reg=reg,
                        tar=tar,
                        k=float(interaction_k),
                        h=None,
                        n=int(hill_ns[j]),
                    )

        sergio_grn.set_mrs()

        mr_profile = sergio_rs.MrProfile.from_random(
            sergio_grn,
            num_cell_types=self.num_cell_types,
            low_range=self.mr_low_range,
            high_range=self.mr_high_range,
            seed=int(rng.integers(0, 2**31)),
        )

        return sergio_grn, mr_profile, idx_to_name, genes_in_grn

    def _simulate_sergio(
        self,
        sergio_grn: sergio_rs.GRN,
        mr_profile: sergio_rs.MrProfile,
        sim_seed: int,
        idx_to_name: dict[int, str],
        context_seed: int,
    ) -> NDArray[np.float32] | None:
        """Run SERGIO simulation.

        Returns:
            Expression data array of shape (num_samples, num_genes), or None if
            simulation produces invalid data (zero-count cells/genes).
        """
        rng = np.random.default_rng(sim_seed)
        noise_s = rng.uniform(*self.noise_s_range)

        sim = sergio_rs.Sim(
            sergio_grn,
            num_cells=self.num_samples,
            noise_s=noise_s,
            safety_iter=self.safety_iter,
            scale_iter=self.scale_iter,
            dt=self.dt,
            seed=sim_seed,
        )
        data = sim.simulate(mr_profile)

        gene_names = data["Genes"].to_list()
        name_to_idx = {name: idx for idx, name in idx_to_name.items()}

        data_np = data.drop("Genes").to_numpy()

        row_sums = data_np.sum(axis=1)
        col_sums = data_np.sum(axis=0)

        nonzero_rows = row_sums > 0
        nonzero_cols = col_sums > 0

        if not np.any(nonzero_rows) or not np.all(nonzero_cols):
            return None

        data_filtered = data_np[nonzero_rows, :][:, nonzero_cols]

        noise_rng = np.random.default_rng(context_seed)
        outlier_mu = noise_rng.uniform(*self.outlier_mu_range)
        library_mu = noise_rng.uniform(*self.library_mu_range)
        library_sigma = noise_rng.uniform(*self.library_sigma_range)
        dropout_k = noise_rng.uniform(*self.dropout_k_range)
        dropout_q = noise_rng.uniform(*self.dropout_q_range)

        data_noisy = sergio_rs.add_technical_noise_custom(
            data_filtered,
            outlier_mu=outlier_mu,
            library_mu=library_mu,
            library_sigma=library_sigma,
            dropout_k=dropout_k,
            dropout_q=dropout_q,
            seed=sim_seed,
        )

        data_np = np.zeros_like(data_np)
        data_np[np.ix_(nonzero_rows, nonzero_cols)] = data_noisy

        data_np = data_np.T

        full_data = np.zeros((data_np.shape[0], self.num_genes), dtype=np.float32)

        for col_idx, gene_name in enumerate(gene_names):
            if gene_name in name_to_idx:
                full_data[:, name_to_idx[gene_name]] = data_np[:, col_idx]

        return full_data

    def encode_treatment(self, treatment_idx: int | None) -> NDArray[np.float32]:
        """Encode treatment as one-hot vector."""
        treatment_vec = np.zeros(self.num_genes, dtype=np.float32)

        if treatment_idx is not None:
            treatment_vec[treatment_idx] = 1.0

        return treatment_vec

    def sample_condition(
        self,
        context_seed: int,
        treatment_idx: int | None = None,
    ) -> dict[str, NDArray[np.float32] | str] | None:
        """Sample data for a single (context, treatment) condition.

        Returns:
            Dictionary with batch data, or None if simulation failed.
        """
        rng = np.random.default_rng(context_seed)

        G = self.sample_grn(context_seed)
        sergio_grn, mr_profile, idx_to_name, genes_in_grn = self._grn_to_sergio(G, rng)

        if treatment_idx is not None and treatment_idx in genes_in_grn:
            gene_name = idx_to_name[treatment_idx]
            sergio_grn, mr_profile = sergio_grn.ko_perturbation(gene_name=gene_name, mr_profile=mr_profile)

        sim_seed = int(rng.integers(0, 2**31))
        treatment_offset = 0 if treatment_idx is None else treatment_idx + 1
        sim_seed = sim_seed if self.counterfactual else sim_seed + treatment_offset

        data = self._simulate_sergio(sergio_grn, mr_profile, sim_seed, idx_to_name, context_seed)

        if data is None:
            return None

        treatment_id = str(treatment_idx) if treatment_idx is not None else Values.CONTROL

        return {
            BatchKeys.TREATMENT_ID: treatment_id,
            BatchKeys.CONTEXT_ID: str(context_seed),
            BatchKeys.DATA: data,
            BatchKeys.TREATMENT: self.encode_treatment(treatment_idx),
        }

    def __iter__(self) -> Iterator[dict[str, NDArray[np.float32] | str]]:
        """Iterate over all (context, treatment) conditions."""
        conditions = list(self.conditions)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = len(conditions) // worker_info.num_workers
            remainder = len(conditions) % worker_info.num_workers
            start = worker_info.id * per_worker + min(worker_info.id, remainder)
            end = start + per_worker + (1 if worker_info.id < remainder else 0)
            conditions = conditions[start:end]

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(conditions)

        for context_seed, treatment_idx in conditions:
            result = self.sample_condition(
                int(context_seed),
                treatment_idx=int(treatment_idx) if treatment_idx is not None else None,
            )
            if result is not None:
                yield result

    def __len__(self) -> int:
        """Total number of (context, treatment) conditions including controls."""
        return len(self.conditions)
