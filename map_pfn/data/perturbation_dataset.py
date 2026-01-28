import anndata as ad
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

from map_pfn.data.utils import BatchKeys, ColumnNames, Values


class PerturbationDataset(Dataset):
    """Dataset sampling queries and optionally context conditions.

    For each query (bio_context, treatment), samples num_shots other treatments
    from the same bio_context in the context pool, excluding the query treatment.
    If no context_adata is provided, operates in zero-shot mode.

    Each sample consists of:
    - obs_data: Control cells (num_shots + 1, n_cells, dim) - query last
    - int_data: Treated cells (num_shots + 1, n_cells, dim)
    - treatment: One-hot vectors (num_shots + 1, 1, n_treatments)
    - context_id: Context/cell type name (str)
    - treatment_id: List of treatment names, query last (list[str])
    """

    def __init__(
        self,
        query_adata: ad.AnnData,
        context_adata: ad.AnnData | None = None,
        num_shots: int = 0,
        num_samples: int = 500,
        context_col: str = ColumnNames.CONTEXT,
        treatment_col: str = ColumnNames.TREATMENT,
        control_value: str = Values.CONTROL,
        seed: int | None = None,
    ) -> None:
        """Initialize the ICL dataset.

        Args:
            query_adata: AnnData for queries (val/test split).
            context_adata: AnnData for context demonstrations (train split).
                If None, operates in zero-shot mode.
            num_shots: Number of context demonstrations per query.
            num_samples: Maximum number of cells to sample per population.
            context_col: Column in obs identifying the biological context.
            treatment_col: Column in obs identifying the treatment.
            control_value: Value in treatment_col that indicates control.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.query_adata = query_adata
        self.context_adata = context_adata
        self.num_shots = num_shots
        self.num_samples = num_samples
        self.context_col = context_col
        self.treatment_col = treatment_col
        self.control_value = control_value
        self.rng = np.random.default_rng(seed)

        self._validate(self.query_adata)
        if self.context_adata is not None:
            self._validate(self.context_adata)

        self._build_indices()

    def _validate(self, adata: ad.AnnData) -> None:
        """Validate that required columns exist in adata."""
        for col in [self.context_col, self.treatment_col]:
            if col not in adata.obs.columns:
                raise ValueError(f"Column '{col}' not found in adata.obs")

        if self.treatment_col not in adata.obsm:
            raise ValueError(f"Treatment vectors not found in adata.obsm['{self.treatment_col}']")

    def _build_indices(self) -> None:
        """Build sample indices for both query and context adata."""
        self.query_group_indices = self.query_adata.obs.groupby(
            [self.context_col, self.treatment_col], observed=False
        ).indices

        query_context_to_treatments = self._get_context_to_treatments(self.query_group_indices)

        self.query_samples: list[tuple[str, str]] = [
            (ctx, treat)
            for ctx in sorted(query_context_to_treatments.keys())
            for treat in query_context_to_treatments[ctx]
        ]

        if self.context_adata is not None:
            self.context_group_indices = self.context_adata.obs.groupby(
                [self.context_col, self.treatment_col], observed=False
            ).indices

            self.context_to_treatments = self._get_context_to_treatments(self.context_group_indices)

            self._validate_coverage()
        else:
            self.context_group_indices = {}
            self.context_to_treatments = {}

    def _get_context_to_treatments(
        self,
        group_indices: dict[tuple[str, str], NDArray],
    ) -> dict[str, list[str]]:
        """Extract context -> treatments mapping from group indices."""
        context_to_treatments: dict[str, list[str]] = {}
        contexts_with_control: set[str] = set()

        for (ctx, treat), indices in group_indices.items():
            if len(indices) == 0:
                continue
            if treat == self.control_value:
                contexts_with_control.add(ctx)
            else:
                context_to_treatments.setdefault(ctx, []).append(treat)

        return {ctx: sorted(treats) for ctx, treats in context_to_treatments.items() if ctx in contexts_with_control}

    def _validate_coverage(self) -> None:
        """Warn if any query bio_context has insufficient context samples."""
        query_contexts = {ctx for ctx, _ in self.query_samples}

        for ctx in query_contexts:
            n_available = len(self.context_to_treatments.get(ctx, []))
            if n_available < self.num_shots:
                raise ValueError(
                    f"Insufficient context treatments for bio_context '{ctx}': "
                    f"required {self.num_shots}, available {n_available}"
                )

    def _get_samples(
        self,
        adata: ad.AnnData,
        group_indices: dict[tuple[str, str], NDArray],
        key: tuple[str, str],
    ) -> NDArray[np.float32]:
        """Sample cells for a given (context, treatment) key."""
        indices = group_indices[key]
        replace = len(indices) < self.num_samples
        sampled = self.rng.choice(indices, size=self.num_samples, replace=replace)
        return np.asarray(adata.X[sampled, :], dtype=np.float32)

    def _get_treatment_vector(
        self,
        adata: ad.AnnData,
        group_indices: dict[tuple[str, str], NDArray],
        key: tuple[str, str],
    ) -> NDArray[np.float32]:
        """Fetch treatment one-hot vector for a (context, treatment) key."""
        idx = group_indices[key][0]
        return np.asarray(adata.obsm[self.treatment_col][idx], dtype=np.float32).reshape(1, -1)

    def _sample_context_treatments(
        self,
        bio_context: str,
        exclude_treatment: str,
    ) -> list[str]:
        """Sample context treatment names for a bio_context."""
        if self.context_adata is None:
            return []

        available = [t for t in self.context_to_treatments.get(bio_context, []) if t != exclude_treatment]

        n = min(self.num_shots, len(available))
        return self.rng.choice(available, size=n, replace=False).tolist()

    def __len__(self) -> int:
        """Return the number of query samples."""
        return len(self.query_samples)

    def __getitem__(self, idx: int) -> dict[str, NDArray[np.float32] | list[str] | str | int]:
        """Get a sample consisting of context and query data."""
        bio_context, query_treatment = self.query_samples[idx]
        context_treatments = self._sample_context_treatments(bio_context, query_treatment)

        all_treatments = [*context_treatments, query_treatment]

        x0_list: list[NDArray[np.float32]] = []
        x1_list: list[NDArray[np.float32]] = []
        treat_list: list[NDArray[np.float32]] = []

        for treat in context_treatments:
            x0_list.append(
                self._get_samples(self.context_adata, self.context_group_indices, (bio_context, self.control_value))
            )
            x1_list.append(self._get_samples(self.context_adata, self.context_group_indices, (bio_context, treat)))
            treat_list.append(
                self._get_treatment_vector(self.context_adata, self.context_group_indices, (bio_context, treat))
            )

        x0_list.append(
            self._get_samples(self.query_adata, self.query_group_indices, (bio_context, self.control_value))
        )
        x1_list.append(self._get_samples(self.query_adata, self.query_group_indices, (bio_context, query_treatment)))
        treat_list.append(
            self._get_treatment_vector(self.query_adata, self.query_group_indices, (bio_context, query_treatment))
        )

        return {
            BatchKeys.OBS_DATA: np.stack(x0_list, axis=0),
            BatchKeys.INT_DATA: np.stack(x1_list, axis=0),
            BatchKeys.TREATMENT: np.stack(treat_list, axis=0),
            BatchKeys.CONTEXT_ID: bio_context,
            BatchKeys.TREATMENT_ID: all_treatments,
        }
