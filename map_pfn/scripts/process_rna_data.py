import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from pertpy.data import frangieh_2021_rna

from map_pfn.data.utils import ColumnNames, Values, assign_split
from map_pfn.utils.helpers import git_commit_hash
from map_pfn.utils.logging import logger


def process_rna_data(cache_path: str = "datasets/single_cell", seed: int = 42) -> None:
    """Processes the single cell dataset.

    Args:
        cache_path: Path to the processed dataset.
        seed: Random seed for reproducibility.
    """
    file_path = Path(f"{cache_path}") / "frangieh_rna_new.h5ad"

    if file_path.is_file():
        logger.info(f"Processed dataset already exists at {file_path}. Skipping processing.")
        return

    adata = frangieh_2021_rna()

    adata.obs = adata.obs.rename(
        columns={
            "perturbation_2": ColumnNames.CONTEXT,
            "perturbation": ColumnNames.TREATMENT,
        }
    )

    adata = adata[
        adata.obs.groupby(ColumnNames.TREATMENT, observed=False)[ColumnNames.TREATMENT].transform("size") >= 200
    ].copy()

    adata = adata[:, adata.var_names.isin(adata.obs[ColumnNames.TREATMENT])].copy()

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(
            adata, groupby=ColumnNames.TREATMENT, reference=Values.CONTROL, n_genes=50, rankby_abs=True
        )

    rank_df = sc.get.rank_genes_groups_df(adata, group=None)
    gene_scores = rank_df.groupby("names")["scores"].max()
    top_marker_genes = gene_scores.nlargest(50).index.tolist()
    adata = adata[:, top_marker_genes].copy()

    adata = adata[adata.obs[ColumnNames.TREATMENT].isin([Values.CONTROL, *adata.var_names])].copy()

    one_hot = pd.get_dummies(adata.obs[ColumnNames.TREATMENT]).reindex(columns=adata.var_names, fill_value=0)
    adata.obsm[ColumnNames.TREATMENT] = one_hot.to_numpy().astype(np.float32)

    adata.X = adata.X.toarray().astype(np.float32)
    adata.uns["commit_hash"] = git_commit_hash()

    adata = assign_split(adata, holdout_context="IFNγ", seed=seed)  # noqa: RUF001
    adata.write_h5ad(file_path)


if __name__ == "__main__":
    process_rna_data()
