import warnings
from pathlib import Path
from typing import Literal

import anndata as ad
import click
import numpy as np
import pandas as pd
import scanpy as sc
from pertpy.data import frangieh_2021_rna, papalexi_2021

from map_pfn.data.utils import ColumnNames, Values, assign_split
from map_pfn.utils.helpers import git_commit_hash
from map_pfn.utils.logging import logger

DatasetName = Literal["frangieh", "papalexi"]


def load_dataset(dataset: DatasetName) -> tuple[ad.AnnData, list[str]]:
    """Load dataset and rename columns to standard names.

    Args:
        dataset: Name of the dataset to load.

    Returns:
        Tuple of (AnnData object with standardized column names,
        list of required gene names to include in the final gene set).
    """
    loaders = {
        "frangieh": frangieh_2021_rna,
        "papalexi": papalexi_2021,
    }
    if dataset not in loaders:
        raise ValueError(f"Unknown dataset: {dataset}")

    adata = loaders[dataset]()
    if dataset == "frangieh":
        adata.obs = adata.obs.rename(
            columns={
                "perturbation_2": ColumnNames.CONTEXT,
                "perturbation": ColumnNames.TREATMENT,
            }
        )
    elif dataset == "papalexi":
        mdata = adata
        adata = mdata.mod["rna"].copy()
        adata.obs[ColumnNames.TREATMENT] = mdata.obs["gene_target"].to_numpy()
        adata.obs[ColumnNames.TREATMENT] = adata.obs[ColumnNames.TREATMENT].replace("NT", Values.CONTROL)
        adata.obs[ColumnNames.CONTEXT] = "stimulated"
    else:
        raise NotImplementedError

    treatments = adata.obs[ColumnNames.TREATMENT].unique()
    perturbed_genes = sorted(g for g in treatments if g != Values.CONTROL and g in adata.var_names)

    return adata, perturbed_genes


def _rank_genes(adata: ad.AnnData) -> pd.Series:
    """Normalize, rank genes by differential expression, and return scores."""
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(
            adata,
            groupby=ColumnNames.TREATMENT,
            reference=Values.CONTROL,
            n_genes=adata.n_vars,
            rankby_abs=True,
        )

    rank_df = sc.get.rank_genes_groups_df(adata, group=None)
    return rank_df.groupby("names")["scores"].max()


def _select_genes_from_treatments(adata: ad.AnnData, n_top_genes: int) -> ad.AnnData:
    """Select top marker genes from treatment-matching genes.

    Filters columns to genes that match treatment names, ranks them, then
    keeps the top marker genes and their corresponding rows.
    """
    adata = adata[:, adata.var_names.isin(adata.obs[ColumnNames.TREATMENT])].copy()
    gene_scores = _rank_genes(adata)

    selected = gene_scores.nlargest(n_top_genes).index.tolist()
    row_mask = adata.obs[ColumnNames.TREATMENT].isin([Values.CONTROL, *selected])
    return adata[row_mask, selected].copy()


def _select_genes_with_required(adata: ad.AnnData, required_genes: list[str], n_top_genes: int) -> ad.AnnData:
    """Select genes from a required gene pool, filling with top markers if needed.

    If the pool has more genes than ``n_top_genes``, selects the top by DE score.
    If fewer, includes all pool genes and fills remaining slots with top DE
    markers from the full gene set.
    """
    gene_scores = _rank_genes(adata)

    pool = [g for g in required_genes if g in adata.var_names]

    if len(pool) > n_top_genes:
        pool_scores = gene_scores.reindex(pool).dropna()
        selected = pool_scores.nlargest(n_top_genes).index.tolist()
    elif len(pool) < n_top_genes:
        selected = list(pool)
        selected_set = set(selected)
        top_ranked = gene_scores.nlargest(n_top_genes + len(selected)).index
        filler = [g for g in top_ranked if g not in selected_set]
        selected.extend(filler[: n_top_genes - len(selected)])
    else:
        selected = list(pool)

    logger.info(
        f"Selected {len(selected)} / {len(pool)} pool genes (+ {max(0, len(selected) - len(pool))} marker fill)"
    )

    row_mask = adata.obs[ColumnNames.TREATMENT].isin([Values.CONTROL, *selected])
    return adata[row_mask, selected].copy()


def process_sc_data(
    dataset: DatasetName,
    cache_path: str = "datasets/single_cell",
    seed: int = 42,
    min_samples: int = 200,
    n_top_genes: int = 50,
) -> None:
    """Processes a single cell dataset.

    Args:
        dataset: Name of the dataset to process.
        cache_path: Path to the processed dataset.
        seed: Random seed for reproducibility.
        min_samples: Minimum number of samples per treatment.
        n_top_genes: Number of top marker genes to select.
    """
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    file_path = Path(cache_path) / f"{dataset}.h5ad"

    if file_path.is_file():
        logger.info(f"Processed dataset already exists at {file_path}. Skipping processing.")
        return

    adata, required_genes = load_dataset(dataset)

    counts = adata.obs[ColumnNames.TREATMENT].value_counts()
    keep_treatments = counts[counts >= min_samples].index
    adata = adata[adata.obs[ColumnNames.TREATMENT].isin(keep_treatments)].copy()

    if required_genes:
        adata = _select_genes_with_required(adata, required_genes, n_top_genes)
    else:
        adata = _select_genes_from_treatments(adata, n_top_genes)

    one_hot = pd.get_dummies(adata.obs[ColumnNames.TREATMENT]).reindex(columns=adata.var_names, fill_value=0)
    adata.obsm[ColumnNames.TREATMENT] = one_hot.to_numpy().astype(np.float32)

    adata.X = adata.X.toarray().astype(np.float32)
    adata.uns["commit_hash"] = git_commit_hash()

    adata = assign_split(adata, seed=seed)
    adata.write_h5ad(file_path)
    logger.info(f"Saved processed dataset to {file_path}")


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["frangieh", "papalexi"]),
    required=True,
    help="Name of the dataset to process.",
)
@click.option(
    "--cache-path",
    type=str,
    default="datasets/single_cell",
    help="Path to cache directory for processed data.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--min-samples",
    type=int,
    default=200,
    help="Minimum number of samples per treatment.",
)
@click.option(
    "--n-top-genes",
    type=int,
    default=50,
    help="Number of top marker genes to select.",
)
def main(dataset: DatasetName, cache_path: str, seed: int, min_samples: int, n_top_genes: int) -> None:
    """Process single-cell datasets."""
    process_sc_data(dataset, cache_path, seed, min_samples, n_top_genes)


if __name__ == "__main__":
    main()
