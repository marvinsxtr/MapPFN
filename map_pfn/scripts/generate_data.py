from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from ml_project_template.config import run
from torch.utils.data import DataLoader
from tqdm import tqdm

from map_pfn.configs.data.base_config import DataGeneratorRun
from map_pfn.data.utils import BatchKeys, ColumnNames, assign_split, collate_fn
from map_pfn.utils.helpers import cpu_count, git_commit_hash, register_resolvers, seed_everything
from map_pfn.utils.logging import logger


def generate_data(cfg: DataGeneratorRun) -> None:
    """Generate and split a synthetic linear SCM dataset."""
    data_path = Path(cfg.output_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    file_path = data_path / f"{cfg.name}.h5ad"

    if file_path.is_file():
        logger.info(f"Processed dataset already exists at {file_path}. Skipping processing.")
        return

    dataset = cfg.dataset(seed=cfg.seed)
    num_workers = cpu_count() - 1 if cfg.name == "sergio_grn" else 0

    dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=num_workers,
        persistent_workers=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    num_batches = int(np.ceil(len(dataset) / dataloader.batch_size))
    batches = list(tqdm(dataloader, desc="Generating data", total=num_batches))

    treatment_ids = np.concatenate([b[BatchKeys.TREATMENT_ID] for b in batches])
    context_ids = np.concatenate([b[BatchKeys.CONTEXT_ID] for b in batches])
    treatments = np.concatenate([b[BatchKeys.TREATMENT] for b in batches])
    data = np.concatenate([b[BatchKeys.DATA] for b in batches])

    _, num_samples, num_nodes = data.shape
    X = data.reshape(-1, num_nodes)

    treatment_ids = np.repeat(treatment_ids, num_samples)
    context_ids = np.repeat(context_ids, num_samples)
    treatments = np.repeat(treatments, num_samples, axis=0)

    adata = ad.AnnData(
        X=X,
        obs={
            ColumnNames.CONTEXT: context_ids,
            ColumnNames.TREATMENT: treatment_ids,
        },
        obsm={
            ColumnNames.TREATMENT: treatments,
        },
        uns={
            "commit_hash": git_commit_hash(),
        },
    )

    if cfg.name == "sergio_grn":
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    adata = assign_split(adata, val_share=cfg.val_share, test_share=cfg.test_share, seed=cfg.seed)
    adata.write_h5ad(file_path)


if __name__ == "__main__":
    register_resolvers()

    from map_pfn.configs.data import config_stores  # noqa: F401

    run(generate_data, seed_fn=seed_everything)
