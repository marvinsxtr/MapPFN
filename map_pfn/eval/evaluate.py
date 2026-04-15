from pathlib import Path

import jax.random as jr
import numpy as np
import torch
import wandb
from hydra_zen import builds, instantiate
from lightning import LightningDataModule, LightningModule, Trainer

from map_pfn.configs.train.base_config import (
    DataModuleConfig,
    JaxLightningModuleConfig,
    LRScheduleConfig,
    MapPFNConfig,
    MMDiTConfig,
    PerturbationDatasetConfig,
)
from map_pfn.configs.train.baseline_config import BaselineDataModuleConfig, CondOTModuleConfig, MetaFMModuleConfig
from map_pfn.data.utils import BatchKeys
from map_pfn.eval.metrics import compute_distribution_metrics
from map_pfn.utils.helpers import resolve_checkpoint
from map_pfn.utils.lightning import JaxTrainer, TestMetrics


def evaluate_baselines(
    datamodule: LightningDataModule,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate identity and observed baselines from the datamodule alone.

    Iterates the predict dataloader twice with different seeds to resample
    observational and interventional data independently.

    Args:
        datamodule: Data module providing test data loaders.
        seed: Random seed for metric computation.

    Returns:
        Dictionary with "identity" and "observed" keys, each mapping
        to a dict of metric names to scalar values.
    """
    int_data_rounds = []

    if not hasattr(datamodule, "test_dataset") or datamodule.test_dataset is None:
        datamodule.setup("test")

    for i in range(2):
        datamodule.set_predict_seed(seed + i)
        loader = datamodule.predict_dataloader()
        if isinstance(loader, list):
            loader = loader[0]

        obs_list, int_list = [], []
        for batch in loader:
            obs_list.append(np.asarray(batch[BatchKeys.OBS_DATA][:, -1]))
            int_list.append(np.asarray(batch[BatchKeys.INT_DATA][:, -1]))

        int_data_rounds.append(np.concatenate(int_list))
        if i == 0:
            obs_data = np.concatenate(obs_list)

    key = jr.PRNGKey(seed)
    key, k1, k2 = jr.split(key, 3)

    identity_metrics = compute_distribution_metrics(obs_data, int_data_rounds[0], obs_data, key=k1)
    observed_metrics = compute_distribution_metrics(obs_data, int_data_rounds[0], int_data_rounds[1], key=k2)

    if wandb.run is not None:
        wandb.log({f"baselines/{k}/identity": v for k, v in identity_metrics.items()})
        wandb.log({f"baselines/{k}/observed": v for k, v in observed_metrics.items()})

    return {"identity": identity_metrics, "observed": observed_metrics}


def load_model(
    method: str,
    *,
    run_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    dataset_path: str,
    num_samples: int = 200,
    ood: bool = False,
    num_nodes: int = 50,
    seed: int = 42,
) -> tuple[Trainer, LightningModule, LightningDataModule]:
    """Instantiate and load a model for inference.

    Supports MapPFN (JAX/Equinox), CondOT, and MetaFM baselines.
    Weights can be loaded from a W&B run artifact or a local checkpoint file.

    Args:
        method: Model type, one of "map_pfn", "condot", or "metafm".
            Also accepts full config names like "map_pfn_rna" or "condot_scm".
        run_id: W&B run ID to download the model artifact from.
        checkpoint_path: Path to a local checkpoint file.
        dataset_path: Path to the h5ad dataset file for evaluation.
        num_samples: Number of cell samples per perturbation context.
        ood: Whether to use out-of-distribution test splits.
        num_nodes: Feature dimensionality (e.g., 50 for PCA-reduced scRNA-seq).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (trainer, module, datamodule) ready for inference via
        trainer.predict(module, datamodule=datamodule).

    Raises:
        ValueError: If method is unknown or neither *run_id* nor
            *checkpoint_path* is provided.
    """
    if run_id is None and checkpoint_path is None:
        raise ValueError("Provide either run_id or checkpoint_path.")

    base_method = method.lower().replace("_scm", "").replace("_rna", "").replace("_sergio", "")

    dataset_cfg = PerturbationDatasetConfig(seed=seed, num_samples=num_samples)

    if base_method == "map_pfn":
        mmdit = MMDiTConfig(noise_dim=num_nodes, key=builds(jr.key, seed))
        model_cfg = MapPFNConfig(decoder=mmdit, in_dim=num_nodes, key=builds(jr.key, seed))
        module = instantiate(
            JaxLightningModuleConfig(
                model=model_cfg,
                lr_schedule=LRScheduleConfig(total_steps=50_000),
                guidance=2.0,
                key=builds(jr.key, seed),
            )
        )
        datamodule = instantiate(DataModuleConfig(dataset=dataset_cfg, dataset_path=dataset_path, ood=ood))
        trainer = JaxTrainer(callbacks=[TestMetrics()], enable_model_summary=False)

    elif base_method == "condot":
        module = instantiate(CondOTModuleConfig(input_dim=num_nodes, input_dim_label=num_nodes))
        datamodule = instantiate(
            BaselineDataModuleConfig(
                dataset=dataset_cfg,
                dataset_path=dataset_path,
                ood=ood,
                batch_size=1,
            )
        )
        trainer = Trainer(callbacks=[TestMetrics()], enable_model_summary=False)

    elif base_method == "metafm":
        module = instantiate(MetaFMModuleConfig(dim=num_nodes, num_treat_conditions=num_nodes))
        datamodule = instantiate(
            BaselineDataModuleConfig(
                dataset=dataset_cfg,
                dataset_path=dataset_path,
                ood=ood,
                batch_size=10,
            )
        )
        trainer = Trainer(callbacks=[TestMetrics()], enable_model_summary=False)

    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'map_pfn', 'condot', or 'metafm'.")

    with resolve_checkpoint(checkpoint_path or run_id) as path:
        _load_weights(module, base_method, path)

    return trainer, module, datamodule


def _load_weights(module: LightningModule, method: str, path: Path) -> None:
    """Load model weights from a checkpoint file.

    Dispatches to Equinox deserialization for MapPFN and PyTorch
    state_dict loading for baselines (CondOT / MetaFM).

    Args:
        module: Lightning module to load weights into.
        method: Base method name ("map_pfn", "condot", "metafm").
        path: Path to the checkpoint file.
    """
    if method == "map_pfn":
        module.load_checkpoint(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        module.load_state_dict(ckpt["state_dict"], strict=False)
        module.eval()
