from pathlib import Path

import jax.random as jr
import numpy as np
import torch
import wandb
from hydra_zen import builds, instantiate
from lightning import LightningDataModule, LightningModule, Trainer

from map_pfn.configs.train.base_config import (
    DataModuleConfig,
    DiTConfig,
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


def fit_mean_linear(dataset, rank: int = 8, ridge: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN001
    """Fit the Mean and Linear baselines of Ahlmann-Eltze et al. (2025) on the training pseudobulk.

    Args:
        dataset: Test dataset whose ``context_adata`` holds the training cells.
        rank: Dimension of the PCA gene and perturbation embeddings.
        ridge: Ridge penalty of both least-squares solves.

    Returns:
        The per-gene mean over training perturbations ``(genes,)``, the gene embedding ``(genes, rank)`` and the
        weight ``(rank, rank)`` of the Linear baseline.
    """
    adata = dataset.context_adata
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    profiles, onehots = [], []
    for (_, treatment), idx in dataset.context_group_indices.items():
        if treatment == dataset.control_value or len(idx) == 0:
            continue
        profiles.append(x[idx].mean(axis=0))
        onehots.append(np.asarray(adata.obsm[dataset.treatment_col][idx[0]], dtype=np.float32))
    y = np.stack(profiles, axis=1)
    mean = y.mean(axis=1)
    centered = y - mean[:, None]
    gene_emb = np.linalg.svd(centered, full_matrices=False)[0][:, :rank]
    pert_emb = np.stack(onehots) @ gene_emb
    penalty = ridge * np.eye(rank)
    weight = (
        np.linalg.solve(gene_emb.T @ gene_emb + penalty, gene_emb.T)
        @ centered
        @ pert_emb
        @ np.linalg.inv(pert_emb.T @ pert_emb + penalty)
    )
    return mean, gene_emb, weight


def evaluate_mean_linear(datamodule: LightningDataModule, seed: int = 42) -> dict[str, dict[str, float]]:
    """Evaluate the pseudobulk Mean and Linear baselines of Ahlmann-Eltze et al. (2025) on the test set.

    Args:
        datamodule: Data module providing test data loaders.
        seed: Random seed for resampling and metric computation.

    Returns:
        Dictionary with "mean" and "linear" keys, each mapping to a dict of metric names to scalar values. Metrics
        undefined for a point-mass prediction (W2, MMD, MR) are NaN.
    """
    if not hasattr(datamodule, "test_dataset") or datamodule.test_dataset is None:
        datamodule.setup("test")
    datamodule.set_predict_seed(seed)
    loader = datamodule.predict_dataloader()
    batches = list(loader[0] if isinstance(loader, list) else loader)
    obs_data, int_data, treatment = (
        np.concatenate([np.asarray(b[k][:, -1]) for b in batches])
        for k in (BatchKeys.OBS_DATA, BatchKeys.INT_DATA, BatchKeys.TREATMENT)
    )

    treatment = treatment.reshape(len(obs_data), -1)
    mean, gene_emb, weight = fit_mean_linear(datamodule.test_dataset)
    predictions = {
        "mean": np.broadcast_to(mean, obs_data.shape),
        "linear": np.broadcast_to((mean + treatment @ gene_emb @ weight.T @ gene_emb.T)[:, None], obs_data.shape),
    }
    metrics = {}
    for (name, pred), key in zip(predictions.items(), jr.split(jr.PRNGKey(seed), 2), strict=True):
        metrics[name] = compute_distribution_metrics(obs_data, int_data, pred, key=key)
        metrics[name].update({m: float("nan") for m in ("wasserstein", "mmd", "wasserstein_mag_ratio")})
    return metrics


def load_model(
    method: str,
    *,
    run_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    dataset_path: str,
    num_samples: int = 200,
    num_nodes: int = 50,
    seed: int = 42,
) -> tuple[Trainer, LightningModule, LightningDataModule]:
    """Instantiate and load a model for inference.

    Supports MapPFN (JAX/Equinox), CondOT, and MetaFM baselines.
    Weights can be loaded from a W&B run artifact or a local checkpoint file.

    Args:
        method: Model type, one of "map_pfn", "dit", "condot", or "metafm".
            Also accepts full config names like "map_pfn_rna" or "condot_scm".
        run_id: W&B run ID to download the model artifact from.
        checkpoint_path: Path to a local checkpoint file.
        dataset_path: Path to the h5ad dataset file for evaluation.
        num_samples: Number of cell samples per perturbation context.
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

    if base_method in ("map_pfn", "dit"):
        decoder_cfg = DiTConfig if base_method == "dit" else MMDiTConfig
        decoder = decoder_cfg(noise_dim=num_nodes, key=builds(jr.key, seed))
        model_cfg = MapPFNConfig(decoder=decoder, in_dim=num_nodes, key=builds(jr.key, seed))
        module = instantiate(
            JaxLightningModuleConfig(
                model=model_cfg,
                lr_schedule=LRScheduleConfig(total_steps=50_000),
                guidance=2.0,
                key=builds(jr.key, seed),
            )
        )
        datamodule = instantiate(DataModuleConfig(dataset=dataset_cfg, dataset_path=dataset_path))
        trainer = JaxTrainer(callbacks=[TestMetrics()], enable_model_summary=False)

    elif base_method == "condot":
        module = instantiate(CondOTModuleConfig(input_dim=num_nodes, input_dim_label=num_nodes))
        datamodule = instantiate(
            BaselineDataModuleConfig(
                dataset=dataset_cfg,
                dataset_path=dataset_path,
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
                batch_size=10,
            )
        )
        trainer = Trainer(callbacks=[TestMetrics()], enable_model_summary=False)

    else:
        raise ValueError(f"Unknown method '{method}'. Expected 'map_pfn', 'dit', 'condot', or 'metafm'.")

    with resolve_checkpoint(checkpoint_path or run_id) as path:
        _load_weights(module, base_method, path)

    return trainer, module, datamodule


def _load_weights(module: LightningModule, method: str, path: Path) -> None:
    """Load model weights from a checkpoint file.

    Dispatches to Equinox deserialization for MapPFN and PyTorch
    state_dict loading for baselines (CondOT / MetaFM).

    Args:
        module: Lightning module to load weights into.
        method: Base method name ("map_pfn", "dit", "condot", "metafm").
        path: Path to the checkpoint file.
    """
    if method in ("map_pfn", "dit"):
        module.load_checkpoint(path)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        module.load_state_dict(ckpt["state_dict"], strict=False)
        module.eval()
