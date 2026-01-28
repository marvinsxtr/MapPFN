from typing import Any, NamedTuple

import equinox as eqx
import jax.random as jr
from hydra_zen import MISSING, builds
from hydra_zen.typing import Partial
from lightning import LightningDataModule, LightningModule
from ml_project_template.runs import Job
from ml_project_template.utils import get_output_dir
from ml_project_template.wandb import WandBRun

from map_pfn.data.data_module import DataModule
from map_pfn.data.perturbation_dataset import PerturbationDataset
from map_pfn.loss.loss_fn import fm_loss
from map_pfn.models.map_pfn import MapPFN
from map_pfn.models.mmdit import MMDiT
from map_pfn.train.jax_lightning import JaxLightningModule
from map_pfn.train.wsd_schedule import warmup_stable_decay_schedule
from map_pfn.utils.helpers import cpu_count
from map_pfn.utils.lightning import Checkpointer, JaxTrainer, ModelSummary, Stopper, TestMetrics


class TrainingRun(NamedTuple):
    """Configures a training run."""

    name: str
    module: LightningModule
    datamodule: LightningDataModule
    trainer: Partial[JaxTrainer]
    wandb: WandBRun | None = None
    globals: dict[str, Any] | None = None
    seed: int | None = None
    job: Job | None = None
    debug: bool = False
    load_artifact: str | None = None


GlobalsConfig = builds(dict, in_dim=MISSING, num_steps=50_000, data_commit_hash=MISSING)
TestMetricsConfig = builds(TestMetrics)
ModelSummaryConfig = builds(ModelSummary)
CheckpointerConfig = builds(Checkpointer, dirpath=builds(get_output_dir), filename="model")
StopperConfig = builds(Stopper, max_steps="${globals: num_steps}")

TrainerConfig = builds(
    JaxTrainer,
    callbacks=[CheckpointerConfig, ModelSummaryConfig, StopperConfig, TestMetricsConfig],
    max_steps="${globals: num_steps}",
    val_check_interval=500,
    limit_val_batches=1,
    log_every_n_steps=10,
    check_val_every_n_epoch=None,
    enable_model_summary=False,
    zen_partial=True,
)

PerturbationDatasetConfig = builds(
    PerturbationDataset,
    num_samples=MISSING,
    seed="${cfg.seed}",
    zen_partial=True,
)

DataModuleConfig = builds(
    DataModule,
    dataset_path=MISSING,
    prior_dataset_path=None,
    dataset=PerturbationDatasetConfig,
    ood=True,
    num_shots=4,
    batch_size=32,
    num_workers=cpu_count() - 1,
    persistent_workers=True,
    drop_last=True,
)

SCMDataModuleConfig = {
    "dataset_path": "datasets/synthetic/linear_scm.h5ad",
    "dataset": {"num_samples": 500},
}

RNADataModuleConfig = {
    "dataset_path": "datasets/single_cell/frangieh_rna.h5ad",
    "dataset": {"num_samples": 200},
}

SergioRNADataModuleConfig = {
    "dataset_path": "datasets/single_cell/frangieh_rna.h5ad",
    "prior_dataset_path": '${eval:\'"datasets/synthetic/sergio_grn.h5ad" if "map_pfn" in "${cfg.name}" else None\'}',
    "dataset": {"num_samples": 200},
}

MMDiTConfig = builds(
    MMDiT,
    embed_dim=256,
    cond_dim=256,
    noise_dim="${globals: in_dim}",
    num_heads=4,
    num_blocks=8,
    use_rms_norm=True,
    num_reg_tokens=8,
    key=builds(jr.key, "${cfg.seed}"),
)

MapPFNConfig = builds(
    MapPFN,
    decoder=MMDiTConfig,
    in_dim="${globals: in_dim}",
    cond_dim=256,
    key=builds(jr.key, "${cfg.seed}"),
)

LossConfig = builds(eqx.Partial, fm_loss)

LRScheduleConfig = builds(
    warmup_stable_decay_schedule,
    peak_value=1e-4,
    total_steps="${globals: num_steps}",
    warmup_frac=0.01,
    decay_frac=0.2,
)

JaxLightningModuleConfig = builds(
    JaxLightningModule,
    model=MapPFNConfig,
    loss_fn=LossConfig,
    lr_schedule=LRScheduleConfig,
    gradient_accumulation_steps=8,
    step_size=0.01,
    guidance=2.0,
    key=builds(jr.key, "${cfg.seed}"),
)

MapPFNSCMTrainingRunConfig = builds(
    TrainingRun,
    name="map_pfn_scm",
    module=JaxLightningModuleConfig,
    datamodule=DataModuleConfig,
    trainer=TrainerConfig,
    seed=42,
    wandb=None,
    job=None,
    globals=GlobalsConfig,
    debug=False,
    load_artifact=None,
)

MapPFNRNATrainingRunConfig = MapPFNSCMTrainingRunConfig(
    name="map_pfn_rna", globals=GlobalsConfig(num_steps=400_000), datamodule=DataModuleConfig(num_shots=8)
)
