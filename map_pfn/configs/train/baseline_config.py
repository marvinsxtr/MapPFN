from hydra_zen import MISSING, builds
from lightning import Trainer

from baselines.condot.condot_module import CondOTModule
from baselines.metafm.metafm_module import MetaFMModule
from map_pfn.configs.train.base_config import (
    CheckpointerConfig,
    GlobalsConfig,
    PerturbationDatasetConfig,
    StopperConfig,
    TestMetricsConfig,
    TrainingRun,
)
from map_pfn.data.data_module import DataModule
from map_pfn.data.utils import torch_collate_fn
from map_pfn.utils.helpers import cpu_count

BaselineTrainerConfig = builds(
    Trainer,
    callbacks=[CheckpointerConfig, StopperConfig, TestMetricsConfig],
    max_steps="${globals: num_steps}",
    val_check_interval=500,
    limit_val_batches=1,
    log_every_n_steps=10,
    check_val_every_n_epoch=None,
    enable_model_summary=True,
    zen_partial=True,
)

BaselineDataModuleConfig = builds(
    DataModule,
    dataset_path=MISSING,
    prior_dataset_path=None,
    dataset=PerturbationDatasetConfig,
    ood=True,
    num_shots=0,
    batch_size=MISSING,
    num_workers=cpu_count() - 1,
    persistent_workers=True,
    drop_last=True,
    collate_fn=torch_collate_fn,
)

CondOTModuleConfig = builds(
    CondOTModule,
    input_dim="${globals: in_dim}",
    input_dim_label="${globals: in_dim}",
    hidden_units=128,
    num_layers=4,
    lr=1e-4,
    softplus_wz_kernels=False,
    fnorm_penalty=1,
    betas=(0.5, 0.9),
)

MetaFMModuleConfig = builds(
    MetaFMModule,
    dim="${globals: in_dim}",
    num_hidden_gnn=64,
    flow_lr=1e-4,
    gnn_lr=1e-4,
    knn_k=50,
    num_treat_conditions="${globals: in_dim}",
)

CondOTSCMTrainingRunConfig = builds(
    TrainingRun,
    name="condot_scm",
    module=CondOTModuleConfig,
    datamodule=BaselineDataModuleConfig(batch_size=1),
    trainer=BaselineTrainerConfig,
    seed=42,
    wandb=None,
    job=None,
    globals=GlobalsConfig,
    debug=False,
)

CondOTRNATrainingRunConfig = CondOTSCMTrainingRunConfig(name="condot_rna", globals=GlobalsConfig(num_steps=100_000))

MetaFMSCMTrainingRunConfig = builds(
    TrainingRun,
    name="metafm_scm",
    module=MetaFMModuleConfig,
    datamodule=BaselineDataModuleConfig(batch_size=10),
    trainer=BaselineTrainerConfig,
    seed=42,
    wandb=None,
    job=None,
    globals=GlobalsConfig,
    debug=False,
)

MetaFMRNATrainingRunConfig = MetaFMSCMTrainingRunConfig(name="metafm_rna", globals=GlobalsConfig(num_steps=100_000))
