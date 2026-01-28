from hydra_zen import builds
from ml_project_template.wandb import WandBRun

from map_pfn.utils.run import Job, SweepJob, TrainingSlurmParams

LongGPURunSlurmConfig = builds(
    TrainingSlurmParams,
    partition="gpu-7d",
    time_hours=168,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=1,
    gpus_per_node=1,
    mem_gb=128,
    constraint="80gb|h100",
    exclude="head074,head021",
)

MediumGPURunSlurmConfig = builds(
    TrainingSlurmParams,
    partition="gpu-2d",
    time_hours=48,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=1,
    gpus_per_node=1,
    mem_gb=128,
    constraint="80gb|h100",
    exclude="head074,head021",
)

MediumCPURunSlurmConfig = builds(
    TrainingSlurmParams,
    partition="cpu-2d",
    time_hours=48,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=0,
    gpus_per_node=0,
    mem_gb=128,
)

WandBConfig = builds(WandBRun, group=None, mode="online", entity="map-pfn", project="map-pfn")

GPUJobConfig = builds(Job, slurm_params=MediumGPURunSlurmConfig)
CPUJobConfig = builds(Job, slurm_params=MediumCPURunSlurmConfig)

SweepMethodsSCMConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=18,
    parameters={
        "cfg": ["map_pfn_scm", "condot_scm", "metafm_scm"],
        "cfg.seed": [100, 200, 300],
        "cfg/datamodule": ["scm"],
        "cfg.datamodule.ood": [True, False],
    },
)

SweepMapPFNSCMConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=6,
    parameters={
        "cfg": ["map_pfn_scm"],
        "cfg.seed": [100, 200, 300],
        "cfg/datamodule": ["scm"],
        "cfg.datamodule.ood": [True, False],
        "cfg.globals.num_steps": [200_000],
    },
)

SweepMethodsSergioConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=18,
    parameters={
        "cfg": ["map_pfn_rna", "condot_rna", "metafm_rna"],
        "cfg.seed": [100, 200, 300],
        "cfg/datamodule": ["sergio_rna"],
        "cfg.datamodule.ood": [True, False],
    },
)

SweepMapPFNSergioConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=6,
    parameters={
        "cfg": ["map_pfn_rna"],
        "cfg.seed": [100, 200, 300],
        "cfg/datamodule": ["sergio_rna"],
        "cfg.datamodule.ood": [True, False],
    },
)

SweepCondOTSCMConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=9,
    parameters={
        "cfg": ["condot_scm"],
        "cfg.module.hidden_units": [64, 128, 256],
        "cfg.module.num_layers": [2, 3, 4],
        "cfg/datamodule": ["scm"],
    },
)

SweepMetaFMSCMConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=12,
    parameters={
        "cfg": ["metafm_scm"],
        "cfg.module.knn_k": [0, 10, 50, 100],
        "cfg.module.num_hidden_gnn": [64, 128, 256],
        "cfg/datamodule": ["scm"],
    },
)
