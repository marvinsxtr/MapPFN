from hydra_zen import builds
from ml_project_template.wandb import WandBRun

from map_pfn.utils.run import Job, SweepJob, TrainingSlurmParams

MediumGPURunSlurmConfig = builds(
    TrainingSlurmParams,
    partition="gpu-2d",
    time_hours=48,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=1,
    gpus_per_node=1,
    mem_gb=256,
    constraint="80gb|h100",
    exclude="head074,head021",
)

MediumMultiGPURunSlurmConfig = builds(
    TrainingSlurmParams,
    partition="gpu-2d",
    time_hours=48,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=4,
    gpus_per_node=4,
    mem_gb=256,
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
    mem_gb=256,
)

WandBConfig = builds(WandBRun, group=None, mode="online")

GPUJobConfig = builds(Job, slurm_params=MediumGPURunSlurmConfig)
MultiGPUJobConfig = builds(Job, slurm_params=MediumMultiGPURunSlurmConfig)
CPUJobConfig = builds(Job, slurm_params=MediumCPURunSlurmConfig)

SweepMethodsSCMConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=3,
    parameters={
        "cfg": ["map_pfn_scm", "condot_scm", "metafm_scm"],
        "cfg/datamodule": ["scm"],
    },
)

SweepMethodsSergioConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=4,
    parameters={
        "cfg": ["metafm_rna", "condot_rna"],
        "cfg/datamodule": ["frangieh", "papalexi"],
    },
)

SweepMapPFNSergioConfig = builds(
    SweepJob,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=1,
    parameters={
        "cfg": ["map_pfn_rna"],
        "cfg/datamodule": ["frangieh"],
    },
)
