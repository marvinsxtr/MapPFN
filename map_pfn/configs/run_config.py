from hydra_zen import builds
from ml_project_template.wandb import WandBRun

from ml_project_template.runs import Job, SlurmParams, SweepJob

IMAGE = "${oc.env:JOB_IMAGE,oras://ghcr.io/marvinsxtr/mappfn:latest-sif}"

MediumGPURunSlurmConfig = builds(
    SlurmParams,
    partition="gpu-2d",
    time_hours=48,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=1,
    gpus_per_node=1,
    mem_gb=256,
    constraint="80gb|h100",
    exclude="head021,head023,head025,head074",
)

MediumCPURunSlurmConfig = builds(
    SlurmParams,
    partition="cpu-2d",
    time_hours=48,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=16,
    gpus_per_task=0,
    gpus_per_node=0,
    mem_gb=256,
)

GPUTestRunSlurmConfig = MediumGPURunSlurmConfig(partition="gpu-2h", time_hours=2, cpus_per_task=8, mem_gb=64)

WandBConfig = builds(WandBRun, group=None, mode="online", id=None, resume=None)

GPUJobConfig = builds(Job, image=IMAGE, slurm_params=MediumGPURunSlurmConfig)
CPUJobConfig = builds(Job, image=IMAGE, slurm_params=MediumCPURunSlurmConfig)
GPUTestJobConfig = builds(Job, image=IMAGE, slurm_params=GPUTestRunSlurmConfig)

SweepMethodsSCMConfig = builds(
    SweepJob,
    image=IMAGE,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=3,
    parameters={
        "cfg": ["map_pfn_scm", "condot_scm", "metafm_scm"],
        "cfg/datamodule": ["scm"],
    },
)

SweepMethodsSergioConfig = builds(
    SweepJob,
    image=IMAGE,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=4,
    parameters={
        "cfg": ["metafm_rna", "condot_rna"],
        "cfg/datamodule": ["frangieh", "papalexi"],
    },
)

SweepMapPFNSergioConfig = builds(
    SweepJob,
    image=IMAGE,
    slurm_params=MediumGPURunSlurmConfig,
    num_workers=1,
    parameters={
        "cfg": ["map_pfn_rna"],
        "cfg/datamodule": ["frangieh"],
    },
)
