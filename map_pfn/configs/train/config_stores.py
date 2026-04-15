from hydra_zen import MISSING, builds, store

from map_pfn.configs.run_config import (
    CPUJobConfig,
    GPUJobConfig,
    MultiGPUJobConfig,
    SweepMapPFNSergioConfig,
    SweepMethodsSCMConfig,
    SweepMethodsSergioConfig,
    WandBConfig,
)
from map_pfn.configs.train.base_config import (
    FrangiehDataModuleConfig,
    FrangiehFinetuneDataModuleConfig,
    MapPFNRNATrainingRunConfig,
    MapPFNSCMTrainingRunConfig,
    PapalexiDataModuleConfig,
    PapalexiFinetuneDataModuleConfig,
    SCMDataModuleConfig,
)
from map_pfn.configs.train.baseline_config import (
    CondOTRNATrainingRunConfig,
    CondOTSCMTrainingRunConfig,
    MetaFMRNATrainingRunConfig,
    MetaFMSCMTrainingRunConfig,
)
from map_pfn.scripts.train import train

MainConfig = builds(train, cfg=MISSING, populate_full_signature=True)

store(
    MainConfig,
    name="root",
    hydra_defaults=[
        "_self_",
        {"cfg": "map_pfn_rna"},
        {"cfg/datamodule": "frangieh"},
        {"cfg/wandb": None},
        {"cfg/job": None},
    ],
)

cfg_store = store(group="cfg")
cfg_store(MapPFNSCMTrainingRunConfig, name="map_pfn_scm")
cfg_store(MapPFNRNATrainingRunConfig, name="map_pfn_rna")
cfg_store(CondOTSCMTrainingRunConfig, name="condot_scm")
cfg_store(CondOTRNATrainingRunConfig, name="condot_rna")
cfg_store(MetaFMSCMTrainingRunConfig, name="metafm_scm")
cfg_store(MetaFMRNATrainingRunConfig, name="metafm_rna")

datamodule_store = store(group="cfg/datamodule")
datamodule_store(SCMDataModuleConfig, name="scm")
datamodule_store(FrangiehDataModuleConfig, name="frangieh")
datamodule_store(PapalexiDataModuleConfig, name="papalexi")
datamodule_store(FrangiehFinetuneDataModuleConfig, name="frangieh_finetune")
datamodule_store(PapalexiFinetuneDataModuleConfig, name="papalexi_finetune")

wandb_store = store(group="cfg/wandb")
wandb_store(WandBConfig, name="base")

job_store = store(group="cfg/job")
job_store(GPUJobConfig, name="gpu")
job_store(MultiGPUJobConfig, name="multi_gpu")
job_store(CPUJobConfig, name="cpu")
job_store(SweepMethodsSCMConfig, name="methods_scm")
job_store(SweepMethodsSergioConfig, name="methods_sergio")
job_store(SweepMapPFNSergioConfig, name="map_pfn_sergio")
