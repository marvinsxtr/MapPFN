from hydra_zen import MISSING, builds, store

from map_pfn.configs.run_config import (
    CPUJobConfig,
    GPUJobConfig,
    SweepCondOTSCMConfig,
    SweepMetaFMSCMConfig,
    SweepMapPFNSCMConfig,
    SweepMapPFNSergioConfig,
    SweepMethodsSCMConfig,
    SweepMethodsSergioConfig,
    WandBConfig,
)
from map_pfn.configs.train.base_config import (
    MapPFNRNATrainingRunConfig,
    MapPFNSCMTrainingRunConfig,
    RNADataModuleConfig,
    SCMDataModuleConfig,
    SergioRNADataModuleConfig,
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
        {"cfg": "map_pfn_scm"},
        {"cfg/datamodule": "scm"},
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
datamodule_store(RNADataModuleConfig, name="rna")
datamodule_store(SergioRNADataModuleConfig, name="sergio_rna")

wandb_store = store(group="cfg/wandb")
wandb_store(WandBConfig, name="base")

job_store = store(group="cfg/job")
job_store(GPUJobConfig, name="gpu")
job_store(CPUJobConfig, name="cpu")
job_store(SweepMethodsSCMConfig, name="methods_scm")
job_store(SweepMapPFNSCMConfig, name="map_pfn_scm")
job_store(SweepMethodsSergioConfig, name="methods_sergio")
job_store(SweepMapPFNSergioConfig, name="map_pfn_sergio")
job_store(SweepCondOTSCMConfig, name="condot_scm")
job_store(SweepMetaFMSCMConfig, name="metafm_scm")
