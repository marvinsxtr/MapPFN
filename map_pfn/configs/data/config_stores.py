from hydra_zen import MISSING, builds, store

from map_pfn.configs.data.base_config import LinearDataGeneratorRunConfig, SergioDataGeneratorRunConfig
from map_pfn.configs.run_config import CPUJobConfig, WandBConfig
from map_pfn.scripts.generate_data import generate_data

MainConfig = builds(generate_data, cfg=MISSING, populate_full_signature=True)

store(
    MainConfig,
    name="root",
    hydra_defaults=[
        "_self_",
        {"cfg": "linear"},
        {"cfg/wandb": None},
        {"cfg/job": None},
    ],
)

cfg_store = store(group="cfg")
cfg_store(LinearDataGeneratorRunConfig, name="linear")
cfg_store(SergioDataGeneratorRunConfig, name="sergio")

wandb_store = store(group="cfg/wandb")
wandb_store(WandBConfig, name="base")

job_store = store(group="cfg/job")
job_store(CPUJobConfig, name="cpu")
