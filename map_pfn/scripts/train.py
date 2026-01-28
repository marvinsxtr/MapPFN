from __future__ import annotations

from lightning.pytorch.loggers import WandbLogger

from map_pfn.utils.config import run
from map_pfn.utils.helpers import (
    configure_jax,
    download_artifact,
    get_output_dir,
    register_resolvers,
    seed_everything,
)


def train(cfg: TrainingRun) -> None:
    """Run a main function from a config."""
    wandb_logger = WandbLogger(save_dir=get_output_dir()) if cfg.wandb is not None else None
    trainer = cfg.trainer(logger=wandb_logger, fast_dev_run=cfg.debug)

    if cfg.load_artifact is None:
        trainer.fit(cfg.module, datamodule=cfg.datamodule)
        trainer.test(cfg.module, datamodule=cfg.datamodule)
    else:
        with download_artifact(f"map-pfn/map-pfn/model-{cfg.load_artifact}:latest") as artifact:
            cfg.module.load_checkpoint(artifact)

        trainer.test(cfg.module, datamodule=cfg.datamodule)


if __name__ == "__main__":
    configure_jax()
    register_resolvers()

    from map_pfn.configs.train import config_stores  # noqa: F401
    from map_pfn.configs.train.base_config import TrainingRun  # Imports jax

    run(train, seed_fn=seed_everything)
