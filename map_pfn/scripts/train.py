from __future__ import annotations

from lightning.pytorch.loggers import WandbLogger

from map_pfn.utils.config import run
from map_pfn.utils.helpers import (
    configure_jax,
    debug_overrides,
    get_output_dir,
    register_resolvers,
    resolve_checkpoint,
    seed_everything,
)


def train(cfg: TrainingRun) -> None:
    """Run a main function from a config."""
    wandb_logger = WandbLogger(save_dir=get_output_dir()) if cfg.wandb is not None else None
    trainer = cfg.trainer(logger=wandb_logger, **(debug_overrides() if cfg.debug else {}))

    if cfg.load_checkpoint is not None:
        with resolve_checkpoint(cfg.load_checkpoint) as path:
            cfg.module.load_checkpoint(path)

    trainer.fit(cfg.module, datamodule=cfg.datamodule)
    trainer.test(cfg.module, datamodule=cfg.datamodule)

    evaluate_baselines(datamodule=cfg.datamodule, seed=cfg.seed)


if __name__ == "__main__":
    configure_jax()
    register_resolvers()

    from map_pfn.configs.train import config_stores  # noqa: F401
    from map_pfn.configs.train.base_config import TrainingRun
    from map_pfn.eval.evaluate import evaluate_baselines

    run(train, seed_fn=seed_everything)
