from pathlib import Path
from typing import Any, override

import anndata as ad
import equinox as eqx
import jax
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.callbacks import Checkpoint
from ml_project_template.utils import get_output_dir

import wandb
from map_pfn.data.utils import BatchKeys
from map_pfn.eval.metrics import compute_distribution_metrics
from map_pfn.utils.logging import logger


class Stopper(Callback):
    """Stops training after a certain number of steps."""

    def __init__(self, max_steps: int) -> None:
        """Args:
        max_steps: Maximum number of training steps.
        """
        self.max_steps = max_steps

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, *_: Any, **__: Any) -> None:
        """Called when the train batch begins."""
        if pl_module.global_step >= self.max_steps:
            trainer.should_stop = True
            logger.info(f"Stopping training after {self.max_steps} steps.")


class ModelSummary(Callback):
    """Generates a model summary."""

    @override
    def on_fit_start(self, trainer: Trainer, module: LightningModule) -> None:
        if not hasattr(trainer.model, "model"):
            raise ValueError("Model not found.")

        total_num_params = sum(x.size for x in jax.tree_util.tree_leaves(trainer.model.model) if eqx.is_array(x))

        logger.info(f"The model has a total of {total_num_params:,} parameters.")


class Checkpointer(Checkpoint):
    """Used to save a checkpoint on exception and train epoch end."""

    FILE_EXTENSION = ".ckpt"

    def __init__(self, dirpath: _PATH, filename: str = "model") -> None:
        """Args:
            dirpath: directory to save the checkpoint file.
            filename: checkpoint filename. This must not include the extension.

        Raises:
            ValueError:
                If `filename` is empty.
        """
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename

    @property
    def ckpt_path(self) -> str:
        """Return the checkpoint path."""
        return Path(self.dirpath) / (self.filename + self.FILE_EXTENSION)

    def save_artifact(self) -> None:
        """Save the model as an artifact to WandB."""
        if (run := wandb.run) is not None:
            model_id = f"model-{run.id}"
            artifact = wandb.Artifact(model_id, type="model")
            artifact.add_file(self.ckpt_path)
            wandb.log_artifact(artifact)

            api = wandb.Api(overrides={"project": run.project, "entity": run.entity, "run": run.id})

            try:
                for version in api.artifacts("model", model_id):
                    # Clean up all versions that don't have an alias such as 'latest'.
                    if len(version.aliases) == 0:
                        version.delete()
            except Exception as e:
                logger.error(f"Error cleaning up WandB artifacts: {e}")

    @override
    def on_exception(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        trainer.save_checkpoint(self.ckpt_path)
        self.save_artifact()

    @override
    def on_train_end(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        trainer.save_checkpoint(self.ckpt_path)
        self.save_artifact()

    @override
    def on_validation_end(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        trainer.save_checkpoint(self.ckpt_path)
        self.save_artifact()

    @override
    def teardown(self, trainer: Trainer, *_: Any, **__: Any) -> None:
        trainer.strategy.remove_checkpoint(self.ckpt_path)


class JaxTrainer(Trainer):
    """Lightning Trainer with Equinox serialization for JAX/Equinox models."""

    @override
    def save_checkpoint(
        self, filepath: str | Path, weights_only: bool = False, storage_options: Any | None = None
    ) -> None:
        """Save checkpoint to a file.

        Args:
            filepath: Path where checkpoint will be saved.
            weights_only: Ignored for compatibility.
            storage_options: Ignored for compatibility.
        """
        filepath = Path(filepath)
        self.model.save_checkpoint(filepath)

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Load a checkpoint from a file.

        Args:
            filepath: Path to load the checkpoint from.
        """
        filepath = Path(filepath)
        self.model.load_checkpoint(filepath)


class TestMetrics(Callback):
    """Collect test predictions and compute epoch-level metrics."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._outputs = {}

    @override
    def on_test_epoch_start(self, trainer: Trainer, module: LightningModule) -> None:
        """Initialize storage for test outputs."""
        self._outputs.clear()

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        module: LightningModule,
        outputs: dict[str, torch.Tensor],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Collect outputs from each test batch."""
        if dataloader_idx not in self._outputs:
            self._outputs[dataloader_idx] = []
        self._outputs[dataloader_idx].append(outputs)

    @override
    def on_test_epoch_end(self, trainer: Trainer, module: LightningModule) -> None:
        """Aggregate test outputs and compute metrics."""
        base_path = get_output_dir()

        for dataloader_idx, outputs in self._outputs.items():
            obs_data = np.concatenate([b[BatchKeys.OBS_DATA] for b in outputs], axis=0)
            int_data = np.concatenate([b[BatchKeys.INT_DATA] for b in outputs], axis=0)
            pred_int_data = np.concatenate([b[BatchKeys.PRED_INT_DATA] for b in outputs], axis=0)
            treatment = np.concatenate([b[BatchKeys.TREATMENT] for b in outputs], axis=0)
            context_id = np.concatenate([b[BatchKeys.CONTEXT_ID] for b in outputs], axis=0)
            treatment_id = np.concatenate([b[BatchKeys.TREATMENT_ID] for b in outputs], axis=0)

            metrics_key = jax.random.PRNGKey(self.seed)
            dist_metrics = compute_distribution_metrics(obs_data, int_data, pred_int_data, key=metrics_key)

            suffix = "" if dataloader_idx == 0 else "/prior"
            module.log_dict({f"test/{k}{suffix}": v for k, v in dist_metrics.items()}, add_dataloader_idx=False)

            _, n_cells, n_features = int_data.shape

            adata = ad.AnnData(
                X=int_data.reshape(-1, n_features),
                obs={
                    BatchKeys.CONTEXT_ID: np.repeat(context_id, n_cells),
                    BatchKeys.TREATMENT_ID: np.repeat(treatment_id, n_cells),
                },
                obsm={
                    BatchKeys.TREATMENT: np.repeat(treatment, n_cells, axis=0),
                },
                layers={"pred": pred_int_data.reshape(-1, n_features)},
            )

            suffix = "" if dataloader_idx == 0 else "_prior"
            filepath = base_path / f"test_predictions{suffix}.h5ad"
            adata.write_h5ad(filepath)

            if wandb.run is not None:
                wandb.save(str(filepath), base_path=str(base_path))

        self._outputs.clear()
