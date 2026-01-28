from collections.abc import Callable
from pathlib import Path
from typing import Literal, override

import equinox as eqx
import jax.random as jr
import numpy as np
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from lightning import LightningModule

from map_pfn.data.utils import BatchKeys, SplitNames, filter_batch, unpack_batch
from map_pfn.eval.metrics import compute_distribution_metrics
from map_pfn.models.utils import solve_ode


class JaxLightningModule(LightningModule):
    """Jax Lightning module."""

    def __init__(
        self,
        model: Callable,
        loss_fn: eqx.Partial[Callable],
        lr_schedule: optax.Schedule,
        b1: float = 0.9,
        b2: float = 0.995,
        weight_decay: float = 1e-5,
        gradient_accumulation_steps: int = 8,
        gradient_clipping: float = 1.0,
        step_size: float = 0.01,
        guidance: float = 2.0,
        ema_decay: float = 0.999,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """Initialize module.

        Args:
            model: Model to train.
            loss_fn: Loss function to use.
            lr_schedule: Learning rate schedule.
            b1: AdamW beta 1 hyperparameter.
            b2: AdamW beta 2 hyperparameter.
            weight_decay: Weight decay hyperparameter.
            gradient_accumulation_steps: Number of steps to accumulate gradients for.
            gradient_clipping: Value to use for gradient clipping.
            step_size: Initial step size for ODE solver.
            guidance: Classifier-free guidance weight.
            ema_decay: Exponential moving average decay rate.
            key: Random key.
        """
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.loss_fn = loss_fn
        self.lr_schedule = lr_schedule
        self.b1 = b1
        self.b2 = b2
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clipping = gradient_clipping
        self.step_size = step_size
        self.guidance = guidance
        self.optimizer = None
        self.opt_state = None
        self.ema_decay = ema_decay
        self.ema_state = None
        self.ema_model = None
        self.global_step_ = 0
        self.key, self.train_key, self.sample_key, self.metrics_key = jr.split(key, num=4)

    @property
    def global_step(self) -> int:
        """Get the current global step."""
        return self.global_step_

    def step(
        self,
        batch: dict[str, Float[Array, " batch_size ..."]],
        split: Literal["train", "val"],
        dataloader_idx: int = 0,
    ) -> None:
        """Execute a single step."""
        self.model = eqx.nn.inference_mode(self.model, value=split == SplitNames.VAL)
        batch = filter_batch(batch, keys=[BatchKeys.OBS_DATA, BatchKeys.INT_DATA, BatchKeys.TREATMENT])

        self.train_key, train_key = jr.split(self.train_key)
        loss, aux, self.model, self.opt_state, self.ema_state = JaxLightningModule.make_step(
            self.model, batch, self.loss_fn, self.opt_state, self.optimizer, self.ema_state, self.ema, train_key
        )
        self.ema_model = eqx.combine(self.ema_state.ema, eqx.filter(self.model, eqx.is_array, inverse=True))
        obs_data, int_data, obs_data_cond, int_data_cond, treatment = aux

        metrics = {f"{split}/loss": loss.item()}

        if split == SplitNames.TRAIN:
            metrics[f"{split}/lr"] = self.lr_schedule(self.global_step).item()
            metrics[f"{split}/global_step"] = self.global_step
        elif split == SplitNames.VAL:
            self.sample_key, sample_key = jr.split(self.sample_key)
            b, _, s, d = int_data.shape
            generated_int_data = solve_ode(
                self.ema_model,
                noise_shape=(b, s, d),
                obs_data=obs_data_cond,
                int_data=int_data_cond,
                treatment=treatment,
                guidance=self.guidance,
                step_size=self.step_size,
                key=sample_key,
            )

            obs_data_eval = obs_data[:, -1]
            int_data_eval = int_data[:, -1]

            self.metrics_key, metrics_key = jr.split(self.metrics_key)
            dist_metrics = compute_distribution_metrics(
                obs_data_eval, int_data_eval, generated_int_data, key=metrics_key
            )

            metrics.update({f"{split}/{k}": v for k, v in dist_metrics.items()})

        suffix = "" if dataloader_idx == 0 else "/prior"
        metrics = {f"{k}{suffix}": v for k, v in metrics.items()}

        self.log_dict(
            metrics,
            batch_size=1,
            prog_bar=True,
            on_step=(split == SplitNames.TRAIN),
            on_epoch=(split == SplitNames.VAL),
            add_dataloader_idx=False,
        )

    @override
    def training_step(
        self, batch: dict[str, Float[Array, " batch_size ..."]], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Execute single training step with flow matching loss."""
        self.step(batch, split=SplitNames.TRAIN, dataloader_idx=dataloader_idx)
        self.global_step_ += 1

    @override
    def validation_step(
        self, batch: dict[str, Float[Array, " batch_size ..."]], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Execute single validation step with flow matching loss."""
        self.step(batch, split=SplitNames.VAL, dataloader_idx=dataloader_idx)

    @override
    def test_step(
        self, batch: dict[str, Float[Array, " batch_size ..."]], batch_idx: int, dataloader_idx: int = 0
    ) -> dict[str, np.ndarray]:
        """Execute single test step and return predictions with metadata."""
        self.ema_model = eqx.nn.inference_mode(self.ema_model, value=True)

        obs_data, int_data, obs_data_cond, int_data_cond, treatment = unpack_batch(batch)

        self.sample_key, sample_key = jr.split(self.sample_key)
        b, _, s, d = int_data.shape
        predictions = solve_ode(
            self.ema_model,
            noise_shape=(b, s, d),
            obs_data=obs_data_cond,
            int_data=int_data_cond,
            treatment=treatment,
            guidance=self.guidance,
            step_size=self.step_size,
            key=sample_key,
        )

        obs_data_eval = obs_data[:, -1]
        int_data_eval = int_data[:, -1]
        treatment_eval = treatment[:, -1]

        context_id = batch[BatchKeys.CONTEXT_ID]
        treatment_id = batch[BatchKeys.TREATMENT_ID][:, -1]

        return {
            BatchKeys.PRED_INT_DATA: np.asarray(predictions),
            BatchKeys.INT_DATA: np.asarray(int_data_eval),
            BatchKeys.OBS_DATA: np.asarray(obs_data_eval),
            BatchKeys.TREATMENT: np.asarray(treatment_eval),
            BatchKeys.CONTEXT_ID: np.asarray(context_id),
            BatchKeys.TREATMENT_ID: np.asarray(treatment_id),
        }

    @override
    def configure_optimizers(self) -> None:
        """Configure AdamW optimizer and initialize optimizer state."""
        if self.optimizer is not None or self.opt_state is not None:
            return

        optimizer = optax.adamw(
            self.lr_schedule,
            b1=self.b1,
            b2=self.b2,
            weight_decay=self.weight_decay,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.gradient_clipping),
            optax.MultiSteps(optimizer, every_k_schedule=self.gradient_accumulation_steps),
        )
        params = eqx.filter(self.model, eqx.is_array)
        self.opt_state = self.optimizer.init(params)
        self.ema = optax.ema(decay=self.ema_decay)
        self.ema_state = self.ema.init(params)
        self.ema_model = self.model

    @staticmethod
    @eqx.filter_jit
    def make_step(
        model: eqx.Module,
        batch: Float[Array, " batch_size ..."],
        loss_fn: Callable,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
        ema_state: optax.EmaState,
        ema: optax.GradientTransformation,
        key: PRNGKeyArray,
    ) -> tuple[Array, tuple, eqx.Module, optax.OptState, optax.EmaState]:
        """Perform one optimization step using computed gradients.

        JIT-compiled function that computes loss and gradients, applies optimizer
        updates, and returns updated model state.

        Args:
            model: Current velocity field model parameters.
            batch: Training batch of target data points.
            loss_fn: Loss function to use.
            opt_state: Current optimizer state.
            optimizer: Optax optimizer transformation.
            ema_state: Current EMA state.
            ema: EMA transformation.
            key: Random key for loss computation.

        Returns:
            Tuple of (loss, aux, updated_model, updated_opt_state, updated_ema_state).
        """
        loss_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = loss_fn(model, batch=batch, key=key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        _, ema_state = ema.update(eqx.filter(model, eqx.is_array), ema_state)
        return loss, aux, model, opt_state, ema_state

    def save_checkpoint(self, filepath: str | Path) -> None:
        """Save checkpoint to a file.

        Args:
            filepath: Path where checkpoint will be saved.
        """
        filepath = Path(filepath)
        eqx.tree_serialise_leaves(
            filepath,
            {
                "model": self.model,
                "opt_state": self.opt_state,
                "ema_state": self.ema_state,
            },
        )

    def load_checkpoint(self, filepath: str | Path, evaluate: bool = True) -> None:
        """Load a checkpoint from a file.

        Args:
            filepath: Path to load the checkpoint from.
            evaluate: Whether to load the model in eval mode.
        """
        filepath = Path(filepath)
        self.configure_optimizers()
        state_dict = eqx.tree_deserialise_leaves(
            filepath,
            {
                "model": self.model,
                "opt_state": self.opt_state,
                "ema_state": self.ema_state,
            },
        )
        self.model = eqx.nn.inference_mode(state_dict["model"], value=evaluate)
        self.opt_state = state_dict["opt_state"]
        self.ema_state = state_dict["ema_state"]
        self.ema_model = eqx.combine(self.ema_state.ema, eqx.filter(self.model, eqx.is_array, inverse=True))
