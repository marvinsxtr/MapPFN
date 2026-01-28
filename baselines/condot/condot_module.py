"""
Adapted from https://github.com/bunnech/condot/blob/main/condot/train/train.py
"""

from lightning import LightningModule
import torch
import numpy as np
import jax.random as jr

from baselines.condot.picnn import PICNN
from map_pfn.data.utils import BatchKeys
from map_pfn.eval.metrics import compute_distribution_metrics


class CondOTModule(LightningModule):
    """
    Conditional Optimal Transport model using dual PICNNs as a LightningModule.
    
    Matches the original CondOT training dynamics:
    - Two networks: f (dual potential) and g (transport potential)
    - Alternating optimization with n_inner_iters for g
    - Loss functions matching the Kantorovich dual formulation
    """

    def __init__(
        self,
        input_dim: int,
        input_dim_label: int,
        hidden_units: int = 64,
        num_layers: int = 4,
        activation: str = "leakyrelu",
        softplus_wz_kernels: bool = False,
        softplus_beta: float = 1.0,
        fnorm_penalty: float = 1.0,
        lr_f: float = 1e-4,
        lr_g: float = 1e-4,
        betas: tuple[float, float] = (0.5, 0.9),
        weight_decay: float = 0.0,
        n_inner_iters: int = 10,
        **kwargs,
    ):
        """
        Args:
            input_dim: Dimension of the input data (x)
            input_dim_label: Dimension of the conditioning label (y)
            hidden_units: Hidden layer size (default: 64)
            num_layers: Number of hidden layers (default: 4)
            activation: Activation function ("relu" or "leakyrelu")
            softplus_wz_kernels: Whether to use softplus for non-negative weights
            softplus_beta: Beta parameter for softplus
            fnorm_penalty: Frobenius norm penalty for weight regularization
            lr_f: Learning rate for f network
            lr_g: Learning rate for g network
            betas: Adam betas (default: (0.5, 0.9)
            weight_decay: Weight decay for optimizer
            n_inner_iters: Number of inner iterations for g optimization per f step
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        hidden_units = [hidden_units] * num_layers

        self.input_dim = input_dim
        self.input_dim_label = input_dim_label
        self.lr_f = lr_f
        self.lr_g = lr_g
        self.betas = betas
        self.weight_decay = weight_decay
        self.n_inner_iters = n_inner_iters
        self._train_iter = None

        # f: dual potential network
        self.f = PICNN(
            input_dim=input_dim,
            input_dim_label=input_dim_label,
            hidden_units=hidden_units,
            activation=activation,
            softplus_wz_kernels=softplus_wz_kernels,
            softplus_beta=softplus_beta,
            fnorm_penalty=fnorm_penalty,
            **kwargs,
        )
        
        # g: transport network (gradient defines transport map)
        self.g = PICNN(
            input_dim=input_dim,
            input_dim_label=input_dim_label,
            hidden_units=hidden_units,
            activation=activation,
            softplus_wz_kernels=softplus_wz_kernels,
            softplus_beta=softplus_beta,
            fnorm_penalty=fnorm_penalty,
            **kwargs,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the dual potential f(x, y).
        
        Args:
            x: Input samples (batch_size, input_dim)
            y: Conditioning labels (batch_size, input_dim_label) or (input_dim_label,)
            
        Returns:
            Potential values (batch_size, 1)
        """
        return self.f(x, y)

    def transport(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Transport samples x to target distribution conditioned on y.
        
        The transport map is the gradient of g's convex potential.
        
        Args:
            x: Input samples (batch_size, input_dim), requires_grad=True
            y: Conditioning labels (batch_size, input_dim_label) or (input_dim_label,)
            
        Returns:
            Transported samples (batch_size, input_dim)
        """
        return self.g.transport(x, y)

    def compute_loss_g(
        self, 
        source: torch.Tensor, 
        condition: torch.Tensor,
        transport: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for g network (transport network).
        
        L_g = f(T(x), y) - <x, T(x)>
        
        where T(x) = ∇g(x, y) is the transport map.
        
        Args:
            source: Source samples (batch_size, input_dim)
            condition: Conditioning labels (batch_size, input_dim_label)
            transport: Pre-computed transport (optional)
            
        Returns:
            Loss value (to minimize, which maximizes the dual objective for g)
        """
        if transport is None:
            transport = self.g.transport(source, condition)
        
        # L_g = f(T(x), y) - <x, T(x)>
        loss = self.f(transport, condition) - torch.multiply(
            source, transport
        ).sum(-1, keepdim=True)
        
        return loss

    def compute_loss_f(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
        transport: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute loss for f network (dual potential).
        
        L_f = -f(T(x), y) + f(y_target, y)
        
        Args:
            source: Source samples (batch_size, input_dim)
            target: Target samples (batch_size, input_dim)
            condition: Conditioning labels (batch_size, input_dim_label)
            transport: Pre-computed transport (optional)
            
        Returns:
            Loss value
        """
        if transport is None:
            transport = self.g.transport(source, condition)
        
        # L_f = -f(T(x), y) + f(target, y)
        loss = -self.f(transport, condition) + self.f(target, condition)
        
        return loss

    def on_train_epoch_start(self):
        """Initialize the training iterator at the start of each epoch."""
        self._train_iter = iter(self.trainer.train_dataloader)

    def _unpack_batch(self, batch):
        """
        Unpack a single batch.
        
        Returns:
            x0: Source samples
            x1: Target samples
            treat_cond: Treatment condition
        """
        x0 = batch[BatchKeys.OBS_DATA].squeeze(1).squeeze(0)
        x1 = batch[BatchKeys.INT_DATA].squeeze(1).squeeze(0)
        treat_cond = batch[BatchKeys.TREATMENT].repeat(1, 1, x0.shape[0], 1)
        treat_cond = treat_cond.squeeze(1).squeeze(0)

        return x0, x1, treat_cond

    def get_train_batch(self):
        """Fetch a fresh training batch from the iterator."""
        try:
            batch = next(self._train_iter)
        except StopIteration:
            # Reinitialize iterator if exhausted
            self._train_iter = iter(self.trainer.train_dataloader)
            batch = next(self._train_iter)

        return {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def training_step(self, batch, batch_idx):
        """
        Training step matching original CondOT implementation exactly.
        
        Alternates between:
        1. n_inner_iters updates of g (transport network)
        2. Single update of f (dual potential)
        
        Expects batch to be a dict or tuple with source (x0), target (x1), 
        and condition (y) tensors.
        """
        opt_f, opt_g = self.optimizers()
        source_f, target_f, condition_f = self._unpack_batch(batch)

        # Only count steps for the outer loop optimizing f
        opt_g._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
        opt_g._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")

        # Inner loop: optimize g (transport network) for n_inner_iters
        for _ in range(self.n_inner_iters):
            batch_g = self.get_train_batch()
            source_g, _, condition_g = self._unpack_batch(batch_g)

            source_g.requires_grad_(True)
            opt_g.zero_grad()
            
            loss_g = self.compute_loss_g(source_g, condition_g).mean()
            
            # Add weight penalty if using fnorm regularization
            if not self.g.softplus_wz_kernels and self.g.fnorm_penalty > 0:
                loss_g = loss_g + self.g.penalize_w()

            self.manual_backward(loss_g)
            opt_g.step()

        source_f.requires_grad_(True)
        opt_f.zero_grad()
        
        loss_f = self.compute_loss_f(source_f, target_f, condition_f).mean()
        
        self.manual_backward(loss_f)
        opt_f.step()
        
        self.f.clamp_w()

        self.log("train/loss_g", loss_g, batch_size=1, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss_f", loss_f, batch_size=1, prog_bar=True, on_step=True, on_epoch=False)
        
        return {"loss_g": loss_g, "loss_f": loss_f}

    def validation_step(self, batch, batch_idx):
        """Validation step computing W2 distance and losses."""
        source, target, condition = self._unpack_batch(batch)

        with torch.enable_grad():
            source.requires_grad_(True)
            transport = self.g.transport(source, condition)

        transport_detached = transport.detach()

        with torch.no_grad():
            loss_g = self.compute_loss_g(source, condition, transport_detached).mean()
            loss_f = self.compute_loss_f(source, target, condition, transport_detached).mean()

        key = jr.PRNGKey(0)
        dist_metrics = compute_distribution_metrics(
            source.unsqueeze(0).detach().cpu().numpy(), 
            target.unsqueeze(0).detach().cpu().numpy(), 
            transport_detached.unsqueeze(0).detach().cpu().numpy(), 
            key=key
        )

        metrics = {
            "val/loss_g": loss_g,
            "val/loss_f": loss_f,
        }
        metrics.update({f"val/{k}": v for k, v in dist_metrics.items()})

        self.log_dict(metrics, batch_size=1, prog_bar=True, on_step=True, on_epoch=False)
        
        return {"loss_g": loss_g, "loss_f": loss_f}

    def configure_optimizers(self):
        """Configure separate optimizers for f and g networks."""
        opt_f = torch.optim.Adam(
            self.f.parameters(),
            lr=self.lr_f,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        opt_g = torch.optim.Adam(
            self.g.parameters(),
            lr=self.lr_g,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return [opt_f, opt_g]

    def test_step(self, batch, batch_idx) -> dict[str, np.ndarray]:
        """Transport samples and return predictions with metadata."""
        x0, x1, treat_cond = self._unpack_batch(batch)

        with torch.inference_mode(False):
            x0_grad = x0.clone().requires_grad_(True)
            treat_cond_clone = treat_cond.clone()
            predictions = self.g.transport(x0_grad, treat_cond_clone)

        return {
            BatchKeys.PRED_INT_DATA: predictions.detach().cpu().numpy()[np.newaxis, :],
            BatchKeys.INT_DATA: x1.detach().cpu().numpy()[np.newaxis, :],
            BatchKeys.OBS_DATA: x0.detach().cpu().numpy()[np.newaxis, :],
            BatchKeys.TREATMENT: batch[BatchKeys.TREATMENT].detach().cpu().numpy().squeeze(1),
            BatchKeys.CONTEXT_ID: np.asarray(batch[BatchKeys.CONTEXT_ID]),
            BatchKeys.TREATMENT_ID: np.asarray(batch[BatchKeys.TREATMENT_ID]),
        }