"""
Adapted from https://github.com/lazaratan/meta-flow-matching/blob/main/src/models/trellis_module.py
"""

import lightning.pytorch as pl
import torch
from torchdyn.core import NeuralODE
import numpy as np
import jax.random as jr

from baselines.metafm.gnn import GlobalGNN
from baselines.metafm.mlp import torch_wrapper, torch_wrapper_gnn_flow_cond
from map_pfn.data.utils import BatchKeys
from map_pfn.eval.metrics import compute_distribution_metrics


class MetaFMModule(pl.LightningModule):
    """MetaFM Lightning Module."""

    def __init__(
        self,
        dim: int,
        num_hidden: int = 512,
        num_layers_decoder: int = 7,
        num_hidden_gnn: int = 128,
        num_layers_gnn: int = 2,
        knn_k: int = 100,
        skip_connections: bool = True,
        flow_lr: float = 1e-4,
        gnn_lr: float = 1e-4,
        num_treat_conditions: int = None,
        num_cell_conditions: int = None,
        base: str = "source",
        integrate_time_steps: int = 500,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.dim = dim
        self.flow_lr = flow_lr
        self.gnn_lr = gnn_lr
        self.num_treat_conditions = num_treat_conditions
        self.integrate_time_steps = integrate_time_steps
        
        assert base in ["source", "gaussian"], "base must be 'source' or 'gaussian'"
        self.base = base

        self.model = GlobalGNN(
            D=dim,
            num_hidden_decoder=num_hidden,
            num_layers_decoder=num_layers_decoder,
            num_hidden_gnn=num_hidden_gnn,
            num_layers_gnn=num_layers_gnn,
            knn_k=knn_k,
            skip_connections=skip_connections,
            num_treat_conditions=num_treat_conditions,
            num_cell_conditions=num_cell_conditions,
        )
        
    def compute_loss(self, embedding, source_samples, target_samples, treat_cond=None):
        """
        Compute the flow matching loss.
                
        Args:
            embedding: Context embedding (N, hidden)
            source_samples: Source samples (N, D)
            target_samples: Target samples (N, D)
            treat_cond: Optional treatment (N, num_treat_conditions)
        """
        t = torch.rand_like(source_samples[..., 0, None])
        
        if self.base == "source":
            y = (1.0 - t) * source_samples + t * target_samples
            u = target_samples - source_samples
            
            if treat_cond is not None:
                y_input = torch.cat((y, treat_cond), dim=-1)
            else:
                y_input = y
            
            b = self.model.flow(embedding, t.squeeze(-1), y_input)
            loss = b.norm(dim=-1) ** 2 - 2.0 * (b * u).sum(dim=-1)
        elif self.base == "gaussian":
            z = torch.randn_like(target_samples)
            y = (1.0 - t) * z + t * target_samples
            u = target_samples - z
            
            if treat_cond is not None:
                y_input = torch.cat((y, treat_cond), dim=-1)
            else:
                y_input = y
            
            b = self.model.flow(embedding, t.squeeze(-1), y_input)
            loss = ((b - u) ** 2).sum(dim=-1)
        else:
            raise ValueError(f"unknown base: {self.base}")
        
        loss = loss.mean()
        return loss
    
    def get_embeddings(self, source_samples):
        """
        Compute embeddings for source samples.
        
        Args:
            source_samples: Source samples (B, N, D) or (N, D)
            
        Returns:
            embedding: (B, N, hidden) or (N, hidden) expanded embedding
        """
        if source_samples.dim() == 3 and source_samples.shape[0] > 1:  # batched replicas
            embedding_batch = []
            for i in range(source_samples.shape[0]):
                embedding = self.model.embed_source(source_samples[i]).detach()
                embedding_batch.append(embedding.expand(source_samples.shape[1], -1))
            return torch.stack(embedding_batch)
        else:
            embedding = self.model.embed_source(source_samples).detach()
            return embedding
    
    def flow_step(self, batch, optimizer):
        """
        Optimization step for flow network only.
        """
        x0, x1, treat_cond = self._unpack_batch(batch)
        
        embedding = self.get_embeddings(x0)
        
        loss = self.compute_loss(
            embedding.reshape(-1, embedding.shape[-1]),
            x0.reshape(-1, x0.shape[-1]),
            x1.reshape(-1, x1.shape[-1]),
            treat_cond.reshape(-1, treat_cond.shape[-1]),
        )
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        return loss
    
    def gnn_step(self, batch, optimizer):
        """Optimization step for GNN encoder only."""
        x0, x1, treat_cond = self._unpack_batch(batch)
        
        embedding = self.model.embed_source(x0)
        embedding = embedding.unsqueeze(1).expand(-1, x0.shape[1], -1)
        
        loss = self.compute_loss(
            embedding.reshape(-1, embedding.shape[-1]),
            x0.reshape(-1, x0.shape[-1]),
            x1.reshape(-1, x1.shape[-1]),
            treat_cond.reshape(-1, treat_cond.shape[-1]),
        )
        
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        
        return loss
    
    def _unpack_batch(self, batch):
        """
        Unpack a single batch.
        
        Returns:
            x0: Source samples
            x1: Target samples
            treat_cond: Treatment condition
        """
        x0 = batch[BatchKeys.OBS_DATA].squeeze(1)
        x1 = batch[BatchKeys.INT_DATA].squeeze(1)
        treat_cond = batch[BatchKeys.TREATMENT].repeat(1, 1, x0.shape[1], 1).squeeze(1)

        return x0, x1, treat_cond
    
    def training_step(self, batch, batch_idx):
        flow_opt, gnn_opt = self.optimizers()

        if (batch_idx + 1) % 2 == 0:
            loss = self.gnn_step(batch, gnn_opt)
            self.log("train/gnn_loss", loss, prog_bar=True)
        else:
            loss = self.flow_step(batch, flow_opt)
            self.log("train/flow_loss", loss, prog_bar=True)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x0, x1, treat_cond = self._unpack_batch(batch)

        embedding = self.model.embed_source(x0)
        embedding = embedding.unsqueeze(1).expand(-1, x0.shape[1], -1)
        
        loss = self.compute_loss(
            embedding.reshape(-1, embedding.shape[-1]),
            x0.reshape(-1, x0.shape[-1]),
            x1.reshape(-1, x1.shape[-1]),
            treat_cond.reshape(-1, treat_cond.shape[-1]),
        )

        predictions = []
        for i in range(x0.shape[0]):
            transported = self.transport(
                source_samples=x0[i],
                context_samples=x0[i],
                treat_cond=treat_cond[i],
            )
            predictions.append(transported)
        
        predictions = torch.stack(predictions)

        key = jr.PRNGKey(0)
        dist_metrics = compute_distribution_metrics(
            x0.detach().cpu().numpy(), x1.detach().cpu().numpy(), predictions.detach().cpu().numpy(), key=key
        )

        metrics = {"val/loss": loss}        
        metrics.update({f"val/{k}": v for k, v in dist_metrics.items()})

        self.log_dict(metrics, batch_size=1, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
        flow_params = list(self.model.decoder.parameters())
        gnn_params = list(self.model.gcn_convs.parameters())
        
        flow_optimizer = torch.optim.Adam(flow_params, lr=self.flow_lr)
        gnn_optimizer = torch.optim.Adam(gnn_params, lr=self.gnn_lr)
        
        return [flow_optimizer, gnn_optimizer]
    
    @torch.no_grad()
    def transport(
        self,
        source_samples: torch.Tensor,
        context_samples: torch.Tensor,
        treat_cond: torch.Tensor = None,
        n_steps: int = None,
        solver: str = "dopri5",
    ) -> torch.Tensor:
        """
        Transport source samples to target distribution using learned flow.
        
        Args:
            source_samples: Source samples to transport (N, D)
            context_samples: Context samples for embedding (M, D)
            treat_cond: Optional treatment condition (N, num_treat_conditions)
            n_steps: Number of ODE solver steps (defaults to integrate_time_steps)
            solver: ODE solver type
            
        Returns:
            Transported samples (N, D)
        """
        self.eval()
        
        if n_steps is None:
            n_steps = self.integrate_time_steps
        
        # Update embedding for inference
        self.model.update_embedding_for_inference(context_samples)
        
        # Handle base distribution
        if self.base == "gaussian":
            x0 = torch.randn_like(source_samples)
        else:
            x0 = source_samples
        
        # Create wrapped model for ODE solver
        if treat_cond is not None and self.num_treat_conditions is not None:
            wrapped_model = torch_wrapper_gnn_flow_cond(self.model)
            # Concatenate treatment to x0 for ODE
            x0_with_treat = torch.cat([x0, treat_cond], dim=-1)
            
            node = NeuralODE(
                wrapped_model,
                solver=solver,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )
            time_span = torch.linspace(0, 1, n_steps, device=x0.device)
            traj = node.trajectory(x0_with_treat, t_span=time_span)
            # Extract only the data dimensions (remove treatment)
            transported = traj[-1, :, :source_samples.shape[-1]]
        else:
            wrapped_model = torch_wrapper(self.model)
            
            node = NeuralODE(
                wrapped_model,
                solver=solver,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )
            time_span = torch.linspace(0, 1, n_steps, device=x0.device)
            traj = node.trajectory(x0, t_span=time_span)
            transported = traj[-1]
        
        return transported
    
    @torch.no_grad()
    def get_trajectory(
        self,
        source_samples: torch.Tensor,
        context_samples: torch.Tensor,
        treat_cond: torch.Tensor = None,
        n_steps: int = None,
        solver: str = "dopri5",
    ) -> torch.Tensor:
        """
        Get full trajectory of transport.
        
        Returns:
            Trajectory tensor of shape (n_steps, N, D)
        """
        self.eval()
        
        if n_steps is None:
            n_steps = self.integrate_time_steps
        
        # Update embedding for inference
        self.model.update_embedding_for_inference(context_samples)
        
        # Handle base distribution
        if self.base == "gaussian":
            x0 = torch.randn_like(source_samples)
        else:
            x0 = source_samples
        
        if treat_cond is not None and self.num_treat_conditions is not None:
            wrapped_model = torch_wrapper_gnn_flow_cond(self.model)
            x0_with_treat = torch.cat([x0, treat_cond], dim=-1)
            
            node = NeuralODE(
                wrapped_model,
                solver=solver,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )
            time_span = torch.linspace(0, 1, n_steps, device=x0.device)
            traj = node.trajectory(x0_with_treat, t_span=time_span)
            # Extract only the data dimensions
            trajectory = traj[:, :, :source_samples.shape[-1]]
        else:
            wrapped_model = torch_wrapper(self.model)
            
            node = NeuralODE(
                wrapped_model,
                solver=solver,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )
            time_span = torch.linspace(0, 1, n_steps, device=x0.device)
            trajectory = node.trajectory(x0, t_span=time_span)
        
        return trajectory

    def test_step(self, batch, batch_idx) -> dict[str, np.ndarray]:
        """Transport samples and return predictions with metadata."""
        x0, x1, treat_cond = self._unpack_batch(batch)
        
        predictions = []
        for i in range(x0.shape[0]):
            transported = self.transport(
                source_samples=x0[i],
                context_samples=x0[i],
                treat_cond=treat_cond[i],
            )
            predictions.append(transported)
        
        predictions = torch.stack(predictions)

        return {
            BatchKeys.PRED_INT_DATA: predictions.detach().cpu().numpy(),
            BatchKeys.INT_DATA: x1.detach().cpu().numpy(),
            BatchKeys.OBS_DATA: x0.detach().cpu().numpy(),
            BatchKeys.TREATMENT: batch[BatchKeys.TREATMENT].detach().cpu().numpy().squeeze(1),
            BatchKeys.CONTEXT_ID: np.asarray(batch[BatchKeys.CONTEXT_ID]),
            BatchKeys.TREATMENT_ID: np.asarray(batch[BatchKeys.TREATMENT_ID]),
        }