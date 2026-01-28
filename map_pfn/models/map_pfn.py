from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from map_pfn.models.mmdit import MMDiT


class MapPFN(eqx.Module):
    """In-context perturbation model with MMDiT decoder."""

    decoder: MMDiT

    null_cond: Float[Array, " cond_dim"]

    cond_in: eqx.nn.Linear
    treatment_in: eqx.nn.Linear

    obs_data_emb: Float[Array, " cond_dim"]
    int_data_emb: Float[Array, " cond_dim"]
    treatment_emb: Float[Array, " cond_dim"]

    context_emb: Float[Array, " max_num_contexts cond_dim"]
    query_emb: Float[Array, " cond_dim"]

    def __init__(
        self,
        decoder: Callable,
        cond_dim: int,
        in_dim: int,
        max_num_contexts: int = 10,
        *,
        key: PRNGKeyArray,
    ) -> None:
        keys = jr.split(key, 8)

        self.decoder = decoder

        self.null_cond = jr.normal(keys[0], (cond_dim,)) / jnp.sqrt(cond_dim)

        self.treatment_in = eqx.nn.Linear(in_dim, cond_dim, key=keys[1])
        self.cond_in = eqx.nn.Linear(in_dim, cond_dim, key=keys[2])

        self.obs_data_emb = jr.normal(keys[3], (cond_dim,)) / jnp.sqrt(cond_dim)
        self.int_data_emb = jr.normal(keys[4], (cond_dim,)) / jnp.sqrt(cond_dim)
        self.treatment_emb = jr.normal(keys[5], (cond_dim,)) / jnp.sqrt(cond_dim)

        self.context_emb = jr.normal(keys[6], (max_num_contexts, cond_dim)) / jnp.sqrt(cond_dim)
        self.query_emb = jr.normal(keys[7], (cond_dim,)) / jnp.sqrt(cond_dim)

    def cfg_dropout(
        self, cond: Float[Array, "... cond_dim"], cfg_dropout: float, drop_cond: bool, key: PRNGKeyArray | None = None
    ) -> Float[Array, "... cond_dim"]:
        """Generate dropout mask for classifier-free guidance.

        Args:
            cond: Condition tensor.
            cfg_dropout: Dropout rate.
            drop_cond: Whether to drop the condition.
            key: PRNG key.

        Returns:
            Dropout mask.
        """
        use_null = drop_cond or (key is not None and jr.uniform(key) < cfg_dropout)
        null_cond = jnp.broadcast_to(self.null_cond, cond.shape)
        cond = jnp.where(use_null, null_cond, cond)
        return cond

    def assemble_context(
        self,
        obs_data: Float[Array, "1 num_rows cond_dim"],
        int_data: Float[Array, "num_contexts-1 num_rows cond_dim"] | None,
        treatment: Float[Array, "num_contexts num_treatments cond_dim"],
    ) -> tuple[Float[Array, "cond_seq_len cond_dim"], Float[Array, "treatment_seq_len cond_dim"]]:
        """Assemble context by concatenating observed data, treatment, and interventional data.

        Args:
            obs_data: Observed data tensor
            int_data: Interventional data tensor
            treatment: Treatment tensor

        Returns:
            Concatenated context tensor
        """
        num_contexts = treatment.shape[0]

        if int_data is None:
            int_data = jnp.broadcast_to(self.null_cond, (num_contexts - 1, *treatment.shape[1:]))

        obs_data = obs_data + jnp.expand_dims(self.obs_data_emb, axis=(0, 1))
        int_data = int_data + jnp.expand_dims(self.int_data_emb, axis=(0, 1))
        treatment = treatment + jnp.expand_dims(self.treatment_emb, axis=(0, 1))

        obs_data = obs_data + jnp.expand_dims(self.context_emb[num_contexts - 1 : num_contexts], axis=1)
        int_data = int_data + jnp.expand_dims(self.context_emb[: num_contexts - 1], axis=1)
        treatment = treatment + jnp.expand_dims(self.context_emb[:num_contexts], axis=1)

        obs_data = obs_data.at[-1].add(self.query_emb)
        treatment = treatment.at[-1].add(self.query_emb)

        obs_data_seq = obs_data.reshape(-1, obs_data.shape[-1])
        int_data_seq = int_data.reshape(-1, int_data.shape[-1])
        treatment_seq = treatment.reshape(-1, treatment.shape[-1])

        cond_seq = jnp.concatenate([obs_data_seq, int_data_seq], axis=0)

        return cond_seq, treatment_seq

    @eqx.filter_jit
    def __call__(
        self,
        noise: Float[Array, "seq_len_noise dim"],
        obs_data: Float[Array, "1 seq_len dim"],
        int_data: Float[Array, "num_contexts-1 seq_len dim"] | None,
        treatment: Float[Array, "num_contexts num_treatments dim"],
        t: Float[Array, ""],
        cfg_dropout: float = 0.0,
        drop_cond: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len_noise dim"]:
        """Forward pass through MMDiT.

        Args:
            noise: Noise input tensor
            obs_data: Observed data tensor
            int_data: Interventional data tensor
            treatment: Treatment tensor
            t: Time values
            cfg_dropout: Dropout for classifier-free guidance.
            drop_cond: Whether to drop the condition.
            key: PRNGKey used for classifier-free guidance dropout.

        Returns:
            Processed noise tensor
        """
        obs_data = jax.vmap(jax.vmap(self.cond_in))(obs_data)
        int_data = jax.vmap(jax.vmap(self.cond_in))(int_data) if int_data is not None else None
        treatment = jax.vmap(jax.vmap(self.treatment_in))(treatment)

        cond, treatment = self.assemble_context(obs_data, int_data, treatment)

        cond = self.cfg_dropout(cond, cfg_dropout, drop_cond, key=key)
        treatment = self.cfg_dropout(treatment, cfg_dropout, drop_cond, key=key)

        return self.decoder(noise, cond, treatment, t)
