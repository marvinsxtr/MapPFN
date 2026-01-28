from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray

from map_pfn.data.utils import unpack_batch


def fm_loss(
    model: eqx.Module,
    batch: dict[str, Float[Array, " batch_size num_contexts num_rows num_features"]],
    cfg_dropout: float = 0.2,
    *,
    key: PRNGKeyArray,
) -> tuple[Float[Array, " 1"], tuple]:
    """Compute flow matching loss.

    Args:
        model: Velocity field neural network model.
        batch: Batch of data points.
        cfg_dropout: Dropout for classifier-free guidance.
        key: Random key for sampling, split across batch dimension.

    Returns:
        Mean loss across the batch and auxiliary outputs as a tuple.
    """
    time_key, noise_key, cfg_key = jr.split(key, num=3)
    obs_data, int_data, obs_data_cond, int_data_cond, treatment = unpack_batch(batch)

    x_1 = int_data[:, -1]
    x_0 = jr.normal(key=noise_key, shape=x_1.shape)
    t = jax.nn.sigmoid(jr.normal(key=time_key, shape=(x_1.shape[0], 1, 1)))

    x_t = (1 - t) * x_0 + t * x_1
    v = x_1 - x_0

    cfg_keys = jr.split(cfg_key, num=x_1.shape[0])
    cfg_model = partial(model, cfg_dropout=cfg_dropout)

    v_pred = jax.vmap(cfg_model)(x_t, obs_data_cond, int_data_cond, treatment, t.squeeze(), key=cfg_keys)
    v_loss = jnp.mean((v_pred - v) ** 2)

    return v_loss, (obs_data, int_data, obs_data_cond, int_data_cond, treatment)
