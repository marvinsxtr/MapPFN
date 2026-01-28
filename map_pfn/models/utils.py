import math
from functools import partial

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    """Multi-layer perceptron with GELU activation."""

    linear_1: eqx.nn.Linear
    linear_2: eqx.nn.Linear

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, *, key: PRNGKeyArray) -> None:
        """Initialize MLP.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            key: Random key for initialization
        """
        key1, key2 = jr.split(key)
        self.linear_1 = eqx.nn.Linear(input_dim, hidden_dim, key=key1)
        self.linear_2 = eqx.nn.Linear(hidden_dim, output_dim, key=key2)

    def __call__(self, x: Float[Array, "batch_size input_dim"]) -> Float[Array, "batch_size output_dim"]:
        """Forward pass through MLP."""
        x = self.linear_1(x)
        x = jax.nn.gelu(x)
        x = self.linear_2(x)
        return x


class SinusoidalPosEmb(eqx.Module):
    """Sinusoidal positional embedding."""

    emb: Float[Array, " dim"]

    def __init__(self, dim: int) -> None:
        """Initialize sinusoidal positional embedding.

        Args:
            dim: Dimension of the embedding.
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = jnp.exp(jnp.arange(half_dim) * -emb)

    def __call__(self, x: Float[Array, " batch_size"]) -> Float[Array, "batch_size dim"]:
        """Forward pass for positional embedding.

        Args:
            x: Input tensor of shape (batch_size,)

        Returns:
            Positional embeddings of shape (batch_size, dim)
        """
        emb = x * self.emb
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


def flash_attention(
    q: Float[Array, " seq_len num_heads head_dim"],
    k: Float[Array, " seq_len num_heads head_dim"],
    v: Float[Array, " seq_len num_heads head_dim"],
    pad_multiple: int = 64,
) -> Float[Array, " seq_len num_heads head_dim"]:
    """Attention with cuDNN backend, handling padding and dtype conversion.

    Args:
        q: Query tensor of shape (seq_len, num_heads, head_dim).
        k: Key tensor of shape (seq_len, num_heads, head_dim).
        v: Value tensor of shape (seq_len, num_heads, head_dim).
        pad_multiple: Pad sequence length to this multiple for cuDNN compatibility.

    Returns:
        Attention output of shape (seq_len, num_heads, head_dim).
    """
    orig_dtype = q.dtype
    seq_len = q.shape[0]
    pad_len = (pad_multiple - seq_len % pad_multiple) % pad_multiple

    if pad_len > 0:
        q = jnp.pad(q, ((0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, pad_len), (0, 0), (0, 0)))

    out = jax.nn.dot_product_attention(
        q[None].astype(jnp.bfloat16),
        k[None].astype(jnp.bfloat16),
        v[None].astype(jnp.bfloat16),
        implementation="cudnn",
    )[0].astype(orig_dtype)

    return out[:seq_len]


def zero_init_linear(linear: eqx.nn.Linear) -> eqx.nn.Linear:
    """Zero-initialize weight and bias of a linear layer.

    Args:
        linear: Linear layer to be zero-initialized.

    Returns:
        Linear layer with zero-initialized weight and bias.
    """
    return eqx.tree_at(
        lambda l: (l.weight, l.bias), linear, (jnp.zeros_like(linear.weight), jnp.zeros_like(linear.bias))
    )


@eqx.filter_jit
def solve_ode(
    model: eqx.Module,
    noise_shape: tuple,
    obs_data: Float[Array, " batch_size ..."],
    int_data: Float[Array, " batch_size ..."] | None,
    treatment: Float[Array, " batch_size ..."],
    guidance: float,
    step_size: float,
    time_grid: Float[Array, " time_steps"] | None = None,
    *,
    key: PRNGKeyArray,
) -> Float[Array, "time_steps batch_size ..."] | Float[Array, "batch_size ..."]:
    """Sample trajectories using diffrax ODE solver.

    Args:
        model: Velocity field model to integrate.
        noise_shape: Shape of the input noise.
        obs_data: Observed data to pass to the model.
        int_data: Interventional data to pass to the model.
        treatment: Treatment to apply.
        guidance: Classifier-free guidance weight.
        step_size: Initial step size.
        time_grid: Time points for integration. If none, the last time will be returned.
        key: Random key.

    Returns:
        Final points or full trajectory depending on return_intermediates.
    """
    x_init = jr.normal(key, noise_shape)
    in_axes = (0, 0, None if int_data is None else 0, 0)

    def vector_field(t: float, x: Float[Array, "batch_size ..."], args: tuple) -> Float[Array, "batch_size ..."]:  # noqa: ARG001
        t = jnp.array(t)

        model_t_cond = jax.vmap(partial(model, t=t, drop_cond=False), in_axes=in_axes)
        model_t_uncond = jax.vmap(partial(model, t=t, drop_cond=True), in_axes=in_axes)

        v_cond = model_t_cond(x, obs_data, int_data, treatment)
        v_uncond = model_t_uncond(x, obs_data, int_data, treatment)

        return guidance * v_cond + (1 - guidance) * v_uncond

    solution = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(vector_field),
        solver=diffrax.Dopri5(),
        t0=0.0,
        t1=1.0,
        dt0=step_size,
        y0=x_init,
        saveat=diffrax.SaveAt(ts=time_grid) if time_grid is not None else diffrax.SaveAt(t1=True),
    )

    return solution.ys.squeeze() if time_grid is None else solution.ys
