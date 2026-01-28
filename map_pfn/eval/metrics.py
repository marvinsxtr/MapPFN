from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence


def compute_distribution_metrics(
    obs_data: Float[Array, "batch_size num_samples dim"],
    int_data: Float[Array, "batch_size num_samples dim"],
    gen_int_data: Float[Array, "batch_size num_samples dim"],
    key: PRNGKeyArray, # noqa: ARG001
) -> dict[str, float]:
    """Computes distribution metrics as a dictionary.

    Args:
        obs_data: Array of shape [batch size, num_samples, dim]
        int_data: Array of shape [batch size, num_samples, dim]
        gen_int_data: Array of shape [batch size, num_samples, dim]
        treatments: [batch size]
        key: Random key

    Returns:
    Dict containing the metrics as floats.
    """
    batch_size = int_data.shape[0]
    metrics = {}

    # Aggregated metrics
    metrics["mean_rmse"] = jax.vmap(mean_rmse)(int_data, gen_int_data).mean().item()
    metrics["var_rmse"] = jax.vmap(var_rmse)(int_data, gen_int_data).mean().item()

    # Correlation metrics
    metrics["mean_r2"] = jax.vmap(mean_r2)(int_data, gen_int_data).mean().item()
    metrics["var_r2"] = jax.vmap(var_r2)(int_data, gen_int_data).mean().item()

    # Distribution metrics
    metrics["wasserstein"] = jax.vmap(wasserstein_distance)(int_data, gen_int_data).mean().item()
    metrics["mmd"] = jax.vmap(multiscale_mmd)(int_data, gen_int_data).mean().item()

    # Magnitude ratio
    metrics["wasserstein_mag_ratio"] = jax.vmap(wasserstein_mag_ratio)(obs_data, int_data, gen_int_data).mean().item()

    if batch_size > 1:
        rank, transposed_rank = compute_rank_metrics(int_data, gen_int_data)
        metrics["rank"] = rank.item()
        metrics["transposed_rank"] = transposed_rank.item()

    return metrics


def wasserstein_mag_ratio(
    obs_data: Float[Array, "num_samples dim"],
    int_data: Float[Array, "num_samples dim"],
    gen_int_data: Float[Array, "num_samples dim"],
) -> float:
    """Computes scale-invariant intervention prediction quality using Wasserstein magnitude ratio.

    Args:
        obs_data: Observational samples from original distribution.
        int_data: True interventional samples from target distribution.
        gen_int_data: Generated/predicted interventional samples.

    Returns:
    See `mmd_mag_ratio`.
    """
    return wasserstein_distance(obs_data, gen_int_data) / (wasserstein_distance(obs_data, int_data) + 1e-8)


@jax.jit
def mean_rmse(x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"]) -> Float[Array, ""]:
    """Compute the root mean squared error (RMSE) of the means of x and y.

    Args:
        x: An array of shape [num_samples, dim].
        y: An array of shape [num_samples, dim].

    Returns:
    The root mean squared error (RMSE) between the means.
    """
    return jnp.sqrt(jnp.mean((x.mean(axis=0) - y.mean(axis=0)) ** 2))


@jax.jit
def var_rmse(x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"]) -> Float[Array, ""]:
    """Compute the root mean squared error (RMSE) of the variances of x and y.

    Args:
        x: An array of shape [num_samples, dim].
        y: An array of shape [num_samples, dim].

    Returns:
    The root mean squared error (RMSE) between the variances.
    """
    return jnp.sqrt(jnp.mean((x.var(axis=0) - y.var(axis=0)) ** 2))


@jax.jit
def wasserstein_distance(
    x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"], epsilon: float = 0.1
) -> Float[Array, ""]:
    """Compute the Sinkhorn divergence between x and y.

    Args:
        x: An array of shape [num_samples, dim].
        y: An array of shape [num_samples, dim].
        epsilon: Regularization parameter.

    Returns:
    The sinkhorn divergence.
    """
    return sinkhorn_divergence(
        pointcloud.PointCloud,
        x=x,
        y=y,
        cost_fn=costs.SqEuclidean(),
        epsilon=epsilon,
        scale_cost=1.0,
    )[0]


@jax.jit
def pairwise_squared_distances(x: Float[Array, "n dim"], y: Float[Array, "m dim"]) -> Float[Array, "n m"]:
    """Compute pairwise squared Euclidean distances efficiently.

    Args:
        x: Array of shape [n, dim].
        y: Array of shape [m, dim].

    Returns:
        Matrix of squared distances of shape [n, m].
    """
    xx = jnp.sum(x**2, axis=1)
    yy = jnp.sum(y**2, axis=1)
    xy = x @ y.T
    return xx[:, None] + yy - 2 * xy


@jax.jit
def rbf_kernel(
    x: Float[Array, "num_samples_x dim"], y: Float[Array, "num_samples_y dim"], gamma: float
) -> Float[Array, "num_samples_x num_samples_y"]:
    """Compute RBF kernel matrix.

    Args:
        x: Points of shape [n_samples_x, dim].
        y: Points of shape [n_samples_y, dim].
        gamma: Kernel bandwidth parameter.

    Returns:
    Kernel matrix of shape [n_samples_x, n_samples_y].
    """
    sq_distances = pairwise_squared_distances(x, y)
    return jnp.exp(-gamma * sq_distances)


def mmd(x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"], gamma: float = 0.1) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y.

    Args:
        x: An array of shape [num_samples, num_features].
        y: An array of shape [num_samples, num_features].
        gamma: Parameter for the rbf kernel.

    Returns:
    A scalar denoting the squared maximum mean discrepancy loss.
    """
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def multiscale_mmd(x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"]) -> float:
    """Computes intervention MMD for multiple gammas.

    Args:
        x: An array of shape [num_samples, dim].
        y: An array of shape [num_samples, dim].
        key: Random key.

    Returns:
        Multi-scale MMD
    """
    gammas = jnp.logspace(1, -3, 5)

    mmd_partial = partial(mmd, x=x, y=y)
    mmds = jax.vmap(mmd_partial)(gamma=gammas)

    return jnp.mean(mmds)


def mean_r2(x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"]) -> float:
    """Correlation coefficient based on feature means.

    Args:
        x: Samples with shape (num_samples, dim).
        y: Samples with shape (num_samples, dim).

    Returns:
        Correlation coefficient in [-1, 1], where 1.0 indicates perfect
        positive correlation of means. Returns NaN if correlation cannot
        be computed.
    """
    mean_x = jnp.mean(x, axis=0)
    mean_y = jnp.mean(y, axis=0)

    return jnp.corrcoef(jnp.stack([mean_x, mean_y]))[0, 1]


def var_r2(x: Float[Array, "num_samples dim"], y: Float[Array, "num_samples dim"]) -> float:
    """Correlation coefficient based on feature variances.

    Args:
        x: Samples with shape (num_samples, dim).
        y: Samples with shape (num_samples, dim).

    Returns:
        Correlation coefficient in [-1, 1], where 1.0 indicates perfect
        positive correlation of variances. Returns NaN if correlation cannot
        be computed.
    """
    var_x = jnp.var(x, axis=0)
    var_y = jnp.var(y, axis=0)

    return jnp.corrcoef(jnp.stack([var_x, var_y]))[0, 1]


@eqx.filter_jit
def compute_rank_metrics(
    int_data: Float[Array, "batch_size num_samples dim"],
    gen_int_data: Float[Array, "batch_size num_samples dim"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Computes the rank and transposed rank metrics.

    Args:
        int_data: True interventional data (batch_size is number of perturbations).
        gen_int_data: Predicted interventional data.

    Returns:
        Tuple of (rank_metric, transposed_rank_metric).
    """
    mu_true = jnp.mean(int_data, axis=1)
    mu_pred = jnp.mean(gen_int_data, axis=1)

    p = mu_true.shape[0]

    dist_matrix = pairwise_squared_distances(mu_true, mu_pred)

    diagonals = jnp.diagonal(dist_matrix)
    off_diagonal_mask = ~jnp.eye(p, dtype=bool)

    row_comparisons = dist_matrix <= diagonals[:, None]
    rank_counts = jnp.sum(row_comparisons * off_diagonal_mask, axis=1)
    rank_metric = jnp.mean(rank_counts / (p - 1))

    col_comparisons = dist_matrix <= diagonals[None, :]
    transposed_rank_counts = jnp.sum(col_comparisons * off_diagonal_mask, axis=0)
    transposed_rank_metric = jnp.mean(transposed_rank_counts / (p - 1))

    return rank_metric, transposed_rank_metric
