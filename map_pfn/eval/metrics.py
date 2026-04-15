from functools import partial

import anndata as ad
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scanpy as sc
from jaxtyping import Array, Float, PRNGKeyArray
from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics import average_precision_score, precision_recall_curve

from map_pfn.utils.helpers import suppress_output


def compute_distribution_metrics(
    obs_data: Float[Array, "batch_size num_samples dim"],
    int_data: Float[Array, "batch_size num_samples dim"],
    gen_int_data: Float[Array, "batch_size num_samples dim"],
    key: PRNGKeyArray, # noqa: ARG001
    treatment_indices: Float[Array, "batch_size dim"] | None = None,
) -> dict[str, float]:
    """Computes distribution metrics as a dictionary.

    Args:
        obs_data: Array of shape [batch size, num_samples, dim]
        int_data: Array of shape [batch size, num_samples, dim]
        gen_int_data: Array of shape [batch size, num_samples, dim]
        key: Random key
        treatment_indices: Optional one-hot treatment array of shape [batch_size, dim].
            When provided, a "target_only_auprc" metric is also computed.

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
    metrics["deg_auprc"] = deg_auprc(obs_data, int_data, gen_int_data)["ap"]

    if treatment_indices is not None:
        result = deg_auprc(obs_data, int_data, gen_int_data, treatment_indices=treatment_indices)
        metrics["target_only_auprc"] = result["target_only_ap"]

    # Magnitude ratio
    metrics["wasserstein_mag_ratio"] = jax.vmap(wasserstein_mag_ratio)(obs_data, int_data, gen_int_data).mean().item()

    # Perturbation discrimination score
    if batch_size > 1:
        metrics["pds"] = perturbation_discrimination_score(int_data, gen_int_data).item()

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
def perturbation_discrimination_score(
    int_data: Float[Array, "batch_size num_samples dim"], gen_int_data: Float[Array, "batch_size num_samples dim"]
) -> Float[Array, ""]:
    """Computes the perturbation discrimination score (PDS).

    Args:
        int_data: True interventional data (batch_size is number of perturbations).
        gen_int_data: Predicted interventional data.

    Returns:
        Perturbation discrimination score.
    """
    mu_true = jnp.mean(int_data, axis=1)
    mu_pred = jnp.mean(gen_int_data, axis=1)

    p = mu_true.shape[0]

    dist_matrix = pairwise_squared_distances(mu_true, mu_pred)

    diagonals = jnp.diagonal(dist_matrix)
    off_diagonal_mask = ~jnp.eye(p, dtype=bool)

    col_comparisons = dist_matrix <= diagonals[None, :]
    transposed_rank_counts = jnp.sum(col_comparisons * off_diagonal_mask, axis=0)
    perturbation_discrimination_score = jnp.mean(transposed_rank_counts / (p - 1))

    return perturbation_discrimination_score


def deg_auprc(
    obs_data: Float[Array, "batch_size num_samples dim"],
    int_data: Float[Array, "batch_size num_samples dim"],
    gen_int_data: Float[Array, "batch_size num_samples dim"],
    pval_threshold: float = 0.01,
    lfc_threshold: float = 0.2,
    treatment_indices: Float[Array, "batch_size dim"] | None = None,
) -> dict:
    """Compute AUPRC for differentially expressed gene detection across all treatments.

    Args:
        obs_data: Control/observational samples of shape [batch_size, num_samples, dim].
        int_data: True interventional samples of shape [batch_size, num_samples, dim].
        gen_int_data: Generated interventional samples of shape [batch_size, num_samples, dim].
        pval_threshold: P-value threshold for calling a gene differentially expressed.
        lfc_threshold: Log fold change threshold for calling a gene differentially expressed.
        treatment_indices: Optional one-hot treatment array of shape [batch_size, dim].
            When provided, a "target-only" baseline AUPRC is also computed (predicts
            score=1 for the knocked-out gene, 0 elsewhere).

    Returns:
        Dict with keys: ap, baseline, n_degs, target_only_ap, curve.
    """
    batch_size = int_data.shape[0]

    obs_np = np.asarray(obs_data)
    int_np = np.asarray(int_data)
    gen_np = np.asarray(gen_int_data)

    treatment_np = np.asarray(treatment_indices) if treatment_indices is not None else None

    all_y_true = []
    all_y_score = []
    all_target_score = []

    for i in range(batch_size):
        with suppress_output():
            adata_ctrl = ad.AnnData(X=obs_np[i])
            adata_ctrl.obs["condition"] = "control"

            adata_true = ad.AnnData(X=int_np[i])
            adata_true.obs["condition"] = "treatment"

            adata_pred = ad.AnnData(X=gen_np[i])
            adata_pred.obs["condition"] = "treatment"

            adata_true_combined = ad.concat([adata_true, adata_ctrl])
            adata_pred_combined = ad.concat([adata_pred, adata_ctrl])

            sc.tl.rank_genes_groups(
                adata_true_combined,
                groupby="condition",
                reference="control",
                method="wilcoxon",
            )
            sc.tl.rank_genes_groups(
                adata_pred_combined,
                groupby="condition",
                reference="control",
                method="wilcoxon",
            )

        df_true = sc.get.rank_genes_groups_df(adata_true_combined, group="treatment").set_index("names")
        df_pred = sc.get.rank_genes_groups_df(adata_pred_combined, group="treatment").set_index("names")

        true_pvals = df_true["pvals_adj"].to_numpy()
        true_lfc = df_true["logfoldchanges"].to_numpy()
        y_true = ((true_pvals < pval_threshold) & (np.abs(true_lfc) > lfc_threshold)).astype(int)

        pred_pvals = df_pred.loc[df_true.index, "pvals_adj"].to_numpy()
        pred_lfc = df_pred.loc[df_true.index, "logfoldchanges"].to_numpy()

        pred_nan_mask = np.isnan(pred_pvals) | np.isnan(pred_lfc)
        pred_pvals = np.where(pred_nan_mask, 1.0, pred_pvals)
        pred_lfc = np.where(pred_nan_mask, 0.0, pred_lfc)

        y_score = np.abs(pred_lfc) * (pred_pvals < pval_threshold).astype(float)

        all_y_true.append(y_true)
        all_y_score.append(y_score)

        if treatment_np is not None:
            gene_order = df_true.index.astype(int).to_numpy()
            all_target_score.append(treatment_np[i][gene_order])

    if len(all_y_true) == 0 or np.concatenate(all_y_true).sum() == 0:
        return {"ap": float("nan"), "baseline": float("nan"), "n_degs": 0, "target_only_ap": float("nan")}

    all_y_true = np.concatenate(all_y_true)
    all_y_score = np.concatenate(all_y_score)

    ap = float(average_precision_score(all_y_true, all_y_score))
    baseline = float(all_y_true.sum() / len(all_y_true))

    target_only_ap = float("nan")
    if len(all_target_score) > 0:
        all_target_score = np.concatenate(all_target_score)
        target_only_ap = float(average_precision_score(all_y_true, all_target_score))

    precision, recall, _ = precision_recall_curve(all_y_true, all_y_score)

    return {
        "ap": ap,
        "baseline": baseline,
        "n_degs": int(all_y_true.sum()),
        "target_only_ap": target_only_ap,
        "curve": (precision, recall),
    }
