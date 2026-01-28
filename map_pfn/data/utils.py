from dataclasses import dataclass, fields
from typing import Final

import anndata as ad
import numpy as np
import torch
from jaxtyping import Array, Float
from numpy.typing import NDArray
from pandas import CategoricalDtype
from sklearn.model_selection import train_test_split


@dataclass
class Values:
    """Standardized values."""

    CONTROL: Final[str] = "control"


@dataclass
class BatchKeys:
    """Standardized keys for batched data dictionaries."""

    TREATMENT: Final[str] = "treatment"
    DATA: Final[str] = "data"
    OBS_DATA: Final[str] = "obs_data"
    INT_DATA: Final[str] = "int_data"
    PRED_INT_DATA: Final[str] = "pred_int_data"
    CONTEXT_ID: Final[str] = "context_id"
    TREATMENT_ID: Final[str] = "treatment_id"


@dataclass
class ColumnNames:
    """Standardized column names for AnnData objects."""

    CONTEXT: Final[str] = "context"
    TREATMENT: Final[str] = "treatment"
    SPLIT: Final[str] = "split"
    TECH_DUP_SPLIT: Final[str] = "tech_dup_split"
    TECH_DUP_MEAN: Final[str] = "tech_dup_mean"
    DATA_MEAN: Final[str] = "data_mean"
    CONTROL_MEAN: Final[str] = "control_mean"


@dataclass
class SplitNames:
    """Standardized split names for training, validation, and testing."""

    TRAIN: Final[str] = "train"
    VAL: Final[str] = "val"
    TEST: Final[str] = "test"

    OTHER: Final[str] = "other"
    CONTROL: Final[str] = "control"
    TEST_OTHER: Final[str] = "test_other"


def collate_fn(
    batch: list[dict[str, NDArray[np.float64]]],
) -> dict[str, NDArray[np.float64]]:
    """Stack a list of sample dictionaries into a single batch.

    Args:
        batch: List of sample dictionaries from the dataset

    Returns:
        Dictionary with batched arrays
    """
    return {key: np.stack([sample[key] for sample in batch]) for key in batch[0]}


def torch_collate_fn(
    batch: list[dict[str, NDArray[np.float64]]],
) -> dict[str, torch.Tensor | NDArray]:
    """Stack a list of sample dictionaries into a single batch of torch tensors.

    String arrays are kept as numpy arrays and not converted to tensors.

    Args:
        batch: List of sample dictionaries from the dataset

    Returns:
        Dictionary with batched torch tensors (or numpy arrays for string types)
    """
    result = {}
    for key in batch[0]:
        stacked = np.stack([sample[key] for sample in batch])
        if np.issubdtype(stacked.dtype, np.str_) or np.issubdtype(stacked.dtype, np.object_):
            result[key] = stacked
        else:
            result[key] = torch.from_numpy(stacked).float()
    return result


def filter_batch(batch: dict[str, NDArray[np.float64]], keys: list) -> dict[str, NDArray[np.float64]]:
    """Filter a batch dictionary to only include specific keys.

    Args:
        batch: Dictionary with batched arrays
        keys: Keys to filter for

    Returns:
        Filtered batch dictionary
    """
    return {key: value for key, value in batch.items() if key in keys}


def unpack_batch(
    batch: dict[str, Float[Array, " batch_size num_contexts num_rows num_features"]],
) -> tuple[
    Float[Array, " batch_size num_contexts num_rows num_features"],
    Float[Array, " batch_size num_contexts num_rows num_features"] | None,
    Float[Array, " batch_size 1 num_rows num_features"],
    Float[Array, " batch_size num_contexts-1 num_rows num_features"] | None,
    Float[Array, " batch_size num_contexts num_treatments num_features"],
]:
    """Unpack batch into observation data, intervention data, and treatment.

    Args:
        batch: Batch of data points.

    Returns:
        obs_data: Observational data tensor.
        int_data: Interventional data tensor.
        obs_data_cond: Observation condition tensor.
        int_data_cond: Intervention condition tensor.
        treatment: Treatment tensor.
    """
    obs_data = batch[BatchKeys.OBS_DATA]
    int_data = batch[BatchKeys.INT_DATA]
    treatment = batch[BatchKeys.TREATMENT]

    obs_data_cond = obs_data[:, -1:]
    int_data_cond = int_data[:, :-1] if int_data.shape[1] > 1 else None

    return obs_data, int_data, obs_data_cond, int_data_cond, treatment


def assign_split(
    adata: ad.AnnData,
    val_share: float = 0.1,
    test_share: float = 0.5,
    holdout_context: str | None = None,
    seed: int = 42,
) -> ad.AnnData:
    """Split AnnData into train/val/test with consistent test split across o.o.d./i.d. modes.

    Splitting is done at the condition (context + treatment) level.
    The test split within the holdout context is identical in both modes.

    o.o.d. mode: holdout context -> test/ignore, other contexts -> train/val
    i.d. mode: holdout context -> test/train, other contexts -> train/val

    Args:
        adata: The AnnData object containing the data.
        val_share: Fraction of conditions for validation (in non-holdout contexts).
        test_share: Fraction of conditions for test (in holdout context).
        holdout_context: The context to hold out. If None, chooses the context with the most treatments.
        seed: Random seed for reproducibility.

    Returns:
        AnnData with split column added.
    """
    if holdout_context is None:
        holdout_context = (
            adata.obs.groupby(ColumnNames.CONTEXT, observed=True)[ColumnNames.TREATMENT].nunique().idxmax()
        )

    adata.obs[ColumnNames.SPLIT] = None

    control_mask = adata.obs[ColumnNames.TREATMENT] == Values.CONTROL
    holdout_mask = adata.obs[ColumnNames.CONTEXT] == holdout_context

    cond = adata.obs[[ColumnNames.CONTEXT, ColumnNames.TREATMENT]].apply(tuple, axis=1)

    unique_cond = cond[~control_mask & ~holdout_mask].drop_duplicates()
    train_cond, val_cond = train_test_split(unique_cond, test_size=val_share, random_state=seed)

    adata.obs.loc[cond.isin(train_cond), ColumnNames.SPLIT] = SplitNames.TRAIN
    adata.obs.loc[cond.isin(val_cond), ColumnNames.SPLIT] = SplitNames.VAL

    holdout_cond = cond[~control_mask & holdout_mask].drop_duplicates()
    other_cond, test_cond = train_test_split(holdout_cond, test_size=test_share, random_state=seed)

    adata.obs.loc[cond.isin(test_cond), ColumnNames.SPLIT] = SplitNames.TEST
    adata.obs.loc[cond.isin(other_cond), ColumnNames.SPLIT] = SplitNames.TEST_OTHER

    adata.obs.loc[control_mask, ColumnNames.SPLIT] = SplitNames.CONTROL

    adata.obs[ColumnNames.SPLIT] = adata.obs[ColumnNames.SPLIT].astype(
        CategoricalDtype(categories=[str(f.default) for f in fields(SplitNames)])
    )

    return adata


def split_dataset(adata: ad.AnnData, ood: bool) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """Split AnnData into training, validation, and test sets based on split column.

    Args:
        adata: The AnnData object containing the data with split column.
        ood: If True, convert to o.o.d. split. If False, convert to i.d. split.

    Returns:
        A tuple of (train_adata, val_adata, test_adata).
    """
    if ColumnNames.SPLIT not in adata.obs.columns:
        raise ValueError(f"Column '{ColumnNames.SPLIT}' not found in adata.obs.")

    adata.obs.loc[adata.obs[ColumnNames.SPLIT] == SplitNames.TEST_OTHER, ColumnNames.SPLIT] = (
        SplitNames.OTHER if ood else SplitNames.TRAIN
    )

    train_adata = adata[adata.obs[ColumnNames.SPLIT].isin([SplitNames.TRAIN, SplitNames.CONTROL])].copy()
    val_adata = adata[adata.obs[ColumnNames.SPLIT].isin([SplitNames.VAL, SplitNames.CONTROL])].copy()
    test_adata = adata[adata.obs[ColumnNames.SPLIT].isin([SplitNames.TEST, SplitNames.CONTROL])].copy()

    return train_adata, val_adata, test_adata
