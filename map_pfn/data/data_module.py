from collections.abc import Callable

import anndata as ad
from hydra_zen.typing import Partial
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from map_pfn.data.utils import collate_fn as default_collate_fn
from map_pfn.data.utils import split_dataset


class DataModule(LightningDataModule):
    """Lightning DataModule for AnnData perturbation datasets."""

    def __init__(
        self,
        dataset_path: str,
        dataset: Partial[Dataset],
        prior_dataset_path: str | None = None,
        ood: bool = False,
        num_shots: int = 4,
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        drop_last: bool = True,
        collate_fn: Callable | None = None,
    ) -> None:
        """Initialize the DataModule.

        Args:
            dataset_path: Path to the h5ad file containing the data.
            dataset: Partial dataset to instantiate (e.g., Partial[PerturbationDataset]).
            prior_dataset_path: Optional path to a prior dataset used for training and additional evaluation.
            ood: Whether to use out-of-distribution splits.
            num_shots: Number of context demonstrations per query.
            batch_size: Batch size for dataloaders.
            num_workers: Number of workers for dataloaders.
            persistent_workers: Whether to keep workers alive between epochs.
            drop_last: Whether to drop the last incomplete batch.
            collate_fn: Collate function to use. Defaults to numpy collate_fn.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "collate_fn"])

        self.dataset_path = dataset_path
        self.prior_dataset_path = prior_dataset_path
        self.dataset = dataset
        self.ood = ood
        self.num_shots = num_shots
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self._collate_fn = collate_fn if collate_fn is not None else default_collate_fn
        self.add_prior = prior_dataset_path is not None

        self.adata = ad.read_h5ad(self.dataset_path)
        self.prior_adata = ad.read_h5ad(self.prior_dataset_path) if self.add_prior else None

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

        self.prior_train_dataset: Dataset | None = None
        self.prior_val_dataset: Dataset | None = None
        self.prior_test_dataset: Dataset | None = None

    def setup(self, stage: str) -> None:
        """Set up datasets for the given stage.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
        """
        train_adata, val_adata, test_adata = split_dataset(self.adata, ood=self.ood)
        num_shots = 0 if self.ood else self.num_shots

        if self.add_prior:
            prior_train_adata, prior_val_adata, prior_test_adata = split_dataset(self.prior_adata, ood=self.ood)

        if stage == "fit":
            if self.add_prior:
                self.prior_train_dataset = self.dataset(
                    query_adata=prior_train_adata, context_adata=prior_train_adata, num_shots=num_shots
                )
            else:
                self.train_dataset = self.dataset(
                    query_adata=train_adata, context_adata=train_adata, num_shots=num_shots
                )

        if stage in ["fit", "validate"]:
            self.val_dataset = self.dataset(query_adata=val_adata, context_adata=train_adata, num_shots=num_shots)

            if self.add_prior:
                self.prior_val_dataset = self.dataset(
                    query_adata=prior_val_adata, context_adata=prior_train_adata, num_shots=num_shots
                )

        if stage in ["test", "predict"]:
            self.test_dataset = self.dataset(query_adata=test_adata, context_adata=train_adata, num_shots=num_shots)

            if self.add_prior:
                self.prior_test_dataset = self.dataset(
                    query_adata=prior_test_adata, context_adata=prior_train_adata, num_shots=num_shots
                )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        return DataLoader(
            self.prior_train_dataset if self.add_prior else self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """Create the validation dataloader(s)."""
        loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "collate_fn": self._collate_fn,
            "drop_last": False,
        }

        if self.add_prior:
            return [
                DataLoader(self.val_dataset, **loader_config),
                DataLoader(self.prior_val_dataset, **loader_config),
            ]

        return DataLoader(self.val_dataset, **loader_config)

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """Create the test dataloader(s)."""
        loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "persistent_workers": self.persistent_workers,
            "collate_fn": self._collate_fn,
            "drop_last": False,
        }

        if self.add_prior:
            return [
                DataLoader(self.test_dataset, **loader_config),
                DataLoader(self.prior_test_dataset, **loader_config),
            ]

        return DataLoader(self.test_dataset, **loader_config)

    def predict_dataloader(self) -> DataLoader:
        """Create the predict dataloader (uses test dataset)."""
        return self.test_dataloader()
