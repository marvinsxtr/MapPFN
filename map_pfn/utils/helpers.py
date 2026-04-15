import contextlib
import logging
import os
import platform
import random
import subprocess
import sys
import warnings
from collections.abc import Generator, Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from ml_project_template.utils import get_output_dir
from ml_project_template.wandb import WandBConfig
from omegaconf import OmegaConf

import wandb


@contextlib.contextmanager
def download_artifact(identifier: str) -> Iterator[Path]:
    """Temporarily download a WandB artifact and return the path.

    Args:
        identifier: Artifact identifier of the form `entity/project/model-<run id>:v<version>`.

    Yields:
    The path where the artifact was downloaded to.
    """
    api = wandb.Api()
    artifact = api.artifact(identifier, type="model")
    download_path = Path(artifact.file(root=get_output_dir()))

    try:
        yield download_path
    finally:
        if download_path.exists():
            download_path.unlink()


@contextlib.contextmanager
def download_file(run_path: str, filename: str) -> Iterator[Path]:
    """Temporarily download a file from a WandB run.

    Args:
        run_path: Run identifier of the form entity/project/run_id.
        filename: Name of the file to download.

    Yields:
    The path where the file was downloaded to.
    """
    api = wandb.Api()
    run = api.run(run_path)

    file_handle = run.file(filename).download(root=Path(get_output_dir()), replace=True)
    download_path = Path(file_handle.name)

    try:
        yield download_path
    finally:
        if download_path.exists():
            download_path.unlink()


def register_resolvers() -> None:
    """Registers globals and eval resolvers."""

    def _globals_resolver(ref: str) -> str:
        """Resolves references to global parameters.

        Args:
            ref: Reference to resolve. Must be a dot-separated string.
        """
        return f"${{cfg.globals.{ref}}}"

    OmegaConf.register_new_resolver("globals", _globals_resolver, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def cpu_count() -> int:
    """Returns the number of CPU cores.

    Returns:
    Number of CPU cores.
    """
    return len(os.sched_getaffinity(0))


def seed_everything(seed: int) -> None:
    """Seed all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def configure_jax() -> None:
    """Configure JAX settings for compatibility with PyTorch."""
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before setting environment variables.")

    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

    if platform.machine() == "aarch64":
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"

    if any("job" in argument for argument in sys.argv):
        os.environ["JAX_PLATFORMS"] = "cpu"

    if any(torch_cfg in argument for torch_cfg in ["condot", "metafm"] for argument in sys.argv):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def debug_overrides() -> dict:
    """Trainer overrides for fast debug runs (without disabling loggers)."""
    return {
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "limit_test_batches": 1,
        "limit_predict_batches": 1,
        "max_epochs": 1,
    }


def git_commit_hash() -> str:
    """Get the current git commit short hash."""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()


def get_wandb_project() -> str:
    """Read W&B entity/project from environment variables.

    Returns:
        W&B project string in the form ``"entity/project"``.

    Raises:
        RuntimeError: If environment variables are not set.
    """
    env_config = WandBConfig.from_env()
    if env_config is None:
        raise RuntimeError("Set WANDB_ENTITY and WANDB_PROJECT in .env to download W&B artifacts.")
    return f"{env_config.WANDB_ENTITY}/{env_config.WANDB_PROJECT}"


@contextlib.contextmanager
def resolve_checkpoint(checkpoint: str | Path) -> Iterator[Path]:
    """Resolve a checkpoint to a local path.

    Args:
        checkpoint: Local file path or W&B run ID.

    Yields:
        Path to the checkpoint file.
    """
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        yield checkpoint_path
    else:
        wandb_project = get_wandb_project()
        with download_artifact(f"{wandb_project}/model-{checkpoint}:latest") as artifact:
            yield artifact


@contextlib.contextmanager
def suppress_output() -> Generator:
    """Suppress all warnings, stdout/stderr, and logging from anndata/scanpy."""
    anndata_logger = logging.getLogger("anndata")
    old_level = anndata_logger.level
    anndata_logger.setLevel(logging.ERROR)

    with warnings.catch_warnings(), Path.open(os.devnull, "w") as devnull:
        warnings.simplefilter("ignore")
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                yield
            finally:
                anndata_logger.setLevel(old_level)
