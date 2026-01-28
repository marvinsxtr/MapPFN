import contextlib
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from ml_project_template.utils import get_output_dir
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
    os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_runtime=false"

    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before setting XLA_PYTHON_CLIENT_PREALLOCATE.")

    if any(torch_cfg in argument for torch_cfg in ["condot", "metafm"] for argument in sys.argv):
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def git_commit_hash() -> str:
    """Get the current git commit short hash."""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
