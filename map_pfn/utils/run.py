import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from ml_project_template.runs import SlurmParams
from ml_project_template.utils import ConfigKeys, get_output_dir, logger
from ml_project_template.wandb import WandBConfig
from submitit import AutoExecutor
from submitit.helpers import CommandFunction


@dataclass
class Job:
    """Job to run code on a cluster using apptainer."""

    image: str = "oras://ghcr.io/marvinsxtr/mappfn:latest-sif"
    dataset: str | None = None
    cluster: str = "slurm"
    slurm_params: SlurmParams = field(default_factory=SlurmParams)
    wait_for_job: bool = False
    timeout_min: int = 5

    def __post_init__(self) -> None:
        """Run the job."""
        self.run()
        sys.exit(0)

    def filter_args(self, args: list[str]) -> list[str]:
        """Filter args to prevent recursive jobs on the cluster."""
        return [arg for arg in args if f"{ConfigKeys.CONFIG}/{ConfigKeys.JOB}" not in arg]

    @property
    def python_command(self) -> str:
        """Python command used by the job."""
        if self.dataset is not None:
            if Path(self.dataset).suffix == ".sqfs":
                bind_mount = f" -B {self.dataset}:/srv/data:image-src=/"
            else:
                bind_mount = f" -B {self.dataset}:/srv/data"
        else:
            bind_mount = ""

        return f"apptainer run{bind_mount} --nv {self.image} python"

    def run(self) -> None:
        """Run the job on the cluster."""
        hydra_run_dir = "./outputs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S-%f}"

        command = [
            "python",
            *self.filter_args(sys.argv),
            "cfg/wandb=base",
            f"hydra.run.dir={hydra_run_dir}",
        ]

        function = CommandFunction(command)
        executor = AutoExecutor(
            folder=get_output_dir(),
            cluster=self.cluster,
            slurm_python=self.python_command,
        )

        executor.update_parameters(
            timeout_min=self.timeout_min,
            **self.slurm_params.to_submitit_params(),
        )
        job = executor.submit(function)

        logger.info(f"Submitted job {job.job_id}")

        if self.wait_for_job:
            logger.info(f"\n{job.result()}")


@dataclass
class SweepJob(Job):
    """Job to run a sweep on a cluster."""

    num_workers: int = 1
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    metric_name: str = "loss"
    metric_goal: Literal["maximize", "minimize"] = "minimize"

    def register_sweep(self, sweep_config: dict) -> str:
        """Register a wandb sweep from a config."""
        if (wandb_config := WandBConfig.from_env()) is None:
            raise RuntimeError("No WandB config found in environment.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "sweep_config.yaml"

            with Path.open(config_path, "w") as config_file:
                yaml.dump(sweep_config, config_file)

            output = subprocess.run(
                ["wandb", "sweep", "--project", wandb_config.WANDB_PROJECT, str(config_path)],
                check=True,
                text=True,
                capture_output=True,
            ).stderr

            sweep_id = output.split(" ")[-1].strip()

            for line in output.splitlines():
                logger.info(line)

        return sweep_id

    def run(self) -> None:
        """Run the sweep on the cluster."""
        parameters = {cfg_key: {"values": list(values)} for cfg_key, values in self.parameters.items()}
        metric = {"goal": self.metric_goal, "name": self.metric_name}
        program, args = sys.argv[0], self.filter_args(sys.argv[1:])

        folder_path = get_output_dir()
        dummy_sweep_id = "sweep_started_" + Path(folder_path).parts[-2] + "_" + Path(folder_path).parts[-1]
        hydra_run_dir = "./outputs/sweeps/" + dummy_sweep_id + "/${now:%H-%M-%S-%f}"

        command = [
            "${env}",
            "${interpreter}",
            "${program}",
            *args,
            "cfg/wandb=base",
            f"hydra.run.dir={hydra_run_dir}",
            "${args_no_hyphens}",
        ]

        sweep_config = {
            "program": program,
            "method": "grid",
            "metric": metric,
            "parameters": parameters,
            "command": command,
        }

        sweep_id = self.register_sweep(sweep_config)

        function = CommandFunction(["wandb", "agent"])
        executor = AutoExecutor(
            folder=folder_path,
            cluster=self.cluster,
            slurm_python=self.python_command,
        )
        executor.update_parameters(
            timeout_min=self.timeout_min,
            slurm_array_parallelism=self.num_workers,
            **self.slurm_params.to_submitit_params(),
        )

        jobs = executor.map_array(function, [sweep_id] * self.num_workers)

        for job in jobs:
            logger.info(f"Submitted job {job.job_id}")


@dataclass
class TrainingSlurmParams(SlurmParams):
    """Parameters of a training job."""

    gpus_per_node: int | None = None
    exclude: str | None = None
    additional_parameters: dict[str, str] | None = None
