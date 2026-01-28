from typing import NamedTuple

from hydra_zen import builds
from hydra_zen.typing import Partial
from ml_project_template.runs import Job
from ml_project_template.wandb import WandBRun
from torch.utils.data import Dataset

from map_pfn.data.scm_dataset import LinearSCMDataset
from map_pfn.data.sergio_dataset import SergioDataset


class DataGeneratorRun(NamedTuple):
    """Configures a data generation run."""

    name: str
    dataset: Partial[Dataset]
    output_dir: str
    val_share: float
    test_share: float
    seed: int | None = None
    job: Job | None = None
    wandb: WandBRun | None = None


LinearDatasetConfig = builds(
    LinearSCMDataset,
    num_nodes=20,
    edge_prob=0.5,
    weight_range=[0.5, 2.0],
    noise_std=1.0,
    num_samples=500,
    num_contexts=1000,
    counterfactual=False,
    zen_partial=True,
)

LinearDataGeneratorRunConfig = builds(
    DataGeneratorRun,
    name="linear_scm",
    dataset=LinearDatasetConfig,
    val_share=0.1,
    test_share=0.5,
    output_dir="datasets/synthetic",
    seed=42,
    job=None,
    wandb=None,
)

SergioDatasetConfig = builds(
    SergioDataset,
    num_genes=50,
    num_samples=200,
    num_contexts=6000,
    num_groups_range=[1, 3],
    regulators_per_gene_range=[1.5, 3.0],
    delta_in_range=[10.0, 300.0],
    delta_out_range=[1.0, 30.0],
    modularity_range=[1.0, 900.0],
    num_cell_types=1,
    interaction_k_range=[1.0, 5.0],
    decay_range=[0.5, 1.0],
    hill_n_range=[1.5, 2.5],
    noise_s_range=[0.5, 1.5],
    safety_iter=150,
    scale_iter=10,
    dt=0.01,
    mr_low_range=[0.5, 2.0],
    mr_high_range=[3.0, 5.0],
    outlier_mu_range=[0.8, 5.0],
    library_mu_range=[4.5, 6.0],
    library_sigma_range=[0.3, 0.7],
    dropout_k_range=[8.0, 8.0],
    dropout_q_range=[45.0, 82.0],
    counterfactual=True,
    zen_partial=True,
)

SergioDataGeneratorRunConfig = builds(
    DataGeneratorRun,
    name="sergio_grn",
    dataset=SergioDatasetConfig,
    val_share=0.1,
    test_share=0.5,
    output_dir="datasets/synthetic",
    seed=42,
    job=None,
    wandb=None,
)
