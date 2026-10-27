import click
from huggingface_hub import hf_hub_download

REPO_ID = "marvinsxtr/MapPFN"
DATASETS = ("frangieh", "papalexi")


def download_data(seeds: tuple[int, ...], include_prior: bool) -> None:
    """Download the pre-trained model, the single-cell splits and optionally the synthetic prior from Hugging Face.

    Args:
        seeds: Split seeds of the single-cell datasets to download.
        include_prior: Whether to also download the synthetic prior used for pre-training.
    """
    hf_hub_download(REPO_ID, "model.ckpt", local_dir="checkpoints", repo_type="model")
    for dataset in DATASETS:
        for seed in seeds:
            hf_hub_download(
                REPO_ID, f"{dataset}/split_{seed}.h5ad", local_dir="datasets/single_cell", repo_type="dataset"
            )
    if include_prior:
        hf_hub_download(REPO_ID, "sergio.h5ad", local_dir="datasets/synthetic", repo_type="dataset")


@click.command()
@click.option("--seeds", type=int, multiple=True, default=(10, 20, 30, 40, 50), help="Split seeds to download.")
@click.option("--include-prior", is_flag=True, help="Also download the synthetic prior used for pre-training.")
def main(seeds: tuple[int, ...], include_prior: bool) -> None:
    """Download model and datasets."""
    download_data(seeds, include_prior)


if __name__ == "__main__":
    main()
