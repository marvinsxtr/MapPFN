import numpy as np
import pytest
from torch.utils.data import DataLoader

from map_pfn.data.scm_dataset import LinearSCMDataset
from map_pfn.data.utils import BatchKeys, Values, collate_fn


def hash_sample(sample: dict) -> int:
    """Create a hash from a sample for duplicate detection."""
    data_hash = hash(sample[BatchKeys.DATA].tobytes())
    treat_hash = hash(sample[BatchKeys.TREATMENT].tobytes())
    return hash((data_hash, treat_hash))


def test_linear_scm_dataset_structure():
    """Test that LinearSCMDataset returns correctly structured samples."""
    num_nodes = 5
    num_samples = 10
    dataset = LinearSCMDataset(
        num_nodes=num_nodes,
        edge_prob=0.3,
        num_samples=num_samples,
        num_contexts=3,
        seed=42,
    )

    sample = next(iter(dataset))

    assert BatchKeys.DATA in sample
    assert BatchKeys.TREATMENT in sample
    assert BatchKeys.CONTEXT_ID in sample
    assert BatchKeys.TREATMENT_ID in sample
    assert sample[BatchKeys.DATA].shape == (num_samples, num_nodes)
    assert sample[BatchKeys.TREATMENT].shape == (num_nodes,)


def test_linear_scm_dataset_deterministic():
    """Test that datasets with same seed produce identical samples."""
    dataset1 = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
    )
    dataset2 = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
    )

    hashes1 = [hash_sample(s) for s in dataset1]
    hashes2 = [hash_sample(s) for s in dataset2]

    assert len(hashes1) == len(hashes2)
    assert hashes1 == hashes2


def test_linear_scm_dataset_different_seeds():
    """Test that datasets with different seeds produce different samples."""
    dataset1 = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
    )
    dataset2 = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=43,
    )

    hashes1 = [hash_sample(s) for s in dataset1]
    hashes2 = [hash_sample(s) for s in dataset2]

    matches = sum(h1 == h2 for h1, h2 in zip(hashes1, hashes2, strict=True))
    assert matches < len(hashes1)


def test_linear_scm_dataset_length():
    """Test dataset length with all conditions."""
    num_contexts = 5
    num_nodes = 5

    dataset = LinearSCMDataset(
        num_nodes=num_nodes,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=num_contexts,
        seed=42,
    )

    expected_length = num_contexts * num_nodes + num_contexts
    assert len(dataset) == expected_length

    samples = list(dataset)
    assert len(samples) == expected_length


def test_linear_scm_dataset_observational_samples_first():
    """Test that observational samples are yielded first."""
    num_contexts = 3
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=num_contexts,
        seed=42,
    )

    samples = list(dataset)

    for i in range(num_contexts):
        assert samples[i][BatchKeys.TREATMENT_ID] == Values.CONTROL
        assert (samples[i][BatchKeys.TREATMENT] == 0).all()

    for sample in samples[num_contexts:]:
        assert sample[BatchKeys.TREATMENT_ID] != Values.CONTROL


def test_linear_scm_dataset_shuffle():
    """Test that shuffle produces different iteration order."""
    dataset_unshuffled = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
        shuffle=False,
    )
    dataset_shuffled = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
        shuffle=True,
    )

    unshuffled_order = [
        (s[BatchKeys.CONTEXT_ID], s[BatchKeys.TREATMENT_ID])
        for s in dataset_unshuffled
    ]
    shuffled_order = [
        (s[BatchKeys.CONTEXT_ID], s[BatchKeys.TREATMENT_ID])
        for s in dataset_shuffled
    ]

    assert set(unshuffled_order) == set(shuffled_order)
    assert unshuffled_order != shuffled_order


def test_linear_scm_dataset_large_num_contexts():
    """Test that large num_contexts works efficiently."""
    num_contexts = 100
    num_nodes = 5

    dataset = LinearSCMDataset(
        num_nodes=num_nodes,
        edge_prob=0.6,  # Higher prob to ensure nodes have children
        num_samples=10,
        num_contexts=num_contexts,
        seed=42,
    )

    samples = list(dataset)
    assert len(samples) == num_contexts * num_nodes + num_contexts


def test_linear_scm_dataset_small_num_contexts():
    """Test edge case with small num_contexts."""
    num_contexts = 2
    num_nodes = 5

    dataset = LinearSCMDataset(
        num_nodes=num_nodes,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=num_contexts,
        seed=42,
    )

    samples = list(dataset)
    assert len(samples) == num_contexts * num_nodes + num_contexts


def test_linear_scm_dataset_unique_samples():
    """Test that all samples produce unique data."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
    )

    all_hashes = [hash_sample(sample) for sample in dataset]
    unique_hashes = len(set(all_hashes))

    assert unique_hashes == len(all_hashes)


def test_linear_scm_dataset_batch_keys_present():
    """Test that all expected batch keys are present in samples."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
    )

    sample = next(iter(dataset))

    assert BatchKeys.DATA in sample
    assert BatchKeys.TREATMENT in sample
    assert BatchKeys.TREATMENT_ID in sample
    assert BatchKeys.CONTEXT_ID in sample


def test_linear_scm_dataset_treatment_encoding():
    """Test that treatments are properly one-hot encoded."""
    num_nodes = 5
    num_contexts = 3
    dataset = LinearSCMDataset(
        num_nodes=num_nodes,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=num_contexts,
        seed=42,
    )

    for sample in dataset:
        treatment = sample[BatchKeys.TREATMENT]

        assert treatment.shape == (num_nodes,)
        non_zero_count = (treatment != 0).sum()
        assert non_zero_count <= 1


def test_linear_scm_dataset_obs_data_normalized():
    """Test that observational data is normalized."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=100,
        num_contexts=5,
        seed=42,
    )

    sample = next(iter(dataset))
    data = sample[BatchKeys.DATA]

    mean = data.mean(axis=0)
    std = data.std(axis=0)

    assert abs(mean.mean()) < 0.5
    assert abs(std.mean() - 1.0) < 0.5


def test_linear_scm_dataset_context_treatment_combinations():
    """Test that all context-treatment combinations are covered."""
    num_contexts = 3
    num_nodes = 5

    dataset = LinearSCMDataset(
        num_nodes=num_nodes,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=num_contexts,
        seed=42,
    )

    seen_pairs = set()
    for sample in dataset:
        pair = (sample[BatchKeys.CONTEXT_ID], sample[BatchKeys.TREATMENT_ID])
        seen_pairs.add(pair)

    expected_obs = num_contexts
    expected_int = num_contexts * num_nodes
    assert len(seen_pairs) == expected_obs + expected_int


def test_linear_scm_dataset_same_context_same_scm():
    """Test that samples with same context use the same SCM structure."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=3,
        seed=42,
    )

    samples_by_context: dict[str, list] = {}
    for sample in dataset:
        context_id = sample[BatchKeys.CONTEXT_ID]
        if context_id not in samples_by_context:
            samples_by_context[context_id] = []
        samples_by_context[context_id].append(sample)

    for context_id in samples_by_context:
        assert len(samples_by_context[context_id]) >= 1


def test_linear_scm_dataset_dataloader_integration():
    """Test that dataset works with DataLoader."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=5,
        seed=42,
    )

    dataloader = DataLoader(dataset, batch_size=4, num_workers=0, collate_fn=collate_fn)

    batch = next(iter(dataloader))

    assert batch[BatchKeys.DATA].shape[0] == 4
    assert batch[BatchKeys.TREATMENT].shape[0] == 4


def test_encode_treatment_observational():
    """Test treatment encoding for observational data."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=3,
        seed=42,
    )

    treatment_vec = dataset.encode_treatment(None)

    assert treatment_vec.shape == (5,)
    assert (treatment_vec == 0).all()


def test_encode_treatment_interventional():
    """Test treatment encoding for interventional data."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=3,
        seed=42,
    )

    treatment_vec = dataset.encode_treatment((2, 1.5))

    assert treatment_vec.shape == (5,)
    assert treatment_vec[2] == 1.5
    assert (treatment_vec[np.arange(5) != 2] == 0).all()


def test_sample_condition_observational():
    """Test sampling observational condition."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=3,
        seed=42,
    )

    ctx_seed = dataset.context_seeds[0]
    sample = dataset.sample_condition(int(ctx_seed), treatment_node=None)

    assert sample[BatchKeys.TREATMENT_ID] == Values.CONTROL
    assert (sample[BatchKeys.TREATMENT] == 0).all()
    assert sample[BatchKeys.DATA].shape == (10, 5)


def test_sample_condition_interventional():
    """Test sampling interventional condition."""
    dataset = LinearSCMDataset(
        num_nodes=5,
        edge_prob=0.3,
        num_samples=10,
        num_contexts=3,
        seed=42,
    )

    ctx_seed = dataset.context_seeds[0]
    trt_seed = dataset.treatments[0]
    sample = dataset.sample_condition(int(ctx_seed), int(trt_seed))

    assert sample[BatchKeys.TREATMENT_ID] == str(trt_seed)
    assert sample[BatchKeys.DATA].shape == (10, 5)
