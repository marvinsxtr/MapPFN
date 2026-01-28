"""Tests for LinearSCM class."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from map_pfn.data.linear_scm import LinearSCM


# === Construction tests ===

def test_adjacency_matrix_is_dag() -> None:
    """Adjacency matrix should be upper triangular after permutation (DAG)."""
    scm = LinearSCM(num_nodes=10, edge_prob=0.5, seed=42)
    
    A = scm.adjacency_matrix
    A_power = A.copy()
    for _ in range(scm.num_nodes):
        A_power = A_power @ A
    
    assert_array_equal(A_power, 0)


def test_weights_match_adjacency_structure() -> None:
    """Non-zero weights should only exist where adjacency matrix has edges."""
    scm = LinearSCM(num_nodes=10, edge_prob=0.5, seed=42)
    
    weight_mask = scm.weights != 0
    assert np.all(weight_mask == scm.adjacency_matrix.astype(bool))


def test_edge_probability_respected() -> None:
    """Average edge density should approximate edge_prob."""
    edge_probs = []
    for seed in range(50):
        scm = LinearSCM(num_nodes=20, edge_prob=0.3, seed=seed)
        max_edges = 20 * 19 / 2
        actual_edges = scm.adjacency_matrix.sum()
        edge_probs.append(actual_edges / max_edges)
    
    assert_allclose(np.mean(edge_probs), 0.3, atol=0.05)


# === Sampling tests ===

def test_observational_shape() -> None:
    """Observational samples should have correct shape."""
    scm = LinearSCM(num_nodes=5, seed=42)
    obs = scm.sample_observations(100)
    
    assert obs.shape == (100, 5)


def test_interventional_shape() -> None:
    """Interventional samples should have correct shape."""
    scm = LinearSCM(num_nodes=5, seed=42)
    intv = scm.sample_intervention(100, intervention_node=0, intervention_value=2.0)
    
    assert intv.shape == (100, 5)


def test_intervention_fixes_target_node() -> None:
    """Intervened node should have constant value."""
    scm = LinearSCM(num_nodes=5, seed=42)
    intv = scm.sample_intervention(100, intervention_node=2, intervention_value=3.5)
    
    assert_allclose(intv[:, 2], 3.5, atol=1e-10)


def test_intervention_changes_downstream_mean() -> None:
    """Intervention should shift mean of descendants."""
    scm1 = LinearSCM(num_nodes=10, edge_prob=0.8, seed=42)
    scm2 = LinearSCM(num_nodes=10, edge_prob=0.8, seed=42)

    # Nodes with children: column sum > 0 (outgoing edges)
    nodes_with_children = np.where(scm1.adjacency_matrix.sum(axis=0) > 0)[0]
    intv_node = nodes_with_children[0]
    
    # Children: nodes i where A[i, intv_node] > 0 (intv_node → i)
    children = np.where(scm1.adjacency_matrix[:, intv_node] > 0)[0]

    obs = scm1.sample_observations(5000, noise_std=1.0)
    intv = scm2.sample_intervention(5000, intv_node, intervention_value=5.0, noise_std=1.0)

    differences = [abs(intv[:, c].mean() - obs[:, c].mean()) for c in children]
    assert max(differences) > 0.1 or len(children) == 0


def test_intervention_preserves_parent_distribution() -> None:
    """Parents should have similar distributions under intervention."""
    scm1 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=123)
    scm2 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=123)

    # Find a node with parents
    nodes_with_parents = np.where(scm1.adjacency_matrix.sum(axis=1) > 0)[0]
    intv_node = nodes_with_parents[-1]
    
    # Parents: nodes j where A[intv_node, j] > 0 (j → intv_node)
    parents = np.where(scm1.adjacency_matrix[intv_node, :] > 0)[0]

    if len(parents) > 0:
        obs = scm1.sample_observations(5000, noise_std=1.0)
        intv = scm2.sample_intervention(5000, intv_node, 5.0, noise_std=1.0)

        for p in parents:
            assert abs(obs[:, p].mean()) < 0.2
            assert abs(intv[:, p].mean()) < 0.2


def test_intervention_changes_distribution() -> None:
    """Intervention should change downstream distribution."""
    scm1 = LinearSCM(num_nodes=10, edge_prob=0.8, seed=42)
    scm2 = LinearSCM(num_nodes=10, edge_prob=0.8, seed=42)

    # Nodes with children: column sum > 0
    nodes_with_children = np.where(scm1.adjacency_matrix.sum(axis=0) > 0)[0]
    intv_node = nodes_with_children[0]
    
    # Children of intv_node: A[:, intv_node] > 0
    children = np.where(scm1.adjacency_matrix[:, intv_node] > 0)[0]

    obs = scm1.sample_observations(5000, noise_std=1.0)
    intv = scm2.sample_intervention(5000, intv_node, intervention_value=5.0, noise_std=1.0)

    for child in children:
        obs_mean = obs[:, child].mean()
        intv_mean = intv[:, child].mean()
        assert abs(intv_mean - obs_mean) > 0.5


def test_intervention_does_not_affect_parents() -> None:
    """Intervention should not affect parents."""
    scm1 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=123)
    scm2 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=123)

    # Nodes with parents: row sum > 0
    nodes_with_parents = np.where(scm1.adjacency_matrix.sum(axis=1) > 0)[0]
    intv_node = nodes_with_parents[-1]
    
    # Parents of intv_node: A[intv_node, :] > 0
    parents = np.where(scm1.adjacency_matrix[intv_node, :] > 0)[0]

    if len(parents) > 0:
        obs = scm1.sample_observations(5000, noise_std=1.0)
        intv = scm2.sample_intervention(5000, intv_node, 5.0, noise_std=1.0)

        for p in parents:
            assert_allclose(obs[:, p].mean(), intv[:, p].mean(), atol=0.1)


# === Reproducibility tests ===

def test_same_seed_produces_same_scm() -> None:
    """Same seed should produce identical SCM."""
    scm1 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=42)
    scm2 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=42)
    
    assert_array_equal(scm1.adjacency_matrix, scm2.adjacency_matrix)
    assert_array_equal(scm1.weights, scm2.weights)


def test_different_seeds_produce_different_scms() -> None:
    """Different seeds should produce different SCMs."""
    scm1 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=42)
    scm2 = LinearSCM(num_nodes=10, edge_prob=0.5, seed=43)
    
    assert not np.array_equal(scm1.weights, scm2.weights)


@pytest.mark.parametrize("context_seed,treatment_seed", [
    (42, 100),
    (42, 200),
    (123, 100),
    (123, 200),
    (0, 0),
    (2**30, 2**30),
])
def test_reproducibility_parametrized(context_seed: int, treatment_seed: int) -> None:
    """Parametrized test for (context, treatment) reproducibility."""
    scm1 = LinearSCM(num_nodes=8, edge_prob=0.6, seed=context_seed)
    scm2 = LinearSCM(num_nodes=8, edge_prob=0.6, seed=context_seed)
    
    trt1 = scm1.sample_treatment(treatment_seed)
    trt2 = scm2.sample_treatment(treatment_seed)
    
    assert_array_equal(scm1.adjacency_matrix, scm2.adjacency_matrix)
    assert_array_equal(scm1.weights, scm2.weights)
    assert trt1 == trt2


# === Treatment sampling tests ===

def test_treatment_value_in_range() -> None:
    """Treatment value should be within specified range."""
    scm = LinearSCM(
        num_nodes=10,
        edge_prob=0.5,
        treatment_range=(-2.0, 2.0),
        seed=42,
    )
    
    for node in range(9):
        _, value = scm.sample_treatment(treatment_node=node)
        assert -2.0 <= value <= 2.0


# === Edge case tests ===

def test_high_edge_probability() -> None:
    """SCM should handle dense graphs."""
    scm = LinearSCM(num_nodes=10, edge_prob=0.95, seed=42)
    obs = scm.sample_observations(100)
    
    assert np.isfinite(obs).all()


def test_low_edge_probability() -> None:
    """SCM should handle sparse graphs."""
    scm = LinearSCM(num_nodes=10, edge_prob=0.1, seed=42)
    obs = scm.sample_observations(100)
    
    assert np.isfinite(obs).all()


def test_small_graph() -> None:
    """SCM should work with minimal graph size."""
    scm = LinearSCM(num_nodes=2, edge_prob=1.0, seed=42)
    obs = scm.sample_observations(100)
    
    assert obs.shape == (100, 2)
    assert np.isfinite(obs).all()


def test_large_noise_std() -> None:
    """SCM should handle large noise values."""
    scm = LinearSCM(num_nodes=5, seed=42)
    obs = scm.sample_observations(100, noise_std=10.0)
    
    assert np.isfinite(obs).all()


def test_zero_noise_std() -> None:
    """Zero noise should produce deterministic observations (all zeros)."""
    scm = LinearSCM(num_nodes=5, seed=42)
    obs = scm.sample_observations(100, noise_std=0.0)
    
    assert_allclose(obs, 0.0, atol=1e-10)