"""Tests for PICNN numerical equivalence with NPICNN (identity init).

PICNN is equivalent to NPICNN with identity initialization. Both models use
w_0 initialized to identity matrix, so softmax(10 * I @ y) ≈ y for one-hot
inputs, making the models numerically equivalent.
"""

import pytest
import torch
import torch.nn as nn

from baselines.condot.original.npicnn import NPICNN
from baselines.condot.picnn import PICNN


def small_init(weight: torch.Tensor) -> None:
    """Small weight initialization for near-zero contribution."""
    nn.init.normal_(weight, mean=0.0, std=1e-4)


class IdentityEmbedding:
    """Identity embedding that returns one-hot for string labels, passthrough for tensors."""
    
    def __init__(self, input_dim_label: int):
        self.input_dim_label = input_dim_label
        self._labels = [torch.eye(input_dim_label)[i] for i in range(input_dim_label)]
    
    def forward(self, y):
        # If y is a string label like "treatment_0", return one-hot
        if isinstance(y, str):
            idx = int(y.split("_")[-1])
            return self._labels[idx]
        # If y is already a tensor, return as-is
        return y


def copy_weights(npicnn: nn.Module, picnn: nn.Module) -> None:
    """Copy weights from NPICNN to PICNN.
    
    Both models now have identical architecture.
    """
    # Copy w_0 layer
    picnn.w_0.weight.data.copy_(npicnn.w_0.weight.data)
    
    # Copy w layers
    for n_layer, s_layer in zip(npicnn.w, picnn.w):
        s_layer.weight.data.copy_(n_layer.weight.data)
        s_layer.bias.data.copy_(n_layer.bias.data)
    
    # Copy w_z0
    picnn.w_z0.weight.data.copy_(npicnn.w_z0.weight.data)
    picnn.w_z0.bias.data.copy_(npicnn.w_z0.bias.data)
    
    # Copy wz layers
    for n_layer, s_layer in zip(npicnn.wz, picnn.wz):
        s_layer.weight.data.copy_(n_layer.weight.data)
    
    # Copy wzu layers
    for n_layer, s_layer in zip(npicnn.wzu, picnn.wzu):
        s_layer.weight.data.copy_(n_layer.weight.data)
        s_layer.bias.data.copy_(n_layer.bias.data)
    
    # Copy wx layers
    for n_layer, s_layer in zip(npicnn.wx, picnn.wx):
        s_layer.weight.data.copy_(n_layer.weight.data)
        s_layer.bias.data.copy_(n_layer.bias.data)
    
    # Copy wxu layers
    for n_layer, s_layer in zip(npicnn.wxu, picnn.wxu):
        s_layer.weight.data.copy_(n_layer.weight.data)
        s_layer.bias.data.copy_(n_layer.bias.data)
    
    # Copy wu layers
    for n_layer, s_layer in zip(npicnn.wu, picnn.wu):
        s_layer.weight.data.copy_(n_layer.weight.data)


class TestPICNNEquivalence:
    """Test that PICNN matches NPICNN with identity initialization.
    
    Both models have identical architecture with w_0 initialized to identity.
    With identity init, softmax(10 * I @ y) ≈ y for one-hot inputs.
    """

    @pytest.fixture
    def dims(self):
        """Common dimensions for tests."""
        return {
            "input_dim": 10,
            "input_dim_label": 5,
            "hidden_units": [32, 32],
        }

    @pytest.fixture
    def npicnn_model(self, dims):
        """Create NPICNN with identity initialization."""
        treatments = [f"treatment_{i}" for i in range(dims["input_dim_label"])]
        embedding = IdentityEmbedding(dims["input_dim_label"])
        model = NPICNN(
            input_dim=dims["input_dim"],
            input_dim_label=dims["input_dim_label"],
            hidden_units=dims["hidden_units"],
            activation="leakyrelu",
            softplus_wz_kernels=False,
            kernel_init_fxn=small_init,
            combinator=False,
            embedding=embedding,
            init_type="identity",
            init_inputs=treatments,
            num_labels=dims["input_dim_label"],
            name=None,
        )
        # Disable embedding for forward pass (just use y directly)
        model.embedding = False
        return model

    @pytest.fixture
    def picnn_model(self, dims):
        """Create PICNN."""
        model = PICNN(
            input_dim=dims["input_dim"],
            input_dim_label=dims["input_dim_label"],
            hidden_units=dims["hidden_units"],
            activation="leakyrelu",
            softplus_wz_kernels=False,
            kernel_init_fxn=small_init,
        )
        return model

    def test_same_architecture(self, npicnn_model, picnn_model, dims):
        """Test that both models have the same number of parameters."""
        npicnn_params = sum(p.numel() for p in npicnn_model.parameters())
        picnn_params = sum(p.numel() for p in picnn_model.parameters())
        
        assert npicnn_params == picnn_params, (
            f"Parameter count mismatch: NPICNN={npicnn_params}, PICNN={picnn_params}"
        )

    def test_shared_layer_structure(self, npicnn_model, picnn_model):
        """Test that shared layers have matching structures."""
        # Check w layers
        assert len(npicnn_model.w) == len(picnn_model.w)
        for i, (n_layer, s_layer) in enumerate(zip(npicnn_model.w, picnn_model.w)):
            assert n_layer.weight.shape == s_layer.weight.shape, f"w[{i}] shape mismatch"

        # Check wz layers
        assert len(npicnn_model.wz) == len(picnn_model.wz)
        for i, (n_layer, s_layer) in enumerate(zip(npicnn_model.wz, picnn_model.wz)):
            assert n_layer.weight.shape == s_layer.weight.shape, f"wz[{i}] shape mismatch"

        # Check wzu layers
        assert len(npicnn_model.wzu) == len(picnn_model.wzu)
        for i, (n_layer, s_layer) in enumerate(zip(npicnn_model.wzu, picnn_model.wzu)):
            assert n_layer.weight.shape == s_layer.weight.shape, f"wzu[{i}] shape mismatch"

        # Check wx layers
        assert len(npicnn_model.wx) == len(picnn_model.wx)
        for i, (n_layer, s_layer) in enumerate(zip(npicnn_model.wx, picnn_model.wx)):
            assert n_layer.weight.shape == s_layer.weight.shape, f"wx[{i}] shape mismatch"

        # Check wxu layers
        assert len(npicnn_model.wxu) == len(picnn_model.wxu)
        for i, (n_layer, s_layer) in enumerate(zip(npicnn_model.wxu, picnn_model.wxu)):
            assert n_layer.weight.shape == s_layer.weight.shape, f"wxu[{i}] shape mismatch"

        # Check wu layers
        assert len(npicnn_model.wu) == len(picnn_model.wu)
        for i, (n_layer, s_layer) in enumerate(zip(npicnn_model.wu, picnn_model.wu)):
            assert n_layer.weight.shape == s_layer.weight.shape, f"wu[{i}] shape mismatch"

    def test_identity_init_w_z0(self, npicnn_model, picnn_model, dims):
        """Test that w_z0 (PosDefPotentials) is initialized identically."""
        # Both should have identity matrices for factors
        assert torch.allclose(npicnn_model.w_z0.weight, picnn_model.w_z0.weight), (
            "w_z0 weight (factors) mismatch"
        )
        # Both should have zero means
        assert torch.allclose(npicnn_model.w_z0.bias, picnn_model.w_z0.bias), (
            "w_z0 bias (means) mismatch"
        )

    def test_npicnn_w_0_is_identity(self, npicnn_model, dims):
        """Test that NPICNN has w_0 initialized to identity.
        
        PICNN uses default PyTorch initialization for w_0 (learned layer for
        arbitrary context vectors, not necessarily one-hot).
        """
        expected = torch.eye(dims["input_dim_label"])
        assert torch.allclose(npicnn_model.w_0.weight, expected), "NPICNN w_0 not identity"

    def test_forward_equivalence_single_condition(self, npicnn_model, picnn_model, dims):
        """Test forward pass equivalence with a single condition vector."""
        copy_weights(npicnn_model, picnn_model)

        torch.manual_seed(42)
        x = torch.randn(16, dims["input_dim"])
        y = torch.zeros(dims["input_dim_label"])
        y[0] = 1.0  # One-hot for first treatment

        npicnn_model.eval()
        picnn_model.eval()

        with torch.no_grad():
            npicnn_out = npicnn_model(x, y)
            picnn_out = picnn_model(x, y)

        assert torch.allclose(npicnn_out, picnn_out, atol=1e-6), (
            f"Forward output mismatch: max diff = {(npicnn_out - picnn_out).abs().max()}"
        )

    def test_forward_equivalence_batch_onehot(self, npicnn_model, picnn_model, dims):
        """Test forward pass equivalence with batched one-hot condition vectors."""
        copy_weights(npicnn_model, picnn_model)

        torch.manual_seed(123)
        batch_size = 32
        x = torch.randn(batch_size, dims["input_dim"])
        # Random one-hot conditions
        y_indices = torch.randint(0, dims["input_dim_label"], (batch_size,))
        y = torch.zeros(batch_size, dims["input_dim_label"])
        y.scatter_(1, y_indices.unsqueeze(1), 1.0)

        npicnn_model.eval()
        picnn_model.eval()

        with torch.no_grad():
            npicnn_out = npicnn_model(x, y)
            picnn_out = picnn_model(x, y)

        # Close due to softmax(10 * I @ onehot) ≈ onehot
        assert torch.allclose(npicnn_out, picnn_out, atol=1e-4), (
            f"Forward output mismatch: max diff = {(npicnn_out - picnn_out).abs().max()}"
        )

    def test_transport_equivalence_onehot(self, npicnn_model, picnn_model, dims):
        """Test transport map equivalence with one-hot conditions."""
        copy_weights(npicnn_model, picnn_model)

        torch.manual_seed(456)
        x = torch.randn(16, dims["input_dim"], requires_grad=True)
        y = torch.zeros(dims["input_dim_label"])
        y[2] = 1.0  # One-hot for third treatment

        npicnn_model.eval()
        picnn_model.eval()

        npicnn_transport = npicnn_model.transport(x, y)
        
        # Need fresh x for picnn model
        x2 = x.detach().clone().requires_grad_(True)
        picnn_transport = picnn_model.transport(x2, y)

        # Close due to softmax(10 * I @ onehot) ≈ onehot
        assert torch.allclose(npicnn_transport, picnn_transport, atol=1e-3), (
            f"Transport mismatch: max diff = {(npicnn_transport - picnn_transport).abs().max()}"
        )

    def test_clamp_w_equivalence(self, npicnn_model, picnn_model):
        """Test that clamp_w works identically on shared wz layers."""
        copy_weights(npicnn_model, picnn_model)

        # Set some negative weights
        for i, wz in enumerate(npicnn_model.wz):
            wz.weight.data = torch.randn_like(wz.weight.data)
            picnn_model.wz[i].weight.data = wz.weight.data.clone()

        npicnn_model.clamp_w()
        picnn_model.clamp_w()

        for n_wz, s_wz in zip(npicnn_model.wz, picnn_model.wz):
            assert torch.allclose(n_wz.weight, s_wz.weight), "clamp_w mismatch"
            assert (n_wz.weight >= 0).all(), "NPICNN has negative weights after clamp"
            assert (s_wz.weight >= 0).all(), "PICNN has negative weights after clamp"

    def test_penalize_w_equivalence(self, npicnn_model, picnn_model):
        """Test that penalize_w computes the same penalty on shared layers."""
        copy_weights(npicnn_model, picnn_model)

        # Set some negative weights for non-zero penalty
        for i, wz in enumerate(npicnn_model.wz):
            wz.weight.data = torch.randn_like(wz.weight.data) - 0.5
            picnn_model.wz[i].weight.data = wz.weight.data.clone()

        # Set same fnorm_penalty
        npicnn_model.fnorm_penalty = 0.1
        picnn_model.fnorm_penalty = 0.1

        npicnn_penalty = npicnn_model.penalize_w()
        picnn_penalty = picnn_model.penalize_w()

        assert torch.allclose(npicnn_penalty, picnn_penalty, atol=1e-6), (
            f"Penalty mismatch: NPICNN={npicnn_penalty}, PICNN={picnn_penalty}"
        )


class TestPICNNWithSoftplus:
    """Test PICNN with softplus non-negative weights."""

    @pytest.fixture
    def dims(self):
        return {"input_dim": 8, "input_dim_label": 4, "hidden_units": [24, 24]}

    @pytest.fixture
    def npicnn_softplus(self, dims):
        treatments = [f"t_{i}" for i in range(dims["input_dim_label"])]
        embedding = IdentityEmbedding(dims["input_dim_label"])
        model = NPICNN(
            input_dim=dims["input_dim"],
            input_dim_label=dims["input_dim_label"],
            hidden_units=dims["hidden_units"],
            activation="leakyrelu",
            softplus_wz_kernels=True,
            softplus_beta=1.0,
            kernel_init_fxn=small_init,
            combinator=False,
            embedding=embedding,
            init_type="identity",
            init_inputs=treatments,
            num_labels=dims["input_dim_label"],
        )
        # Disable embedding for forward pass
        model.embedding = False
        return model

    @pytest.fixture
    def picnn_softplus(self, dims):
        return PICNN(
            input_dim=dims["input_dim"],
            input_dim_label=dims["input_dim_label"],
            hidden_units=dims["hidden_units"],
            activation="leakyrelu",
            softplus_wz_kernels=True,
            softplus_beta=1.0,
            kernel_init_fxn=small_init,
        )

    def test_forward_equivalence_softplus(self, npicnn_softplus, picnn_softplus, dims):
        """Test forward equivalence with softplus weights and one-hot conditions."""
        copy_weights(npicnn_softplus, picnn_softplus)

        torch.manual_seed(789)
        x = torch.randn(20, dims["input_dim"])
        y = torch.zeros(dims["input_dim_label"])
        y[1] = 1.0

        npicnn_softplus.eval()
        picnn_softplus.eval()

        with torch.no_grad():
            npicnn_out = npicnn_softplus(x, y)
            picnn_out = picnn_softplus(x, y)

        # Close due to softmax(10 * I @ onehot) ≈ onehot
        assert torch.allclose(npicnn_out, picnn_out, atol=1e-4)

    def test_clamp_w_noop_with_softplus(self, picnn_softplus):
        """Test that clamp_w is a no-op when using softplus."""
        original_weights = [wz.weight.data.clone() for wz in picnn_softplus.wz]
        picnn_softplus.clamp_w()
        for orig, wz in zip(original_weights, picnn_softplus.wz):
            assert torch.equal(orig, wz.weight.data), "clamp_w modified weights with softplus"


class TestPICNNStandalone:
    """Standalone tests for PICNN functionality."""

    def test_output_shape(self):
        """Test output shape is correct."""
        model = PICNN(
            input_dim=10,
            input_dim_label=5,
            hidden_units=[32, 32],
            kernel_init_fxn=small_init,
        )
        x = torch.randn(16, 10)
        y = torch.randn(5)

        out = model(x, y)
        assert out.shape == (16, 1), f"Expected (16, 1), got {out.shape}"

    def test_convexity_in_x(self):
        """Test that the output is convex in x for fixed y."""
        model = PICNN(
            input_dim=4,
            input_dim_label=3,
            hidden_units=[16, 16],
            kernel_init_fxn=small_init,
        )
        model.clamp_w()  # Ensure convexity

        y = torch.zeros(3)
        y[0] = 1.0

        # Test convexity: f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)
        torch.manual_seed(42)
        x1 = torch.randn(1, 4)
        x2 = torch.randn(1, 4)
        t = 0.3

        x_interp = t * x1 + (1 - t) * x2

        with torch.no_grad():
            f_x1 = model(x1, y)
            f_x2 = model(x2, y)
            f_interp = model(x_interp, y)

        convex_bound = t * f_x1 + (1 - t) * f_x2
        assert f_interp <= convex_bound + 1e-5, (
            f"Convexity violated: f(interp)={f_interp.item()}, bound={convex_bound.item()}"
        )

    def test_gradient_exists(self):
        """Test that gradients flow through the model."""
        model = PICNN(
            input_dim=6,
            input_dim_label=4,
            hidden_units=[20, 20],
            kernel_init_fxn=small_init,
        )
        x = torch.randn(8, 6, requires_grad=True)
        y = torch.randn(4)

        out = model(x, y)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "No gradient for x"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"

    def test_different_hidden_sizes(self):
        """Test with different hidden layer configurations.
        
        Note: NPICNN architecture requires all hidden layers to have the same width
        (except the final output which is 1). This is a constraint of the original
        architecture where wzu layers are square matrices.
        """
        configs = [
            [16, 16],
            [32, 32],
            [64, 64, 64],
            [128, 128, 128, 128],
        ]
        for hidden_units in configs:
            model = PICNN(
                input_dim=8,
                input_dim_label=4,
                hidden_units=hidden_units,
                kernel_init_fxn=small_init,
            )
            x = torch.randn(4, 8)
            y = torch.randn(4)
            out = model(x, y)
            assert out.shape == (4, 1), f"Failed for hidden_units={hidden_units}"

    def test_relu_activation(self):
        """Test with ReLU activation."""
        model = PICNN(
            input_dim=6,
            input_dim_label=3,
            hidden_units=[16, 16],
            activation="relu",
            kernel_init_fxn=small_init,
        )
        x = torch.randn(8, 6)
        y = torch.randn(3)
        out = model(x, y)
        assert out.shape == (8, 1)
        assert out.shape == (8, 1)

    def test_neural_embedding(self):
        """Test PICNN with neural embedding (linear + sigmoid)."""
        input_dim = 8
        input_dim_label = 4
        embed_dim = 10
        
        model = PICNN(
            input_dim=input_dim,
            input_dim_label=input_dim_label,
            hidden_units=[32, 32],
            neural_embedding=(input_dim_label, embed_dim),
            kernel_init_fxn=small_init,
        )
        
        x = torch.randn(16, input_dim)
        y = torch.randn(16, input_dim_label)  # Raw condition, will be embedded
        
        out = model(x, y)
        assert out.shape == (16, 1), f"Expected (16, 1), got {out.shape}"
        
        # Test that gradient flows through embedding
        x_grad = torch.randn(8, input_dim, requires_grad=True)
        y_grad = torch.randn(8, input_dim_label)
        out = model(x_grad, y_grad)
        loss = out.sum()
        loss.backward()
        assert x_grad.grad is not None, "No gradient for x with neural embedding"

    def test_neural_embedding_transport(self):
        """Test transport works with neural embedding."""
        input_dim = 6
        input_dim_label = 3
        embed_dim = 8
        
        model = PICNN(
            input_dim=input_dim,
            input_dim_label=input_dim_label,
            hidden_units=[24, 24],
            neural_embedding=(input_dim_label, embed_dim),
            kernel_init_fxn=small_init,
        )
        
        x = torch.randn(10, input_dim, requires_grad=True)
        y = torch.randn(10, input_dim_label)
        
        transport = model.transport(x, y)
        assert transport.shape == (10, input_dim), f"Expected (10, {input_dim}), got {transport.shape}"
