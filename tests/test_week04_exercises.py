"""Day 04 tests: ReLU, MLP forward, and manual MLP backpropagation."""

import numpy as np
import pytest
import torch

from exercises.week04 import (
    relu_numpy,
    relu_backward_numpy,
    two_layer_mlp_numpy,
    two_layer_mlp_mse_gradients,
)


@pytest.mark.exercise
def test_relu_numpy_forward_values_and_shape():
    x = np.array([[-4.0, 0.0, 2.0], [3.0, -1.0, 0.0]], dtype=np.float64)
    before = x.copy()
    actual = relu_numpy(x)
    expected = np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]])
    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)
    np.testing.assert_array_equal(x, before)


@pytest.mark.exercise
def test_relu_backward_numpy_chain_rule_and_zero():
    z = np.array([[-2.0, 0.0, 3.0], [4.0, -1.0, 0.0]], dtype=np.float64)
    upstream = np.array([[5.0, 7.0, -2.0], [3.0, 4.0, -9.0]], dtype=np.float64)
    z_original, upstream_original = z.copy(), upstream.copy()
    actual = relu_backward_numpy(z, upstream)
    expected = np.array([[0.0, 0.0, -2.0], [3.0, 0.0, 0.0]])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(z, z_original)
    np.testing.assert_array_equal(upstream, upstream_original)


@pytest.mark.exercise
def test_two_layer_mlp_forward_against_torch():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(4, 3)).astype(np.float64)
    w1 = rng.normal(size=(3, 5)).astype(np.float64)
    b1 = rng.normal(size=(5,)).astype(np.float64)
    w2 = rng.normal(size=(5, 2)).astype(np.float64)
    b2 = rng.normal(size=(2,)).astype(np.float64)
    actual = two_layer_mlp_numpy(x, w1, b1, w2, b2)
    reference = torch.relu(torch.from_numpy(x) @ torch.from_numpy(w1) + torch.from_numpy(b1)) @ torch.from_numpy(w2) + torch.from_numpy(b2)
    assert actual.shape == (4, 2)
    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.exercise
def test_two_layer_mlp_backprop_against_torch_autograd():
    rng = np.random.default_rng(2030)
    x = rng.normal(size=(6, 3)).astype(np.float64)
    w1 = rng.normal(size=(3, 4)).astype(np.float64)
    b1 = rng.normal(size=(4,)).astype(np.float64) + 0.13
    w2 = rng.normal(size=(4, 2)).astype(np.float64)
    b2 = rng.normal(size=(2,)).astype(np.float64)
    targets = rng.normal(size=(6, 2)).astype(np.float64)

    actual_loss, dw1, db1, dw2, db2 = two_layer_mlp_mse_gradients(
        x, w1, b1, w2, b2, targets
    )
    xt = torch.from_numpy(x)
    wt1 = torch.tensor(w1, requires_grad=True)
    bt1 = torch.tensor(b1, requires_grad=True)
    wt2 = torch.tensor(w2, requires_grad=True)
    bt2 = torch.tensor(b2, requires_grad=True)
    yt = torch.from_numpy(targets)
    pred = torch.relu(xt @ wt1 + bt1) @ wt2 + bt2
    reference_loss = 0.5 * ((pred - yt) ** 2).sum(dim=-1).mean()
    reference_loss.backward()

    assert isinstance(actual_loss, float)
    assert (dw1.shape, db1.shape, dw2.shape, db2.shape) == (
        w1.shape, b1.shape, w2.shape, b2.shape
    )
    np.testing.assert_allclose(actual_loss, reference_loss.item(), rtol=1e-10, atol=1e-10)
    for actual, reference in [(dw1, wt1.grad), (db1, bt1.grad), (dw2, wt2.grad), (db2, bt2.grad)]:
        np.testing.assert_allclose(actual, reference.detach().numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.exercise
def test_mlp_gradient_one_step_reduces_loss():
    rng = np.random.default_rng(37)
    x = rng.normal(size=(8, 2)).astype(np.float64)
    w1 = rng.normal(size=(2, 3)).astype(np.float64)
    b1 = np.array([0.6, -0.5, 0.4], dtype=np.float64)
    w2 = rng.normal(size=(3, 1)).astype(np.float64)
    b2 = np.array([0.1], dtype=np.float64)
    targets = np.array(np.sin(x[:, 0:1]), dtype=np.float64)
    before, dw1, db1, dw2, db2 = two_layer_mlp_mse_gradients(
        x, w1, b1, w2, b2, targets
    )
    eta = 1e-4
    after, _, _, _, _ = two_layer_mlp_mse_gradients(
        x, w1 - eta * dw1, b1 - eta * db1, w2 - eta * dw2, b2 - eta * db2, targets
    )
    assert np.isfinite(before) and np.isfinite(after)
    assert after < before
