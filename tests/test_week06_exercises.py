"""Day 06 tests: 3D sequence linear layers and shared-parameter gradients."""

import numpy as np
import pytest
import torch

from exercises.week06 import linear_3d_numpy, linear_3d_mse_gradients


@pytest.mark.exercise
def test_linear_3d_numpy_matches_torch_and_preserves_inputs():
    rng = np.random.default_rng(6001)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    weight = rng.normal(size=(4, 5)).astype(np.float64)
    bias = rng.normal(size=(5,)).astype(np.float64)
    x_before, weight_before, bias_before = x.copy(), weight.copy(), bias.copy()

    actual = linear_3d_numpy(x, weight, bias)
    reference = torch.from_numpy(x) @ torch.from_numpy(weight) + torch.from_numpy(bias)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == (2, 3, 5)
    np.testing.assert_allclose(actual, reference.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(weight, weight_before)
    np.testing.assert_array_equal(bias, bias_before)


@pytest.mark.exercise
def test_linear_3d_numpy_broadcasts_one_bias_to_every_token():
    x = np.zeros((2, 3, 4), dtype=np.float64)
    weight = np.arange(20, dtype=np.float64).reshape(4, 5)
    bias = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)

    actual = linear_3d_numpy(x, weight, bias)
    expected = np.broadcast_to(bias, (2, 3, 5))

    assert actual.shape == (2, 3, 5)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


@pytest.mark.exercise
def test_linear_3d_mse_gradients_match_torch_autograd():
    rng = np.random.default_rng(6002)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    weight = rng.normal(size=(4, 5)).astype(np.float64)
    bias = rng.normal(size=(5,)).astype(np.float64)
    targets = rng.normal(size=(2, 3, 5)).astype(np.float64)

    loss, grad_weight, grad_bias = linear_3d_mse_gradients(
        x, weight, bias, targets
    )

    xt = torch.from_numpy(x)
    wt = torch.tensor(weight, requires_grad=True)
    bt = torch.tensor(bias, requires_grad=True)
    yt = torch.from_numpy(targets)

    predictions = xt @ wt + bt
    reference_loss = 0.5 * ((predictions - yt) ** 2).sum(dim=-1).mean()
    reference_loss.backward()

    assert isinstance(loss, float)
    assert grad_weight.shape == weight.shape
    assert grad_bias.shape == bias.shape
    np.testing.assert_allclose(loss, reference_loss.item(), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(
        grad_weight, wt.grad.detach().numpy(), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        grad_bias, bt.grad.detach().numpy(), rtol=1e-11, atol=1e-12
    )


@pytest.mark.exercise
def test_linear_3d_mse_loss_sums_features_and_averages_tokens():
    x = np.zeros((2, 2, 3), dtype=np.float64)
    weight = np.zeros((3, 2), dtype=np.float64)
    bias = np.zeros((2,), dtype=np.float64)
    targets = np.ones((2, 2, 2), dtype=np.float64)

    loss, _, _ = linear_3d_mse_gradients(x, weight, bias, targets)

    # Each of the B*T=4 tokens contributes 0.5 * (1^2 + 1^2) = 1.
    # Averaging the four token losses therefore gives exactly 1.
    assert loss == pytest.approx(1.0)


@pytest.mark.exercise
def test_linear_3d_gradient_one_step_reduces_loss():
    rng = np.random.default_rng(6003)
    x = rng.normal(size=(3, 4, 2)).astype(np.float64)
    weight = rng.normal(size=(2, 3)).astype(np.float64)
    bias = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    targets = rng.normal(size=(3, 4, 3)).astype(np.float64)

    before, grad_weight, grad_bias = linear_3d_mse_gradients(
        x, weight, bias, targets
    )

    learning_rate = 1e-3
    after, _, _ = linear_3d_mse_gradients(
        x,
        weight - learning_rate * grad_weight,
        bias - learning_rate * grad_bias,
        targets,
    )

    assert np.isfinite(before) and np.isfinite(after)
    assert after < before
