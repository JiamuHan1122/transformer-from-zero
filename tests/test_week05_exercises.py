"""Day 05 tests: numerical stability, activation values and backpropagation."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week03 import finite_difference_gradient
from exercises.week05 import (
    sigmoid_numpy,
    sigmoid_backward_numpy,
    tanh_numpy,
    tanh_backward_numpy,
    gelu_tanh_numpy,
    gelu_tanh_backward_numpy,
    activation_gradient_profiles,
)


@pytest.mark.exercise
def test_sigmoid_numpy_stable_values_and_shape():
    x = np.array([[-1000.0, -20.0, -2.0, 0.0],
                  [1.0, 2.0, 20.0, 1000.0]], dtype=np.float64)
    original = x.copy()
    # No huge exponentials should be computed even in discarded branches.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = sigmoid_numpy(x)
    reference = torch.sigmoid(torch.from_numpy(x)).numpy()
    assert actual.shape == x.shape
    np.testing.assert_array_equal(x, original)
    assert np.all(np.isfinite(actual))
    assert np.all((actual >= 0.0) & (actual <= 1.0))
    np.testing.assert_allclose(actual, reference, atol=1e-15, rtol=1e-14)
    assert actual[0, 0] == 0.0
    assert actual[1, 3] == 1.0


@pytest.mark.exercise
def test_sigmoid_backward_numpy_against_torch():
    x = np.array([[-8.0, -1.5, 0.0, 2.0], [0.6, 4.0, -0.7, 8.0]], dtype=np.float64)
    upstream = np.array([[2.0, -3.0, 4.0, 0.4], [1.2, 5.0, -2.1, 0.3]], dtype=np.float64)
    x_original, upstream_original = x.copy(), upstream.copy()
    tx = torch.tensor(x, requires_grad=True)
    (torch.sigmoid(tx) * torch.from_numpy(upstream)).sum().backward()
    actual = sigmoid_backward_numpy(x, upstream)
    np.testing.assert_allclose(actual, tx.grad.detach().numpy(), rtol=1e-12, atol=1e-13)
    np.testing.assert_array_equal(x, x_original)
    np.testing.assert_array_equal(upstream, upstream_original)


@pytest.mark.exercise
def test_tanh_numpy_forward_against_torch():
    x = np.array([[-1000.0, -3.0, -1.0, 0.0],
                  [0.5, 1.0, 4.0, 1000.0]], dtype=np.float64)
    original = x.copy()
    actual = tanh_numpy(x)
    reference = torch.tanh(torch.from_numpy(x)).numpy()
    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(x, original)


@pytest.mark.exercise
def test_tanh_backward_numpy_against_torch():
    x = np.array([[-6.0, -1.0, 0.0], [0.25, 1.0, 6.0]], dtype=np.float64)
    upstream = np.array([[1.1, -2.0, 3.0], [-4.0, 0.7, 5.0]], dtype=np.float64)
    tx = torch.tensor(x, requires_grad=True)
    (torch.tanh(tx) * torch.from_numpy(upstream)).sum().backward()
    actual = tanh_backward_numpy(x, upstream)
    np.testing.assert_allclose(actual, tx.grad.detach().numpy(), rtol=1e-12, atol=1e-12)


@pytest.mark.exercise
def test_gelu_tanh_numpy_forward_against_torch():
    x = np.linspace(-8.0, 8.0, 65, dtype=np.float64).reshape(5, 13)
    original = x.copy()
    actual = gelu_tanh_numpy(x)
    reference = F.gelu(torch.from_numpy(x), approximate="tanh").numpy()
    assert actual.shape == x.shape
    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(x, original)


@pytest.mark.exercise
def test_gelu_tanh_backward_numpy_against_torch():
    x = np.array([[-5.0, -2.0, -1.0, -0.5, 0.0],
                  [0.1, 0.5, 1.0, 2.0, 5.0]], dtype=np.float64)
    upstream = np.array([[1.0, -2.0, 0.1, -0.8, 4.0],
                         [-2.0, 3.0, -1.0, 0.7, 1.4]], dtype=np.float64)
    original = x.copy()
    tx = torch.tensor(x, requires_grad=True)
    (F.gelu(tx, approximate="tanh") * torch.from_numpy(upstream)).sum().backward()
    actual = gelu_tanh_backward_numpy(x, upstream)
    np.testing.assert_allclose(actual, tx.grad.detach().numpy(), rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(x, original)


@pytest.mark.exercise
def test_activation_gradient_profiles_values_and_reuse():
    x = np.array([-8.0, -1.0, 0.0, 1.0, 8.0], dtype=np.float64)
    original = x.copy()
    profiles = activation_gradient_profiles(x)
    assert isinstance(profiles, dict)
    assert set(profiles) == {"sigmoid", "tanh", "relu", "gelu"}
    for values in profiles.values():
        assert isinstance(values, np.ndarray)
        assert values.shape == x.shape
        assert np.all(np.isfinite(values))
    np.testing.assert_allclose(
        [profiles["sigmoid"][2], profiles["tanh"][2],
         profiles["relu"][2], profiles["gelu"][2]],
        [0.25, 1.0, 0.0, 0.5],
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(profiles["relu"], [0, 0, 0, 1, 1], atol=0, rtol=0)
    np.testing.assert_array_equal(x, original)


@pytest.mark.exercise
def test_gelu_backward_matches_day03_numerical_gradients():
    # Day 03 central differences independently check the Day 05 chain rule.
    x = np.array([[-2.25, -1.1, -0.3, 0.0, 0.4, 1.2, 2.9]], dtype=np.float64)
    upstream = np.array([[0.2, -0.7, 1.0, 2.0, -1.5, 0.9, 0.1]], dtype=np.float64)
    def scalar_loss(z):
        return float(np.sum(gelu_tanh_numpy(z) * upstream))
    numeric = finite_difference_gradient(scalar_loss, x, epsilon=1e-5)
    analytic = gelu_tanh_backward_numpy(x, upstream)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)
