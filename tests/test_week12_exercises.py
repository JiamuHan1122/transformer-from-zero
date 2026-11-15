"""Day 12 tests: LayerNorm forward, manual backward and invariants."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week03 import finite_difference_gradient
from exercises.week12 import (
    layer_norm_backward_numpy,
    layer_norm_forward_numpy,
)


EPS = 1e-5


@pytest.mark.exercise
def test_layer_norm_forward_matches_torch_reference():
    rng = np.random.default_rng(1201)
    x = rng.normal(size=(2, 3, 5)).astype(np.float64)
    gamma = rng.normal(size=(5,)).astype(np.float64)
    beta = rng.normal(size=(5,)).astype(np.float64)

    actual = layer_norm_forward_numpy(x, gamma, beta, eps=EPS)
    reference = F.layer_norm(
        torch.from_numpy(x),
        normalized_shape=(5,),
        weight=torch.from_numpy(gamma),
        bias=torch.from_numpy(beta),
        eps=EPS,
    ).numpy()

    assert actual.shape == x.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, reference, rtol=1e-11, atol=1e-12)


@pytest.mark.exercise
def test_layer_norm_normalizes_each_token_over_feature_axis():
    rng = np.random.default_rng(1202)
    x = rng.normal(size=(2, 4, 8)).astype(np.float64)
    gamma = np.ones(8, dtype=np.float64)
    beta = np.zeros(8, dtype=np.float64)

    normalized = layer_norm_forward_numpy(x, gamma, beta, eps=1e-8)

    means = np.mean(normalized, axis=-1)
    variances = np.var(normalized, axis=-1)

    assert np.max(np.abs(means)) < 1e-12
    assert np.max(np.abs(variances - 1.0)) < 1e-6


@pytest.mark.exercise
def test_layer_norm_affine_parameters_broadcast_over_batch_and_tokens():
    rng = np.random.default_rng(1203)
    x = rng.normal(size=(3, 2, 4)).astype(np.float64)
    gamma = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    beta = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float64)

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + EPS)
    expected = x_hat * gamma + beta

    actual = layer_norm_forward_numpy(x, gamma, beta, eps=EPS)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.exercise
def test_layer_norm_does_not_mix_different_tokens():
    rng = np.random.default_rng(1204)
    x = rng.normal(size=(2, 4, 6)).astype(np.float64)
    gamma = rng.normal(size=(6,)).astype(np.float64)
    beta = rng.normal(size=(6,)).astype(np.float64)

    original_output = layer_norm_forward_numpy(x, gamma, beta)
    changed_x = x.copy()
    changed_x[0, 2, :] += np.linspace(0.5, 2.0, 6)
    changed_output = layer_norm_forward_numpy(changed_x, gamma, beta)

    difference = np.max(np.abs(changed_output - original_output), axis=-1)
    assert difference[0, 2] > 1e-8
    difference[0, 2] = 0.0
    assert np.max(difference) == 0.0


@pytest.mark.exercise
def test_layer_norm_forward_preserves_all_inputs():
    rng = np.random.default_rng(1205)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    gamma = rng.normal(size=(4,)).astype(np.float64)
    beta = rng.normal(size=(4,)).astype(np.float64)
    before = (x.copy(), gamma.copy(), beta.copy())

    layer_norm_forward_numpy(x, gamma, beta)

    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(gamma, before[1])
    np.testing.assert_array_equal(beta, before[2])


@pytest.mark.exercise
@pytest.mark.parametrize(
    "bad_x,bad_gamma,bad_beta",
    [
        (np.ones((2, 4)), np.ones(4), np.zeros(4)),
        (np.ones((2, 3, 4)), np.ones(3), np.zeros(4)),
        (np.ones((2, 3, 4)), np.ones(4), np.zeros(3)),
        (np.ones((2, 3, 4), dtype=np.int64), np.ones(4), np.zeros(4)),
    ],
)
def test_layer_norm_forward_rejects_invalid_shapes_and_dtypes(
    bad_x, bad_gamma, bad_beta
):
    with pytest.raises(ValueError):
        layer_norm_forward_numpy(bad_x, bad_gamma, bad_beta)


@pytest.mark.exercise
@pytest.mark.parametrize("bad_eps", [0.0, -1.0, True, float("nan"), float("inf")])
def test_layer_norm_rejects_invalid_epsilon(bad_eps):
    x = np.ones((2, 3, 4), dtype=np.float64)
    gamma = np.ones(4, dtype=np.float64)
    beta = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError):
        layer_norm_forward_numpy(x, gamma, beta, eps=bad_eps)


@pytest.mark.exercise
def test_layer_norm_backward_matches_torch_autograd():
    rng = np.random.default_rng(1206)
    x = rng.normal(size=(2, 3, 5)).astype(np.float64)
    gamma = rng.normal(size=(5,)).astype(np.float64)
    beta = rng.normal(size=(5,)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)

    tx = torch.tensor(x, requires_grad=True)
    tgamma = torch.tensor(gamma, requires_grad=True)
    tbeta = torch.tensor(beta, requires_grad=True)
    ty = F.layer_norm(tx, (5,), tgamma, tbeta, eps=EPS)
    (ty * torch.from_numpy(upstream)).sum().backward()

    grad_x, grad_gamma, grad_beta = layer_norm_backward_numpy(
        upstream, x, gamma, eps=EPS
    )

    np.testing.assert_allclose(
        grad_x, tx.grad.detach().numpy(), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        grad_gamma, tgamma.grad.detach().numpy(), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        grad_beta, tbeta.grad.detach().numpy(), rtol=1e-10, atol=1e-11
    )


@pytest.mark.exercise
def test_layer_norm_backward_matches_finite_differences():
    rng = np.random.default_rng(1207)
    x = rng.normal(size=(1, 2, 3)).astype(np.float64)
    gamma = rng.normal(size=(3,)).astype(np.float64)
    beta = rng.normal(size=(3,)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)

    def loss_x(value):
        return float(np.sum(layer_norm_forward_numpy(value, gamma, beta) * upstream))

    def loss_gamma(value):
        return float(np.sum(layer_norm_forward_numpy(x, value, beta) * upstream))

    def loss_beta(value):
        return float(np.sum(layer_norm_forward_numpy(x, gamma, value) * upstream))

    numeric_x = finite_difference_gradient(loss_x, x, epsilon=1e-6)
    numeric_gamma = finite_difference_gradient(loss_gamma, gamma, epsilon=1e-6)
    numeric_beta = finite_difference_gradient(loss_beta, beta, epsilon=1e-6)
    grad_x, grad_gamma, grad_beta = layer_norm_backward_numpy(
        upstream, x, gamma
    )

    np.testing.assert_allclose(grad_x, numeric_x, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(grad_gamma, numeric_gamma, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(grad_beta, numeric_beta, rtol=2e-6, atol=2e-7)


@pytest.mark.exercise
def test_layer_norm_backward_preserves_inputs_and_gradient_shape():
    rng = np.random.default_rng(1208)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    gamma = rng.normal(size=(4,)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    before = (x.copy(), gamma.copy(), upstream.copy())

    grad_x, grad_gamma, grad_beta = layer_norm_backward_numpy(
        upstream, x, gamma
    )

    assert grad_x.shape == x.shape
    assert grad_gamma.shape == gamma.shape
    assert grad_beta.shape == gamma.shape
    np.testing.assert_array_equal(x, before[0])
    np.testing.assert_array_equal(gamma, before[1])
    np.testing.assert_array_equal(upstream, before[2])
