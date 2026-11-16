"""Day 13 tests: Residual Add, Linear Residual Block and gradients."""

import numpy as np
import pytest
import torch

from exercises.week03 import finite_difference_gradient
from exercises.week13 import (
    residual_add_backward_numpy,
    residual_add_forward_numpy,
    residual_linear_backward_numpy,
    residual_linear_forward_numpy,
)


@pytest.mark.exercise
def test_residual_add_forward_matches_torch_and_preserves_inputs():
    rng = np.random.default_rng(1301)
    x = rng.normal(size=(2, 3, 5)).astype(np.float64)
    branch = rng.normal(size=x.shape).astype(np.float64)
    x0, branch0 = x.copy(), branch.copy()
    actual = residual_add_forward_numpy(x, branch)
    reference = torch.from_numpy(x) + torch.from_numpy(branch)
    assert actual.shape == x.shape and actual.dtype == np.float64
    np.testing.assert_allclose(actual, reference.numpy(), rtol=0, atol=0)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(branch, branch0)


@pytest.mark.exercise
def test_residual_add_forward_is_identity_for_zero_branch():
    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    np.testing.assert_array_equal(
        residual_add_forward_numpy(x, np.zeros_like(x)), x
    )


@pytest.mark.exercise
@pytest.mark.parametrize("bad_x,bad_z", [
    (np.ones((2, 3)), np.ones((2, 3))),
    (np.ones((2, 3, 4)), np.ones((2, 3, 5))),
    (np.ones((2, 3, 4), dtype=np.int64), np.ones((2, 3, 4))),
])
def test_residual_add_rejects_invalid_inputs(bad_x, bad_z):
    with pytest.raises(ValueError):
        residual_add_forward_numpy(bad_x, bad_z)


@pytest.mark.exercise
def test_residual_add_backward_sends_gradient_to_both_branches():
    upstream = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    before = upstream.copy()
    gx, gz = residual_add_backward_numpy(upstream)
    np.testing.assert_array_equal(gx, upstream)
    np.testing.assert_array_equal(gz, upstream)
    assert gx.dtype == gz.dtype == np.float64
    assert not np.shares_memory(gx, upstream)
    assert not np.shares_memory(gz, upstream)
    assert not np.shares_memory(gx, gz)
    np.testing.assert_array_equal(upstream, before)


@pytest.mark.exercise
def test_residual_add_backward_matches_finite_differences():
    rng = np.random.default_rng(1304)
    x = rng.normal(size=(1, 2, 3)).astype(np.float64)
    z = rng.normal(size=x.shape).astype(np.float64)
    g = rng.normal(size=x.shape).astype(np.float64)
    loss_x = lambda value: float(np.sum(residual_add_forward_numpy(value, z) * g))
    loss_z = lambda value: float(np.sum(residual_add_forward_numpy(x, value) * g))
    nx = finite_difference_gradient(loss_x, x, epsilon=1e-6)
    nz = finite_difference_gradient(loss_z, z, epsilon=1e-6)
    gx, gz = residual_add_backward_numpy(g)
    np.testing.assert_allclose(gx, nx, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(gz, nz, rtol=1e-8, atol=1e-9)


@pytest.mark.exercise
def test_residual_linear_forward_matches_torch_and_preserves_inputs():
    rng = np.random.default_rng(1305)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    w = rng.normal(size=(4, 4)).astype(np.float64)
    b = rng.normal(size=4).astype(np.float64)
    x0, w0, b0 = x.copy(), w.copy(), b.copy()
    actual = residual_linear_forward_numpy(x, w, b)
    xt = torch.from_numpy(x)
    expected = xt + xt @ torch.from_numpy(w) + torch.from_numpy(b)
    assert actual.shape == x.shape and actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, x0)
    np.testing.assert_array_equal(w, w0)
    np.testing.assert_array_equal(b, b0)


@pytest.mark.exercise
def test_residual_linear_backward_matches_torch_autograd():
    rng = np.random.default_rng(1306)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    w = rng.normal(size=(4, 4)).astype(np.float64)
    b = rng.normal(size=4).astype(np.float64)
    g = rng.normal(size=x.shape).astype(np.float64)
    tx = torch.tensor(x, requires_grad=True)
    tw = torch.tensor(w, requires_grad=True)
    tb = torch.tensor(b, requires_grad=True)
    y = tx + tx @ tw + tb
    (y * torch.from_numpy(g)).sum().backward()
    gx, gw, gb = residual_linear_backward_numpy(g, x, w)
    np.testing.assert_allclose(gx, tx.grad.numpy(), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(gw, tw.grad.numpy(), rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(gb, tb.grad.numpy(), rtol=1e-11, atol=1e-12)


@pytest.mark.exercise
def test_residual_linear_backward_matches_finite_differences():
    rng = np.random.default_rng(1307)
    x = rng.normal(size=(1, 2, 3)).astype(np.float64)
    w = rng.normal(size=(3, 3)).astype(np.float64)
    b = rng.normal(size=3).astype(np.float64)
    g = rng.normal(size=x.shape).astype(np.float64)
    loss_x = lambda v: float(np.sum(residual_linear_forward_numpy(v, w, b) * g))
    loss_w = lambda v: float(np.sum(residual_linear_forward_numpy(x, v, b) * g))
    loss_b = lambda v: float(np.sum(residual_linear_forward_numpy(x, w, v) * g))
    nx = finite_difference_gradient(loss_x, x, epsilon=1e-6)
    nw = finite_difference_gradient(loss_w, w, epsilon=1e-6)
    nb = finite_difference_gradient(loss_b, b, epsilon=1e-6)
    gx, gw, gb = residual_linear_backward_numpy(g, x, w)
    np.testing.assert_allclose(gx, nx, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(gw, nw, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(gb, nb, rtol=2e-6, atol=2e-7)


@pytest.mark.exercise
def test_zero_weight_leaves_only_identity_path_in_input_gradient():
    rng = np.random.default_rng(1308)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    w = np.zeros((4, 4), dtype=np.float64)
    g = rng.normal(size=x.shape).astype(np.float64)
    gx, _, _ = residual_linear_backward_numpy(g, x, w)
    np.testing.assert_array_equal(gx, g)


@pytest.mark.exercise
def test_residual_linear_rejects_invalid_shapes():
    x = np.ones((2, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError):
        residual_linear_forward_numpy(x, np.ones((4, 5)), np.zeros(4))
    with pytest.raises(ValueError):
        residual_linear_forward_numpy(x, np.eye(4), np.zeros(5))
    with pytest.raises(ValueError):
        residual_linear_backward_numpy(np.ones((2, 3, 5)), x, np.eye(4))
