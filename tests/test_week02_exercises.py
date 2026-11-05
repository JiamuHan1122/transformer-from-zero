"""Week 02 exercise tests (initial failure is intentional).

Run one task:
    python -m pytest -q -m exercise tests/test_week02_exercises.py::test_scalar_linear_gradients

Run all Week 02 tests:
    python -m pytest -q -m exercise tests/test_week02_exercises.py
"""

import numpy as np
import pytest
import torch

from exercises.week02 import (
    scalar_linear_gradients,
    sgd_step,
    batch_linear_mse_gradients,
)


@pytest.mark.exercise
def test_scalar_linear_gradients():
    # Day 2 paper exercise: x=2, w=1, b=0, y=5.
    loss, grad_w, grad_b = scalar_linear_gradients(2.0, 1.0, 0.0, 5.0)
    np.testing.assert_allclose([loss, grad_w, grad_b], [4.5, -6.0, -3.0], rtol=0, atol=1e-12)


@pytest.mark.exercise
def test_sgd_step_one_update():
    w_new, b_new = sgd_step(1.0, 0.0, -6.0, -3.0, learning_rate=0.1)
    np.testing.assert_allclose([w_new, b_new], [1.6, 0.3], rtol=0, atol=1e-12)
    new_prediction = w_new * 2.0 + b_new
    new_loss = 0.5 * (new_prediction - 5.0) ** 2
    assert new_loss == pytest.approx(1.125, abs=1e-12)


@pytest.mark.exercise
def test_batch_linear_mse_gradients_against_autograd():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(5, 3)).astype(np.float64)
    w = rng.normal(size=(3, 2)).astype(np.float64)
    b = rng.normal(size=(2,)).astype(np.float64)
    y = rng.normal(size=(5, 2)).astype(np.float64)

    actual_loss, actual_dw, actual_db = batch_linear_mse_gradients(x, w, b, y)

    xt = torch.from_numpy(x)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(b, dtype=torch.float64, requires_grad=True)
    yt = torch.from_numpy(y)
    prediction = xt @ wt + bt
    reference_loss = 0.5 * ((prediction - yt) ** 2).sum(dim=-1).mean()
    reference_loss.backward()

    assert isinstance(actual_loss, float)
    assert actual_dw.shape == w.shape
    assert actual_db.shape == b.shape
    np.testing.assert_allclose(actual_loss, reference_loss.item(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_dw, wt.grad.detach().numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_db, bt.grad.detach().numpy(), rtol=1e-10, atol=1e-10)


@pytest.mark.exercise
def test_batch_gradient_descent_reduces_loss():
    # A small step in the negative gradient direction should reduce this loss.
    x = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    w = np.array([[0.2], [-0.1]], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    targets = np.array([[2.0], [1.0], [-1.0]], dtype=np.float64)
    loss_before, dw, db = batch_linear_mse_gradients(x, w, b, targets)
    eta = 0.05
    loss_after, _, _ = batch_linear_mse_gradients(
        x, w - eta * dw, b - eta * db, targets
    )
    assert np.isfinite(loss_before) and np.isfinite(loss_after)
    assert loss_after < loss_before
