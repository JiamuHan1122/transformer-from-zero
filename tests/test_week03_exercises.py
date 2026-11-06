"""Week 03 exercise tests. Failures are expected until you finish Tasks A-C."""

import numpy as np
import pytest

from exercises.week03 import (
    finite_difference_gradient,
    gradient_check_linear_mse,
    train_linear_regression_numpy,
)


@pytest.mark.exercise
def test_finite_difference_gradient_quadratic():
    p = np.array([-1.2, 0.3, 3.0], dtype=np.float64)
    center = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    before = p.copy()

    def loss_fn(v):
        return float(0.5 * np.sum((v - center) ** 2))

    actual = finite_difference_gradient(loss_fn, p)
    np.testing.assert_allclose(actual, p - center, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(p, before)


@pytest.mark.exercise
def test_finite_difference_gradient_matrix():
    p = np.array([[0.1, 0.5, -0.8], [1.0, -0.2, 2.0]], dtype=np.float64)
    actual = finite_difference_gradient(lambda v: float(np.sin(v).sum()), p)
    assert actual.shape == p.shape
    np.testing.assert_allclose(actual, np.cos(p), rtol=1e-6, atol=1e-6)


@pytest.mark.exercise
def test_gradient_check_linear_mse():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(7, 3)).astype(np.float64)
    w = rng.normal(size=(3, 2)).astype(np.float64)
    b = rng.normal(size=(2,)).astype(np.float64)
    targets = rng.normal(size=(7, 2)).astype(np.float64)
    w_before, b_before = w.copy(), b.copy()

    err_w, err_b = gradient_check_linear_mse(x, w, b, targets)
    assert isinstance(err_w, float) and isinstance(err_b, float)
    assert np.isfinite(err_w) and np.isfinite(err_b)
    assert err_w < 1e-7, f"weight gradient relative error too high: {err_w}"
    assert err_b < 1e-7, f"bias gradient relative error too high: {err_b}"
    np.testing.assert_array_equal(w, w_before)
    np.testing.assert_array_equal(b, b_before)


@pytest.mark.exercise
def test_numpy_training_learns_y_equals_3x_plus_2():
    x = np.linspace(-2.0, 2.0, 31, dtype=np.float64).reshape(-1, 1)
    targets = 3.0 * x + 2.0
    initial_w = np.array([[-0.5]], dtype=np.float64)
    initial_b = np.array([0.1], dtype=np.float64)
    w_copy, b_copy = initial_w.copy(), initial_b.copy()

    w, b, history = train_linear_regression_numpy(
        x, targets, initial_w, initial_b, learning_rate=0.1, steps=200
    )

    assert w.shape == (1, 1) and b.shape == (1,)
    assert isinstance(history, np.ndarray) and history.shape == (201,)
    assert np.all(np.isfinite(history))
    assert history[0] > history[-1]
    assert history[-1] < 1e-8
    np.testing.assert_allclose(w, [[3.0]], atol=1e-3)
    np.testing.assert_allclose(b, [2.0], atol=1e-3)
    np.testing.assert_array_equal(initial_w, w_copy)
    np.testing.assert_array_equal(initial_b, b_copy)
