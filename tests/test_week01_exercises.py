"""Deliberately failing tests until Week 01 exercises are implemented.

Use: python -m pytest -q -m exercise
"""

import numpy as np
import pytest
import torch

from exercises.week01 import linear_numpy, softmax_numpy, cross_entropy_numpy


@pytest.mark.exercise
def test_linear_numpy_against_torch():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(5, 3))
    w = rng.normal(size=(3, 4))
    b = rng.normal(size=(4,))
    expected = (torch.from_numpy(x) @ torch.from_numpy(w) + torch.from_numpy(b)).numpy()
    np.testing.assert_allclose(linear_numpy(x, w, b), expected, rtol=1e-10, atol=1e-10)


@pytest.mark.exercise
def test_softmax_stays_finite_for_large_logits():
    logits = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
    actual = softmax_numpy(logits)
    expected = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(axis=-1), np.ones(2), atol=1e-12)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.exercise
def test_cross_entropy_against_torch():
    logits = np.array([[1000.0, 1001.0, 999.0], [0.1, -2.0, 1.3]], dtype=np.float64)
    targets = np.array([1, 2], dtype=np.int64)
    actual = cross_entropy_numpy(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        torch.from_numpy(logits), torch.from_numpy(targets)
    ).item()
    assert np.isfinite(actual)
    assert abs(actual - expected) < 1e-10
