"""Day 07 tests: stable softmax and single-head self-attention."""

import numpy as np
import pytest
import torch

from exercises.week07 import (
    scaled_dot_product_attention_numpy,
    self_attention_numpy,
    softmax_numpy,
)


@pytest.mark.exercise
def test_softmax_numpy_known_values_and_row_sums():
    x = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]], dtype=np.float64)
    before = x.copy()

    actual = softmax_numpy(x, axis=-1)
    expected = np.array(
        [
            [0.09003057317038046, 0.24472847105479764, 0.6652409557748218],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )

    assert isinstance(actual, np.ndarray)
    assert actual.shape == x.shape
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(x, before)


@pytest.mark.exercise
def test_softmax_numpy_supports_arbitrary_axis_and_matches_torch():
    rng = np.random.default_rng(7001)
    x = (rng.normal(size=(2, 3, 4)) * 100.0).astype(np.float64)
    before = x.copy()

    for axis in (0, 1, 2, -1):
        actual = softmax_numpy(x, axis=axis)
        expected = torch.softmax(torch.from_numpy(x), dim=axis).numpy()
        assert actual.shape == x.shape
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual.sum(axis=axis), 1.0, atol=1e-12)

    np.testing.assert_array_equal(x, before)


@pytest.mark.exercise
def test_softmax_numpy_is_stable_for_large_positive_and_negative_values():
    x = np.array(
        [[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]],
        dtype=np.float64,
    )
    before = x.copy()

    actual = softmax_numpy(x, axis=-1)
    expected_row = np.array(
        [0.09003057317038046, 0.24472847105479764, 0.6652409557748218]
    )

    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual[0], expected_row, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected_row, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, before)


@pytest.mark.exercise
def test_scaled_dot_product_attention_matches_torch_and_shapes():
    rng = np.random.default_rng(7002)
    q = rng.normal(size=(2, 3, 4)).astype(np.float64)
    k = rng.normal(size=(2, 5, 4)).astype(np.float64)
    v = rng.normal(size=(2, 5, 6)).astype(np.float64)
    q_before, k_before, v_before = q.copy(), k.copy(), v.copy()

    actual_output, actual_weights = scaled_dot_product_attention_numpy(q, k, v)

    qt, kt, vt = map(torch.from_numpy, (q, k, v))
    scores = (qt @ kt.transpose(1, 2)) / np.sqrt(q.shape[-1])
    expected_weights = torch.softmax(scores, dim=-1)
    expected_output = expected_weights @ vt

    assert actual_output.shape == (2, 3, 6)
    assert actual_weights.shape == (2, 3, 5)
    np.testing.assert_allclose(
        actual_weights, expected_weights.numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        actual_output, expected_output.numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(actual_weights.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(q, q_before)
    np.testing.assert_array_equal(k, k_before)
    np.testing.assert_array_equal(v, v_before)


@pytest.mark.exercise
def test_scaled_dot_product_attention_one_key_returns_its_value():
    q = np.array([[[2.0, -1.0], [0.5, 3.0]]], dtype=np.float64)
    k = np.array([[[1.0, 4.0]]], dtype=np.float64)
    v = np.array([[[7.0, -2.0, 5.0]]], dtype=np.float64)

    output, weights = scaled_dot_product_attention_numpy(q, k, v)

    np.testing.assert_allclose(weights, np.ones((1, 2, 1)))
    np.testing.assert_allclose(output, np.broadcast_to(v, (1, 2, 3)))


@pytest.mark.exercise
def test_self_attention_matches_torch_and_supports_different_dv_and_dk():
    rng = np.random.default_rng(7003)
    x = rng.normal(size=(2, 4, 3)).astype(np.float64)
    w_q = rng.normal(size=(3, 2)).astype(np.float64)
    w_k = rng.normal(size=(3, 2)).astype(np.float64)
    w_v = rng.normal(size=(3, 5)).astype(np.float64)
    originals = [a.copy() for a in (x, w_q, w_k, w_v)]

    actual_output, actual_weights = self_attention_numpy(x, w_q, w_k, w_v)

    xt, wqt, wkt, wvt = map(torch.from_numpy, (x, w_q, w_k, w_v))
    q = xt @ wqt
    k = xt @ wkt
    v = xt @ wvt
    scores = (q @ k.transpose(1, 2)) / np.sqrt(q.shape[-1])
    expected_weights = torch.softmax(scores, dim=-1)
    expected_output = expected_weights @ v

    assert actual_output.shape == (2, 4, 5)
    assert actual_weights.shape == (2, 4, 4)
    np.testing.assert_allclose(
        actual_weights, expected_weights.numpy(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        actual_output, expected_output.numpy(), rtol=1e-12, atol=1e-12
    )
    for current, original in zip((x, w_q, w_k, w_v), originals):
        np.testing.assert_array_equal(current, original)


@pytest.mark.exercise
def test_attention_actually_mixes_information_between_tokens():
    # Equal scores give each query a uniform average over the two values.
    q = np.zeros((1, 2, 1), dtype=np.float64)
    k = np.zeros((1, 2, 1), dtype=np.float64)
    v = np.array([[[1.0], [3.0]]], dtype=np.float64)

    output, weights = scaled_dot_product_attention_numpy(q, k, v)

    np.testing.assert_allclose(weights, np.full((1, 2, 2), 0.5))
    np.testing.assert_allclose(output, np.full((1, 2, 1), 2.0))
