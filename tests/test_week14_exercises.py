"""Day 14 tests: FFN forward, backward, tensor shapes and reuse."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week03 import finite_difference_gradient
from exercises.week05 import gelu_tanh_numpy
from exercises.week14 import ffn_forward_numpy, ffn_backward_numpy


@pytest.mark.exercise
def test_ffn_forward_matches_torch_gelu_tanh():
    rng = np.random.default_rng(1401)
    x = rng.normal(size=(2, 3, 4))
    w1, b1 = rng.normal(size=(4, 7)), rng.normal(size=7)
    w2, b2 = rng.normal(size=(7, 4)), rng.normal(size=4)

    actual = ffn_forward_numpy(x, w1, b1, w2, b2)
    xt = torch.from_numpy(x)
    expected = (
        F.gelu(xt @ torch.from_numpy(w1) + torch.from_numpy(b1), approximate="tanh")
        @ torch.from_numpy(w2)
        + torch.from_numpy(b2)
    ).numpy()

    assert actual.shape == (2, 3, 4)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


@pytest.mark.exercise
def test_ffn_forward_reuses_day05_gelu():
    rng = np.random.default_rng(1402)
    x = rng.normal(size=(1, 2, 3))
    w1, b1 = rng.normal(size=(3, 5)), rng.normal(size=5)
    w2, b2 = rng.normal(size=(5, 3)), rng.normal(size=3)

    expected = gelu_tanh_numpy(x @ w1 + b1) @ w2 + b2
    np.testing.assert_allclose(
        ffn_forward_numpy(x, w1, b1, w2, b2),
        expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.exercise
def test_ffn_is_independent_across_tokens():
    rng = np.random.default_rng(1403)
    x = rng.normal(size=(2, 4, 6))
    w1, b1 = rng.normal(size=(6, 12)), rng.normal(size=12)
    w2, b2 = rng.normal(size=(12, 6)), rng.normal(size=6)

    original = ffn_forward_numpy(x, w1, b1, w2, b2)
    altered = x.copy()
    altered[0, 2, :] += np.linspace(0.3, 1.2, 6)
    updated = ffn_forward_numpy(altered, w1, b1, w2, b2)

    delta = np.max(np.abs(updated - original), axis=-1)
    assert delta[0, 2] > 1e-8
    delta[0, 2] = 0
    assert np.max(delta) == 0


@pytest.mark.exercise
def test_ffn_forward_preserves_inputs_and_promotes_to_float64():
    rng = np.random.default_rng(1404)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    w1 = rng.normal(size=(4, 8)).astype(np.float32)
    b1 = rng.normal(size=8).astype(np.float32)
    w2 = rng.normal(size=(8, 4)).astype(np.float32)
    b2 = rng.normal(size=4).astype(np.float32)
    inputs = (x, w1, b1, w2, b2)
    copies = tuple(v.copy() for v in inputs)

    result = ffn_forward_numpy(*inputs)
    assert result.shape == x.shape
    assert result.dtype == np.float64
    for value, original in zip(inputs, copies):
        np.testing.assert_array_equal(value, original)


@pytest.mark.exercise
@pytest.mark.parametrize("bad_name", ["x", "w1", "b1", "w2", "b2"])
def test_ffn_forward_rejects_invalid_shapes(bad_name):
    arrays = {
        "x": np.ones((2, 3, 4), dtype=np.float64),
        "w1": np.ones((4, 6), dtype=np.float64),
        "b1": np.zeros(6, dtype=np.float64),
        "w2": np.ones((6, 4), dtype=np.float64),
        "b2": np.zeros(4, dtype=np.float64),
    }
    replacements = {
        "x": np.ones((2, 4), dtype=np.float64),
        "w1": np.ones((5, 6), dtype=np.float64),
        "b1": np.zeros(5, dtype=np.float64),
        "w2": np.ones((6, 5), dtype=np.float64),
        "b2": np.zeros(5, dtype=np.float64),
    }
    arrays[bad_name] = replacements[bad_name]
    with pytest.raises(ValueError):
        ffn_forward_numpy(**arrays)


@pytest.mark.exercise
def test_ffn_backward_matches_torch_autograd():
    rng = np.random.default_rng(1405)
    x = rng.normal(size=(2, 3, 4))
    w1, b1 = rng.normal(size=(4, 5)), rng.normal(size=5)
    w2, b2 = rng.normal(size=(5, 4)), rng.normal(size=4)
    upstream = rng.normal(size=x.shape)

    tx, tw1, tb1, tw2, tb2 = [
        torch.tensor(v, dtype=torch.float64, requires_grad=True)
        for v in (x, w1, b1, w2, b2)
    ]
    y = F.gelu(tx @ tw1 + tb1, approximate="tanh") @ tw2 + tb2
    (y * torch.from_numpy(upstream)).sum().backward()

    gradients = ffn_backward_numpy(upstream, x, w1, b1, w2)
    references = [v.grad.detach().numpy() for v in (tx, tw1, tb1, tw2, tb2)]
    for actual, expected in zip(gradients, references):
        assert actual.shape == expected.shape
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-10)


@pytest.mark.exercise
def test_ffn_backward_matches_finite_differences():
    rng = np.random.default_rng(1406)
    x = rng.normal(size=(1, 2, 2))
    w1, b1 = rng.normal(size=(2, 3)), rng.normal(size=3)
    w2, b2 = rng.normal(size=(3, 2)), rng.normal(size=2)
    upstream = rng.normal(size=x.shape)

    def loss(xv, w1v, b1v, w2v, b2v):
        return float(np.sum(ffn_forward_numpy(xv, w1v, b1v, w2v, b2v) * upstream))

    numerical = [
        finite_difference_gradient(lambda v: loss(v, w1, b1, w2, b2), x, epsilon=1e-6),
        finite_difference_gradient(lambda v: loss(x, v, b1, w2, b2), w1, epsilon=1e-6),
        finite_difference_gradient(lambda v: loss(x, w1, v, w2, b2), b1, epsilon=1e-6),
        finite_difference_gradient(lambda v: loss(x, w1, b1, v, b2), w2, epsilon=1e-6),
        finite_difference_gradient(lambda v: loss(x, w1, b1, w2, v), b2, epsilon=1e-6),
    ]
    analytic = ffn_backward_numpy(upstream, x, w1, b1, w2)
    for actual, expected in zip(analytic, numerical):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)


@pytest.mark.exercise
def test_ffn_backward_preserves_inputs_and_shapes():
    rng = np.random.default_rng(1407)
    x = rng.normal(size=(2, 3, 4))
    w1, b1 = rng.normal(size=(4, 6)), rng.normal(size=6)
    w2 = rng.normal(size=(6, 4))
    upstream = rng.normal(size=x.shape)
    inputs = (upstream, x, w1, b1, w2)
    copies = tuple(v.copy() for v in inputs)

    gradients = ffn_backward_numpy(*inputs)
    shapes = [x.shape, w1.shape, b1.shape, w2.shape, (4,)]
    for actual, expected in zip(gradients, shapes):
        assert actual.shape == expected
        assert actual.dtype == np.float64
    for value, original in zip(inputs, copies):
        np.testing.assert_array_equal(value, original)


@pytest.mark.exercise
def test_zero_output_projection_has_zero_input_gradient():
    rng = np.random.default_rng(1408)
    x = rng.normal(size=(2, 3, 4))
    w1 = rng.normal(size=(4, 6))
    b1 = rng.normal(size=6)
    w2 = np.zeros((6, 4), dtype=np.float64)
    upstream = rng.normal(size=x.shape)

    gx, gw1, gb1, _, _ = ffn_backward_numpy(upstream, x, w1, b1, w2)
    np.testing.assert_array_equal(gx, np.zeros_like(x))
    np.testing.assert_array_equal(gw1, np.zeros_like(w1))
    np.testing.assert_array_equal(gb1, np.zeros_like(b1))


@pytest.mark.exercise
@pytest.mark.parametrize("bad_grad", [
    np.zeros((2, 3, 5), dtype=np.float64),
    np.zeros((2, 3, 4), dtype=np.int64),
])
def test_ffn_backward_rejects_invalid_upstream(bad_grad):
    x = np.ones((2, 3, 4), dtype=np.float64)
    w1, b1 = np.ones((4, 6)), np.zeros(6)
    w2 = np.ones((6, 4))
    with pytest.raises(ValueError):
        ffn_backward_numpy(bad_grad, x, w1, b1, w2)
