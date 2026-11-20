"""Day 17: verify manual Softmax and Multi-Head Attention backward."""

import numpy as np
import pytest
import torch

from exercises.week07 import softmax_numpy
from exercises.week08 import multi_head_attention_numpy
from exercises.week17 import (
    multi_head_attention_backward_numpy,
    softmax_backward_numpy,
)


def _make_case(seed=1701, b=2, t=4, d=6, heads=3, dk=2, dv=4, d_out=5):
    rng = np.random.default_rng(seed)
    scale = 0.24
    x = rng.normal(scale=0.6, size=(b, t, d)).astype(np.float64)
    w_q = rng.normal(scale=scale, size=(d, heads * dk)).astype(np.float64)
    w_k = rng.normal(scale=scale, size=(d, heads * dk)).astype(np.float64)
    w_v = rng.normal(scale=scale, size=(d, heads * dv)).astype(np.float64)
    w_o = rng.normal(scale=scale, size=(heads * dv, d_out)).astype(np.float64)
    grad_output = rng.normal(size=(b, t, d_out)).astype(np.float64)
    return grad_output, x, w_q, w_k, w_v, w_o, heads


def _torch_reference(grad_output, x, w_q, w_k, w_v, w_o, heads):
    arrays = [
        torch.tensor(arr, dtype=torch.float64, requires_grad=True)
        for arr in (x, w_q, w_k, w_v, w_o)
    ]
    xt, qt, kt, vt, ot = arrays
    b, t, _ = x.shape
    dk = w_q.shape[1] // heads
    dv = w_v.shape[1] // heads

    def split(z, head_dim):
        return z.reshape(b, t, heads, head_dim).transpose(1, 2)

    qh = split(xt @ qt, dk)
    kh = split(xt @ kt, dk)
    vh = split(xt @ vt, dv)
    scores = (qh @ kh.transpose(-1, -2)) / np.sqrt(dk)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ vh
    merged = context.transpose(1, 2).contiguous().reshape(b, t, heads * dv)
    output = merged @ ot
    loss = (output * torch.from_numpy(grad_output)).sum()
    loss.backward()
    return tuple(tensor.grad.detach().numpy() for tensor in arrays)


@pytest.mark.exercise
def test_softmax_backward_matches_torch_autograd():
    rng = np.random.default_rng(1710)
    logits = rng.normal(size=(2, 3, 4, 5))
    upstream = rng.normal(size=logits.shape)
    weights = softmax_numpy(logits, axis=-1)
    actual = softmax_backward_numpy(weights, upstream)

    z = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    a = torch.softmax(z, dim=-1)
    (a * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(actual, z.grad.numpy(), rtol=1e-12, atol=1e-12)
    assert actual.dtype == np.float64


@pytest.mark.exercise
@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 4, 5)])
def test_softmax_backward_rows_sum_to_zero(shape):
    rng = np.random.default_rng(1711)
    weights = softmax_numpy(rng.normal(size=shape), axis=-1)
    upstream = rng.normal(size=shape)
    grads = softmax_backward_numpy(weights, upstream)
    np.testing.assert_allclose(
        grads.sum(axis=-1), np.zeros(shape[:-1]), rtol=0, atol=1e-13
    )


@pytest.mark.exercise
def test_softmax_backward_uniform_upstream_has_zero_score_gradient():
    rng = np.random.default_rng(1712)
    weights = softmax_numpy(rng.normal(size=(3, 6)), axis=-1)
    grad_w = np.full_like(weights, 3.5)
    grad_s = softmax_backward_numpy(weights, grad_w)
    np.testing.assert_allclose(grad_s, np.zeros_like(grad_s), atol=1e-14)


@pytest.mark.exercise
def test_softmax_backward_float32_inputs_give_float64_result():
    z = np.array([[-1., 1., 2.]], dtype=np.float32)
    w = softmax_numpy(z).astype(np.float32)
    g = np.array([[.2, -.3, .7]], dtype=np.float32)
    w_old, g_old = w.copy(), g.copy()
    result = softmax_backward_numpy(w, g)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(w, w_old)
    np.testing.assert_array_equal(g, g_old)


@pytest.mark.exercise
@pytest.mark.parametrize("kind", ["shape", "type", "dtype", "empty"])
def test_softmax_backward_rejects_invalid_arrays(kind):
    w = np.array([[0.5, 0.5]], dtype=np.float64)
    g = np.ones_like(w)
    if kind == "shape":
        g = np.ones((1, 3), dtype=np.float64)
    elif kind == "type":
        g = [[1, 1]]
    elif kind == "dtype":
        g = np.ones((1, 2), dtype=np.int64)
    elif kind == "empty":
        g = np.empty((0, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        softmax_backward_numpy(w, g)


@pytest.mark.exercise
@pytest.mark.parametrize("heads,dk,dv,d_out", [
    (1, 4, 4, 6), (2, 3, 2, 6), (3, 2, 4, 5),
])
def test_multi_head_backward_matches_torch(heads, dk, dv, d_out):
    args = _make_case(
        seed=1720 + heads, d=6, heads=heads, dk=dk, dv=dv, d_out=d_out
    )
    actual = multi_head_attention_backward_numpy(*args)
    expected = _torch_reference(*args)
    for a, e in zip(actual, expected):
        assert a.dtype == np.float64
        np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-11)


@pytest.mark.exercise
def test_backward_gradient_shapes_and_return_order():
    grad, x, wq, wk, wv, wo, heads = _make_case(seed=1724)
    result = multi_head_attention_backward_numpy(
        grad, x, wq, wk, wv, wo, heads
    )
    assert len(result) == 5
    for a, source in zip(result, (x, wq, wk, wv, wo)):
        assert a.shape == source.shape
        assert a.dtype == np.float64
        assert np.isfinite(a).all()


@pytest.mark.exercise
def test_backward_consistent_with_forward_scalar_objective():
    args = _make_case(seed=1725)
    grad, x, wq, wk, wv, wo, heads = args
    output, attn = multi_head_attention_numpy(x, wq, wk, wv, wo, heads)
    assert attn.shape == (2, heads, 4, 4)
    assert np.isfinite(np.sum(output * grad))
    result = multi_head_attention_backward_numpy(*args)
    ref = _torch_reference(*args)
    for a, b in zip(result, ref):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)


@pytest.mark.exercise
@pytest.mark.parametrize("parameter_index,element_index", [
    (1, (0, 1, 2)),
    (2, (0, 0)),
    (3, (1, 2)),
    (4, (3, 5)),
    (5, (2, 1)),
])
def test_finite_difference_selected_gradient(parameter_index, element_index):
    # Index 1 refers to x; indices 2..5 refer to Wq, Wk, Wv, Wo.
    args = _make_case(seed=1731)
    upstream, x, wq, wk, wv, wo, heads = args
    arrs = [x, wq, wk, wv, wo]
    analytic = multi_head_attention_backward_numpy(*args)
    original = arrs[parameter_index - 1]

    eps = 1e-6
    def scalar_at(delta):
        cloned = [arr.copy() for arr in arrs]
        cloned[parameter_index - 1][element_index] += delta
        out, _ = multi_head_attention_numpy(*cloned, heads)
        return float(np.sum(out * upstream))

    numerical = (scalar_at(eps) - scalar_at(-eps)) / (2 * eps)
    actual = analytic[parameter_index - 1][element_index]
    np.testing.assert_allclose(actual, numerical, rtol=4e-5, atol=1e-7)
    # Also check original inputs were not modified by finite differences.
    np.testing.assert_array_equal(
        original, arrs[parameter_index - 1]
    )


@pytest.mark.exercise
def test_input_gradient_accumulates_q_k_and_v_paths():
    args = _make_case(seed=1735)
    grad_x = multi_head_attention_backward_numpy(*args)[0]
    expected_x = _torch_reference(*args)[0]
    np.testing.assert_allclose(grad_x, expected_x, rtol=1e-11, atol=1e-12)
    # An ablation to a zero V projection should block gradient through all
    # paths to the input for this bias-free attention.
    upstream, x, wq, wk, wv, wo, heads = args
    zero_v = np.zeros_like(wv)
    actual = multi_head_attention_backward_numpy(
        upstream, x, wq, wk, zero_v, wo, heads
    )
    np.testing.assert_allclose(actual[0], np.zeros_like(x), atol=1e-14)


@pytest.mark.exercise
def test_multi_head_backward_no_mutation_float32_inputs():
    args = list(_make_case(seed=1736))
    args[:6] = [a.astype(np.float32) for a in args[:6]]
    preserved = [a.copy() for a in args[:6]]
    result = multi_head_attention_backward_numpy(*args)
    for original, current in zip(preserved, args[:6]):
        np.testing.assert_array_equal(original, current)
    assert all(value.dtype == np.float64 for value in result)


@pytest.mark.exercise
@pytest.mark.parametrize("invalid", ["heads", "qshape", "kshape", "vshape", "oshape", "gradshape", "dtype"])
def test_multi_head_backward_rejects_bad_inputs(invalid):
    args = list(_make_case(seed=1737))
    if invalid == "heads":
        args[-1] = 4
    elif invalid == "qshape":
        args[2] = np.ones((5, 6))
    elif invalid == "kshape":
        args[3] = np.ones((6, 9))
    elif invalid == "vshape":
        args[4] = np.ones((6, 11))
    elif invalid == "oshape":
        args[5] = np.ones((6, 5))
    elif invalid == "gradshape":
        args[0] = np.ones((2, 4, 6))
    elif invalid == "dtype":
        args[1] = args[1].astype(np.int64)
    with pytest.raises(ValueError):
        multi_head_attention_backward_numpy(*args)
