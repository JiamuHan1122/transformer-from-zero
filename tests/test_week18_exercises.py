"""Day 18: independent checks for full Encoder Block backward."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week15 import transformer_encoder_block_forward_numpy
from exercises.week18 import transformer_encoder_block_backward_numpy


# Day 15 forward order: x, Wq, Wk, Wv, Wo, H,
# gamma_attn, beta_attn, W1, b1, W2, b2, gamma_ffn, beta_ffn.
# Gradient order: same array arguments, but no derivative for H.
_ARRAY_POSITIONS = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13)
_ARRAY_NAMES = (
    "x", "w_q", "w_k", "w_v", "w_o",
    "gamma_attn", "beta_attn",
    "w1", "b1", "w2", "b2",
    "gamma_ffn", "beta_ffn",
)


def _case(seed=1801, batch=2, tokens=4, d=8, dff=13, h=2):
    rng = np.random.default_rng(seed)
    scale = 1 / np.sqrt(d)
    x = rng.normal(size=(batch, tokens, d))
    q = rng.normal(scale=scale, size=(d, d))
    k = rng.normal(scale=scale, size=(d, d))
    v = rng.normal(scale=scale, size=(d, d))
    o = rng.normal(scale=scale, size=(d, d))
    gamma_attn = rng.uniform(0.7, 1.3, size=d)
    beta_attn = rng.normal(scale=0.1, size=d)
    w1 = rng.normal(scale=0.18, size=(d, dff))
    b1 = rng.normal(scale=0.08, size=dff)
    w2 = rng.normal(scale=0.18, size=(dff, d))
    b2 = rng.normal(scale=0.08, size=d)
    gamma_ffn = rng.uniform(0.8, 1.2, size=d)
    beta_ffn = rng.normal(scale=0.07, size=d)
    params = (
        x, q, k, v, o, h,
        gamma_attn, beta_attn,
        w1, b1, w2, b2,
        gamma_ffn, beta_ffn,
    )
    upstream = rng.normal(size=(batch, tokens, d))
    return upstream, params


def _torch_reference(upstream, args, norm_first):
    # Independent PyTorch graph: does not call any NumPy backward helper.
    torch_arrays = [
        torch.tensor(args[i], dtype=torch.float64, requires_grad=True)
        for i in _ARRAY_POSITIONS
    ]
    (
        x, wq, wk, wv, wo, ga, ba,
        w1, b1, w2, b2, gf, bf
    ) = torch_arrays
    heads = args[5]
    batch, seq, d = x.shape
    dh = d // heads

    def mha(z):
        q = (z @ wq).reshape(batch, seq, heads, dh).transpose(1, 2)
        k = (z @ wk).reshape(batch, seq, heads, dh).transpose(1, 2)
        v = (z @ wv).reshape(batch, seq, heads, dh).transpose(1, 2)
        score = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
        attention = torch.softmax(score, dim=-1)
        merged = (attention @ v).transpose(1, 2).contiguous()
        return merged.reshape(batch, seq, d) @ wo

    def ln(z, gamma, beta):
        return F.layer_norm(z, (d,), gamma, beta, eps=1e-5)

    def ffn(z):
        hidden = F.gelu(z @ w1 + b1, approximate="tanh")
        return hidden @ w2 + b2

    if norm_first:
        h = x + mha(ln(x, ga, ba))
        y = h + ffn(ln(h, gf, bf))
    else:
        h = ln(x + mha(x), ga, ba)
        y = ln(h + ffn(h), gf, bf)

    loss = torch.sum(y * torch.from_numpy(upstream.astype(np.float64)))
    loss.backward()
    return tuple(a.grad.detach().numpy() for a in torch_arrays)


def _objective(upstream, args, norm_first):
    out, _ = transformer_encoder_block_forward_numpy(
        *args, norm_first=norm_first
    )
    return float(np.sum(out * upstream))


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("seed", [1802, 1803, 1804])
def test_backward_matches_pytorch_reference(norm_first, seed):
    upstream, args = _case(seed=seed)
    actual = transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=norm_first
    )
    expected = _torch_reference(upstream, args, norm_first)
    assert len(actual) == 13
    for name, a, e in zip(_ARRAY_NAMES, actual, expected):
        assert a.dtype == np.float64, name
        np.testing.assert_allclose(
            a, e, rtol=3e-9, atol=2e-10, err_msg=name
        )


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_gradient_shapes_and_finiteness(norm_first):
    upstream, args = _case(seed=1805, batch=3, tokens=5, dff=11)
    results = transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=norm_first
    )
    expected_shapes = tuple(args[i].shape for i in _ARRAY_POSITIONS)
    assert tuple(grad.shape for grad in results) == expected_shapes
    assert all(v.dtype == np.float64 for v in results)
    assert all(np.isfinite(v).all() for v in results)


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_original_arrays_are_unchanged(norm_first):
    upstream, args = _case(seed=1806)
    preserved = [upstream.copy()] + [
        v.copy() if isinstance(v, np.ndarray) else v for v in args
    ]
    transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=norm_first
    )
    np.testing.assert_array_equal(upstream, preserved[0])
    for original, value in zip(preserved[1:], args):
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original)


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_float32_inputs_produce_float64_gradients(norm_first):
    upstream, args = _case(seed=1807)
    upstream32 = upstream.astype(np.float32)
    args32 = tuple(
        v.astype(np.float32) if isinstance(v, np.ndarray) else v
        for v in args
    )
    result = transformer_encoder_block_backward_numpy(
        upstream32, *args32, norm_first=norm_first
    )
    expected = _torch_reference(
        upstream32, args32, norm_first
    )
    for actual, ref in zip(result, expected):
        assert actual.dtype == np.float64
        np.testing.assert_allclose(actual, ref, rtol=3e-9, atol=2e-10)


@pytest.mark.exercise
def test_pre_ln_zero_branches_keep_identity_input_gradient():
    upstream, original_args = _case(seed=1808)
    args = list(original_args)
    args[3] = np.zeros_like(args[3])  # w_v
    args[4] = np.zeros_like(args[4])  # w_o
    args[10] = np.zeros_like(args[10])  # w2
    args[11] = np.zeros_like(args[11])  # b2
    grads = transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=True
    )
    np.testing.assert_allclose(grads[0], upstream, rtol=0, atol=1e-13)


@pytest.mark.exercise
def test_post_ln_zero_branches_keep_layernorm_gradient_path():
    upstream, original_args = _case(seed=1809)
    args = list(original_args)
    args[3] = np.zeros_like(args[3])
    args[4] = np.zeros_like(args[4])
    args[10] = np.zeros_like(args[10])
    args[11] = np.zeros_like(args[11])
    actual = transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=False
    )
    expected = _torch_reference(upstream, args, False)
    np.testing.assert_allclose(
        actual[0], expected[0], rtol=1e-9, atol=1e-10
    )
    assert not np.allclose(actual[0], upstream)


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_nonconstant_gamma_beta_gradients(norm_first):
    upstream, args = _case(seed=1810)
    result = transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=norm_first
    )
    reference = _torch_reference(upstream, args, norm_first)
    for i in (5, 6, 11, 12):
        np.testing.assert_allclose(
            result[i], reference[i], atol=1e-10, rtol=1e-9
        )


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("gradient_index,coord", [
    (0, (0, 1, 2)),       # x
    (1, (1, 2)),          # Wq
    (2, (3, 4)),          # Wk
    (3, (0, 1)),          # Wv
    (4, (2, 5)),          # Wo
    (5, (1,)),            # gamma_attn
    (6, (3,)),            # beta_attn
    (7, (1, 4)),          # FFN W1
    (8, (4,)),            # FFN b1
    (9, (5, 2)),          # FFN W2
    (10, (5,)),           # FFN b2
    (11, (2,)),           # gamma_ffn
    (12, (4,)),           # beta_ffn
])
def test_central_finite_difference(norm_first, gradient_index, coord):
    upstream, args = _case(seed=1811, batch=1, tokens=3)
    analytic = transformer_encoder_block_backward_numpy(
        upstream, *args, norm_first=norm_first
    )[gradient_index][coord]
    source_pos = _ARRAY_POSITIONS[gradient_index]
    epsilon = 1e-6

    def value(delta):
        changed = list(args)
        changed[source_pos] = args[source_pos].copy()
        changed[source_pos][coord] += delta
        return _objective(upstream, changed, norm_first)

    numerical = (value(epsilon) - value(-epsilon)) / (2 * epsilon)
    np.testing.assert_allclose(
        analytic, numerical, rtol=5e-5, atol=3e-7
    )


@pytest.mark.exercise
@pytest.mark.parametrize("field", [
    "grad_shape",
    "grad_type",
    "grad_dtype",
    "heads",
    "gamma_shape",
    "ffn_shape",
    "eps",
    "norm_first",
])
def test_invalid_inputs_rejected(field):
    upstream, args = _case(seed=1812)
    args = list(args)
    if field == "grad_shape":
        upstream = np.ones((2, 4, 7))
    elif field == "grad_type":
        upstream = upstream.tolist()
    elif field == "grad_dtype":
        upstream = upstream.astype(np.int64)
    elif field == "heads":
        args[5] = 3
    elif field == "gamma_shape":
        args[6] = np.ones(7)
    elif field == "ffn_shape":
        args[10] = np.ones((12, 8))
    elif field == "eps":
        with pytest.raises(ValueError):
            transformer_encoder_block_backward_numpy(
                upstream, *args, eps=0
            )
        return
    elif field == "norm_first":
        with pytest.raises(ValueError):
            transformer_encoder_block_backward_numpy(
                upstream, *args, norm_first="pre"
            )
        return
    with pytest.raises(ValueError):
        transformer_encoder_block_backward_numpy(upstream, *args)


@pytest.mark.exercise
def test_encoder_backward_has_exact_documented_gradient_order():
    upstream, args = _case(seed=1813)
    gradients = transformer_encoder_block_backward_numpy(upstream, *args)
    assert len(gradients) == len(_ARRAY_NAMES)
    ref = _torch_reference(upstream, args, True)
    for name, actual, expected in zip(_ARRAY_NAMES, gradients, ref):
        np.testing.assert_allclose(
            actual, expected, atol=1e-10, rtol=1e-9,
            err_msg=f"incorrect ordering or derivative for {name}",
        )
