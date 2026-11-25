"""Day 19 tests: reverse depth traversal and tied parameter aggregation."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week16 import _LAYER_KEYS, encoder_stack_forward_numpy
from exercises.week18 import transformer_encoder_block_backward_numpy
from exercises.week19 import (
    encoder_stack_backward_numpy,
    aggregate_tied_parameter_gradients_numpy,
)


def _make_case(seed=1901, depth=3, widths=None, dtype=np.float64):
    rng = np.random.default_rng(seed)
    b, t, d = 2, 4, 8
    x = rng.normal(size=(b, t, d)).astype(dtype)
    upstream = rng.normal(size=(b, t, d)).astype(dtype)
    if widths is None:
        widths = [12 + 4 * l for l in range(depth)]
    layers = []
    for width in widths:
        scale = 1 / np.sqrt(d)
        layers.append({
            "w_q": rng.normal(scale=scale, size=(d, d)).astype(dtype),
            "w_k": rng.normal(scale=scale, size=(d, d)).astype(dtype),
            "w_v": rng.normal(scale=scale, size=(d, d)).astype(dtype),
            "w_o": rng.normal(scale=scale, size=(d, d)).astype(dtype),
            "gamma_attn": rng.uniform(0.85, 1.15, size=d).astype(dtype),
            "beta_attn": rng.normal(scale=.08, size=d).astype(dtype),
            "w1": rng.normal(scale=.18, size=(d, width)).astype(dtype),
            "b1": rng.normal(scale=.06, size=width).astype(dtype),
            "w2": rng.normal(scale=.18, size=(width, d)).astype(dtype),
            "b2": rng.normal(scale=.06, size=d).astype(dtype),
            "gamma_ffn": rng.uniform(.8, 1.2, size=d).astype(dtype),
            "beta_ffn": rng.normal(scale=.07, size=d).astype(dtype),
        })
    return x, layers, upstream


def _torch_gradients(x, layers, upstream, norm_first):
    # Independent autograd reference: no NumPy backward functions are used.
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    lookup = {}
    tensors = []
    for layer in layers:
        p = {}
        for k in _LAYER_KEYS:
            orig = layer[k]
            oid = id(orig)
            if oid not in lookup:
                lookup[oid] = torch.tensor(orig, dtype=torch.float64,
                                            requires_grad=True)
            p[k] = lookup[oid]
        tensors.append(p)
    current = tx
    b, t, d = x.shape
    heads = 2
    dh = d // heads

    def block(inp, p):
        def ln(z, gamma, beta):
            return F.layer_norm(z, (d,), p[gamma], p[beta], eps=1e-5)

        def attn(z):
            q = (z @ p["w_q"]).reshape(b, t, heads, dh).transpose(1, 2)
            k = (z @ p["w_k"]).reshape(b, t, heads, dh).transpose(1, 2)
            v = (z @ p["w_v"]).reshape(b, t, heads, dh).transpose(1, 2)
            a = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(dh), dim=-1)
            merged = (a @ v).transpose(1, 2).contiguous().reshape(b, t, d)
            return merged @ p["w_o"]

        def ffn(z):
            return F.gelu(z @ p["w1"] + p["b1"], approximate="tanh") @ p["w2"] + p["b2"]

        if norm_first:
            h = inp + attn(ln(inp, "gamma_attn", "beta_attn"))
            return h + ffn(ln(h, "gamma_ffn", "beta_ffn"))
        h = ln(inp + attn(inp), "gamma_attn", "beta_attn")
        return ln(h + ffn(h), "gamma_ffn", "beta_ffn")

    hidden = [tx]
    for p in tensors:
        current = block(current, p)
        current.retain_grad()
        hidden.append(current)
    loss = (current * torch.as_tensor(upstream, dtype=torch.float64)).sum()
    loss.backward()
    per_layer = [
        {k: p[k].grad.detach().numpy() for k in _LAYER_KEYS} for p in tensors
    ]
    hidden_grads = np.stack([t.grad.detach().numpy() for t in hidden])
    return tx.grad.detach().numpy(), per_layer, hidden_grads


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_stack_matches_independent_autograd(norm_first, depth):
    x, layers, upstream = _make_case(seed=1900 + depth, depth=depth)
    actual_x, actual_layers, actual_hidden = encoder_stack_backward_numpy(
        upstream, x, layers, num_heads=2, norm_first=norm_first
    )
    ref_x, ref_layers, ref_hidden = _torch_gradients(
        x, layers, upstream, norm_first
    )
    np.testing.assert_allclose(actual_x, ref_x, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(actual_hidden, ref_hidden, atol=1e-9, rtol=1e-9)
    assert len(actual_layers) == depth
    for layer, reference in zip(actual_layers, ref_layers):
        assert tuple(layer) == _LAYER_KEYS
        for key in _LAYER_KEYS:
            np.testing.assert_allclose(layer[key], reference[key],
                                       atol=1e-9, rtol=1e-9)


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_one_layer_matches_day18_exactly(norm_first):
    x, layers, upstream = _make_case(depth=1)
    g, p, history = encoder_stack_backward_numpy(
        upstream, x, layers, 2, norm_first=norm_first
    )
    expected = transformer_encoder_block_backward_numpy(
        upstream, x, num_heads=2, norm_first=norm_first, **layers[0]
    )
    np.testing.assert_allclose(g, expected[0], rtol=0, atol=0)
    for key, value in zip(_LAYER_KEYS, expected[1:]):
        np.testing.assert_allclose(p[0][key], value, rtol=0, atol=0)
    np.testing.assert_array_equal(history[0], g)
    np.testing.assert_array_equal(history[-1], upstream)


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_float32_upcasts_without_modifying_inputs(norm_first):
    x, layers, upstream = _make_case(dtype=np.float32)
    originals = [x.copy(), upstream.copy()] + [
        a.copy() for p in layers for a in p.values()
    ]
    gx, pg, history = encoder_stack_backward_numpy(
        upstream, x, layers, 2, norm_first=norm_first
    )
    assert gx.dtype == np.float64
    assert history.dtype == np.float64
    assert all(a.dtype == np.float64 for p in pg for a in p.values())
    arrays = [x, upstream] + [a for p in layers for a in p.values()]
    for a, original in zip(arrays, originals):
        np.testing.assert_array_equal(a, original)


@pytest.mark.exercise
def test_distinct_equal_arrays_not_tied():
    x, layers, upstream = _make_case(depth=2)
    layers[1]["w_q"] = layers[0]["w_q"].copy()
    _, per, _ = encoder_stack_backward_numpy(upstream, x, layers, 2)
    aggregated = aggregate_tied_parameter_gradients_numpy(layers, per)
    np.testing.assert_array_equal(aggregated[0]["w_q"], per[0]["w_q"])
    np.testing.assert_array_equal(aggregated[1]["w_q"], per[1]["w_q"])


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_reused_entire_dictionary_sums_shared_gradients(norm_first):
    x, base, upstream = _make_case(depth=1)
    layers = [base[0], base[0], base[0]]
    _, contributions, _ = encoder_stack_backward_numpy(
        upstream, x, layers, 2, norm_first=norm_first
    )
    aggregate = aggregate_tied_parameter_gradients_numpy(layers, contributions)
    _, ref, _ = _torch_gradients(x, layers, upstream, norm_first)
    for key in _LAYER_KEYS:
        expected = sum((g[key] for g in contributions))
        np.testing.assert_allclose(aggregate[0][key], expected, atol=1e-10)
        np.testing.assert_allclose(aggregate[1][key], expected, atol=1e-10)
        np.testing.assert_allclose(aggregate[2][key], ref[0][key], atol=1e-9)


@pytest.mark.exercise
def test_partial_parameter_tying_across_dictionaries():
    x, layers, upstream = _make_case(depth=2)
    layers[1]["w_q"] = layers[0]["w_q"]
    layers[1]["gamma_attn"] = layers[0]["gamma_attn"]
    _, contributions, _ = encoder_stack_backward_numpy(
        upstream, x, layers, 2
    )
    aggregate = aggregate_tied_parameter_gradients_numpy(layers, contributions)
    for key in ("w_q", "gamma_attn"):
        expected = contributions[0][key] + contributions[1][key]
        for p in aggregate:
            np.testing.assert_allclose(p[key], expected, atol=1e-10)
    for key in ("w_k", "w_v", "gamma_ffn", "w1"):
        for idx in (0, 1):
            np.testing.assert_array_equal(
                aggregate[idx][key], contributions[idx][key]
            )


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_finite_difference_for_first_and_last_layer(norm_first):
    x, layers, upstream = _make_case(depth=3)
    g_x, p, _ = encoder_stack_backward_numpy(
        upstream, x, layers, 2, norm_first=norm_first
    )
    eps = 1e-6
    for layer_id, name, idx in (
        (0, "w_q", (1, 2)),
        (2, "w2", (1, 3)),
    ):
        def objective(sign):
            changed = [{k: a.copy() for k, a in t.items()} for t in layers]
            changed[layer_id][name][idx] += sign * eps
            y, _, _ = encoder_stack_forward_numpy(
                x, changed, 2, norm_first=norm_first
            )
            return float(np.sum(y * upstream))
        numerical = (objective(1) - objective(-1)) / (2 * eps)
        np.testing.assert_allclose(p[layer_id][name][idx], numerical,
                                   atol=5e-7, rtol=3e-5)


@pytest.mark.exercise
@pytest.mark.parametrize("invalid", ["grad_shape", "grad_type", "grad_dtype", "layers", "heads", "eps", "mode"])
def test_invalid_stack_inputs(invalid):
    x, layers, g = _make_case(depth=2)
    args = dict(grad_output=g, x=x, layers=layers, num_heads=2)
    if invalid == "grad_shape":
        args["grad_output"] = g[..., :-1]
    elif invalid == "grad_type":
        args["grad_output"] = g.tolist()
    elif invalid == "grad_dtype":
        args["grad_output"] = g.astype(np.int64)
    elif invalid == "layers":
        args["layers"] = []
    elif invalid == "heads":
        args["num_heads"] = 3
    elif invalid == "eps":
        args["eps"] = 0
    else:
        args["norm_first"] = "pre"
    with pytest.raises(ValueError):
        encoder_stack_backward_numpy(**args)


@pytest.mark.exercise
@pytest.mark.parametrize("invalid", ["length", "missing", "shape", "dtype"])
def test_invalid_aggregation_inputs(invalid):
    x, layers, upstream = _make_case(depth=2)
    _, per, _ = encoder_stack_backward_numpy(upstream, x, layers, 2)
    if invalid == "length":
        per = per[:-1]
    elif invalid == "missing":
        per[0].pop("w_q")
    elif invalid == "shape":
        per[0]["w_q"] = np.ones((3, 3))
    else:
        per[0]["w_q"] = per[0]["w_q"].astype(np.int64)
    with pytest.raises(ValueError):
        aggregate_tied_parameter_gradients_numpy(layers, per)
