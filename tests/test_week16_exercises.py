"""Day 16 tests: independently parameterized multi-layer Encoder."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week15 import transformer_encoder_block_forward_numpy
from exercises.week16 import encoder_stack_forward_numpy


def _make_layers(seed=1616, depth=3, d_model=8, d_ffs=None):
    rng = np.random.default_rng(seed)
    if d_ffs is None:
        d_ffs = [16] * depth
    assert len(d_ffs) == depth
    scale = 1.0 / np.sqrt(d_model)
    layers = []
    for d_ff in d_ffs:
        layers.append(
            {
                "w_q": rng.normal(scale=scale, size=(d_model, d_model)),
                "w_k": rng.normal(scale=scale, size=(d_model, d_model)),
                "w_v": rng.normal(scale=scale, size=(d_model, d_model)),
                "w_o": rng.normal(scale=scale, size=(d_model, d_model)),
                "gamma_attn": np.ones(d_model, dtype=np.float64),
                "beta_attn": np.zeros(d_model, dtype=np.float64),
                "w1": rng.normal(scale=0.20, size=(d_model, d_ff)),
                "b1": rng.normal(scale=0.05, size=(d_ff,)),
                "w2": rng.normal(scale=0.20, size=(d_ff, d_model)),
                "b2": rng.normal(scale=0.05, size=(d_model,)),
                "gamma_ffn": np.ones(d_model, dtype=np.float64),
                "beta_ffn": np.zeros(d_model, dtype=np.float64),
            }
        )
    return layers


def _torch_block(x, params, num_heads, norm_first, eps=1e-5):
    d_model = x.shape[-1]
    batch_size, seq_len, _ = x.shape
    head_dim = d_model // num_heads
    p = {
        key: torch.from_numpy(value.astype(np.float64))
        for key, value in params.items()
    }

    def ln(z, g, b):
        return F.layer_norm(z, (d_model,), p[g], p[b], eps=eps)

    def attention(z):
        q = (z @ p["w_q"]).reshape(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        k = (z @ p["w_k"]).reshape(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        v = (z @ p["w_v"]).reshape(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        weights = torch.softmax(
            (q @ k.transpose(-2, -1)) / np.sqrt(head_dim),
            dim=-1,
        )
        context = (weights @ v).transpose(1, 2).contiguous().reshape(
            batch_size, seq_len, d_model
        )
        return context @ p["w_o"], weights

    def ffn(z):
        return (
            F.gelu(z @ p["w1"] + p["b1"], approximate="tanh")
            @ p["w2"] + p["b2"]
        )

    if norm_first:
        attn_out, weights = attention(ln(x, "gamma_attn", "beta_attn"))
        h = x + attn_out
        output = h + ffn(ln(h, "gamma_ffn", "beta_ffn"))
    else:
        attn_out, weights = attention(x)
        h = ln(x + attn_out, "gamma_attn", "beta_attn")
        output = ln(h + ffn(h), "gamma_ffn", "beta_ffn")
    return output, weights


def _torch_stack(x, layers, num_heads, norm_first):
    current = torch.from_numpy(x.astype(np.float64))
    states = [current]
    weights = []
    for params in layers:
        current, w = _torch_block(
            current, params, num_heads, norm_first
        )
        states.append(current)
        weights.append(w)
    return (
        current.numpy(),
        torch.stack(weights).numpy(),
        torch.stack(states).numpy(),
    )


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_depth_three_matches_torch_reference(norm_first):
    rng = np.random.default_rng(1601)
    x = rng.normal(size=(2, 4, 8))
    layers = _make_layers(seed=1602, depth=3)

    actual = encoder_stack_forward_numpy(
        x, layers, num_heads=2, norm_first=norm_first
    )
    expected = _torch_stack(x, layers, num_heads=2, norm_first=norm_first)

    for value, ref in zip(actual, expected):
        np.testing.assert_allclose(value, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.exercise
def test_depth_one_exactly_matches_day15():
    rng = np.random.default_rng(1603)
    x = rng.normal(size=(2, 5, 8))
    layer = _make_layers(seed=1604, depth=1)[0]

    final, weights, states = encoder_stack_forward_numpy(
        x, [layer], num_heads=2
    )
    ref_out, ref_weights = transformer_encoder_block_forward_numpy(
        x, num_heads=2, **layer
    )

    np.testing.assert_array_equal(final, ref_out)
    np.testing.assert_array_equal(weights[0], ref_weights)
    np.testing.assert_array_equal(states[0], x)
    np.testing.assert_array_equal(states[1], final)


@pytest.mark.exercise
def test_shapes_and_depth_axis_are_preserved():
    rng = np.random.default_rng(1605)
    x = rng.normal(size=(3, 6, 8))
    layers = _make_layers(seed=1606, depth=4)
    final, attention, states = encoder_stack_forward_numpy(
        x, layers, num_heads=2
    )
    assert final.shape == (3, 6, 8)
    assert attention.shape == (4, 3, 2, 6, 6)
    assert states.shape == (5, 3, 6, 8)
    assert final.dtype == attention.dtype == states.dtype == np.float64
    np.testing.assert_array_equal(states[0], x)
    np.testing.assert_array_equal(states[-1], final)


@pytest.mark.exercise
def test_next_layer_receives_previous_layer_output():
    rng = np.random.default_rng(1607)
    x = rng.normal(size=(2, 3, 8))
    layer0, layer1 = _make_layers(seed=1608, depth=2)
    one, _, _ = encoder_stack_forward_numpy(x, [layer0], num_heads=2)
    two, _, states = encoder_stack_forward_numpy(
        x, [layer0, layer1], num_heads=2
    )
    ref, _ = transformer_encoder_block_forward_numpy(
        one, num_heads=2, **layer1
    )
    np.testing.assert_allclose(two, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(states[1], one)


@pytest.mark.exercise
def test_layer_specific_parameters_change_result():
    rng = np.random.default_rng(1609)
    x = rng.normal(size=(2, 3, 8))
    layers = _make_layers(seed=1610, depth=3)
    original, _, states_before = encoder_stack_forward_numpy(
        x, layers, num_heads=2
    )
    modified = [{k: v.copy() for k, v in layer.items()} for layer in layers]
    modified[1]["w_o"] *= 1.2
    changed, _, states_after = encoder_stack_forward_numpy(
        x, modified, num_heads=2
    )
    np.testing.assert_array_equal(states_before[:2], states_after[:2])
    assert np.linalg.norm(changed - original) > 1e-9


@pytest.mark.exercise
def test_attention_weights_are_separate_for_each_layer():
    rng = np.random.default_rng(1611)
    x = rng.normal(size=(2, 4, 8))
    layers = _make_layers(seed=1612, depth=3)
    _, weights, _ = encoder_stack_forward_numpy(x, layers, num_heads=2)

    np.testing.assert_allclose(
        weights.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12
    )
    assert not np.allclose(weights[0], weights[1])


@pytest.mark.exercise
def test_layers_may_use_different_hidden_ffn_widths():
    rng = np.random.default_rng(1613)
    x = rng.normal(size=(1, 3, 8))
    layers = _make_layers(
        seed=1614, depth=3, d_ffs=[12, 20, 24]
    )
    final, attn, states = encoder_stack_forward_numpy(
        x, layers, num_heads=2
    )
    ref = _torch_stack(x, layers, num_heads=2, norm_first=True)
    np.testing.assert_allclose(final, ref[0], rtol=1e-10, atol=1e-11)
    assert attn.shape == (3, 1, 2, 3, 3)
    assert states.shape == (4, 1, 3, 8)


@pytest.mark.exercise
def test_shared_dictionary_means_tied_weights():
    rng = np.random.default_rng(1615)
    x = rng.normal(size=(1, 4, 8))
    layer = _make_layers(seed=1616, depth=1)[0]
    final, _, states = encoder_stack_forward_numpy(
        x, [layer, layer], num_heads=2
    )
    ref, _ = transformer_encoder_block_forward_numpy(
        states[1], num_heads=2, **layer
    )
    np.testing.assert_allclose(final, ref, rtol=0.0, atol=0.0)


@pytest.mark.exercise
def test_input_and_all_parameter_arrays_are_unchanged():
    rng = np.random.default_rng(1617)
    x = rng.normal(size=(2, 4, 8)).astype(np.float32)
    layers = _make_layers(seed=1618, depth=2)
    x_before = x.copy()
    before = [
        {key: value.copy() for key, value in layer.items()}
        for layer in layers
    ]
    final, attn, states = encoder_stack_forward_numpy(
        x, layers, num_heads=2
    )
    assert final.dtype == attn.dtype == states.dtype == np.float64
    np.testing.assert_array_equal(x, x_before)
    for current, original in zip(layers, before):
        for key in current:
            np.testing.assert_array_equal(current[key], original[key])


@pytest.mark.exercise
def test_pre_ln_zero_branches_preserve_input_at_every_depth():
    rng = np.random.default_rng(1619)
    x = rng.normal(size=(2, 4, 8))
    layers = _make_layers(seed=1620, depth=3)
    for layer in layers:
        layer["w_o"][:] = 0.0
        layer["w2"][:] = 0.0
        layer["b2"][:] = 0.0
    final, _, states = encoder_stack_forward_numpy(
        x, layers, num_heads=2, norm_first=True
    )
    np.testing.assert_array_equal(final, x)
    for layer_state in states:
        np.testing.assert_array_equal(layer_state, x)


@pytest.mark.exercise
def test_post_ln_zero_branches_do_not_give_identity():
    rng = np.random.default_rng(1621)
    x = rng.normal(size=(2, 4, 8))
    layers = _make_layers(seed=1622, depth=2)
    for layer in layers:
        layer["w_o"][:] = 0.0
        layer["w2"][:] = 0.0
        layer["b2"][:] = 0.0
    final, _, _ = encoder_stack_forward_numpy(
        x, layers, num_heads=2, norm_first=False
    )
    assert not np.allclose(final, x)


@pytest.mark.exercise
def test_perturbed_token_can_influence_other_tokens_at_depth():
    rng = np.random.default_rng(1623)
    x = rng.normal(size=(1, 5, 8))
    layers = _make_layers(seed=1624, depth=3)
    _, _, original_states = encoder_stack_forward_numpy(
        x, layers, num_heads=2
    )
    changed_x = x.copy()
    changed_x[0, 2, :] += np.linspace(0.4, 1.1, 8)
    _, _, changed_states = encoder_stack_forward_numpy(
        changed_x, layers, num_heads=2
    )
    delta = np.linalg.norm(
        changed_states - original_states, axis=-1
    )[:, 0, :]
    np.testing.assert_array_equal(
        delta[0, [0, 1, 3, 4]], np.zeros(4)
    )
    assert np.max(delta[1:, [0, 1, 3, 4]]) > 1e-9


@pytest.mark.exercise
@pytest.mark.parametrize("invalid_layers", [[], None, {}, [None], [{}]])
def test_invalid_layers_raise_value_error(invalid_layers):
    x = np.ones((1, 2, 8), dtype=np.float64)
    with pytest.raises(ValueError):
        encoder_stack_forward_numpy(
            x, invalid_layers, num_heads=2
        )


@pytest.mark.exercise
@pytest.mark.parametrize("num_heads", [0, -1, 3, True])
def test_invalid_heads_raise_value_error(num_heads):
    x = np.ones((1, 2, 8), dtype=np.float64)
    layer = _make_layers(depth=1)[0]
    with pytest.raises(ValueError):
        encoder_stack_forward_numpy(x, [layer], num_heads)


@pytest.mark.exercise
@pytest.mark.parametrize("bad_norm_first", [0, 1, "pre", None])
def test_norm_first_requires_boolean(bad_norm_first):
    x = np.ones((1, 2, 8), dtype=np.float64)
    layer = _make_layers(depth=1)[0]
    with pytest.raises(ValueError):
        encoder_stack_forward_numpy(
            x, [layer], 2, norm_first=bad_norm_first
        )


@pytest.mark.exercise
@pytest.mark.parametrize("eps", [0.0, -1e-5, float("nan"), True])
def test_epsilon_must_be_valid(eps):
    x = np.ones((1, 2, 8), dtype=np.float64)
    layer = _make_layers(depth=1)[0]
    with pytest.raises(ValueError):
        encoder_stack_forward_numpy(
            x, [layer], 2, eps=eps
        )


@pytest.mark.exercise
def test_invalid_single_layer_shape_is_rejected():
    x = np.ones((1, 2, 8), dtype=np.float64)
    layers = _make_layers(depth=2)
    layers[1]["w_o"] = np.ones((8, 7), dtype=np.float64)
    with pytest.raises(ValueError):
        encoder_stack_forward_numpy(x, layers, num_heads=2)
