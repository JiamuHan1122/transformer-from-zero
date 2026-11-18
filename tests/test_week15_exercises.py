"""Day 15 tests: complete Transformer Encoder Block composition."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exercises.week15 import transformer_encoder_block_forward_numpy


def _make_case(seed=1501, batch_size=2, seq_len=4, d_model=8, d_ff=12):
    rng = np.random.default_rng(seed)
    num_heads = 2
    scale = 1.0 / np.sqrt(d_model)

    x = rng.normal(size=(batch_size, seq_len, d_model)).astype(np.float64)
    w_q = rng.normal(scale=scale, size=(d_model, d_model))
    w_k = rng.normal(scale=scale, size=(d_model, d_model))
    w_v = rng.normal(scale=scale, size=(d_model, d_model))
    w_o = rng.normal(scale=scale, size=(d_model, d_model))

    gamma_attn = np.ones(d_model, dtype=np.float64)
    beta_attn = np.zeros(d_model, dtype=np.float64)
    gamma_ffn = np.ones(d_model, dtype=np.float64)
    beta_ffn = np.zeros(d_model, dtype=np.float64)

    w1 = rng.normal(scale=0.2, size=(d_model, d_ff))
    b1 = rng.normal(scale=0.1, size=d_ff)
    w2 = rng.normal(scale=0.2, size=(d_ff, d_model))
    b2 = rng.normal(scale=0.1, size=d_model)

    return (
        x, w_q, w_k, w_v, w_o, num_heads,
        gamma_attn, beta_attn,
        w1, b1, w2, b2,
        gamma_ffn, beta_ffn,
    )


def _torch_attention(x, w_q, w_k, w_v, w_o, num_heads):
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    q = (x @ w_q).reshape(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)
    k = (x @ w_k).reshape(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)
    v = (x @ w_v).reshape(
        batch_size, seq_len, num_heads, head_dim
    ).transpose(1, 2)

    scores = (q @ k.transpose(-2, -1)) / np.sqrt(head_dim)
    weights = torch.softmax(scores, dim=-1)
    context = weights @ v
    merged = context.transpose(1, 2).contiguous().reshape(
        batch_size, seq_len, d_model
    )
    return merged @ w_o, weights


def _torch_reference(arrays, norm_first, eps=1e-5):
    (
        x, w_q, w_k, w_v, w_o, num_heads,
        gamma_attn, beta_attn,
        w1, b1, w2, b2,
        gamma_ffn, beta_ffn,
    ) = arrays

    tensors = [
        torch.tensor(value, dtype=torch.float64)
        if isinstance(value, np.ndarray)
        else value
        for value in arrays
    ]
    (
        tx, tq, tk, tv, to,
        _, tga, tba, tw1, tb1, tw2, tb2, tgf, tbf,
    ) = tensors

    def ln(value, gamma, beta):
        return F.layer_norm(
            value, (value.shape[-1],), gamma, beta, eps=eps
        )

    def ffn(value):
        hidden = F.gelu(value @ tw1 + tb1, approximate="tanh")
        return hidden @ tw2 + tb2

    if norm_first:
        attention_input = ln(tx, tga, tba)
        attention_output, weights = _torch_attention(
            attention_input, tq, tk, tv, to, num_heads
        )
        h = tx + attention_output
        ffn_output = ffn(ln(h, tgf, tbf))
        output = h + ffn_output
    else:
        attention_output, weights = _torch_attention(
            tx, tq, tk, tv, to, num_heads
        )
        h = ln(tx + attention_output, tga, tba)
        output = ln(h + ffn(h), tgf, tbf)

    return output.detach().numpy(), weights.detach().numpy()


@pytest.mark.exercise
@pytest.mark.parametrize("norm_first", [True, False])
def test_encoder_block_matches_torch_reference(norm_first):
    arrays = _make_case(seed=1502)
    actual, actual_weights = transformer_encoder_block_forward_numpy(
        *arrays, eps=1e-5, norm_first=norm_first
    )
    expected, expected_weights = _torch_reference(arrays, norm_first)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(
        actual_weights, expected_weights, rtol=1e-10, atol=1e-11
    )


@pytest.mark.exercise
def test_encoder_block_output_and_attention_shapes():
    arrays = _make_case(seed=1503, batch_size=3, seq_len=5, d_model=8)
    output, weights = transformer_encoder_block_forward_numpy(*arrays)

    assert output.shape == (3, 5, 8)
    assert weights.shape == (3, 2, 5, 5)
    assert output.dtype == np.float64
    assert weights.dtype == np.float64


@pytest.mark.exercise
def test_attention_rows_are_probability_distributions():
    arrays = _make_case(seed=1504)
    _, weights = transformer_encoder_block_forward_numpy(*arrays)

    np.testing.assert_allclose(
        np.sum(weights, axis=-1),
        np.ones(weights.shape[:-1]),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(weights >= 0.0)
    assert np.all(weights > 0.0)


@pytest.mark.exercise
def test_encoder_attention_is_bidirectional_not_causal():
    arrays = _make_case(seed=1505)
    _, weights = transformer_encoder_block_forward_numpy(*arrays)

    # The current Day 08 encoder attention is unmasked:
    # later-key columns are not blocked for earlier queries.
    assert np.all(weights[..., 0, 1:] > 0.0)
    assert np.all(weights[..., 1, 2:] > 0.0)


@pytest.mark.exercise
def test_pre_ln_zero_branches_reduce_to_identity():
    arrays = list(_make_case(seed=1506))
    x = arrays[0]
    arrays[3] = np.zeros_like(arrays[3])  # w_v = 0
    arrays[4] = np.zeros_like(arrays[4])  # w_o = 0
    arrays[10] = np.zeros_like(arrays[10])  # w2 = 0
    arrays[11] = np.zeros_like(arrays[11])  # b2 = 0

    output, _ = transformer_encoder_block_forward_numpy(
        *arrays, norm_first=True
    )
    np.testing.assert_allclose(output, x, rtol=0.0, atol=0.0)


@pytest.mark.exercise
def test_post_ln_zero_branches_still_apply_layer_norm():
    arrays = list(_make_case(seed=1507))
    x = arrays[0]
    arrays[3] = np.zeros_like(arrays[3])
    arrays[4] = np.zeros_like(arrays[4])
    arrays[10] = np.zeros_like(arrays[10])
    arrays[11] = np.zeros_like(arrays[11])

    output, _ = transformer_encoder_block_forward_numpy(
        *arrays, norm_first=False
    )

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    first_norm = (x - mean) / np.sqrt(var + 1e-5)
    mean2 = np.mean(first_norm, axis=-1, keepdims=True)
    var2 = np.mean((first_norm - mean2) ** 2, axis=-1, keepdims=True)
    expected = (first_norm - mean2) / np.sqrt(var2 + 1e-5)

    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
    assert not np.allclose(output, x)


@pytest.mark.exercise
def test_pre_ln_and_post_ln_are_not_the_same_architecture():
    arrays = _make_case(seed=1508)
    pre, _ = transformer_encoder_block_forward_numpy(
        *arrays, norm_first=True
    )
    post, _ = transformer_encoder_block_forward_numpy(
        *arrays, norm_first=False
    )
    assert np.linalg.norm(pre - post) > 1e-8


@pytest.mark.exercise
def test_attention_mixes_information_across_tokens():
    arrays = list(_make_case(seed=1509))
    original, _ = transformer_encoder_block_forward_numpy(
        *arrays, norm_first=True
    )

    changed = arrays[0].copy()
    changed[0, 2, :] += np.linspace(0.4, 1.1, changed.shape[-1])
    changed_arrays = [changed] + arrays[1:]
    updated, _ = transformer_encoder_block_forward_numpy(
        *changed_arrays, norm_first=True
    )

    other_positions = np.arange(changed.shape[1]) != 2
    other_delta = np.max(
        np.abs(updated[0, other_positions] - original[0, other_positions])
    )
    assert other_delta > 1e-10


@pytest.mark.exercise
def test_encoder_block_preserves_all_inputs():
    arrays = _make_case(seed=1510)
    copies = [
        value.copy() if isinstance(value, np.ndarray) else value
        for value in arrays
    ]

    transformer_encoder_block_forward_numpy(*arrays)

    for value, original in zip(arrays, copies):
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original)


@pytest.mark.exercise
def test_encoder_block_rejects_invalid_head_count():
    arrays = list(_make_case(seed=1511))
    arrays[5] = 3  # d_model=8 is not divisible by 3
    with pytest.raises(ValueError):
        transformer_encoder_block_forward_numpy(*arrays)


@pytest.mark.exercise
@pytest.mark.parametrize("bad_norm_first", [0, 1, "pre", None])
def test_encoder_block_rejects_non_boolean_norm_first(bad_norm_first):
    arrays = _make_case(seed=1512)
    with pytest.raises(ValueError):
        transformer_encoder_block_forward_numpy(
            *arrays, norm_first=bad_norm_first
        )


@pytest.mark.exercise
def test_encoder_block_rejects_incompatible_ffn_dimensions():
    arrays = list(_make_case(seed=1513))
    arrays[10] = np.ones((arrays[8].shape[1] + 1, arrays[0].shape[-1]))
    with pytest.raises(ValueError):
        transformer_encoder_block_forward_numpy(*arrays)
