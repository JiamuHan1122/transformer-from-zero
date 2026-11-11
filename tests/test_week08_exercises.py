"""Day 08 tests: Split/Merge Heads and Multi-Head Self-Attention."""

import numpy as np
import pytest
import torch

from exercises.week07 import self_attention_numpy
from exercises.week08 import (
    merge_heads_numpy,
    multi_head_attention_numpy,
    split_heads_numpy,
)


def _torch_multi_head_reference(x, w_q, w_k, w_v, w_o, num_heads):
    xt = torch.from_numpy(x)
    wqt, wkt, wvt, wot = map(torch.from_numpy, (w_q, w_k, w_v, w_o))

    batch_size, seq_len, _ = x.shape
    q = (xt @ wqt).reshape(batch_size, seq_len, num_heads, -1).transpose(1, 2)
    k = (xt @ wkt).reshape(batch_size, seq_len, num_heads, -1).transpose(1, 2)
    v = (xt @ wvt).reshape(batch_size, seq_len, num_heads, -1).transpose(1, 2)

    dk = q.shape[-1]
    scores = (q @ k.transpose(-1, -2)) / np.sqrt(dk)
    weights = torch.softmax(scores, dim=-1)
    head_outputs = weights @ v

    merged = head_outputs.transpose(1, 2).contiguous().reshape(
        batch_size, seq_len, -1
    )
    output = merged @ wot
    return output.numpy(), weights.numpy()


@pytest.mark.exercise
def test_split_heads_shape_order_and_values():
    x = np.arange(1 * 2 * 8, dtype=np.float64).reshape(1, 2, 8)
    before = x.copy()

    actual = split_heads_numpy(x, num_heads=2)
    expected = x.reshape(1, 2, 2, 4).transpose(0, 2, 1, 3)

    assert actual.shape == (1, 2, 2, 4)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, before)


@pytest.mark.exercise
def test_split_heads_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        split_heads_numpy(np.zeros((2, 3)), num_heads=2)

    with pytest.raises(ValueError):
        split_heads_numpy(np.zeros((2, 3, 7)), num_heads=2)

    with pytest.raises(ValueError):
        split_heads_numpy(np.zeros((2, 3, 8)), num_heads=0)


@pytest.mark.exercise
def test_merge_heads_restores_original_tensor():
    x = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    before = x.copy()

    actual = merge_heads_numpy(x)
    expected = x.transpose(0, 2, 1, 3).reshape(2, 4, 15)

    assert actual.shape == (2, 4, 15)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(x, before)


@pytest.mark.exercise
def test_split_and_merge_are_inverse_operations():
    rng = np.random.default_rng(8001)
    x = rng.normal(size=(2, 5, 24)).astype(np.float64)

    split = split_heads_numpy(x, num_heads=6)
    restored = merge_heads_numpy(split)

    assert split.shape == (2, 6, 5, 4)
    assert restored.shape == x.shape
    np.testing.assert_array_equal(restored, x)


@pytest.mark.exercise
def test_multi_head_attention_matches_torch_reference():
    rng = np.random.default_rng(8002)
    batch_size, seq_len, model_dim, num_heads = 2, 4, 8, 2

    x = rng.normal(size=(batch_size, seq_len, model_dim)).astype(np.float64)
    w_q = rng.normal(size=(model_dim, model_dim)).astype(np.float64)
    w_k = rng.normal(size=(model_dim, model_dim)).astype(np.float64)
    w_v = rng.normal(size=(model_dim, model_dim)).astype(np.float64)
    w_o = rng.normal(size=(model_dim, model_dim)).astype(np.float64)
    originals = [a.copy() for a in (x, w_q, w_k, w_v, w_o)]

    actual_output, actual_weights = multi_head_attention_numpy(
        x, w_q, w_k, w_v, w_o, num_heads
    )
    expected_output, expected_weights = _torch_multi_head_reference(
        x, w_q, w_k, w_v, w_o, num_heads
    )

    assert actual_output.shape == (2, 4, 8)
    assert actual_weights.shape == (2, 2, 4, 4)
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_weights.sum(axis=-1), 1.0, atol=1e-12)

    for current, original in zip((x, w_q, w_k, w_v, w_o), originals):
        np.testing.assert_array_equal(current, original)


@pytest.mark.exercise
def test_multi_head_attention_supports_different_dk_and_dv():
    rng = np.random.default_rng(8003)
    batch_size, seq_len, model_dim, num_heads = 2, 3, 6, 3
    dk, dv = 2, 4

    x = rng.normal(size=(batch_size, seq_len, model_dim)).astype(np.float64)
    w_q = rng.normal(size=(model_dim, num_heads * dk)).astype(np.float64)
    w_k = rng.normal(size=(model_dim, num_heads * dk)).astype(np.float64)
    w_v = rng.normal(size=(model_dim, num_heads * dv)).astype(np.float64)
    w_o = rng.normal(size=(num_heads * dv, model_dim)).astype(np.float64)

    actual_output, actual_weights = multi_head_attention_numpy(
        x, w_q, w_k, w_v, w_o, num_heads
    )
    expected_output, expected_weights = _torch_multi_head_reference(
        x, w_q, w_k, w_v, w_o, num_heads
    )

    assert actual_output.shape == (batch_size, seq_len, model_dim)
    assert actual_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_weights, expected_weights, rtol=1e-12, atol=1e-12)


@pytest.mark.exercise
def test_single_head_case_matches_day07_attention():
    rng = np.random.default_rng(8004)
    x = rng.normal(size=(1, 4, 4)).astype(np.float64)
    w_q = rng.normal(size=(4, 4)).astype(np.float64)
    w_k = rng.normal(size=(4, 4)).astype(np.float64)
    w_v = rng.normal(size=(4, 4)).astype(np.float64)
    w_o = np.eye(4, dtype=np.float64)

    expected_output, expected_weights = self_attention_numpy(x, w_q, w_k, w_v)
    actual_output, actual_weights = multi_head_attention_numpy(
        x, w_q, w_k, w_v, w_o, num_heads=1
    )

    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual_weights[:, 0], expected_weights, rtol=1e-12, atol=1e-12
    )


@pytest.mark.exercise
def test_attention_heads_have_independent_weight_patterns():
    x = np.array(
        [[[1.0, 0.0, 0.0, 1.0],
          [0.0, 1.0, 1.0, 0.0],
          [1.0, 1.0, -1.0, 1.0]]],
        dtype=np.float64,
    )
    # Head 0 uses the first two feature channels; Head 1 uses the last two.
    w_q = np.eye(4, dtype=np.float64)
    w_k = np.eye(4, dtype=np.float64)
    w_v = np.eye(4, dtype=np.float64)
    w_o = np.eye(4, dtype=np.float64)

    _, weights = multi_head_attention_numpy(x, w_q, w_k, w_v, w_o, num_heads=2)

    assert weights.shape == (1, 2, 3, 3)
    assert not np.allclose(weights[:, 0], weights[:, 1])
