"""Day 09 tests: causal/padding masks and masked multi-head attention."""

import numpy as np
import pytest
import torch

from exercises.week09 import (
    causal_mask_numpy,
    masked_softmax_numpy,
    multi_head_attention_with_mask_numpy,
    padding_mask_numpy,
)


def _torch_masked_reference(x, w_q, w_k, w_v, w_o, num_heads, mask):
    xt = torch.from_numpy(x)
    wqt, wkt, wvt, wot = map(torch.from_numpy, (w_q, w_k, w_v, w_o))
    b, t, _ = x.shape
    q = (xt @ wqt).reshape(b, t, num_heads, -1).transpose(1, 2)
    k = (xt @ wkt).reshape(b, t, num_heads, -1).transpose(1, 2)
    v = (xt @ wvt).reshape(b, t, num_heads, -1).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1])
    blocked = torch.as_tensor(np.broadcast_to(mask, scores.shape), dtype=torch.bool)
    scores = scores.masked_fill(blocked, float("-inf"))
    # The test fixtures avoid fully masked rows for the torch reference.
    weights = torch.softmax(scores, dim=-1)
    merged = weights @ v
    merged = merged.transpose(1, 2).contiguous().reshape(b, t, -1)
    return (merged @ wot).numpy(), weights.numpy()


@pytest.mark.exercise
def test_causal_mask_is_strict_upper_triangle():
    actual = causal_mask_numpy(4)
    expected = np.triu(np.ones((4, 4), dtype=bool), k=1)
    assert actual.shape == (4, 4)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.exercise
@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4", None])
def test_causal_mask_rejects_invalid_lengths(bad):
    with pytest.raises(ValueError):
        causal_mask_numpy(bad)


@pytest.mark.exercise
def test_padding_mask_marks_only_pad_positions_and_preserves_input():
    ids = np.array([[4, 5, 0, 0], [7, 0, 8, 9]], dtype=np.int64)
    before = ids.copy()
    actual = padding_mask_numpy(ids, pad_token_id=0)
    expected = np.array([[False, False, True, True], [False, True, False, False]])
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(ids, before)


@pytest.mark.exercise
def test_masked_softmax_normalizes_visible_entries_and_zeros_masked_entries():
    scores = np.array([[1.0, 2.0, 3.0], [4.0, -2.0, 0.5]])
    mask = np.array([[False, True, False], [True, False, False]])
    actual = masked_softmax_numpy(scores, mask)
    assert actual.shape == scores.shape
    assert np.all(actual[mask] == 0.0)
    np.testing.assert_allclose(actual.sum(axis=-1), 1.0, atol=1e-12)


@pytest.mark.exercise
def test_masked_softmax_all_masked_row_returns_zeros_without_nan():
    scores = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, True], [False, True]])
    actual = masked_softmax_numpy(scores, mask)
    np.testing.assert_array_equal(actual[0], np.zeros(2))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[1], np.array([1.0, 0.0]))


@pytest.mark.exercise
def test_masked_softmax_is_stable_for_extreme_scores():
    scores = np.array([[10000.0, 9999.0, -10000.0]])
    mask = np.array([[False, False, True]])
    actual = masked_softmax_numpy(scores, mask)
    assert np.isfinite(actual).all()
    assert actual[0, 2] == 0.0
    np.testing.assert_allclose(actual.sum(axis=-1), 1.0, atol=1e-12)


@pytest.mark.exercise
def test_masked_softmax_supports_broadcasting_and_rejects_bad_masks():
    scores = np.zeros((2, 3, 4, 4), dtype=np.float64)
    mask = np.triu(np.ones((4, 4), dtype=bool), k=1)
    actual = masked_softmax_numpy(scores, mask)
    assert actual.shape == scores.shape
    assert np.all(actual[..., mask] == 0.0)
    np.testing.assert_allclose(actual.sum(axis=-1), 1.0, atol=1e-12)
    with pytest.raises((ValueError, TypeError)):
        masked_softmax_numpy(scores, np.zeros((4, 4), dtype=np.int64))
    with pytest.raises(ValueError):
        masked_softmax_numpy(scores, np.zeros((3, 7), dtype=bool))


@pytest.mark.exercise
def test_masked_multi_head_attention_matches_torch_reference():
    rng = np.random.default_rng(9001)
    b, t, d, h, dk, dv = 2, 4, 8, 2, 3, 4
    x = rng.normal(size=(b, t, d))
    w_q = rng.normal(size=(d, h * dk))
    w_k = rng.normal(size=(d, h * dk))
    w_v = rng.normal(size=(d, h * dv))
    w_o = rng.normal(size=(h * dv, d))
    mask = np.triu(np.ones((t, t), dtype=bool), k=1)[None, None, :, :]
    actual_y, actual_a = multi_head_attention_with_mask_numpy(
        x, w_q, w_k, w_v, w_o, h, mask
    )
    expected_y, expected_a = _torch_masked_reference(
        x, w_q, w_k, w_v, w_o, h, mask
    )
    assert actual_y.shape == (b, t, d)
    assert actual_a.shape == (b, h, t, t)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(actual_a, expected_a, rtol=1e-11, atol=1e-11)


@pytest.mark.exercise
def test_causal_mask_prevents_future_token_from_changing_past_outputs():
    rng = np.random.default_rng(9002)
    b, t, d, h = 1, 5, 8, 2
    x = rng.normal(size=(b, t, d))
    w_q = rng.normal(size=(d, d))
    w_k = rng.normal(size=(d, d))
    w_v = rng.normal(size=(d, d))
    w_o = rng.normal(size=(d, d))
    mask = causal_mask_numpy(t)[None, None, :, :]
    y1, _ = multi_head_attention_with_mask_numpy(x, w_q, w_k, w_v, w_o, h, mask)
    changed = x.copy()
    changed[:, 4, :] += 100.0
    y2, _ = multi_head_attention_with_mask_numpy(
        changed, w_q, w_k, w_v, w_o, h, mask
    )
    np.testing.assert_allclose(y1[:, :4], y2[:, :4], atol=1e-12, rtol=1e-12)


@pytest.mark.exercise
def test_causal_and_padding_masks_combine_and_zero_all_blocked_keys():
    ids = np.array([[11, 12, 0, 0]], dtype=np.int64)
    pad = padding_mask_numpy(ids, pad_token_id=0)[:, None, None, :]
    causal = causal_mask_numpy(ids.shape[1])[None, None, :, :]
    combined = causal | pad
    rng = np.random.default_rng(9003)
    x = rng.normal(size=(1, 4, 8))
    w_q = rng.normal(size=(8, 8))
    w_k = rng.normal(size=(8, 8))
    w_v = rng.normal(size=(8, 8))
    w_o = rng.normal(size=(8, 8))
    _, weights = multi_head_attention_with_mask_numpy(
        x, w_q, w_k, w_v, w_o, 2, combined
    )
    broadcast = np.broadcast_to(combined, weights.shape)
    assert np.all(weights[broadcast] == 0.0)
    assert np.isfinite(weights).all()
