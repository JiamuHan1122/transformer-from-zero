"""Day 11 tests: decoder input pipeline integration."""

import numpy as np
import pytest

from exercises.week11 import (
    causal_padding_mask_numpy,
    prepare_decoder_input_numpy,
    token_embedding_lookup_numpy,
)


@pytest.mark.exercise
def test_embedding_lookup_shape_values_and_dtype():
    ids = np.array([[1, 3, 2], [4, 0, 1]], dtype=np.int64)
    table = np.arange(30, dtype=np.float64).reshape(5, 6)
    actual = token_embedding_lookup_numpy(ids, table)
    assert actual.shape == (2, 3, 6)
    np.testing.assert_array_equal(actual, table[ids])


@pytest.mark.exercise
@pytest.mark.parametrize("bad_ids", [
    np.array([[1.0, 2.0]]),
    np.array([[True, False]]),
    np.array([["1", "2"]]),
    np.array([1, 2]),
])
def test_embedding_lookup_rejects_invalid_ids(bad_ids):
    with pytest.raises(ValueError):
        token_embedding_lookup_numpy(bad_ids, np.zeros((4, 3)))


@pytest.mark.exercise
def test_embedding_lookup_rejects_out_of_range_ids():
    table = np.zeros((4, 3))
    with pytest.raises(ValueError):
        token_embedding_lookup_numpy(np.array([[0, 4]]), table)
    with pytest.raises(ValueError):
        token_embedding_lookup_numpy(np.array([[-1, 2]]), table)


@pytest.mark.exercise
def test_embedding_lookup_does_not_modify_inputs():
    ids = np.array([[1, 2], [0, 3]], dtype=np.int64)
    table = np.arange(20, dtype=np.float64).reshape(5, 4)
    ids_before, table_before = ids.copy(), table.copy()
    token_embedding_lookup_numpy(ids, table)
    np.testing.assert_array_equal(ids, ids_before)
    np.testing.assert_array_equal(table, table_before)


@pytest.mark.exercise
def test_combined_mask_shape_and_semantics():
    ids = np.array([[5, 6, 7, 0], [8, 9, 0, 0]], dtype=np.int64)
    actual = causal_padding_mask_numpy(ids, pad_token_id=0)
    assert actual.shape == (2, 1, 4, 4)
    # Batch 0: query 0 sees only key 0.
    np.testing.assert_array_equal(
        actual[0, 0, 0], np.array([False, True, True, True])
    )
    # Batch 0: query 2 cannot see future key 3, which is also PAD.
    np.testing.assert_array_equal(
        actual[0, 0, 2], np.array([False, False, False, True])
    )
    # Every PAD key is blocked for every query.
    assert actual[1, 0, :, 2:].all()


@pytest.mark.exercise
def test_combined_mask_rejects_non_integer_ids():
    with pytest.raises(ValueError):
        causal_padding_mask_numpy(np.array([[1.0, 0.0]]), 0)


@pytest.mark.exercise
def test_prepare_decoder_input_shapes_and_values():
    ids = np.array([[1, 2, 0], [3, 4, 0]], dtype=np.int64)
    table = np.arange(30, dtype=np.float64).reshape(5, 6) / 10.0
    pe = np.arange(18, dtype=np.float64).reshape(3, 6) / 100.0
    x, mask = prepare_decoder_input_numpy(ids, table, pe, pad_token_id=0)
    assert x.shape == (2, 3, 6)
    assert mask.shape == (2, 1, 3, 3)
    np.testing.assert_allclose(x, table[ids] + pe[np.newaxis, :, :])
    np.testing.assert_array_equal(mask, causal_padding_mask_numpy(ids, 0))


@pytest.mark.exercise
def test_prepare_decoder_input_preserves_all_inputs():
    ids = np.array([[1, 2, 0]], dtype=np.int64)
    table = np.arange(12, dtype=np.float64).reshape(4, 3)
    pe = np.ones((3, 3), dtype=np.float64)
    before = [ids.copy(), table.copy(), pe.copy()]
    prepare_decoder_input_numpy(ids, table, pe, 0)
    for actual, expected in zip((ids, table, pe), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.exercise
def test_prepare_decoder_input_rejects_mismatched_position_shape():
    ids = np.array([[1, 2, 0]], dtype=np.int64)
    table = np.zeros((4, 3))
    with pytest.raises(ValueError):
        prepare_decoder_input_numpy(ids, table, np.zeros((2, 3)), 0)
