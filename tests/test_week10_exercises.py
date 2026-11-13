"""Day 10 tests: sinusoidal positional encoding and embedding addition."""

import numpy as np
import pytest

from exercises.week10 import (
    add_positional_encoding_numpy,
    sinusoidal_positional_encoding_numpy,
)


@pytest.mark.exercise
def test_sinusoidal_encoding_shape_dtype_and_position_zero():
    actual = sinusoidal_positional_encoding_numpy(seq_len=5, d_model=8)

    assert actual.shape == (5, 8)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(
        actual[0],
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        atol=1e-12,
    )


@pytest.mark.exercise
def test_sinusoidal_encoding_matches_formula_at_position_one():
    actual = sinusoidal_positional_encoding_numpy(seq_len=2, d_model=8)

    expected = np.array(
        [
            np.sin(1.0), np.cos(1.0),
            np.sin(0.1), np.cos(0.1),
            np.sin(0.01), np.cos(0.01),
            np.sin(0.001), np.cos(0.001),
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(actual[1], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.exercise
def test_sinusoidal_encoding_supports_odd_model_dimension():
    actual = sinusoidal_positional_encoding_numpy(seq_len=4, d_model=5)

    assert actual.shape == (4, 5)
    assert actual.dtype == np.float64
    expected_last = np.sin(3.0 / (10000.0 ** (4.0 / 5.0)))
    np.testing.assert_allclose(actual[3, 4], expected_last, atol=1e-12)


@pytest.mark.exercise
@pytest.mark.parametrize("bad_seq_len", [0, -1, 2.5, True, "5", None])
def test_sinusoidal_encoding_rejects_invalid_sequence_length(bad_seq_len):
    with pytest.raises(ValueError):
        sinusoidal_positional_encoding_numpy(bad_seq_len, 8)


@pytest.mark.exercise
@pytest.mark.parametrize("bad_d_model", [0, -1, 2.5, False, "8", None])
def test_sinusoidal_encoding_rejects_invalid_model_dimension(bad_d_model):
    with pytest.raises(ValueError):
        sinusoidal_positional_encoding_numpy(5, bad_d_model)


@pytest.mark.exercise
def test_sinusoidal_encoding_contains_only_finite_values():
    actual = sinusoidal_positional_encoding_numpy(seq_len=64, d_model=31)
    assert np.isfinite(actual).all()


@pytest.mark.exercise
def test_add_positional_encoding_broadcasts_across_batch():
    embeddings = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    positional = np.arange(3 * 4, dtype=np.float64).reshape(3, 4) / 10.0

    actual = add_positional_encoding_numpy(embeddings, positional)

    assert actual.shape == (2, 3, 4)
    np.testing.assert_allclose(actual[0], embeddings[0] + positional)
    np.testing.assert_allclose(actual[1], embeddings[1] + positional)


@pytest.mark.exercise
def test_add_positional_encoding_does_not_modify_inputs():
    embeddings = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    positional = np.ones((3, 4), dtype=np.float64)
    embeddings_before = embeddings.copy()
    positional_before = positional.copy()

    _ = add_positional_encoding_numpy(embeddings, positional)

    np.testing.assert_array_equal(embeddings, embeddings_before)
    np.testing.assert_array_equal(positional, positional_before)


@pytest.mark.exercise
@pytest.mark.parametrize(
    "embeddings, positional",
    [
        (np.zeros((3, 4)), np.zeros((3, 4))),
        (np.zeros((2, 3, 4)), np.zeros((4, 3))),
        (np.zeros((2, 3, 4)), np.zeros((3, 5))),
    ],
)
def test_add_positional_encoding_rejects_invalid_or_mismatched_shapes(
    embeddings,
    positional,
):
    with pytest.raises(ValueError):
        add_positional_encoding_numpy(embeddings, positional)


@pytest.mark.exercise
def test_add_positional_encoding_rejects_non_array_inputs():
    with pytest.raises(ValueError):
        add_positional_encoding_numpy([[1.0]], np.zeros((1, 1)))
    with pytest.raises(ValueError):
        add_positional_encoding_numpy(np.zeros((1, 1, 1)), [[1.0]])
