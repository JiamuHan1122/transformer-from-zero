"""Day 10 exercises: sinusoidal positional encoding.

Core implementations must use NumPy only.

Tasks:
    A. sinusoidal_positional_encoding_numpy
    B. add_positional_encoding_numpy

Run:
    python -m pytest -q -m exercise tests/test_week10_exercises.py
"""

from __future__ import annotations

import numpy as np


def sinusoidal_positional_encoding_numpy(
    seq_len: int,
    d_model: int,
) -> np.ndarray:
    """Task A: create sinusoidal positional encodings.

    Args:
        seq_len: positive number of sequence positions.
        d_model: positive embedding dimension.

    Returns:
        A float64 array with shape (seq_len, d_model).

    Formula:
        PE(pos, 2i)   = sin(pos / 10000 ** (2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000 ** (2i / d_model))

    Requirements:
        - Use NumPy only.
        - Support both even and odd d_model.
        - Reject booleans and non-positive/non-integer dimensions.
        - Do not modify caller-owned data.
    """
    if (
        not isinstance(seq_len, (int, np.integer))
        or isinstance(seq_len, (bool, np.bool_))
        or seq_len <= 0
    ):
        raise ValueError(
            f"seq_len must be a positive integer, got {seq_len!r}."
        )
    if (
        not isinstance(d_model, (int, np.integer))
        or isinstance(d_model, (bool, np.bool_))
        or d_model <= 0
    ):
        raise ValueError(
            f"d_model must be a positive integer, got {d_model!r}."
        )

    # Step 1: Create position indices and dimension indices
    positions = np.arange(seq_len)[:, np.newaxis]  # Shape: (seq_len, 1)
    dimensions = np.arange(d_model)[np.newaxis, :]  # Shape: (1, d_model)

    # Step 2: Compute the angle rates
    angle_rates = positions / np.power(
        10000,
        (2 * (dimensions // 2)) / d_model,
    )

    # Step 3: Apply sin to even indices and cos to odd indices
    positional_encoding = np.zeros(
        (seq_len, d_model),
        dtype=np.float64,
    )
    positional_encoding[:, 0::2] = np.sin(angle_rates[:, 0::2])
    positional_encoding[:, 1::2] = np.cos(angle_rates[:, 1::2])

    return positional_encoding


def add_positional_encoding_numpy(
    token_embeddings: np.ndarray,
    positional_encoding: np.ndarray,
) -> np.ndarray:
    """Task B: add positional encoding to token embeddings by broadcasting.

    Args:
        token_embeddings: array with shape (B, T, D).
        positional_encoding: array with shape (T, D).

    Returns:
        Array with shape (B, T, D).

    Requirements:
        - Validate input types, ranks, and matching T/D dimensions.
        - Use broadcasting; do not loop over batch or token positions.
        - Do not modify either caller-owned input array.
    """
    if not isinstance(token_embeddings, np.ndarray):
        raise ValueError("token_embeddings must be a NumPy array.")
    if not isinstance(positional_encoding, np.ndarray):
        raise ValueError("positional_encoding must be a NumPy array.")
    if token_embeddings.ndim != 3:
        raise ValueError(
            "token_embeddings must be 3D with shape (B, T, D), "
            f"got {token_embeddings.ndim}D."
        )
    if positional_encoding.ndim != 2:
        raise ValueError(
            "positional_encoding must be 2D with shape (T, D), "
            f"got {positional_encoding.ndim}D."
        )
    if token_embeddings.shape[1] != positional_encoding.shape[0]:
        raise ValueError(
            f"Number of tokens in token_embeddings ({token_embeddings.shape[1]}) "
            "does not match number of positions in positional_encoding "
            f"({positional_encoding.shape[0]})."
        )
    if token_embeddings.shape[2] != positional_encoding.shape[1]:
        raise ValueError(
            "Embedding dimension in token_embeddings "
            f"({token_embeddings.shape[2]}) does not match embedding dimension "
            f"in positional_encoding ({positional_encoding.shape[1]})."
        )

    return token_embeddings + positional_encoding[np.newaxis, :, :]
