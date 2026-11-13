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
    raise NotImplementedError("Task A: implement sinusoidal positional encoding.")


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
    raise NotImplementedError("Task B: implement embedding + positional encoding.")
