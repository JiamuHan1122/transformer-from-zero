"""Day 11 exercises: assemble the decoder input pipeline.

Core implementations must use NumPy only.
Reuse Day 09 masks and Day 10 positional encoding addition.

Tasks:
    A. token_embedding_lookup_numpy
    B. causal_padding_mask_numpy
    C. prepare_decoder_input_numpy

Run:
    python -m pytest -q -m exercise tests/test_week11_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week09 import (
    causal_mask_numpy,
    padding_mask_numpy,
)

from exercises.week10 import add_positional_encoding_numpy


def token_embedding_lookup_numpy(
    token_ids: np.ndarray,
    embedding_table: np.ndarray,
) -> np.ndarray:
    """Task A: map token IDs (B, T) to embeddings (B, T, D).

    Reject non-integer IDs, Boolean IDs, invalid table shapes/types,
    and IDs outside [0, vocab_size).

    Do not modify either input.
    """

    # Step 1: Validate token_ids
    if not isinstance(token_ids, np.ndarray):
        raise ValueError("token_ids must be a NumPy array.")

    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2D with shape (B, T), "
            f"got {token_ids.ndim}D."
        )

    if (
        not np.issubdtype(token_ids.dtype, np.integer)
        or np.issubdtype(token_ids.dtype, np.bool_)
    ):
        raise ValueError(
            f"token_ids must have an integer dtype, "
            f"got {token_ids.dtype}."
        )

    if np.any(token_ids < 0):
        raise ValueError("token_ids must be non-negative.")

    # Step 2: Validate embedding_table
    if not isinstance(embedding_table, np.ndarray):
        raise ValueError("embedding_table must be a NumPy array.")

    if embedding_table.ndim != 2:
        raise ValueError(
            f"embedding_table must be 2D with shape "
            f"(vocab_size, D), got {embedding_table.ndim}D."
        )

    if not np.issubdtype(embedding_table.dtype, np.floating):
        raise ValueError(
            "embedding_table must have a floating-point dtype."
        )

    vocab_size, embedding_dim = embedding_table.shape

    if vocab_size == 0 or embedding_dim == 0:
        raise ValueError(
            "embedding_table must have non-empty dimensions."
        )

    # Step 3: Validate token ID range
    if np.any(token_ids >= vocab_size):
        raise ValueError(
            f"token_ids must be within [0, {vocab_size})."
        )

    # Step 4: Perform embedding lookup
    # (B, T) -> (B, T, D)
    embeddings = embedding_table[token_ids]

    return embeddings


def causal_padding_mask_numpy(
    token_ids: np.ndarray,
    pad_token_id: int,
) -> np.ndarray:
    """Task B: return combined blocked mask with shape (B, 1, T, T).

    True means blocked.
    Block future keys and all PAD keys.

    Reuse Day 09 causal_mask_numpy and padding_mask_numpy.
    Do not modify token_ids.
    """

    # Step 1: Validate token_ids
    if not isinstance(token_ids, np.ndarray):
        raise ValueError("token_ids must be a NumPy array.")

    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape (B, T)."
        )

    if (
        not np.issubdtype(token_ids.dtype, np.integer)
        or np.issubdtype(token_ids.dtype, np.bool_)
    ):
        raise ValueError(
            "token_ids must have an integer dtype."
        )

    if np.any(token_ids < 0):
        raise ValueError(
            "token_ids must be non-negative."
        )

    # Step 2: Validate PAD token ID
    if (
        not isinstance(pad_token_id, (int, np.integer))
        or isinstance(pad_token_id, (bool, np.bool_))
    ):
        raise ValueError(
            "pad_token_id must be an integer."
        )

    # Step 3: Generate the causal mask
    # Shape: (T, T)
    causal_mask = causal_mask_numpy(
        token_ids.shape[1]
    )

    # Step 4: Generate the padding mask
    # Shape: (B, T)
    padding_mask = padding_mask_numpy(
        token_ids,
        pad_token_id,
    )

    # Step 5: Expand dimensions for broadcasting
    # (T, T) -> (1, 1, T, T)
    causal_mask = causal_mask[
        np.newaxis,
        np.newaxis,
        :,
        :
    ]

    # (B, T) -> (B, 1, 1, T)
    padding_mask = padding_mask[
        :,
        np.newaxis,
        np.newaxis,
        :
    ]

    # Step 6: Combine causal and padding masks
    # True means blocked.
    combined_mask = np.logical_or(
        causal_mask,
        padding_mask,
    )

    # Output: (B, 1, T, T)
    return combined_mask


def prepare_decoder_input_numpy(
    token_ids: np.ndarray,
    embedding_table: np.ndarray,
    positional_encoding: np.ndarray,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Task C: assemble position-aware embeddings and attention mask.

    Args:
        token_ids: (B, T)
        embedding_table: (vocab_size, D)
        positional_encoding: (T, D)
        pad_token_id: padding token ID

    Returns:
        input_embeddings: (B, T, D)
        attention_mask:   (B, 1, T, T)

    Reuse Task A, Task B, and Day 10 add_positional_encoding_numpy.
    Do not modify caller-owned arrays.
    """

    # Step 1: Validate positional encoding
    if not isinstance(positional_encoding, np.ndarray):
        raise ValueError(
            "positional_encoding must be a NumPy array."
        )

    if positional_encoding.ndim != 2:
        raise ValueError(
            "positional_encoding must be 2D with shape (T, D)."
        )

    if not np.issubdtype(
        positional_encoding.dtype,
        np.floating,
    ):
        raise ValueError(
            "positional_encoding must have a floating-point dtype."
        )

    # Step 2: Convert token IDs into embeddings
    # (B, T) -> (B, T, D)
    input_embeddings = token_embedding_lookup_numpy(
        token_ids,
        embedding_table,
    )

    # Step 3: Add positional encoding
    # (B, T, D) + (T, D) -> (B, T, D)
    input_embeddings = add_positional_encoding_numpy(
        input_embeddings,
        positional_encoding,
    )

    # Step 4: Construct causal + padding attention mask
    # Output: (B, 1, T, T)
    attention_mask = causal_padding_mask_numpy(
        token_ids,
        pad_token_id,
    )

    # Step 5: Return decoder inputs
    return input_embeddings, attention_mask
