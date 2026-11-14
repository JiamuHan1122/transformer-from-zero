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

from exercises.week09 import causal_mask_numpy, padding_mask_numpy
from exercises.week10 import add_positional_encoding_numpy


def token_embedding_lookup_numpy(
    token_ids: np.ndarray,
    embedding_table: np.ndarray,
) -> np.ndarray:
    """Task A: map token IDs (B, T) to embeddings (B, T, D).

    Reject non-integer IDs, Boolean IDs, invalid table shapes/types,
    and IDs outside [0, vocab_size). Do not modify either input.
    """
    raise NotImplementedError("Task A: implement token_embedding_lookup_numpy")


def causal_padding_mask_numpy(
    token_ids: np.ndarray,
    pad_token_id: int,
) -> np.ndarray:
    """Task B: return combined blocked mask with shape (B, 1, T, T).

    True means blocked. Block future keys and all PAD keys.
    Reuse Day 09 causal_mask_numpy and padding_mask_numpy.
    Do not modify token_ids.
    """
    raise NotImplementedError("Task B: implement causal_padding_mask_numpy")


def prepare_decoder_input_numpy(
    token_ids: np.ndarray,
    embedding_table: np.ndarray,
    positional_encoding: np.ndarray,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Task C: assemble position-aware embeddings and attention mask.

    Returns:
        input_embeddings: (B, T, D)
        attention_mask:   (B, 1, T, T)

    Reuse Task A, Task B, and Day 10 add_positional_encoding_numpy.
    Do not modify caller-owned arrays.
    """
    raise NotImplementedError("Task C: implement prepare_decoder_input_numpy")
