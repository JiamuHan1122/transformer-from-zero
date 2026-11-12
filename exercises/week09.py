"""Day 09 exercises: causal/padding masks and masked multi-head attention.

Mask convention:
    True  -> blocked / not visible
    False -> allowed / visible

Core implementations use NumPy only. Day 07 and Day 08 helpers are reused.
"""

from __future__ import annotations

import numpy as np

from exercises.week07 import softmax_numpy
from exercises.week08 import merge_heads_numpy, split_heads_numpy


def causal_mask_numpy(seq_len: int) -> np.ndarray:
    """Return a (T, T) Boolean mask that blocks future keys.

    True means blocked. The diagonal and lower triangle remain False.
    """
    if (
        not isinstance(seq_len, (int, np.integer))
        or isinstance(seq_len, (bool, np.bool_))
        or seq_len <= 0
    ):
        raise ValueError(
            f"seq_len must be a positive integer, got {seq_len!r}."
        )

    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)


def padding_mask_numpy(
    token_ids: np.ndarray,
    pad_token_id: int,
) -> np.ndarray:
    """Return a (B, T) Boolean mask; True marks PAD positions."""
    if not isinstance(token_ids, np.ndarray):
        raise ValueError("token_ids must be a NumPy array.")
    if token_ids.ndim != 2:
        raise ValueError(
            f"token_ids must be 2D with shape (B, T), got {token_ids.ndim}D."
        )

    return token_ids == pad_token_id


def masked_softmax_numpy(
    scores: np.ndarray,
    mask: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Numerically stable Softmax with Boolean masking.

    The mask must be broadcastable to scores; True means blocked.
    A fully masked row returns all zeros rather than NaN.
    Inputs are not modified.
    """
    # Step 1: Validate the input arrays and normalization axis.
    if not isinstance(scores, np.ndarray):
        raise ValueError("scores must be a NumPy array.")
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a NumPy array.")
    if scores.ndim < 1:
        raise ValueError("scores must have at least one dimension.")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError("mask must have Boolean dtype.")
    if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise ValueError("scores must contain real numeric values.")
    if (
        not isinstance(axis, (int, np.integer))
        or isinstance(axis, (bool, np.bool_))
        or axis < -scores.ndim
        or axis >= scores.ndim
    ):
        raise ValueError(
            f"axis must be valid for a {scores.ndim}D scores array, got {axis!r}."
        )

    axis = int(axis) % scores.ndim

    # Step 2: Broadcast the mask to the full score shape.
    try:
        broadcast_mask = np.broadcast_to(mask, scores.shape)
    except ValueError as exc:
        raise ValueError(
            f"mask with shape {mask.shape} is not broadcastable "
            f"to scores with shape {scores.shape}."
        ) from exc

    # Step 3: Convert scores to float64 for stable exponentiation.
    scores_float = scores.astype(np.float64, copy=False)

    # Step 4: Identify fully masked rows before subtracting the row maximum.
    fully_masked_rows = np.all(
        broadcast_mask,
        axis=axis,
        keepdims=True,
    )

    # Step 5: Replace blocked scores by -inf, but use 0 as the safe maximum
    # for fully masked rows to avoid the undefined expression -inf - (-inf).
    masked_scores = np.where(broadcast_mask, -np.inf, scores_float)
    row_max = np.max(masked_scores, axis=axis, keepdims=True)
    safe_row_max = np.where(fully_masked_rows, 0.0, row_max)

    # Step 6: Exponentiate only after max subtraction.
    shifted_scores = np.where(
        broadcast_mask,
        -np.inf,
        scores_float - safe_row_max,
    )
    exp_scores = np.exp(shifted_scores)

    # Step 7: Normalize visible entries. A fully masked row has denominator
    # zero, so the initialized zero output is retained for that row.
    denominator = np.sum(exp_scores, axis=axis, keepdims=True)
    weights = np.divide(
        exp_scores,
        denominator,
        out=np.zeros_like(exp_scores),
        where=denominator > 0.0,
    )

    # Step 8: Enforce exact zeros at every blocked position.
    return np.where(broadcast_mask, 0.0, weights)


def multi_head_attention_with_mask_numpy(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bias-free Multi-Head Self-Attention with an optional Boolean mask.

    Shapes:
        x:     (B, T, D)
        w_q:   (D, H * dk)
        w_k:   (D, H * dk)
        w_v:   (D, H * dv)
        w_o:   (H * dv, D_out)
        mask:  broadcastable to (B, H, T, T), True means blocked

    Return:
        output:  (B, T, D_out)
        weights: (B, H, T, T)

    Reuses Day 07 softmax and Day 08 split/merge helpers.
    """
    # Step 1: Compute the fused Q, K, and V projections.
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # Step 2: Split the projected channels into independent heads.
    q_heads = split_heads_numpy(q, num_heads)
    k_heads = split_heads_numpy(k, num_heads)
    v_heads = split_heads_numpy(v, num_heads)

    # Step 3: Compute scaled dot-product attention scores.
    dk = q_heads.shape[-1]
    scores = q_heads @ k_heads.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(dk)

    # Step 4: Apply the mask before Softmax.
    if mask is None:
        weights = softmax_numpy(scores, axis=-1)
    else:
        weights = masked_softmax_numpy(scores, mask, axis=-1)

    # Step 5: Aggregate the Value vectors independently in each head.
    head_outputs = weights @ v_heads

    # Step 6: Merge the head dimension back into the feature dimension.
    merged = merge_heads_numpy(head_outputs)

    # Step 7: Apply the output projection.
    output = merged @ w_o

    return output, weights
