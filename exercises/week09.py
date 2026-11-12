"""Day 09 exercises: Causal and Padding Masks for Attention.

Mask convention:
    True  -> blocked / not visible
    False -> allowed / visible

Core implementations must use NumPy only.
"""

from __future__ import annotations

import numpy as np


def causal_mask_numpy(seq_len: int) -> np.ndarray:
    """Task A: return a (T, T) Boolean mask that blocks future keys.

    True means blocked. The diagonal and lower triangle remain False.
    """
    raise NotImplementedError("Task A: implement causal_mask_numpy")


def padding_mask_numpy(token_ids: np.ndarray, pad_token_id: int) -> np.ndarray:
    """Task B: return a (B, T) Boolean mask; True marks PAD positions."""
    raise NotImplementedError("Task B: implement padding_mask_numpy")


def masked_softmax_numpy(
    scores: np.ndarray,
    mask: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """Task C: numerically stable Softmax with Boolean masking.

    The mask must be broadcastable to scores; True means blocked.
    A fully masked row must return all zeros rather than NaN.
    """
    raise NotImplementedError("Task C: implement masked_softmax_numpy")


def multi_head_attention_with_mask_numpy(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Task D: bias-free Multi-Head Self-Attention with an optional mask.

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

    Reuse split_heads_numpy and merge_heads_numpy from Day 08.
    """
    raise NotImplementedError(
        "Task D: implement multi_head_attention_with_mask_numpy"
    )
