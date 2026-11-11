
"""Day 08 exercises: Multi-Head Self-Attention.

Input convention:
    x:   (B, T, D)
    w_q: (D, H * dk)
    w_k: (D, H * dk)
    w_v: (D, H * dv)
    w_o: (H * dv, D_out)

Output:
    output:  (B, T, D_out)
    weights: (B, H, T, T)

Rules:
- Implement every operation with NumPy.
- Reuse Day 07's softmax_numpy.
- Do not use torch, torch.autograd, or torch.nn.
- Do not modify caller-owned arrays.
- This day implements unmasked, bias-free Multi-Head Self-Attention.
"""

from __future__ import annotations

import numpy as np

from exercises.week07 import softmax_numpy


def split_heads_numpy(
    x: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """Task A: split the final projected feature dimension into heads.

    Input:
        x: (B, T, H * D_head)

    Return:
        (B, H, T, D_head)
    """

    # Step 1: Validate input
    if x.ndim != 3:
        raise ValueError(f"Input x must be 3D, got {x.ndim}D.")

    if (
        not isinstance(num_heads, (int, np.integer))
        or isinstance(num_heads, (bool, np.bool_))
        or num_heads <= 0
    ):
        raise ValueError(
            f"num_heads must be a positive integer, got {num_heads}."
        )

    # Step 2: Validate divisibility
    if x.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Last dimension of x ({x.shape[-1]}) "
            f"is not divisible by num_heads ({num_heads})."
        )

    # Step 3: Reshape (B, T, H * D_head) -> (B, T, H, D_head)
    B, T, _ = x.shape
    D_head = x.shape[-1] // num_heads

    x = x.reshape(B, T, num_heads, D_head)

    # Step 4: Transpose (B, T, H, D_head) -> (B, H, T, D_head)
    x = x.transpose(0, 2, 1, 3)

    return x


def merge_heads_numpy(
    x: np.ndarray,
) -> np.ndarray:
    """Task B: merge heads back into the feature dimension.

    Input:
        x: (B, H, T, D_head)

    Return:
        (B, T, H * D_head)
    """

    # Step 1: Validate input
    if x.ndim != 4:
        raise ValueError(f"Input x must be 4D, got {x.ndim}D.")

    # Step 2: Transpose (B, H, T, D_head) -> (B, T, H, D_head)
    x = x.transpose(0, 2, 1, 3)

    # Step 3: Reshape (B, T, H, D_head) -> (B, T, H * D_head)
    B, T, H, D_head = x.shape

    x = x.reshape(B, T, H * D_head)

    return x


def multi_head_attention_numpy(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    num_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Task C: implement bias-free, unmasked Multi-Head Self-Attention.

    Shapes:
        x:   (B, T, D)
        w_q: (D, H * dk)
        w_k: (D, H * dk)
        w_v: (D, H * dv)
        w_o: (H * dv, D_out)

    Return:
        output:  (B, T, D_out)
        weights: (B, H, T, T)

    Notes:
    - dk is inferred from q_heads.shape[-1].
    - dv is inferred from v_heads.shape[-1].
    - dk and dv may differ.
    - Reuse Day 07's softmax_numpy.
    - Do not modify caller-owned arrays.
    """

    # Step 1: Compute Q, K, V projections
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # Step 2: Split heads
    q_heads = split_heads_numpy(q, num_heads)
    k_heads = split_heads_numpy(k, num_heads)
    v_heads = split_heads_numpy(v, num_heads)

    # Step 3: Compute attention scores
    scores = q_heads @ k_heads.transpose(0, 1, 3, 2)

    # Step 4: Scale scores by sqrt(dk)
    dk = q_heads.shape[-1]
    scores = scores / np.sqrt(dk)

    # Step 5: Compute attention weights
    weights = softmax_numpy(scores, axis=-1)

    # Step 6: Aggregate values
    head_outputs = weights @ v_heads

    # Step 7: Merge heads
    merged = merge_heads_numpy(head_outputs)

    # Step 8: Output projection
    output = merged @ w_o

    return output, weights
