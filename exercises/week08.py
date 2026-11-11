"""Day 08 exercises: Multi-Head Self-Attention.

Input convention:
    x:   (B, T, D)
    w_q: (D, H * dk)
    w_k: (D, H * dk)
    w_v: (D, H * dv)
    w_o: (H * dv, D_out)

Output:
    output: (B, T, D_out)
    weights: (B, H, T, T)

Rules:
- Implement every operation with NumPy.
- Reuse Day 07's softmax_numpy.
- Do not use torch, torch.autograd, or torch.nn in this file.
- Do not modify caller-owned arrays.
- This day implements unmasked, bias-free Multi-Head Self-Attention.

Run:
    python -m pytest -q -m exercise tests/test_week08_exercises.py
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

    Required order:
        1. Validate that x is 3D and num_heads is a positive integer.
        2. Validate that x.shape[-1] is divisible by num_heads.
        3. Reshape (B, T, H * D_head) to (B, T, H, D_head).
        4. Transpose to (B, H, T, D_head).

    Do not modify x.
    """
    raise NotImplementedError("Day 08 Task A: implement split_heads_numpy")


def merge_heads_numpy(x: np.ndarray) -> np.ndarray:
    """Task B: merge the head dimension back into the feature dimension.

    Input:
        x: (B, H, T, D_head)

    Return:
        (B, T, H * D_head)

    Required order:
        1. Validate that x is 4D.
        2. Transpose to (B, T, H, D_head).
        3. Reshape to (B, T, H * D_head).

    Do not modify x.
    """
    raise NotImplementedError("Day 08 Task B: implement merge_heads_numpy")


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

    Computation:
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v

        q_heads = split_heads_numpy(q, num_heads)
        k_heads = split_heads_numpy(k, num_heads)
        v_heads = split_heads_numpy(v, num_heads)

        scores = q_heads @ k_heads.transpose(0, 1, 3, 2)
        scores = scores / sqrt(dk)
        weights = softmax_numpy(scores, axis=-1)

        head_outputs = weights @ v_heads
        merged = merge_heads_numpy(head_outputs)
        output = merged @ w_o

    Return:
        output:  (B, T, D_out)
        weights: (B, H, T, T)

    Notes:
    - dk is inferred from q_heads.shape[-1].
    - dv is inferred from v_heads.shape[-1].
    - dk and dv may differ.
    - Each head has independent projection channels in the fused matrices.
    - Reuse the imported Day 07 softmax_numpy.
    - Do not use torch or automatic differentiation.
    - Do not modify x or any projection matrix.
    """
    raise NotImplementedError(
        "Day 08 Task C: implement multi_head_attention_numpy"
    )
