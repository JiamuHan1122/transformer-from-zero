"""Day 07 exercises: stable softmax and single-head self-attention.

Input convention:
    x: (B, T, D)
    q: (B, Tq, dk)
    k: (B, Tk, dk)
    v: (B, Tk, dv)

Rules:
- Implement every operation with NumPy.
- Do not use torch, torch.autograd, or torch.nn in this file.
- Do not modify caller-owned arrays.
- This day implements unmasked, single-head attention only.

Run:
    python -m pytest -q -m exercise tests/test_week07_exercises.py
"""

from __future__ import annotations

import numpy as np


def softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Task A: numerically stable softmax."""

    # Step 1: Convert input to float64
    x = np.asarray(x, dtype=np.float64)

    # Step 2: Find maximum along axis
    # Remember keepdims=True
    max_values = np.max(x, axis=axis, keepdims=True)

    # Step 3: Subtract maximum
    shifted = x - max_values

    # Step 4: Compute exponentials
    exp_shifted = np.exp(shifted)
    # Step 5: Sum exponentials along axis
    # Remember keepdims=True
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
    # Step 6: Normalize and return
    return exp_shifted / sum_exp


def scaled_dot_product_attention_numpy(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Task B: compute unmasked scaled dot-product attention.

    Shapes:
        q: (B, Tq, dk)
        k: (B, Tk, dk)
        v: (B, Tk, dv)

    Computation:
        scores = q @ k.transpose(0, 2, 1) / sqrt(dk)
        weights = softmax_numpy(scores, axis=-1)
        output = weights @ v

    Return:
        (output, weights)
        output:  (B, Tq, dv)
        weights: (B, Tq, Tk)

    Notes:
    - Tq and Tk may differ.
    - dk and dv may differ.
    - Each query distributes its weights across the Tk key positions.
    - Do not use torch or automatic differentiation.
    - Do not modify q, k, or v.
    """
    # Step 1: 获取 dk
    dk = q.shape[-1]

    # Step 2: 计算 QK^T
    scores = q @ k.transpose(0, 2, 1)

    # Step 3: 使用 sqrt(dk) 进行缩放
    scores = scores / np.sqrt(dk)

    # Step 4: 调用昨天完成的 softmax_numpy
    # 注意 axis=-1
    weights = softmax_numpy(scores, axis=-1)

    # Step 5: 使用 Attention 权重聚合 V
    output = weights @ v

    # Step 6: 返回 output 和 weights
    return output, weights


def self_attention_numpy(
    x: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Task C: build single-head self-attention from a sequence input.

    Shapes:
        x:   (B, T, D)
        w_q: (D, dk)
        w_k: (D, dk)
        w_v: (D, dv)

    Computation:
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        output, weights = scaled_dot_product_attention_numpy(q, k, v)

    Return:
        (output, weights)
        output:  (B, T, dv)
        weights: (B, T, T)

    Notes:
    - There are no projection biases in this first implementation.
    - w_q, w_k, and w_v are independent learned projection matrices.
    - dk and dv do not have to be equal.
    - Do not use torch or automatic differentiation.
    - Do not modify x or any projection matrix.
    """
    # Step 1: 生成 Query
    q = x @ w_q

    # Step 2: 生成 Key
    k = x @ w_k

    # Step 3: 生成 Value
    v = x @ w_v

    # Step 4: 调用 Task B
    output, weights = scaled_dot_product_attention_numpy(q, k, v)

    # Step 5: 返回结果
    return output, weights
