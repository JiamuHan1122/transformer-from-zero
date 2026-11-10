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
    """Task A: numerically stable softmax along one axis.

    Requirements:
    - Support 1D, 2D, 3D (or higher-rank) arrays.
    - Preserve the input shape.
    - Return float64 values.
    - Subtract the maximum along axis before exponentiation.
    - Do not modify the caller's input array.

    Formula:
        shifted = x - max(x, axis=axis, keepdims=True)
        exp_values = exp(shifted)
        probabilities = exp_values / sum(exp_values, axis=axis, keepdims=True)
    """
    raise NotImplementedError("Task A: implement stable softmax")


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
    raise NotImplementedError("Task B: implement scaled dot-product attention")


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
    raise NotImplementedError("Task C: implement single-head self-attention")
