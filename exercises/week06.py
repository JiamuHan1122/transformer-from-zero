"""Day 06 exercises: 3D sequence tensors and shared linear projections.

Input convention:
    x:      (B, T, D)
    weight: (D, H)
    bias:   (H,)
    output: (B, T, H)

Rules:
- Implement every operation with NumPy.
- Do not use torch, torch.autograd, or torch.nn in this file.
- Do not modify caller-owned arrays.
- The same weight and bias are shared by every token in every batch.
- For the loss task, average over B*T tokens and sum over H features.

Run:
    python -m pytest -q -m exercise tests/test_week06_exercises.py
"""

from __future__ import annotations

import numpy as np


def linear_3d_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Task A: apply one shared linear layer to a batch of sequences.

    Shapes:
        x:      (B, T, D)
        weight: (D, H)
        bias:   (H,)
        return: (B, T, H)

    Required computation:
        output = x @ weight + bias

    Concepts being tested:
        - NumPy matmul on the last two dimensions.
        - Bias broadcasting over both B and T.
        - One shared weight matrix for every token.

    Do not reshape x unless you have a clear reason. NumPy's @ already
    supports this batched matrix multiplication.
    """
    return x @ weight + bias



def linear_3d_mse_gradients(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Task B: manually compute loss and gradients for a shared 3D linear layer.

    Shapes:
        x:       (B, T, D)
        weight:  (D, H)
        bias:    (H,)
        targets: (B, T, H)

    Forward:
        predictions = x @ weight + bias
        error = predictions - targets

    Exact loss convention:
        L = 0.5 * sum(error**2) / (B*T)

    Sum over the H feature dimension for each token, then average over all
    B*T tokens. Do NOT divide by B*T*H.

    Return:
        (loss, grad_weight, grad_bias)
        loss is a Python float;
        grad_weight has shape (D, H);
        grad_bias has shape (H,).

    Hints:
        1. Treat B*T as the number of token examples.
        2. Flatten x to (B*T, D) and error to (B*T, H), or use einsum.
        3. The same weight is shared by all tokens, so its gradient must
           accumulate contributions from every batch and token.
        4. The bias gradient must sum over axes 0 and 1.
        5. Reuse the matrix-gradient reasoning from Week 02.

    Do not use torch or automatic differentiation.
    """
    B, T, D = x.shape
    _, _, H = targets.shape

    predictions = linear_3d_numpy(x, weight, bias)
    error = predictions - targets

    loss = 0.5 * np.sum(error**2) / (B * T)

    grad_weight = (x.reshape(B * T, D).T @ error.reshape(B * T, H)) / (B * T)
    grad_bias = np.sum(error, axis=(0, 1)) / (B * T)

    return loss.item(), grad_weight, grad_bias
