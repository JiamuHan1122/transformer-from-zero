"""Day 14 exercises: Position-wise Feed-Forward Network (FFN).

Forward:
    z = x @ w1 + b1
    a = GELU_tanh(z)
    y = a @ w2 + b2

Shapes:
    x:  (B, T, D)
    w1: (D, H)
    b1: (H,)
    w2: (H, D)
    b2: (D,)
    y:  (B, T, D)

Rules:
- Core implementations use NumPy only; no torch or autograd here.
- Reuse Day 05 gelu_tanh_numpy and gelu_tanh_backward_numpy.
- Use float64 for calculations and return values.
- Do not modify caller-owned arrays.
- The same FFN parameters are shared across every batch and token.

Run:
    python -m pytest -q -m exercise tests/test_week14_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week05 import (
    gelu_tanh_numpy,
    gelu_tanh_backward_numpy,
)


def _validate_ffn_inputs(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray | None,
) -> None:
    """Provided helper for forward and backward input validation."""
    arrays = [("x", x, 3), ("w1", w1, 2), ("b1", b1, 1), ("w2", w2, 2)]
    if b2 is not None:
        arrays.append(("b2", b2, 1))

    for name, value, rank in arrays:
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a NumPy array.")
        if value.ndim != rank:
            raise ValueError(
                f"{name} must be {rank}D, got shape {value.shape}."
            )
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{name} must have a floating-point dtype.")
        if any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must not have an empty dimension.")

    d_model = x.shape[-1]
    if w1.shape[0] != d_model:
        raise ValueError(f"w1 must start with input dimension {d_model}.")
    d_ff = w1.shape[1]
    if b1.shape != (d_ff,):
        raise ValueError(f"b1 must have shape ({d_ff},).")
    if w2.shape != (d_ff, d_model):
        raise ValueError(
            f"w2 must have shape ({d_ff}, {d_model}), got {w2.shape}."
        )
    if b2 is not None and b2.shape != (d_model,):
        raise ValueError(f"b2 must have shape ({d_model},).")


def ffn_forward_numpy(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Task A: compute a two-layer position-wise FFN.

    z = x @ w1 + b1          (B, T, H)
    a = gelu_tanh_numpy(z)   (B, T, H)
    y = a @ w2 + b2          (B, T, D)

    Return y as a float64 array of shape (B, T, D).
    This function must NOT add residuals or LayerNorm.
    """
    _validate_ffn_inputs(x, w1, b1, w2, b2)
    # TODO: implement the two linear projections and the GELU activation.
    raise NotImplementedError("Task A: implement ffn_forward_numpy")


def ffn_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Task B: manually backpropagate through the Day 14 FFN.

    Forward equations:
        z = x @ w1 + b1
        a = gelu_tanh_numpy(z)
        y = a @ w2 + b2

    Recompute z and a from the inputs (do not require a saved cache).

    Given grad_output=dL/dy (B,T,D), return in this exact order:
        grad_x:  (B,T,D)
        grad_w1: (D,H)
        grad_b1: (H,)
        grad_w2: (H,D)
        grad_b2: (D,)

    Use the chain rule in reverse:
        grad_w2 = sum_{b,t} outer(a[b,t], grad_output[b,t])
        grad_b2 = sum(grad_output, axis=(0,1))
        grad_a = grad_output @ w2.T
        grad_z = gelu_tanh_backward_numpy(z, grad_a)
        grad_w1 = sum_{b,t} outer(x[b,t], grad_z[b,t])
        grad_b1 = sum(grad_z, axis=(0,1))
        grad_x = grad_z @ w1.T

    Hints: np.einsum("bth,btd->hd", a, grad_output) is useful for grad_w2.
    The b2 value is not needed because d(x+b2)/dx is independent of b2.

    Return float64 arrays. Never modify the provided inputs.
    """
    _validate_ffn_inputs(x, w1, b1, w2, None)
    if not isinstance(grad_output, np.ndarray):
        raise ValueError("grad_output must be a NumPy array.")
    if grad_output.shape != x.shape:
        raise ValueError("grad_output must have the same shape as x.")
    if not np.issubdtype(grad_output.dtype, np.floating):
        raise ValueError("grad_output must have a floating-point dtype.")

    # TODO: recompute z/a and return all five gradients using NumPy.
    raise NotImplementedError("Task B: implement ffn_backward_numpy")
