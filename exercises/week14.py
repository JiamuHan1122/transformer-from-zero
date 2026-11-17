
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
- Core implementations use NumPy only.
- Reuse Day 05 GELU forward and backward functions.
- Use float64 for calculations and return values.
- Do not modify caller-owned arrays.
- Share the same FFN parameters across all tokens.

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
    """Validate FFN input types, ranks and shapes."""

    # Step 1: Collect inputs
    arrays = [
        ("x", x, 3),
        ("w1", w1, 2),
        ("b1", b1, 1),
        ("w2", w2, 2),
    ]

    if b2 is not None:
        arrays.append(("b2", b2, 1))

    # Step 2: Validate array types and dimensions
    for name, value, rank in arrays:

        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"{name} must be a NumPy array."
            )

        if value.ndim != rank:
            raise ValueError(
                f"{name} must be {rank}D, "
                f"got shape {value.shape}."
            )

        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError(
                f"{name} must have a floating-point dtype."
            )

        if any(size == 0 for size in value.shape):
            raise ValueError(
                f"{name} must not have an empty dimension."
            )

    # Step 3: Validate feature dimensions
    d_model = x.shape[-1]

    if w1.shape[0] != d_model:
        raise ValueError(
            f"w1 must start with input dimension {d_model}."
        )

    d_ff = w1.shape[1]

    if b1.shape != (d_ff,):
        raise ValueError(
            f"b1 must have shape ({d_ff},)."
        )

    if w2.shape != (d_ff, d_model):
        raise ValueError(
            f"w2 must have shape ({d_ff}, {d_model}), "
            f"got {w2.shape}."
        )

    if b2 is not None and b2.shape != (d_model,):
        raise ValueError(
            f"b2 must have shape ({d_model},)."
        )


def ffn_forward_numpy(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Task A: compute the position-wise FFN forward pass.

    Computation:
        z = x @ w1 + b1
        a = gelu_tanh_numpy(z)
        y = a @ w2 + b2

    Shapes:
        x:  (B, T, D)
        w1: (D, H)
        b1: (H,)
        w2: (H, D)
        b2: (D,)

    Returns:
        y: (B, T, D), dtype float64.

    Do not add Residual Connection or LayerNorm here.
    """

    # Step 1: Validate inputs
    _validate_ffn_inputs(
        x, w1, b1, w2, b2
    )

    # Step 2: Convert inputs to float64
    x = np.asarray(x, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)
    b2 = np.asarray(b2, dtype=np.float64)

    # Step 3: First linear projection
    # (B, T, D) @ (D, H) + (H,)
    # -> (B, T, H)
    z = x @ w1 + b1

    # Step 4: GELU activation
    # (B, T, H) -> (B, T, H)
    a = gelu_tanh_numpy(z)

    # Step 5: Second linear projection
    # (B, T, H) @ (H, D) + (D,)
    # -> (B, T, D)
    y = a @ w2 + b2

    return y


def ffn_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Task B: manually backpropagate through the FFN.

    Forward:
        z = x @ w1 + b1
        a = gelu_tanh_numpy(z)
        y = a @ w2 + b2

    Given:
        grad_output = dL/dy, shape (B, T, D)

    Returns:
        grad_x:  (B, T, D)
        grad_w1: (D, H)
        grad_b1: (H,)
        grad_w2: (H, D)
        grad_b2: (D,)

    Backpropagation:
        grad_w2 = sum over B,T of outer(a, grad_output)
        grad_b2 = sum over B,T of grad_output
        grad_a = grad_output @ w2.T

        grad_z = gelu_tanh_backward_numpy(z, grad_a)

        grad_w1 = sum over B,T of outer(x, grad_z)
        grad_b1 = sum over B,T of grad_z
        grad_x = grad_z @ w1.T

    Use NumPy only.
    Do not modify caller-owned arrays.
    """

    # Step 1: Validate FFN inputs
    # b2 is not required for backward calculations.
    _validate_ffn_inputs(
        x, w1, b1, w2, None
    )

    # Step 2: Validate upstream gradient
    if not isinstance(grad_output, np.ndarray):
        raise ValueError(
            "grad_output must be a NumPy array."
        )

    if grad_output.shape != x.shape:
        raise ValueError(
            "grad_output must have the same shape as x."
        )

    if not np.issubdtype(
        grad_output.dtype,
        np.floating,
    ):
        raise ValueError(
            "grad_output must have a floating-point dtype."
        )

    # Step 3: Convert inputs to float64
    x = np.asarray(x, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    w2 = np.asarray(w2, dtype=np.float64)

    grad_output = np.asarray(
        grad_output,
        dtype=np.float64,
    )

    # Step 4: Recompute forward intermediate values
    # z: (B, T, H)
    # a: (B, T, H)
    z = x @ w1 + b1
    a = gelu_tanh_numpy(z)

    # Step 5: Backward through the second linear layer

    # grad_w2: (H, D)
    grad_w2 = np.einsum(
        "bth,btd->hd",
        a,
        grad_output,
    )

    # grad_b2: (D,)
    grad_b2 = np.sum(
        grad_output,
        axis=(0, 1),
    )

    # grad_a: (B, T, H)
    grad_a = grad_output @ w2.T

    # Step 6: Backward through GELU
    # grad_z: (B, T, H)
    grad_z = gelu_tanh_backward_numpy(
        z,
        grad_a,
    )

    # Step 7: Backward through the first linear layer

    # grad_w1: (D, H)
    grad_w1 = np.einsum(
        "btd,bth->dh",
        x,
        grad_z,
    )

    # grad_b1: (H,)
    grad_b1 = np.sum(
        grad_z,
        axis=(0, 1),
    )

    # grad_x: (B, T, D)
    grad_x = grad_z @ w1.T

    # Step 8: Return all five gradients
    return (
        grad_x,
        grad_w1,
        grad_b1,
        grad_w2,
        grad_b2,
    )
