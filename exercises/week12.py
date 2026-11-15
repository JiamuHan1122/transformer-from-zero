"""Day 12 exercises: Layer Normalization.

Input convention:
    x:           (B, T, D)
    gamma, beta: (D,)
    output:      (B, T, D)

Normalize each token independently over its final feature dimension.
Use population variance (ddof=0), as in torch.nn.LayerNorm.

Core implementations must use NumPy only.

Run:
    python -m pytest -q -m exercise tests/test_week12_exercises.py
"""

from __future__ import annotations

import numpy as np


def _validate_layer_norm_inputs(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps: float,
) -> None:
    """Shared input validation; this helper is provided."""

    # Step 1: Validate x
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a NumPy array.")

    if x.ndim != 3:
        raise ValueError(
            f"x must have shape (B, T, D), got {x.ndim}D."
        )

    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError(
            "x must have a floating-point dtype."
        )

    if x.shape[-1] == 0:
        raise ValueError(
            "The feature dimension D must be positive."
        )

    # Step 2: Validate gamma
    if not isinstance(gamma, np.ndarray):
        raise ValueError(
            "gamma must be a NumPy array."
        )

    if gamma.ndim != 1 or gamma.shape[0] != x.shape[-1]:
        raise ValueError(
            f"gamma must have shape ({x.shape[-1]},), "
            f"got {gamma.shape}."
        )

    if not np.issubdtype(gamma.dtype, np.floating):
        raise ValueError(
            "gamma must have a floating-point dtype."
        )

    # Step 3: Validate beta
    if beta is not None:
        if not isinstance(beta, np.ndarray):
            raise ValueError(
                "beta must be a NumPy array."
            )

        if beta.ndim != 1 or beta.shape[0] != x.shape[-1]:
            raise ValueError(
                f"beta must have shape ({x.shape[-1]},), "
                f"got {beta.shape}."
            )

        if not np.issubdtype(beta.dtype, np.floating):
            raise ValueError(
                "beta must have a floating-point dtype."
            )

    # Step 4: Validate epsilon
    if (
        not isinstance(
            eps,
            (int, float, np.integer, np.floating)
        )
        or isinstance(eps, (bool, np.bool_))
        or not np.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError(
            "eps must be a finite positive real number."
        )


def layer_norm_forward_numpy(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Task A: implement LayerNorm forward for x shaped (B, T, D).

    For every (b, t), compute mean and population variance over axis=-1:

        mean = mean(x, axis=-1, keepdims=True)

        var = mean((x - mean)**2, axis=-1, keepdims=True)

        x_hat = (x - mean) / sqrt(var + eps)

        output = gamma * x_hat + beta

    gamma and beta have shape (D,) and broadcast over B and T.

    Return:
        float64 array with shape (B, T, D).

    Do not modify caller-owned arrays.
    """

    # Step 1: Validate inputs
    _validate_layer_norm_inputs(
        x,
        gamma,
        beta,
        eps
    )

    # Step 2: Convert inputs to float64
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    # Step 3: Compute mean over the feature dimension
    # (B, T, D) -> (B, T, 1)
    mean = np.mean(
        x,
        axis=-1,
        keepdims=True
    )

    # Step 4: Compute population variance
    # (B, T, D) -> (B, T, 1)
    var = np.mean(
        (x - mean) ** 2,
        axis=-1,
        keepdims=True
    )

    # Step 5: Normalize each token independently
    x_hat = (x - mean) / np.sqrt(var + eps)

    # Step 6: Apply learnable scale and shift
    # gamma, beta: (D,)
    # Broadcasting: (B, T, D)
    output = gamma * x_hat + beta

    return output


def layer_norm_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task B: manually implement LayerNorm backward.

    Inputs:
        grad_output: dL/dY, shape (B, T, D)
        x:           original input, shape (B, T, D)
        gamma:       scale parameter, shape (D,)

    Returns:
        grad_x:      dL/dX, shape (B, T, D)
        grad_gamma:  dL/dgamma, shape (D,)
        grad_beta:   dL/dbeta, shape (D,)

    Recompute mean, variance, x_hat and inv_std from x.

    Let:
        q = grad_output * gamma
        inv_std = 1 / sqrt(var + eps)

    Input gradient:

        grad_x = inv_std * (
            q
            - mean(q, axis=-1, keepdims=True)
            - x_hat * mean(q * x_hat, axis=-1, keepdims=True)
        )

    Parameter gradients:

        grad_gamma = sum(grad_output * x_hat, axis=(0, 1))

        grad_beta = sum(grad_output, axis=(0, 1))

    Do not use torch or automatic differentiation.
    Do not modify caller-owned arrays.
    """

    # Step 1: Validate x and gamma
    _validate_layer_norm_inputs(
        x,
        gamma,
        None,
        eps
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
        np.floating
    ):
        raise ValueError(
            "grad_output must have a floating-point dtype."
        )

    # Step 3: Convert inputs to float64
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    grad_output = np.asarray(
        grad_output,
        dtype=np.float64
    )

    # Step 4: Recompute forward statistics
    # (B, T, D) -> (B, T, 1)
    mean = np.mean(
        x,
        axis=-1,
        keepdims=True
    )

    var = np.mean(
        (x - mean) ** 2,
        axis=-1,
        keepdims=True
    )

    # Step 5: Compute inverse standard deviation
    inv_std = 1.0 / np.sqrt(var + eps)

    # Step 6: Recompute normalized input
    x_hat = (x - mean) * inv_std

    # Step 7: Compute gradient with respect to normalized input
    # q = dL/dY * gamma
    q = grad_output * gamma

    # Step 8: Compute input gradient
    # All means are calculated along the feature dimension.
    grad_x = inv_std * (
        q
        - np.mean(
            q,
            axis=-1,
            keepdims=True
        )
        - x_hat * np.mean(
            q * x_hat,
            axis=-1,
            keepdims=True
        )
    )

    # Step 9: Compute gamma gradient
    # Accumulate over batch and token dimensions.
    grad_gamma = np.sum(
        grad_output * x_hat,
        axis=(0, 1)
    )

    # Step 10: Compute beta gradient
    # Accumulate over batch and token dimensions.
    grad_beta = np.sum(
        grad_output,
        axis=(0, 1)
    )

    return grad_x, grad_gamma, grad_beta
