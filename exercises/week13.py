"""Day 13 exercises: Residual Connections and gradient flow.

Input convention:
    x: (B, T, D)
    branch_output: (B, T, D)
    weight: (D, D)
    bias: (D,)

Core implementations must use NumPy only.

Tasks:
    A1. residual_add_forward_numpy
    A2. residual_add_backward_numpy
    B1. residual_linear_forward_numpy
    B2. residual_linear_backward_numpy
"""

from __future__ import annotations

import numpy as np


def _validate_float_array(
    name: str,
    value: np.ndarray,
    ndim: int,
) -> None:
    """Shared validation helper provided for the exercises."""

    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")

    if value.ndim != ndim:
        raise ValueError(
            f"{name} must be {ndim}D, got {value.shape}."
        )

    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(
            f"{name} must have a floating-point dtype."
        )

    if any(size == 0 for size in value.shape):
        raise ValueError(
            f"{name} must not contain an empty dimension."
        )


def _validate_residual_add_inputs(
    x: np.ndarray,
    branch_output: np.ndarray,
) -> None:
    """Validate inputs for residual addition."""

    _validate_float_array("x", x, 3)
    _validate_float_array("branch_output", branch_output, 3)

    if x.shape != branch_output.shape:
        raise ValueError(
            "x and branch_output must have exactly the same shape."
        )


def _validate_residual_linear_inputs(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> None:
    """Validate inputs for the linear residual block."""

    _validate_float_array("x", x, 3)
    _validate_float_array("weight", weight, 2)
    _validate_float_array("bias", bias, 1)

    d_model = x.shape[-1]

    if weight.shape != (d_model, d_model):
        raise ValueError(
            f"weight must have shape ({d_model}, {d_model})."
        )

    if bias.shape != (d_model,):
        raise ValueError(
            f"bias must have shape ({d_model},)."
        )


def residual_add_forward_numpy(
    x: np.ndarray,
    branch_output: np.ndarray,
) -> np.ndarray:
    """Task A1: implement residual addition.

    Formula:
        y = x + branch_output

    Inputs:
        x:             (B, T, D)
        branch_output: (B, T, D)

    Return:
        output: (B, T, D), float64

    Do not modify caller-owned inputs.
    """

    # Step 1: Validate input shapes and dtypes
    _validate_residual_add_inputs(x, branch_output)

    # Step 2: Convert inputs to float64
    x_float = np.asarray(x, dtype=np.float64)
    branch_float = np.asarray(
        branch_output,
        dtype=np.float64,
    )

    # Step 3: Add the identity path and residual branch
    output = x_float + branch_float

    return output


def residual_add_backward_numpy(
    grad_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Task A2: manually differentiate residual addition.

    Forward:
        y = x + z

    Backward:
        grad_x = grad_output
        grad_z = grad_output

    Return:
        grad_skip:   (B, T, D), float64
        grad_branch: (B, T, D), float64

    The two returned arrays must be independent copies.
    Do not modify caller-owned inputs.
    """

    # Step 1: Validate upstream gradient
    _validate_float_array(
        "grad_output",
        grad_output,
        3,
    )

    # Step 2: Split the upstream gradient into two independent paths
    grad_skip = np.array(
        grad_output,
        dtype=np.float64,
        copy=True,
    )

    grad_branch = np.array(
        grad_output,
        dtype=np.float64,
        copy=True,
    )

    # Step 3: Return independent gradients
    return grad_skip, grad_branch


def residual_linear_forward_numpy(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Task B1: implement a linear residual block.

    Formula:
        y = x + x @ weight + bias

    Shapes:
        x:      (B, T, D)
        weight: (D, D)
        bias:   (D,)

    Return:
        output: (B, T, D), float64

    This linear residual block is a teaching example,
    not a full Feed-Forward Network.

    Do not modify caller-owned inputs.
    """

    # Step 1: Validate input shapes and dtypes
    _validate_residual_linear_inputs(
        x,
        weight,
        bias,
    )

    # Step 2: Convert inputs to float64
    x_float = np.asarray(x, dtype=np.float64)
    weight_float = np.asarray(
        weight,
        dtype=np.float64,
    )
    bias_float = np.asarray(
        bias,
        dtype=np.float64,
    )

    # Step 3: Compute the linear transformation
    branch_output = x_float @ weight_float + bias_float

    # Step 4: Add the identity residual connection
    output = x_float + branch_output

    return output


def residual_linear_backward_numpy(
    grad_output: np.ndarray,
    x: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task B2: manually differentiate the linear residual block.

    Forward:
        y = x + x @ weight + bias

    Shapes:
        grad_output: (B, T, D)
        x:           (B, T, D)
        weight:      (D, D)

    Backward formulas:

        grad_x = grad_output + grad_output @ weight.T

        grad_weight = sum_{b,t}(
            x[b,t,:].T @ grad_output[b,t,:]
        )

        grad_bias = sum_{b,t}(
            grad_output[b,t,:]
        )

    Return:
        grad_x:      (B, T, D)
        grad_weight: (D, D)
        grad_bias:   (D,)

    All returned arrays must be float64.

    Use NumPy only.
    Do not use torch or automatic differentiation.
    Do not modify caller-owned inputs.
    """

    # Step 1: Validate input types, dimensions and dtypes
    _validate_float_array(
        "grad_output",
        grad_output,
        3,
    )

    _validate_float_array(
        "x",
        x,
        3,
    )

    _validate_float_array(
        "weight",
        weight,
        2,
    )

    # Step 2: Validate weight dimensions
    d_model = x.shape[-1]

    if weight.shape != (d_model, d_model):
        raise ValueError(
            f"weight must have shape ({d_model}, {d_model})."
        )

    # Step 3: Validate upstream gradient dimensions
    if grad_output.shape != x.shape:
        raise ValueError(
            "grad_output and x must have identical shapes."
        )

    # Step 4: Convert inputs to float64
    x_float = np.asarray(
        x,
        dtype=np.float64,
    )

    weight_float = np.asarray(
        weight,
        dtype=np.float64,
    )

    grad_output_float = np.asarray(
        grad_output,
        dtype=np.float64,
    )

    # Step 5: Compute gradient with respect to x
    # Identity path + residual branch
    grad_x = (
        grad_output_float
        + grad_output_float @ weight_float.T
    )

    # Step 6: Compute gradient with respect to weight
    # Accumulate outer products over batch and token dimensions
    grad_weight = np.einsum(
        "btd,bte->de",
        x_float,
        grad_output_float,
    )

    # Step 7: Compute gradient with respect to bias
    # Accumulate over batch and token dimensions
    grad_bias = np.sum(
        grad_output_float,
        axis=(0, 1),
    )

    # Step 8: Return all three gradients
    return grad_x, grad_weight, grad_bias
