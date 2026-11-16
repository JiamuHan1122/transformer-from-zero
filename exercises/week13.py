"""Day 13 exercises: Residual Connections and gradient flow."""

from __future__ import annotations

import numpy as np


def _validate_float_array(name: str, value: np.ndarray, ndim: int) -> None:
    """Shared validation helper provided for the exercises."""
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {value.shape}.")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError(f"{name} must have a floating-point dtype.")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must not contain an empty dimension.")


def _validate_residual_add_inputs(x, branch_output) -> None:
    _validate_float_array("x", x, 3)
    _validate_float_array("branch_output", branch_output, 3)
    if x.shape != branch_output.shape:
        raise ValueError("x and branch_output must have exactly the same shape.")


def _validate_residual_linear_inputs(x, weight, bias) -> None:
    _validate_float_array("x", x, 3)
    _validate_float_array("weight", weight, 2)
    _validate_float_array("bias", bias, 1)
    d_model = x.shape[-1]
    if weight.shape != (d_model, d_model):
        raise ValueError(f"weight must have shape ({d_model}, {d_model}).")
    if bias.shape != (d_model,):
        raise ValueError(f"bias must have shape ({d_model},).")


def residual_add_forward_numpy(x, branch_output) -> np.ndarray:
    """Task A1: return x + branch_output as a float64 array.

    Both inputs must be floating NumPy arrays with identical (B,T,D) shapes.
    Do not modify caller-owned inputs.
    """
    _validate_residual_add_inputs(x, branch_output)
    # TODO: implement element-wise residual addition.
    raise NotImplementedError("Implement residual_add_forward_numpy.")


def residual_add_backward_numpy(grad_output) -> tuple[np.ndarray, np.ndarray]:
    """Task A2: split the upstream gradient at y = x + z.

    Return independent float64 arrays (grad_skip, grad_branch).
    Both equal grad_output. Do not share writable memory with the input.
    """
    _validate_float_array("grad_output", grad_output, 3)
    # TODO: implement the gradient split.
    raise NotImplementedError("Implement residual_add_backward_numpy.")


def residual_linear_forward_numpy(x, weight, bias) -> np.ndarray:
    """Task B1: implement y = x + x @ weight + bias.

    Shapes: x=(B,T,D), weight=(D,D), bias=(D,). Return float64.
    This linear residual block is a teaching example, not a full FFN.
    """
    _validate_residual_linear_inputs(x, weight, bias)
    # TODO: compute the branch and add the identity path.
    raise NotImplementedError("Implement residual_linear_forward_numpy.")


def residual_linear_backward_numpy(
    grad_output, x, weight
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task B2: manually differentiate y = x + x @ weight + bias.

    Return (grad_x, grad_weight, grad_bias), with shapes:
        (B,T,D), (D,D), (D,)

    Formulas:
        grad_x = grad_output + grad_output @ weight.T
        grad_weight = sum_{b,t} x[b,t,:].T @ grad_output[b,t,:]
        grad_bias = sum_{b,t} grad_output[b,t,:]

    Use NumPy only; do not modify inputs.
    """
    _validate_float_array("grad_output", grad_output, 3)
    _validate_float_array("x", x, 3)
    _validate_float_array("weight", weight, 2)
    d_model = x.shape[-1]
    if weight.shape != (d_model, d_model):
        raise ValueError(f"weight must have shape ({d_model}, {d_model}).")
    if grad_output.shape != x.shape:
        raise ValueError("grad_output and x must have identical shapes.")
    # TODO: implement all three gradients.
    raise NotImplementedError("Implement residual_linear_backward_numpy.")
