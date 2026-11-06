"""Week 03: numerical gradient checking and a full NumPy training loop.

Rules:
- Implement Tasks A-C yourself using basic Python and NumPy.
- You may reuse your Week 02 batch_linear_mse_gradients implementation.
- Do not use torch.autograd or torch.optim to implement these functions.
- Run tests with: python -m pytest -q -m exercise tests/test_week03_exercises.py
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def finite_difference_gradient(
    loss_fn: Callable[[np.ndarray], float],
    parameters: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Task A: estimate each partial derivative by central differences.

    For each index i:
        grad[i] ~= (loss_fn(p + epsilon*e_i) - loss_fn(p - epsilon*e_i))
                   / (2 * epsilon)

    Args:
        loss_fn: a function mapping an ndarray to one scalar loss.
        parameters: floating-point array of any shape.
        epsilon: small positive perturbation.

    Returns:
        An ndarray of the same shape as parameters, containing derivatives.

    Important:
    - Do NOT modify the caller's parameters array in-place.
    - The shape may be 1D, 2D, or higher.
    - Use float64 working copies for numerical precision.
    """
    raise NotImplementedError("Week 03 Task A: implement finite_difference_gradient")


def gradient_check_linear_mse(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[float, float]:
    """Task B: compare manual analytical gradients to finite differences.

    Reuse your function from exercises.week02:
        batch_linear_mse_gradients(x, weight, bias, targets)

    Use the EXACT SAME loss definition:
        L = 0.5 * sum((X @ W + b - targets)**2) / batch_size

    Compute independent numerical estimates of the gradients with respect
    to weight and bias by calling finite_difference_gradient.

    Relative error for an array g:
        ||g_analytic - g_numeric||_2 /
        max(1e-12, ||g_analytic||_2 + ||g_numeric||_2)

    Returns:
        (relative_error_weight, relative_error_bias), both Python floats.

    Hint: create one loss_fn that varies W while keeping b fixed,
    and one that varies b while keeping W fixed.
    """
    raise NotImplementedError("Week 03 Task B: implement gradient_check_linear_mse")


def train_linear_regression_numpy(
    x: np.ndarray,
    targets: np.ndarray,
    initial_weight: np.ndarray,
    initial_bias: np.ndarray,
    learning_rate: float = 0.1,
    steps: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Task C: train a linear model using full-batch gradient descent.

    Args:
        x: (N, D) float64 inputs.
        targets: (N, C) float64 training targets.
        initial_weight: (D, C), initial parameters.
        initial_bias: (C,), initial bias.
        learning_rate: gradient-descent step size.
        steps: number of parameter updates.

    Returns:
        (trained_weight, trained_bias, loss_history)
        - trained_weight: same shape as initial_weight
        - trained_bias: same shape as initial_bias
        - loss_history: shape (steps + 1,)
          history[0] is the loss BEFORE any update.
          history[-1] is the loss AFTER the last update.

    Reuse the Week 02 manual gradient function; never call torch.optim.
    Do not modify initial_weight or initial_bias in place.
    """
    raise NotImplementedError("Week 03 Task C: implement train_linear_regression_numpy")
