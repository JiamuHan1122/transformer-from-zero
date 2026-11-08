"""Day 05 exercises: Sigmoid, Tanh, GELU and activation gradients.

All implementations must be your own NumPy code. PyTorch is used ONLY in
the separate test file as a reference.

Use float64 for numerical experiments. None of these functions may modify
a caller's input.

Run:
  python -m pytest -q -m exercise tests/test_week05_exercises.py
"""

from __future__ import annotations

import numpy as np

from exercises.week04 import relu_backward_numpy


def sigmoid_numpy(x: np.ndarray) -> np.ndarray:
    
    """Task A1: numerically stable elementwise Sigmoid.

    sigmoid(x) = 1 / (1 + exp(-x))

    Requirements:
      - For x >= 0, compute 1 / (1 + exp(-x)).
      - For x < 0, compute exp(x) / (1 + exp(x)).
      - Do NOT evaluate both exponentials on every element: np.where
        evaluates both branches eagerly. Use a boolean mask or equivalent.
      - Handle values as large as +/-1000 with no overflow warnings.
      - Preserve shape; do not modify x.
    """
    forward = np.zeros_like(x, dtype=np.float64)
    mask = x >= 0
    exp_x = np.exp(x[~mask])
    forward[mask] = 1 / (1 + np.exp(-x[mask]))
    forward[~mask] = exp_x / (1 + exp_x)
    return forward


def sigmoid_backward_numpy(
    pre_activation: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Task A2: manual Sigmoid backward, using the chain rule.

    s = sigmoid_numpy(pre_activation)
    sigmoid'(x) = s * (1 - s)
    grad_input = grad_output * sigmoid'(pre_activation)

    Input and grad_output have the same shape. Do not modify either.
    """
    s = sigmoid_numpy(pre_activation)
    return grad_output * s * (1 - s)


def tanh_numpy(x: np.ndarray) -> np.ndarray:
    """Task B1: elementwise hyperbolic tangent.

    Use np.tanh for a stable float64 implementation.
    Return an array of the same shape without mutating input.
    """
    return np.tanh(x)


def tanh_backward_numpy(
    pre_activation: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Task B2: manual Tanh backward.

    tanh'(x) = 1 - tanh(x)^2
    grad_input = grad_output * tanh'(pre_activation)
    """
    t = tanh_numpy(pre_activation)
    return grad_output * (1 - t**2)


def gelu_tanh_numpy(x: np.ndarray) -> np.ndarray:
    """Task C1: tanh-approximation of GELU.

    f(x) = 0.5 * x * (1 + tanh(u))
    u = sqrt(2/pi) * (x + 0.044715 * x**3)

    This is the popular tanh approximation, NOT the exact erf-based GELU.
    It should match torch.nn.functional.gelu(..., approximate="tanh")
    up to floating-point numerical differences.

    Do not use torch in the implementation.
    """
    a = np.sqrt(2 / np.pi)
    u = a * (x + 0.044715 * x**3)
    t = tanh_numpy(u)
    return 0.5 * x * (1 + t)


def gelu_tanh_backward_numpy(
    pre_activation: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Task C2: backward through the tanh-approximation of GELU.

    Let:
        a = sqrt(2/pi)
        u = a * (x + 0.044715*x**3)
        t = tanh(u)
        du_dx = a * (1 + 3*0.044715*x**2)

    Product rule + chain rule give:
        df_dx = 0.5*(1+t) + 0.5*x*(1-t**2)*du_dx

    grad_input = grad_output * df_dx

    The approximate GELU derivative can be slightly negative; do not
    clamp it to the interval [0,1].
    """
    a = np.sqrt(2 / np.pi)
    u = a * (pre_activation + 0.044715 * pre_activation**3)
    t = tanh_numpy(u)
    du_dx = a * (1 + 3 * 0.044715 * pre_activation**2)
    df_dx = 0.5 * (1 + t) + 0.5 * pre_activation * (1 - t**2) * du_dx
    return grad_output * df_dx


def activation_gradient_profiles(
    x: np.ndarray,
) -> dict[str, np.ndarray]:

    upstream = np.ones_like(x, dtype=np.float64)

    return {
        "sigmoid": sigmoid_backward_numpy(x, upstream),
        "tanh": tanh_backward_numpy(x, upstream),
        "relu": relu_backward_numpy(x, upstream),
        "gelu": gelu_tanh_backward_numpy(x, upstream),
    }