"""Day 04 exercises: ReLU and a two-layer MLP, using NumPy.

Goal: learn how nonlinearity lets a network represent functions that one
affine transformation cannot express.

Rules:
- Write the forward and backward computations by hand with NumPy.
- Do not call torch.autograd, torch.nn or torch.optim in the solutions.
- ReLU derivative at exactly zero is defined as 0 in this exercise.
- Keep the Day 03 loss convention: L = sum(error**2) / (2 * batch_size).

Tests:
    python -m pytest -q -m exercise tests/test_week04_exercises.py
"""

from __future__ import annotations

import numpy as np


def relu_numpy(x: np.ndarray) -> np.ndarray:
    """Task A1: elementwise ReLU.

    Formula: relu(x) = max(0, x)
    Preserve the input shape.
    """
    raise NotImplementedError("Day 04 Task A1: implement relu_numpy")


def relu_backward_numpy(
    pre_activation: np.ndarray,
    grad_output: np.ndarray,
) -> np.ndarray:
    """Task A2: propagate an upstream gradient through ReLU.

    Args:
        pre_activation: inputs BEFORE ReLU, arbitrary shape.
        grad_output: derivative dL/d(relu(pre_activation)), same shape.

    Returns:
        dL/d(pre_activation), same shape.

    Definition for this exercise:
        derivative is 1 for pre_activation > 0, and 0 otherwise.
        In particular, derivative at exactly 0 is defined to be 0.

    Do not modify either caller-supplied array.
    """
    raise NotImplementedError("Day 04 Task A2: implement relu_backward_numpy")


def two_layer_mlp_numpy(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Task B: forward pass through a two-layer MLP.

    Shapes:
        x:  (N, D)
        w1: (D, H)
        b1: (H,)
        w2: (H, C)
        b2: (C,)
        prediction: (N, C)

    Computation:
        z1 = x @ w1 + b1
        hidden = relu_numpy(z1)
        prediction = hidden @ w2 + b2

    Hint: reuse your relu_numpy function from Task A.
    """
    raise NotImplementedError("Day 04 Task B: implement two_layer_mlp_numpy")


def two_layer_mlp_mse_gradients(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Task C: manually backpropagate through the two-layer MLP.

    Same shapes as Task B, plus targets: (N, C).

    Define:
        z1 = x @ w1 + b1
        hidden = ReLU(z1)
        predictions = hidden @ w2 + b2
        error = predictions - targets
        loss = 0.5 * sum(error**2) / N

    Return in this exact order:
        (loss, grad_w1, grad_b1, grad_w2, grad_b2)

    The scalar loss must be a Python float.

    Suggested chain-rule sequence:
        1. dL/d(predictions) = error / N.
        2. Compute gradients for w2 and b2 from hidden.
        3. Propagate dL/d(hidden) through w2.T.
        4. Use relu_backward_numpy(z1, grad_hidden).
        5. Compute gradients for w1 and b1 from x.

    Caution: the denominator is N, NOT N*C.
    """
    raise NotImplementedError("Day 04 Task C: implement two_layer_mlp_mse_gradients")
