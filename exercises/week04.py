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
    return np.maximum(0, x)


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
    grad_input = np.where(pre_activation > 0, grad_output, 0)
    return grad_input

def two_layer_mlp_numpy(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:

# Step 1: 第一层线性变换
    z1 = x @ w1 + b1

# Step 2: 调用刚完成的 relu_numpy()
    hidden = relu_numpy(z1)

# Step 3: 第二层线性变换
    prediction = hidden @ w2 + b2

    return prediction


def two_layer_mlp_mse_gradients(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Step 1: 前向传播
    z1 = x @ w1 + b1
    hidden = relu_numpy(z1)
    predictions = hidden @ w2 + b2

    # Step 2: 计算损失
    N = x.shape[0]
    error = predictions - targets
    loss = 0.5 * np.sum(error ** 2) / N

    # Step 3: 输出层梯度
    grad_predictions = error / N

    # Step 4: 计算第二层参数的梯度
    grad_w2 = hidden.T @ grad_predictions
    grad_b2 = np.sum(grad_predictions, axis=0)

    # Step 5: 计算第一层参数的梯度
    grad_hidden = grad_predictions @ w2.T
    grad_z1 = relu_backward_numpy(z1, grad_hidden)
    grad_w1 = x.T @ grad_z1
    grad_b1 = np.sum(grad_z1, axis=0)

    return float(loss), grad_w1, grad_b1, grad_w2, grad_b2