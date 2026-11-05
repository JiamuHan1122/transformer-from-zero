"""Week 02: gradients and the first weight update, implemented with NumPy.

Rules:
- Tasks A-C must use basic Python/NumPy; no torch.autograd or optimizers.
- Do not modify tests just to make them pass.
- Run one test at a time, then run the full exercise suite.
- See docs/03_week02_tasks.md for the formulas and learning plan.
"""

from __future__ import annotations

import numpy as np


def scalar_linear_gradients(
    x: float, w: float, b: float, target: float
) -> tuple[float, float, float]:
    """Task A: calculate loss and gradients for ONE scalar example.

    Model: prediction = w * x + b
    Loss:  L = 0.5 * (prediction - target) ** 2

    Return: (loss, grad_w, grad_b), all as Python floats.

    Hint: use the chain rule. First find dL/dprediction, and then
    dprediction/dw and dprediction/db.
    """
    y_pred = w * x + b
    L = 0.5 * (y_pred - target) ** 2
    dL_dy = y_pred - target
    grad_w = dL_dy * x
    grad_b = dL_dy
    return L, grad_w, grad_b


def sgd_step(
    w: float,
    b: float,
    grad_w: float,
    grad_b: float,
    learning_rate: float,
) -> tuple[float, float]:
    """Task B: do ONE gradient-descent update of scalar w and b.

    Return: (new_w, new_b), both as Python floats.

    Hint: update each parameter in the direction opposite its gradient.
    """
    new_w = w - learning_rate * grad_w
    new_b = b - learning_rate * grad_b
    return new_w, new_b


def batch_linear_mse_gradients(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:

    # Step 1: 获取 Batch Size
    N = x.shape[0]

    # Step 2: 计算预测
    predictions = x @ weight + bias  # (N, output_dim)

    # Step 3: 计算误差矩阵
    error = predictions - targets

    # Step 4: 计算损失
    # 先对每个样本的输出维度求平方和
    # 然后对全部样本求平均
    loss = 0.5 * np.mean(np.sum(error**2, axis=-1))

    # Step 5: 计算权重梯度
    grad_weight = (x.T @ error) / N

    # Step 6: 计算偏置梯度
    grad_bias = np.sum(error, axis=0) / N

    return float(loss), grad_weight, grad_bias
