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
    # 1. 创建与参数形状相同的梯度数组
    grad = np.zeros_like(parameters, dtype=np.float64)

    # 2. 遍历参数数组中的每一个元素
    for i in np.ndindex(parameters.shape):
    # 3. 仅将当前元素增加 epsilon，计算 loss_plus
        parameters_plus = parameters.astype(np.float64, copy=True)
        parameters_plus[i] += epsilon
        loss_plus = loss_fn(parameters_plus)

    # 4. 仅将当前元素减少 epsilon，计算 loss_minus
        parameters_minus = parameters.astype(np.float64, copy=True)
        parameters_minus[i] -= epsilon
        loss_minus = loss_fn(parameters_minus)

    # 5. 用中心差分计算该元素的梯度
        grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # 6. 返回 grad
    return grad

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
    # Step 1: 解析梯度
    # 复用 Week 02 的 batch_linear_mse_gradients()
    from exercises.week02 import batch_linear_mse_gradients

    # Step 2: 构建 loss_w()
    # 变化 W，固定 x、b、targets
    def loss_w(w):
        predictions = x @ w + bias
        error = predictions - targets
        return 0.5 * np.sum(error ** 2) / x.shape[0]

    # Step 3: 构建 loss_b()
    # 变化 b，固定 x、W、targets
    def loss_b(b):
        predictions = x @ weight + b
        error = predictions - targets
        return 0.5 * np.sum(error ** 2) / x.shape[0]
    
    # Step 4: 利用 Task A 的中心差分
    # 计算 numeric_grad_w 和 numeric_grad_b
    numeric_grad_w = finite_difference_gradient(
    loss_w, weight, epsilon=epsilon
    )
    numeric_grad_b = finite_difference_gradient(
    loss_b, bias, epsilon=epsilon
    )
    # Step 5: 分别计算相对误差
    # 返回两个 Python float
    analytic_loss, analytic_grad_w, analytic_grad_b = batch_linear_mse_gradients(x, weight, bias, targets)
    relative_error_weight = np.linalg.norm(analytic_grad_w - numeric_grad_w) / max(1e-12, np.linalg.norm(analytic_grad_w) + np.linalg.norm(numeric_grad_w))
    relative_error_bias = np.linalg.norm(analytic_grad_b - numeric_grad_b) / max(1e-12, np.linalg.norm(analytic_grad_b) + np.linalg.norm(numeric_grad_b))
    return float(relative_error_weight), float(relative_error_bias)   





from exercises.week02 import batch_linear_mse_gradients

def train_linear_regression_numpy(
    x, targets, initial_weight, initial_bias,
    learning_rate=0.1, steps=200
):
    # Step 1: 复制初始参数
    weight = np.array(initial_weight, dtype=np.float64, copy=True)
    bias = np.array(initial_bias, dtype=np.float64, copy=True)

    # Step 2: 计算并记录初始 Loss 与梯度
    loss, grad_w, grad_b = batch_linear_mse_gradients(
        x, weight, bias, targets
    )
    loss_history = [loss]

    # Step 3: 开始训练
    for step in range(steps):

        # 更新参数
        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        # 使用更新后的参数，计算下一轮 Loss 和梯度
        loss, grad_w, grad_b = batch_linear_mse_gradients(
            x, weight, bias, targets
        )

        # 记录更新后的 Loss
        loss_history.append(loss)

    return weight, bias, np.array(loss_history)


