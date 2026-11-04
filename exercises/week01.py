"""Week 01: implement basic numerical operations *yourself* using NumPy.

Do not replace these exercises with torch.nn or scipy functions.
See docs/02_week01_tasks.md for formulas and acceptance criteria.
"""

from __future__ import annotations

import numpy as np


def linear_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Return x @ weight + bias.

    Shapes: x=[batch, input_dim], weight=[input_dim, output_dim],
    bias=[output_dim]. Return [batch, output_dim].
    输入特征维度决定权重矩阵的行数。
    输出特征维度决定权重矩阵的列数。
    Batch Size 和 Sequence Length 在这种逐 Token 的线性投影中保持不变。
    """
    return x @ weight + bias



def softmax_numpy(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along 'axis'."""

    # 1. 找到每一行的最大值
    max_value = np.max(logits, axis=axis, keepdims=True)

    # 2. 减去最大值，避免指数溢出
    shifted = logits - max_value

    # 3. 计算指数
    exp_values = np.exp(shifted)

    # 4. 归一化
    sum_exp = np.sum(exp_values, axis=axis, keepdims=True)

    return exp_values / sum_exp





def cross_entropy_numpy(
    logits: np.ndarray,
    targets: np.ndarray
) -> float:

    # Step 1: 每行减去最大值，确保数值稳定
    max_per_row = np.max(logits, axis=1, keepdims=True)   # (N, 1)
    shifted = logits - max_per_row                        # (N, C)

    # Step 2: 计算稳定的 log_softmax
    # log_softmax = shifted - log(sum(exp(shifted)))
    sum_exp = np.sum(np.exp(shifted), axis=1, keepdims=True)  # (N, 1)
    log_norm = np.log(sum_exp)                                # (N, 1)
    log_probs = shifted - log_norm                            # (N, C)

    # Step 3: 根据 targets 提取正确类别的 log 概率
    N = logits.shape[0]
    rows = np.arange(N)
    correct_log_probs = log_probs[rows, targets]              # (N,)

    # Step 4: 取负数并求平均，返回 float
    return float(-np.mean(correct_log_probs))

