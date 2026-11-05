# Week 02｜从 Loss 到梯度，再到第一次参数更新

> Transformer From Zero — 365 Days  
> 学习主题：Chain Rule / Gradient Descent / Manual Backpropagation  
> 使用环境：`tfz`（Python 3.11、NumPy、PyTorch；本轮 NumPy 代码在 CPU 上即可）

## 学习目标

本轮不直接调用 `torch.optim` 或 `loss.backward()` 来实现练习，而是先自己推导梯度，再通过 PyTorch 的 Autograd **检验**答案。

完成后应能够：

1. 解释导数/梯度的方向和大小，以及负梯度更新的含义。
2. 对标量模型 `prediction = w*x+b` 和 `L=0.5*(prediction-target)^2` 求梯度。
3. 手动更新一次 `w`、`b`，检验 Loss 是否下降。
4. 从单样本标量推广到批量矩阵模型的梯度，并与 Autograd 对照。

## 例题回顾（已在对话中完成）

已知 `x=2, w=1, b=0, target=5, learning_rate=0.1`。

前向计算：

\[
\hat y=wx+b=2,\qquad L=\frac12(2-5)^2=4.5
\]

链式法则：

\[
\frac{\partial L}{\partial w}=(\hat y-y)x=-6,
\qquad
\frac{\partial L}{\partial b}=\hat y-y=-3
\]

更新后：

\[
w'=1-0.1(-6)=1.6,\qquad b'=0-0.1(-3)=0.3
\]

新的预测：`1.6 * 2 + 0.3 = 3.5`，新损失为 `0.5 * (3.5 - 5)**2 = 1.125`。

**提醒：Loss 在这一示例下降，不意味着任意数据和任意学习率的一步更新都必然下降。**

## Task A：标量线性模型的梯度

文件：`exercises/week02.py` 中的 `scalar_linear_gradients()`。

返回 `(loss, grad_w, grad_b)`。先在纸上推导，再把公式写成 Python。

单独运行测试：

```powershell
python -m pytest -q -m exercise tests/test_week02_exercises.py::test_scalar_linear_gradients
```

## Task B：手写一次 SGD 更新

文件：`exercises/week02.py` 中的 `sgd_step()`。

只处理两个标量参数 `w`、`b`。不要调用 PyTorch Optimizer。

单独运行测试：

```powershell
python -m pytest -q -m exercise tests/test_week02_exercises.py::test_sgd_step_one_update
```

## Task C：推广到批量矩阵求导

输入形状：

| 变量 | 形状 |
| --- | --- |
| `x` | `(B, D)` |
| `weight` | `(D, O)` |
| `bias` | `(O,)` |
| `targets` | `(B, O)` |
| `predictions` | `(B, O)` |

模型为：

\[
P=XW+b
\]

损失的**精确定义**（先在输出维求和，后在 batch 维求平均）：

\[
L=\frac{1}{2B}\sum_{i=1}^B\sum_{j=1}^O(P_{ij}-Y_{ij})^2
\]

需要返回：`(loss: float, grad_weight: ndarray, grad_bias: ndarray)`。梯度形状分别与 `weight`、`bias` 一致。

建议自行推导：

- 令误差矩阵 `E = P - Y`，形状 `(B,O)`。
- 考虑 `W` 的每个元素会影响多个样本、多个输出，因此如何将各样本贡献汇总？
- 注意 Batch 平均带来的 `1/B`。
- 注意 `np.mean((P-Y)**2)` 不等于本任务规定的 Loss，除非 `O=1`。

单独运行测试：

```powershell
python -m pytest -q -m exercise tests/test_week02_exercises.py::test_batch_linear_mse_gradients_against_autograd
python -m pytest -q -m exercise tests/test_week02_exercises.py::test_batch_gradient_descent_reduces_loss
```

测试会使用 PyTorch Autograd 计算**参考梯度**。你的解答应只用 NumPy 计算。

## 一次运行完整测试

在项目根目录（含 `pyproject.toml`）执行：

```powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week02_exercises.py
```

所有任务完成后期望：`4 passed`。

在开始写代码之前，测试报 `NotImplementedError` **是预期行为**，代表练习尚未实现。

注意：项目的 `pyproject.toml` 默认排除了 `exercise` 标记，因此不要省略 `-m exercise`。

## Git 提交

测试通过后：

```powershell
git add exercises/week02.py tests/test_week02_exercises.py docs/03_week02_tasks.md
git commit -m "feat: implement week02 manual gradients"
git push
```

如果 VS Code 的终端仍无法直连 GitHub，可以使用此前成功的一次性代理命令：

```powershell
git -c http.proxy=http://127.0.0.1:7897 push
```

> 如果代理软件端口已经变化，请使用实际端口。

## 本轮结束时必须能回答

1. 为什么 `grad_w` 可能是负数？负梯度更新为什么可能增大参数？
2. 对于 `(B,D) @ (D,O)`，`grad_weight` 的形状为什么必须是 `(D,O)`？
3. 为什么对一个 batch 求平均时，梯度也要按 batch size 归一化？
4. PyTorch Autograd 如何成为你手写梯度的对照，而不是手写实现的替代品？
