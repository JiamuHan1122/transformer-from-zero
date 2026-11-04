# 第 1 周：让一切从可验证的基础开始

**本周目标**：建立项目独立环境，理解矩阵与张量的形状，用 NumPy 手写三个基础函数，并通过测试。

## Day 1：仓库与环境

- 用 VS Code 打开项目文件夹。
- 创建 `.venv`，安装依赖。
- 运行 `python scripts/verify_setup.py`。
- 运行 `python -m pytest -q`，确认默认测试通过。
- `git init -b main` 并提交第一次记录。

## Task A：线性变换 `linear_numpy`

输入：`x ∈ R^(B×D_in)`，`weight ∈ R^(D_in×D_out)`，`bias ∈ R^(D_out)`。  
输出：`y = x @ weight + bias`，形状为 `(B, D_out)`。

检查：尺寸是否相容？bias 是如何广播到 B 行的？

## Task B：稳定 Softmax `softmax_numpy`

定义：`softmax(z_i) = exp(z_i) / Σ_j exp(z_j)`。  
**不能**直接对数值为 1000 的输入调用 `exp`；先减去沿所需轴的最大值。检查每行概率总和是否约等于 1。

## Task C：交叉熵 `cross_entropy_numpy`

输入 `logits: (B, C)` 和 `targets: (B,)`，`targets[b]` 表示第 b 个样本的正确类别索引。

`L = - (1/B) Σ_b log softmax(logits[b])[targets[b]]`

请确保大 logit 下没有 `inf` / `nan`，推荐用稳定的 log-sum-exp 思路。

## 提交前验收

```bash
python -m pytest -q
python -m pytest -q -m exercise
```

第一条检查仓库基础；第二条检查你的三个函数。刚解压时第二条会因 `NotImplementedError` 失败，**这正是第一周要修复的内容**。

## 学习记录（至少写下这 4 个答案）

1. 为什么 `X @ W` 的维度是 `(B,D_out)`？
2. 为什么 Softmax 需要减去最大值，结果为什么不变？
3. Cross-Entropy 为什么应该直接从 logits 稳定计算？
4. PyTorch `autograd` 帮我们做了哪些事情，我们目前还没有自己写什么？

当三项练习测试全部通过后，再进入第二周。无需急着预写 Transformer。
