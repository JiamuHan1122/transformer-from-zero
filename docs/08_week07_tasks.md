# Day 07｜从线性映射到 Self-Attention

> Transformer From Zero — 365 Days  
> Date: 2026-11-10  
> Theme: Stable Softmax / Query-Key-Value / Scaled Dot-Product Attention / Self-Attention  
> Status: Scaffold prepared; implementation and tests are intentionally unfinished.

## 一、今天为什么学习 Attention？

Day 06 的 3D 线性层完成了：

\[
Y=XW+b,\qquad X\in\mathbb{R}^{B\times T\times D}
\]

同一组参数被所有 Token 共享，但每个位置仍然只对自己的输入向量做线性变换。它不会直接读取其他 Token 的特征。

Self-Attention 要解决的问题是：**让每个 Token 根据当前内容，从同一序列的其他 Token 中选择并聚合信息。**

例如，一个代词的含义可能需要结合前文才能确定。Attention 为这种上下文信息交互提供了可学习的机制。

今天只实现单头、无 Mask 的前向计算，不引入 Multi-Head、Causal Mask、Dropout、残差连接或 LayerNorm。

## 二、Q、K、V 分别是什么？

从输入 X 通过三组独立的线性投影得到：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

- **Query**：当前 Token 用来提出查询的表示。
- **Key**：每个 Token 用来与 Query 计算匹配程度的表示。
- **Value**：真正被加权聚合、传递给输出的内容表示。

Q、K、V 是模型学习到的三组表示，不是人工标注的语义类别。它们来自同一个 X，但投影矩阵不同，因此承担不同计算角色。

## 三、Scaled Dot-Product Attention 的完整公式

首先计算 Query 与所有 Key 的点积：

\[
S=QK^\top
\]

再进行缩放：

\[
S_{\mathrm{scaled}}=\frac{QK^\top}{\sqrt{d_k}}
\]

当 Q、K 各维分量近似独立、均值为 0、方差为 1 时，点积的方差随 \(d_k\) 增长，标准差约为 \(\sqrt{d_k}\)。缩放用于控制 Softmax 输入的数值尺度，避免维度增大时分数过大、分布过尖锐。

沿 Key 维度做 Softmax：

\[
A=\operatorname{softmax}(S_{\mathrm{scaled}},\mathrm{axis}=-1)
\]

每个 Query 对所有 Key 的权重和为 1：

\[
\sum_{j=1}^{T_k}A_{ij}=1
\]

最后聚合 Value：

\[
O=AV
\]

对第 i 个 Query：

\[
O_i=\sum_{j=1}^{T_k}A_{ij}V_j
\]

注意：A 是“从哪里取多少信息”的权重矩阵；V 才是被聚合的内容。单独的 A 不是最终的内容输出。

## 四、张量形状推导

### 4.1 Scaled Dot-Product Attention

允许 Query 长度 \(T_q\) 与 Key/Value 长度 \(T_k\) 不同：

| Variable | Shape |
|---|---|
| q | (B, Tq, dk) |
| k | (B, Tk, dk) |
| v | (B, Tk, dv) |
| k.transpose(0, 2, 1) | (B, dk, Tk) |
| scores | (B, Tq, Tk) |
| attention weights | (B, Tq, Tk) |
| output | (B, Tq, dv) |

NumPy 的三维矩阵乘法：

~~~python
scores = q @ k.transpose(0, 2, 1)
output = attention_weights @ v
~~~

不要使用 k.T：对三维数组，.T 会反转所有轴，而这里仅需要交换 Key 的最后两个轴。

### 4.2 Self-Attention

输入与投影矩阵：

| Variable | Shape |
|---|---|
| x | (B, T, D) |
| w_q | (D, dk) |
| w_k | (D, dk) |
| w_v | (D, dv) |
| q, k | (B, T, dk) |
| v | (B, T, dv) |
| output | (B, T, dv) |
| weights | (B, T, T) |

本日不使用投影偏置。注意 dk 与 dv 不必相同。

## 五、Task A — 数值稳定的 Softmax

File: exercises/week07.py  
Function: softmax_numpy(x, axis=-1)

公式：

\[
\operatorname{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

稳定实现：

\[
m=\max(x,\mathrm{axis},\mathrm{keepdims=True})
\]

\[
p_i=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}}
\]

要求：

1. 支持一维、二维、三维及更高维数组。
2. 支持负数轴与不同 axis。
3. 返回与输入相同的形状，dtype 为 float64。
4. 不修改输入数组。
5. 对类似 [1000, 1001, 1002] 的输入保持有限数值。

容易出错的地方：必须在指数运算前减去指定轴上的最大值；分母也必须沿同一个 axis 求和，并使用 keepdims=True 保留广播维度。

## 六、Task B — Scaled Dot-Product Attention

File: exercises/week07.py  
Function: scaled_dot_product_attention_numpy(q, k, v)

输入：

- q: (B, Tq, dk)
- k: (B, Tk, dk)
- v: (B, Tk, dv)

计算：

\[
S=\frac{qk^\top}{\sqrt{d_k}}
\]

\[
A=\operatorname{softmax}(S,\mathrm{axis}=-1)
\]

\[
O=Av
\]

返回 (output, weights)。

要求：

- 只用 NumPy。
- 使用 k.transpose(0, 2, 1)。
- 支持 Tq 不等于 Tk。
- 支持 dk 不等于 dv。
- 权重沿最后一维归一化。
- 不修改 q、k、v。
- 本日不加入 Mask。

## 七、Task C — Self-Attention

File: exercises/week07.py  
Function: self_attention_numpy(x, w_q, w_k, w_v)

输入：

- x: (B, T, D)
- w_q: (D, dk)
- w_k: (D, dk)
- w_v: (D, dv)

按顺序：

1. q = x @ w_q
2. k = x @ w_k
3. v = x @ w_v
4. 调用 Task B。
5. 返回 (output, weights)。

要求：不要复制粘贴另一套 Attention 计算；应复用已经实现的 Task B。

## 八、测试命令

从项目根目录运行：

~~~powershell
conda activate tfz

python -m pytest -q -m exercise tests/test_week07_exercises.py::test_softmax_numpy_known_values_and_row_sums
python -m pytest -q -m exercise tests/test_week07_exercises.py::test_softmax_numpy_supports_arbitrary_axis_and_matches_torch
python -m pytest -q -m exercise tests/test_week07_exercises.py::test_softmax_numpy_is_stable_for_large_positive_and_negative_values
python -m pytest -q -m exercise tests/test_week07_exercises.py::test_scaled_dot_product_attention_matches_torch_and_shapes
python -m pytest -q -m exercise tests/test_week07_exercises.py::test_scaled_dot_product_attention_one_key_returns_its_value
python -m pytest -q -m exercise tests/test_week07_exercises.py::test_self_attention_matches_torch_and_supports_different_dv_and_dk
python -m pytest -q -m exercise tests/test_week07_exercises.py::test_attention_actually_mixes_information_between_tokens
python -m pytest -q -m exercise tests/test_week07_exercises.py
~~~

测试中的 PyTorch 仅作为独立数值参考；exercises/week07.py 不允许导入或调用 PyTorch。初始 NotImplementedError 是有意保留的练习占位符。完成后预期 7 passed。

## 九、最终展示 Demo

完成 Task A–C 并通过测试后运行：

~~~powershell
python scripts/day07_attention_demo.py
~~~

Demo 将展示：

- Q/K/V 与 Attention 输出的张量形状。
- 一条序列的 Attention 权重矩阵。
- 每个 Query 的权重和是否为 1。
- 修改一个输入 Token 后，其他位置的输出是否发生变化。

这是一项前向机制演示，不是模型训练实验。不要把随机初始化的 Attention 权重解释成已经学到的语言语义。

预期本地输出：

- results/day07/metrics.json
- results/day07/attention_weights.png

## 十、思考题与验收

1. 为什么 A 的形状是 (B,Tq,Tk)，而不是 (B,T,D)？
2. 为什么 Attention 权重需要乘以 V？
3. 为什么缩放因子是 \(\sqrt{d_k}\)，而不是 \(d_k\)？
4. 为什么 Softmax 应沿最后一个维度计算？
5. 如果只有一个 Key，Attention 输出为什么必然等于该 Key 对应的 Value？
6. Self-Attention 中，为什么改变一个 Token 可能改变其他 Token 的输出？

完成标准：

- [ ] Task A：Softmax 实现及 3 项测试通过。
- [ ] Task B：Attention 前向实现及 2 项测试通过。
- [ ] Task C：Self-Attention 实现及 2 项测试通过。
- [ ] 全部 7 项测试通过。
- [ ] 运行 Day 07 Demo，检查权重矩阵和跨 Token 信息交互。
- [ ] 根据实际运行结果整理正式学习笔记与知识词云。

完成后再提交实现，不要把未完成的练习框架误认为已经完成的代码。
