# Day 08｜从 Single-Head 到 Multi-Head Attention

> Transformer From Zero — 365 Days  
> Date: 2026-11-11  
> Theme: Multi-Head Attention / Head Splitting / Head Concatenation / Output Projection  
> Status: Learning scaffold prepared; implementation and tests are intentionally unfinished.

## 一、今天为什么要从单头走向多头？

Day 07 已经实现了单头 Self-Attention。它通过一组 Query、Key、Value 投影，在一个特征子空间中计算 Token 之间的注意力分布。

但一个头只有一套投影参数和一张注意力权重矩阵。它需要用同一套匹配方式完成所有信息交互。

Multi-Head Attention 的想法是：**把特征投影到多个不同的子空间，让多个注意力头并行计算，再把各头的结果合并。**

可以把它理解成多个并行的信息读取通道：

- 每个 Head 有自己的一组 Q/K/V 投影参数；
- 每个 Head 独立计算自己的注意力权重；
- 每个 Head 根据自己的权重聚合 Value；
- 最后把所有 Head 的输出拼接，再经过一个输出投影。

不同 Head 有机会学习不同的关系模式，但不能未经分析就把某个 Head 解释为固定的人类语义角色。

## 二、先理解数学定义

设有 $h$ 个注意力头，第 $i$ 个头的计算为：

$$
\operatorname{head}_i
=
\operatorname{Attention}
\left(
QW_i^Q,\,
KW_i^K,\,
VW_i^V
\right)
$$

其中每个头拥有独立的投影矩阵。多头结果沿特征维度拼接，并通过输出矩阵：

$$
\boxed{
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
(\operatorname{head}_1,\ldots,\operatorname{head}_h)
W^O
}
$$

对于 Self-Attention，Q、K、V 都来自同一个输入 X：

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

这里的 $W^Q,W^K,W^V$ 可以看成把各个 Head 的投影矩阵沿输出维度拼接起来得到的融合矩阵。**融合矩阵不是让所有 Head 共用同一套参数；不同输出通道仍对应不同 Head 的独立投影。**

## 三、今天采用的张量约定

为让多头计算可以用 NumPy 的批量矩阵乘法一次完成，我们采用融合投影矩阵。

设：

- $B$：Batch Size；
- $T$：Token 数量；
- $D$：输入特征维度；
- $H$：注意力头数量；
- $d_k$：每个头的 Query/Key 维度；
- $d_v$：每个头的 Value 维度；
- $D_{\mathrm{out}}$：输出特征维度。

输入：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

融合投影矩阵：

$$
W^Q\in\mathbb{R}^{D\times(Hd_k)}
$$

$$
W^K\in\mathbb{R}^{D\times(Hd_k)}
$$

$$
W^V\in\mathbb{R}^{D\times(Hd_v)}
$$

输出投影矩阵：

$$
W^O\in\mathbb{R}^{(Hd_v)\times D_{\mathrm{out}}}
$$

本日不使用投影偏置。通常令 $D_{\mathrm{out}}=D$，使注意力模块的输出维度与输入维度一致。

## 四、最关键的形状变化：Split Heads

### 4.1 先做线性投影

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

形状分别为：

$$
Q,K:(B,T,Hd_k)
$$

$$
V:(B,T,Hd_v)
$$

这时 Head 维度还被合并在最后一个特征维度里。

### 4.2 把 Head 维度拆出来

以 Q 为例，先 reshape：

$$
(B,T,Hd_k)\rightarrow(B,T,H,d_k)
$$

再交换 Token 轴与 Head 轴：

$$
(B,T,H,d_k)\rightarrow(B,H,T,d_k)
$$

因此：

$$
Q_{\mathrm{heads}}\in\mathbb{R}^{B\times H\times T\times d_k}
$$

K 和 V 做同样的操作，得到：

$$
K_{\mathrm{heads}}\in\mathbb{R}^{B\times H\times T\times d_k}
$$

$$
V_{\mathrm{heads}}\in\mathbb{R}^{B\times H\times T\times d_v}
$$

**为什么要把 Head 放到第二维？** 这样每个 Batch 中的每个 Head 都有独立的 $T\times d$ 矩阵，后续可以直接用批量矩阵乘法并行计算。

### 4.3 NumPy 维度操作

Task A 将实现：

~~~~python
def split_heads_numpy(x, num_heads):
    # x: (B, T, H * D_head)
    # return: (B, H, T, D_head)
    ...
~~~~

核心不是改变数据内容，而是重新解释维度并交换轴：

$$
X_{\mathrm{split}}
=
\operatorname{transpose}
\left(
\operatorname{reshape}(X,B,T,H,d),
0,2,1,3
\right)
$$

## 五、每个 Head 独立计算 Attention

拆分后：

$$
Q_h,K_h\in\mathbb{R}^{B\times H\times T\times d_k}
$$

$$
V_h\in\mathbb{R}^{B\times H\times T\times d_v}
$$

每个 Head 的分数矩阵：

$$
S=Q_hK_h^\top
$$

在 NumPy 中，最后两个维度交换即可：

~~~~python
scores = q_heads @ k_heads.transpose(0, 1, 3, 2)
~~~~

形状变化：

$$
(B,H,T,d_k)\times(B,H,d_k,T)
\rightarrow(B,H,T,T)
$$

缩放并归一化：

$$
A=
\operatorname{Softmax}
\left(
\frac{Q_hK_h^\top}{\sqrt{d_k}}
\right)
$$

其中 Softmax 仍然沿最后一个维度计算：

$$
\sum_{j=1}^{T}A_{b,h,i,j}=1
$$

随后聚合 Value：

$$
O_h=AV_h
$$

形状：

$$
(B,H,T,T)\times(B,H,T,d_v)
\rightarrow(B,H,T,d_v)
$$

注意：每个 Head 都有自己的一张注意力矩阵，不能把 Head 维度也放进 Softmax 的归一化轴。

## 六、Merge Heads 与输出投影

每个 Head 的输出为：

$$
O_h\in\mathbb{R}^{B\times H\times T\times d_v}
$$

先把 Head 轴移回 Token 轴之后：

$$
(B,H,T,d_v)\rightarrow(B,T,H,d_v)
$$

再合并最后两个维度：

$$
(B,T,H,d_v)\rightarrow(B,T,Hd_v)
$$

这就是 Task B 的逆操作：

~~~~python
def merge_heads_numpy(x):
    # x: (B, H, T, D_head)
    # return: (B, T, H * D_head)
    ...
~~~~

最后执行输出投影：

$$
\boxed{
Y=\operatorname{MergeHeads}(O_h)W^O
}
$$

因此：

$$
(B,T,Hd_v)\times(Hd_v,D_{\mathrm{out}})
\rightarrow(B,T,D_{\mathrm{out}})
$$

输出投影 $W^O$ 让拼接后的多头表示进一步进行可学习的线性组合。

## 七、把完整计算链串起来

~~~~text
Input X: (B, T, D)
       |
       +------ X @ WQ ------> Q: (B, T, H*dk)
       |
       +------ X @ WK ------> K: (B, T, H*dk)
       |
       +------ X @ WV ------> V: (B, T, H*dv)
                                  |
                         Split Heads
                                  |
             Qh, Kh: (B, H, T, dk)
             Vh:     (B, H, T, dv)
                                  |
                    Qh @ Kh^T / sqrt(dk)
                                  |
                     Softmax(axis=-1)
                                  |
                     Attention Weights
                         (B, H, T, T)
                                  |
                            Weights @ Vh
                                  |
                     Head Outputs (B,H,T,dv)
                                  |
                          Merge Heads
                                  |
                           (B,T,H*dv)
                                  |
                              @ WO
                                  |
                       Output (B,T,D_out)
~~~~

## 八、Task A — 实现 Split Heads

File: exercises/week08.py  
Function: split_heads_numpy(x, num_heads)

输入：

- x: (B, T, H * D_head)
- num_heads: H

输出：

- (B, H, T, D_head)

要求：

1. 只使用 NumPy。
2. 检查输入必须是三维数组。
3. num_heads 必须是正整数。
4. 最后一个维度必须能被 num_heads 整除，否则抛出 ValueError。
5. 不修改输入数组。
6. 正确区分 reshape 与 transpose：先 reshape 成 (B,T,H,D_head)，再 transpose 成 (B,H,T,D_head)。

## 九、Task B — 实现 Merge Heads

File: exercises/week08.py  
Function: merge_heads_numpy(x)

输入：

- x: (B, H, T, D_head)

输出：

- (B, T, H * D_head)

要求：

1. 输入必须是四维数组，否则抛出 ValueError。
2. 先 transpose(0, 2, 1, 3)，再 reshape。
3. 不修改输入数组。
4. 与 Task A 构成可逆操作：对合法输入，merge_heads_numpy(split_heads_numpy(x,H)) 应恢复原始数值和形状。

## 十、Task C — 实现 Multi-Head Self-Attention

File: exercises/week08.py  
Function: multi_head_attention_numpy(x, w_q, w_k, w_v, w_o, num_heads)

输入形状：

| Variable | Shape |
|---|---|
| x | (B, T, D) |
| w_q | (D, H * dk) |
| w_k | (D, H * dk) |
| w_v | (D, H * dv) |
| w_o | (H * dv, D_out) |

按以下顺序实现：

1. 用 x 分别乘 w_q、w_k、w_v，得到融合投影 Q、K、V。
2. 调用 Task A，把 Q、K、V 拆成 Head 维度。
3. 计算每个 Head 的 QK^T，缩放因子为 sqrt(dk)。
4. 复用 Day 07 的 softmax_numpy，沿 axis=-1 计算权重。
5. 使用权重乘 V，得到各 Head 输出。
6. 调用 Task B 合并 Head。
7. 乘 w_o，得到最终输出。
8. 返回 (output, attention_weights)。

返回形状：

| Return | Shape |
|---|---|
| output | (B, T, D_out) |
| attention_weights | (B, H, T, T) |

约束：

- 只在 exercises/week08.py 中使用 NumPy。
- 可以从 exercises.week07 导入并复用 softmax_numpy。
- 不允许在实现文件中导入 torch、torch.nn 或 torch.autograd。
- 不修改 x、w_q、w_k、w_v、w_o。
- 本日不加入 bias、Mask、Dropout、Positional Encoding、Residual 或 LayerNorm。
- dk 与 dv 可以不同；不要错误地把它们写成必须相等。
- 注意力权重沿 Key/Token 轴归一化，即 axis=-1。

## 十一、测试命令

从项目根目录运行：

~~~~powershell
conda activate tfz

python -m pytest -q -m exercise tests/test_week08_exercises.py::test_split_heads_shape_order_and_values
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_split_heads_rejects_invalid_dimensions
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_merge_heads_restores_original_tensor
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_split_and_merge_are_inverse_operations
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_multi_head_attention_matches_torch_reference
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_multi_head_attention_supports_different_dk_and_dv
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_single_head_case_matches_day07_attention
python -m pytest -q -m exercise tests/test_week08_exercises.py::test_attention_heads_have_independent_weight_patterns
python -m pytest -q -m exercise tests/test_week08_exercises.py
~~~~

测试中的 PyTorch 只作为独立参考。实现文件不允许调用 PyTorch。初始 NotImplementedError 是有意保留的练习占位符；完成后预期 8 passed。

## 十二、思考题与验收

1. 为什么多个 Head 需要不同的 Q/K/V 投影矩阵，而不是简单地把同一个 Attention 结果复制 H 份？
2. 为什么 Split Heads 之后要把 Head 维度移动到第二维？
3. 为什么 Attention Weights 的形状是 (B,H,T,T)？
4. 为什么 Softmax 应沿最后一维计算，而不是沿 Head 维度计算？
5. 为什么拼接各 Head 后还需要输出投影矩阵 W^O？
6. 当 H=1、dk=D、dv=D 且 W^O 为单位矩阵时，Multi-Head 实现应退化为 Day 07 的单头 Self-Attention 吗？为什么？
7. 多头机制提供了多个可学习的信息交互子空间，但是否意味着每个 Head 一定会学到不同、可解释的语义关系？

## 十三、完成标准

- [ ] Task A：Split Heads 实现及测试通过。
- [ ] Task B：Merge Heads 实现及测试通过。
- [ ] Task C：Multi-Head Self-Attention 实现及测试通过。
- [ ] 全部 8 项测试通过。
- [ ] 能够独立推导 (B,T,D) 到 (B,H,T,dk) 的形状变化。
- [ ] 能解释每个 Head 的注意力权重为何独立归一化。
- [ ] 能解释输出投影 W^O 的作用。
- [ ] 完成 Demo，检查不同 Head 的注意力矩阵。
- [ ] 整理 Day 08 正式学习笔记，核对所有数学公式的 Markdown 渲染。
- [ ] 将代码、测试和笔记提交 GitHub，并在 VS Code 中 pull 同步。

完成 Task A–C 并通过测试后，再进入 Causal / Padding Mask。不要把尚未实现的练习框架当作已经完成的代码。
