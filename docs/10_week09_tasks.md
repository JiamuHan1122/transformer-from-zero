# Day 09｜Masked Attention：Causal Mask、Padding Mask 与 Masked Softmax

> **项目：** Transformer From Zero · 365 Days  
> **日期：** 2026-11-12  
> **阶段：** Attention 核心机制 · Day 09  
> **今日主题：** Causal Mask、Padding Mask、Masked Softmax、Masked Multi-Head Attention  
> **学习方式：** 先理解可见性约束，再实现掩码与数值稳定的归一化，最后整合到多头注意力。

---

## 一、今天为什么需要 Mask？

Day 08 已经实现了无掩码的 Multi-Head Self-Attention。普通 Self-Attention 允许每个 Query 访问序列中的所有 Key。

但不同任务对信息可见性有不同约束：

1. **自回归生成（GPT）**：位置 i 只能读取位置 i 及其之前的信息，不能读取未来 Token。
2. **变长 Batch**：为了对齐长度而补充的 PAD 位置不是有效内容，不应作为 Key 被关注。
3. **数值稳定性**：遮挡后必须重新正确归一化；整行全部被遮挡时还必须避免 NaN。

今天统一约定：

- Boolean mask 中 \`True\` 表示 blocked / 不可见；
- \`False\` 表示 allowed / 可见；
- Mask 应在 Softmax 之前应用到 Attention Scores。

## 二、Task A：Causal Mask（因果掩码）

### 2.1 可见性约束

对长度为 T 的序列，Query 位置 i 只能读取 Key 位置 j 满足 j ≤ i 的信息。

因此，允许访问矩阵是下三角矩阵；Boolean blocked mask 则是严格上三角矩阵：

$$
M^{\\mathrm{causal}}_{ij}
=
\\begin{cases}
\\mathrm{False}, & j\\le i,\\\\
\\mathrm{True}, & j>i.
\\end{cases}
$$

例如 T=4：

| Query \\ Key | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 允许 | 遮挡 | 遮挡 | 遮挡 |
| 1 | 允许 | 允许 | 遮挡 | 遮挡 |
| 2 | 允许 | 允许 | 允许 | 遮挡 |
| 3 | 允许 | 允许 | 允许 | 允许 |

### 2.2 Mask 如何进入 Attention？

Day 08 的注意力分数：

$$
S=\\frac{QK^\\top}{\\sqrt{d_k}}
$$

把 Boolean mask 转换为加性掩码：允许位置加 0，遮挡位置加负无穷：

$$
\\widetilde S_{ij}
=
\\begin{cases}
S_{ij}, & M_{ij}=\\mathrm{False},\\\\
-\\infty, & M_{ij}=\\mathrm{True}.
\\end{cases}
$$

然后再计算：

$$
A=\\operatorname{Softmax}(\\widetilde S)
$$

因为 exp(-∞)=0，被遮挡位置的权重为 0。**先 Mask、后 Softmax**，才能保证剩余可见位置重新归一化。

### 2.3 实现要求

文件：\`exercises/week09.py\`  
函数：\`causal_mask_numpy(seq_len)\`

- 输入正整数序列长度 T；
- 返回形状为 (T,T) 的 Boolean 数组；
- 严格上三角为 True，对角线及下三角为 False；
- 非法长度（非正整数、布尔值）抛出 ValueError；
- 不依赖 PyTorch。

## 三、Task B：Padding Mask（填充掩码）

### 3.1 为什么需要 Padding Mask？

变长句子组成 Batch 时，通常需要填充到相同长度。例如：

\`\`\`text
样本 A: I    love   AI    PAD
样本 B: I    love   PAD   PAD
\`\`\`

PAD 不是有效语义 Token。作为 Key 时，PAD 位置不应被任何 Query 读取。

给定 token_ids，定义：

$$
M^{\\mathrm{pad}}_{b,j}
=
[\\,token\\_ids_{b,j}=pad\\_token\\_id\\,]
$$

其中方括号表示条件成立时为 True。输出形状为 (B,T)。

### 3.2 广播到注意力分数

多头注意力分数的形状是：

$$
S\\in\\mathbb{R}^{B\\times H\\times T\\times T}
$$

Padding mask 的形状是 (B,T)，需要扩展为：

$$
(B,T)\\rightarrow(B,1,1,T)
$$

这样它遮挡每个 Batch 中的 PAD Key，并在 Head 轴和 Query 轴广播。

Causal mask 的形状 (T,T) 扩展为：

$$
(T,T)\\rightarrow(1,1,T,T)
$$

两者组合时使用逻辑或：

$$
M^{\\mathrm{combined}}
=
M^{\\mathrm{causal}}\\lor M^{\\mathrm{pad}}
$$

**边界说明：** Padding Mask 在今天的实现中屏蔽 PAD Key 列。PAD Query 对应的输出位置不自动等于零；训练时通常还需要在 Loss 中忽略 PAD 目标位置。

### 3.3 实现要求

函数：\`padding_mask_numpy(token_ids, pad_token_id)\`

- token_ids 必须是二维 NumPy 数组 (B,T)；
- 返回同形状 Boolean 数组；
- PAD 位置为 True；
- 不修改输入数组。

## 四、Task C：Masked Softmax（数值稳定的掩码归一化）

这是今天最需要认真处理的数值细节。

普通 Softmax：

$$
\\operatorname{Softmax}(x_i)
=
\\frac{e^{x_i}}{\\sum_j e^{x_j}}
$$

稳定实现通常先减去最大值。但有了 mask 后，不能简单地把整行填成 -∞ 再直接做 max 和减法：**如果整行全部被遮挡，最大值也是 -∞，会出现 -∞ - (-∞) = NaN。**

### 4.1 正确的行为约定

对于每一行：

- 可见位置正常进行稳定 Softmax；
- 被遮挡位置权重严格为 0；
- 如果整行都被遮挡，整行返回全 0；
- 输出必须全部有限，不允许 NaN 或 Inf。

可见位置集合记为 V，则：

$$
A_i=
\\begin{cases}
\\displaystyle
\\frac{\\exp(S_i-m)}
{\\sum_{j\\in V}\\exp(S_j-m)}, & i\\in V,\\\\[8pt]
0, & i\\notin V,
\\end{cases}
$$

其中 m 是该行可见分数的最大值。若 V 为空，则整行输出 0。

### 4.2 实现要求

函数：\`masked_softmax_numpy(scores, mask, axis=-1)\`

- scores 为 NumPy 数组；
- mask 必须是 Boolean 数组，且可广播到 scores；
- True 表示遮挡；
- 沿指定 axis 归一化；
- 遮挡位置严格为 0；
- 全遮挡行返回全 0；
- 对极大正负分数保持数值稳定；
- 对非法 mask 类型或不可广播形状给出清晰异常。

注意：不要把全遮挡行当作普通 Softmax 行处理。

## 五、Task D：Masked Multi-Head Self-Attention

把前三项整合进 Day 08 的多头注意力计算。

函数：\`multi_head_attention_with_mask_numpy(x, w_q, w_k, w_v, w_o, num_heads, mask=None)\`

输入形状：

| 参数 | 形状 |
|---|---|
| x | (B,T,D) |
| w_q | (D,H·dk) |
| w_k | (D,H·dk) |
| w_v | (D,H·dv) |
| w_o | (H·dv,D_out) |
| mask | 可广播到 (B,H,T,T) |

计算流程：

1. 使用融合投影计算 Q、K、V；
2. 复用 Day 08 的 \`split_heads_numpy\`；
3. 计算缩放点积 Attention Scores；
4. 若 mask 不为 None，使用 \`masked_softmax_numpy\`；
5. 用 Attention Weights 聚合 Value；
6. 复用 Day 08 的 \`merge_heads_numpy\`；
7. 进行输出投影；
8. 返回 output 与 attention_weights。

输出形状：

$$
output\\in\\mathbb{R}^{B\\times T\\times D_{out}}
$$

$$
attention\\_weights\\in\\mathbb{R}^{B\\times H\\times T\\times T}
$$

实现限制：

- 核心实现只使用 NumPy；
- 复用 Day 08 的 Split/Merge Heads；
- 不使用 PyTorch 完成前向实现；
- 不修改调用者传入的数组；
- 支持 dk 与 dv 不相等；
- 不加入 Bias、Dropout、Residual、LayerNorm；
- 今天先实现通用 mask 输入，Causal 与 Padding mask 在调用端组合并扩展维度。

## 六、测试与验收

从项目根目录执行：

\`\`\`powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week09_exercises.py
\`\`\`

测试覆盖：

1. Causal Mask 的形状、类型与严格上三角方向；
2. 非法序列长度；
3. Padding Mask 对 PAD Token 的识别；
4. Masked Softmax 的正常归一化与遮挡零权重；
5. 全遮挡行返回全零且无 NaN；
6. 极端分数下的数值稳定性；
7. Mask 广播与非法形状检查；
8. Masked Multi-Head Attention 与独立 PyTorch 参考结果一致；
9. Causal Mask 下未来 Token 不影响过去位置的输出；
10. Causal 与 Padding Mask 组合后，所有被遮挡位置权重为零。

## 七、今天需要真正理解的四件事

1. **Causal Mask 不是删掉未来 Token**，而是限制每个 Query 的信息可见范围。
2. **Padding Mask 与 Causal Mask 的目的不同**：一个排除填充位置，一个阻断未来信息。
3. **Mask 必须在 Softmax 前应用**，这样可见位置才能正确归一化。
4. **全遮挡行需要单独处理**，否则稳定 Softmax 仍可能出现 NaN。

## 八、完成标准

- [ ] Task A：Causal Mask 实现正确。
- [ ] Task B：Padding Mask 实现正确。
- [ ] Task C：Masked Softmax 通过数值稳定性测试。
- [ ] Task D：Masked Multi-Head Attention 与参考实现一致。
- [ ] 全部测试通过。
- [ ] 能解释 mask 的广播形状。
- [ ] 能解释为什么全遮挡行必须返回零。
- [ ] Demo 生成注意力热图与扰动对照。
- [ ] 完成正式学习笔记与统一系列风格词云。
- [ ] 提交 GitHub，并同步 VS Code。

**完成今天的 Masked Attention 后，我们才具备构建 GPT 因果注意力层的关键前提。**
