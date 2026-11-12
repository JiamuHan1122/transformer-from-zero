# Day 09｜Masked Attention：Causal Mask、Padding Mask 与 Masked Softmax

> **项目：** Transformer From Zero · 365 Days  
> **日期：** 2026-11-12  
> **阶段：** Attention 核心机制 · Day 09  
> **今日主题：** Causal Mask、Padding Mask、Masked Softmax、Masked Multi-Head Attention  
> **状态：** 已完成；15 项练习测试通过；Demo 验证成功

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
- Mask 在 Softmax 之前应用到 Attention Scores。

## 二、Task A：Causal Mask（因果掩码）

对长度为 T 的序列，Query 位置 i 只能读取 Key 位置 j 满足 j ≤ i 的信息。

$$
M^{\mathrm{causal}}_{ij}
=
\begin{cases}
\mathrm{False}, & j\le i,\
\mathrm{True}, & j>i.
\end{cases}
$$

因此，允许访问矩阵是下三角（含对角线）；Boolean blocked mask 是严格上三角。

实现要求：

- 输入正整数序列长度 T；
- 返回形状为 (T,T) 的 Boolean 数组；
- 严格上三角为 True，对角线及下三角为 False；
- 非法长度（非正整数、布尔值、浮点数等）抛出 ValueError；
- 不依赖 PyTorch。

## 三、Task B：Padding Mask（填充掩码）

变长句子组成 Batch 时，需要用 PAD 补齐短序列。PAD 作为 Key 时不应被有效 Query 读取。

$$
M^{\mathrm{pad}}_{b,j}
=
[\,token\_ids_{b,j}=pad\_token\_id\,]
$$

Padding Mask 的形状为 (B,T)，需要扩展为：

$$
(B,T)\rightarrow(B,1,1,T).
$$

Causal Mask 的形状 (T,T) 扩展为：

$$
(T,T)\rightarrow(1,1,T,T).
$$

两者组合：

$$
M^{\mathrm{combined}}
=
M^{\mathrm{causal}}\lor M^{\mathrm{pad}}.
$$

**边界说明：** Padding Mask 在今天的实现中屏蔽 PAD Key 列。PAD Query 对应的输出位置不自动等于零；训练时通常还需要在 Loss 中忽略 PAD 目标位置。

## 四、Task C：Masked Softmax

普通 Softmax：

$$
\operatorname{Softmax}(s_i)
=
\frac{\exp(s_i)}{\sum_j\exp(s_j)}.
$$

稳定实现会减去最大值。但如果整行全部被遮挡，最大值也是负无穷，计算会出现：

$$
-\infty-(-\infty)=\mathrm{NaN}.
$$

正确行为：

- 可见位置正常进行稳定 Softmax；
- 被遮挡位置权重严格为 0；
- 如果整行都被遮挡，整行返回全 0；
- 输出不允许出现 NaN / Inf；
- Mask 必须为 Boolean，且可广播到 Scores 的形状。

令可见位置集合为 \(\mathcal{V}\)，且：

$$
m=\max_{j\in\mathcal{V}}s_j.
$$

则：

$$
A_i=
\begin{cases}
\displaystyle
\frac{\exp(s_i-m)}
{\sum_{j\in\mathcal{V}}\exp(s_j-m)},
& i\in\mathcal{V},\[8pt]
0, & i\notin\mathcal{V}.
\end{cases}
$$

若 \(\mathcal{V}\) 为空，整行返回零。

## 五、Task D：Masked Multi-Head Self-Attention

函数：

\`multi_head_attention_with_mask_numpy(x, w_q, w_k, w_v, w_o, num_heads, mask=None)\`

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
4. 若 Mask 不为 None，调用 \`masked_softmax_numpy\`；若没有 Mask，则复用 Day 07 的 \`softmax_numpy\`；
5. 用 Attention Weights 聚合 Value；
6. 复用 Day 08 的 \`merge_heads_numpy\`；
7. 进行输出投影；
8. 返回 output 与 attention_weights。

输出形状：

$$
output\in\mathbb{R}^{B\times T\times D_{\mathrm{out}}}
$$

$$
attention\_weights
\in\mathbb{R}^{B\times H\times T\times T}.
$$

实现限制：

- 核心实现只使用 NumPy；
- 复用 Day 07 Softmax 与 Day 08 Split/Merge Heads；
- 不使用 PyTorch 完成前向实现；
- 不修改调用者传入的数组；
- 支持 \(d_k\) 与 \(d_v\) 不相等；
- 不加入 Bias、Dropout、Residual、LayerNorm；
- Causal 与 Padding Mask 在调用端组合并扩展维度。

## 六、测试与验收结果

运行命令：

\`\`\`powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week09_exercises.py
\`\`\`

实际结果：

\`\`\`text
...............                                      [100%]
15 passed, 1 warning
\`\`\`

15 项测试覆盖：

1. Causal Mask 的形状、类型与严格上三角方向；
2. 非法序列长度；
3. Padding Mask 对 PAD Token 的识别与输入不变性；
4. Masked Softmax 正常归一化与遮挡零权重；
5. 全遮挡行返回全零且无 NaN；
6. 极端分数下的数值稳定性；
7. Mask 广播与非法形状、非法类型检查；
8. Masked Multi-Head Attention 与独立 PyTorch 参考结果一致；
9. Causal Mask 下未来 Token 不影响过去位置的输出；
10. Causal 与 Padding Mask 组合后，所有被遮挡位置权重为零。

唯一 Warning 来自测试辅助代码将只读的 \`np.broadcast_to\` 视图传给 PyTorch，不影响测试通过结果。

## 七、Demo 实验结果

从项目根目录运行：

\`\`\`powershell
python -m scripts.day09_masked_attention_demo
\`\`\`

使用模块方式运行，以确保项目根目录下的 \`exercises\` 包能够正确导入。

实际输出的关键指标：

| 指标 | 结果 |
|---|---:|
| Input Shape | (2,5,8) |
| Output Shape | (2,5,8) |
| Attention Weights Shape | (2,2,5,5) |
| Maximum Blocked Attention Weight | 0.0 |
| Maximum Visible Row-Sum Error | 2.220446049250313e-16 |
| Maximum Past Output Change after Future Perturbation | 0.0 |

稳定 Masked Softmax 示例：

$$
[10000,9999,10001]
\quad\longrightarrow\quad
[0.73105858,0.26894142,0].
$$

第三个位置被遮挡，因此权重严格为零；前两个可见位置重新归一化。

### 因果干预验证

固定投影矩阵，只修改最后一个 Token，再比较此前所有 Query 的输出：

$$
\max_{i<T-1}
\left\|
Y'_{:,i,:}-Y_{:,i,:}
\right\|_{\infty}
=0.
$$

本次实验中，未来 Token 的扰动没有改变任何更早位置的输出，验证了因果 Mask 的信息隔离作用。

实验图与指标保存在：

\`\`\`text
results/day09/
├── metrics.json
└── masked_attention.png
\`\`\`

## 八、完成标准

- [x] Task A：Causal Mask 实现正确。
- [x] Task B：Padding Mask 实现正确。
- [x] Task C：Masked Softmax 通过数值稳定性测试。
- [x] Task D：Masked Multi-Head Attention 与参考实现一致。
- [x] 15 项测试全部通过。
- [x] 能解释 Mask 的广播形状。
- [x] 能解释为什么全遮挡行必须返回零。
- [x] Demo 生成注意力热图与指标。
- [x] 完成正式学习笔记与统一系列风格词云。
- [x] 更新 GitHub 并准备同步 VS Code。

**Day 09 完成。**
