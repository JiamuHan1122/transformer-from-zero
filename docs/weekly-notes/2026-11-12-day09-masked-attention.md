# Day 09 学习笔记｜Masked Attention

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-12  
> **阶段：** Attention 核心机制 · Day 09  
> **今日主题：** Causal Mask、Padding Mask、Masked Softmax、Masked Multi-Head Attention  
> **状态：** 学习脚手架已上传；待完成实现与测试  
> **核心实现：** `exercises/week09.py`  
> **测试：** `tests/test_week09_exercises.py`  
> **实验脚本：** `scripts/day09_masked_attention_demo.py`

---

## 一、今天的核心问题：谁可以看见谁？

Day 08 的普通 Multi-Head Self-Attention 允许每个 Query 读取所有 Key。今天引入 Mask，为注意力建立信息可见性约束。

两个最重要的场景：

- 自回归生成：当前 Token 不得读取未来 Token。
- 变长 Batch：PAD 位置不应作为有效 Key 被关注。

本日统一采用 Boolean mask：True 表示遮挡，False 表示允许访问。

## 二、Causal Mask：阻断未来信息

对 Query 位置 i，只允许读取 Key 位置 j 满足 j ≤ i 的信息。

$$
M^{\mathrm{causal}}_{ij}
=
\begin{cases}
\mathrm{False}, & j\le i,\\
\mathrm{True}, & j>i.
\end{cases}
$$

因此 blocked mask 是严格上三角矩阵。注意它与“允许访问矩阵”正好相反：允许矩阵是下三角（含对角线），blocked mask 是严格上三角。

原始注意力分数：

$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$

加入加性掩码的概念表达：

$$
\widetilde S_{ij}
=
\begin{cases}
S_{ij}, & M_{ij}=\mathrm{False},\\
-\infty, & M_{ij}=\mathrm{True}.
\end{cases}
$$

然后：

$$
A=\operatorname{Softmax}(\widetilde S)
$$

因为 exp(-∞)=0，被遮挡位置的注意力权重为零。

## 三、Padding Mask：排除填充 Key

Padding Mask 标记输入中的 PAD 位置：

$$
M^{\mathrm{pad}}_{b,j}
=
[\,token\_ids_{b,j}=pad\_token\_id\,]
$$

形状由 (B,T) 扩展为 (B,1,1,T)，使它能够广播到 (B,H,T,T) 的注意力分数。

Causal Mask 从 (T,T) 扩展为 (1,1,T,T)。

组合规则：

$$
M^{\mathrm{combined}}
=
M^{\mathrm{causal}}\lor M^{\mathrm{pad}}
$$

Padding Mask 屏蔽 PAD Key；PAD Query 对应的输出通常由训练 Loss Mask 忽略。本日不把二者混为一谈。

## 四、Masked Softmax：必须处理全遮挡行

普通 Softmax：

$$
\operatorname{Softmax}(x_i)
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
$$

数值稳定实现会减去最大值。但若一整行全部填充为 -∞，直接计算最大值和减法会产生：

$$
-\infty-(-\infty)=\mathrm{NaN}
$$

因此需要先识别可见位置。若一行存在可见位置，只在可见位置上做稳定归一化；被遮挡位置权重为零。若整行完全遮挡，则返回全零行。

本日测试要求：

- 被遮挡权重严格等于 0；
- 非全遮挡行权重和为 1；
- 全遮挡行全为 0；
- 输出无 NaN / Inf；
- 极端 logits 下保持稳定。

## 五、Masked Multi-Head Attention 计算链

$$
Q=XW^Q,\quad K=XW^K,\quad V=XW^V
$$

$$
S=\frac{Q_hK_h^\top}{\sqrt{d_k}}
$$

$$
A=\operatorname{MaskedSoftmax}(S,M)
$$

$$
O_h=AV_h
$$

$$
Y=\operatorname{MergeHeads}(O_h)W^O
$$

这里复用 Day 08 的 Split Heads 与 Merge Heads，不重新实现多头张量变换。

## 六、关键验证：未来 Token 是否影响过去输出？

因果约束最有说服力的测试不是只检查矩阵长什么样，而是做干预：

1. 固定模型参数和输入序列；
2. 计算带 Causal Mask 的输出；
3. 只修改最后一个 Token；
4. 再次计算输出；
5. 检查之前位置的输出是否保持不变。

如果因果掩码正确，那么最后位置的变化不应传递到更早 Query 的输出。

该性质是自回归信息流的核心不变量。

## 七、实现与实验边界

本日实现仍是 NumPy 前向计算，不包含训练、反向传播、Dropout、Residual、LayerNorm 或 FFN。PyTorch 仅用于测试中的独立数值参考。

实验脚本预期输出：

```text
results/day09/
├── metrics.json
└── masked_attention.png
```

实验图用于检查 Causal Mask、注意力权重以及 Causal + Padding Mask 的组合效果。

## 八、知识词云

![Day 09 Knowledge Word Cloud](images/day09-knowledge-wordcloud.svg)

核心关键词：

**Causal Mask · Padding Mask · Masked Softmax · Future Information · Visibility Constraint · Strict Upper Triangle · Broadcasting · Fully Masked Row · Numerical Stability · Zero Attention Weight · Autoregressive · NumPy**

---

## Day 09 完成记录

- [ ] Task A：Causal Mask
- [ ] Task B：Padding Mask
- [ ] Task C：Masked Softmax
- [ ] Task D：Masked Multi-Head Attention
- [ ] 全部测试通过
- [ ] 未来 Token 干预测试通过
- [ ] Demo 图与指标生成
- [ ] 词云样式与前几日系列保持一致
- [ ] GitHub 提交并同步 VS Code

**Day 09 当前状态：学习脚手架已准备，等待本地实现与验收。**
