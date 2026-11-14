# Day 11 学习笔记｜Transformer 输入表示与信息可见性机制

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-14  
> **阶段：** Transformer 输入表示与 Attention 整合 · Day 11  
> **今日主题：** Tokenization、Embedding、Positional Information、Causal Mask、Padding Mask  
> **课程定位：** 理论学习笔记；代码练习单独保存在 docs/12_week11_tasks.md

---

## 一、今天要回答的核心问题

前十天，我们已经逐步建立了 Transformer 的若干基本模块：向量与矩阵运算、Softmax、Self-Attention、Multi-Head Attention、Masked Attention，以及 Sinusoidal Positional Encoding。

但真实的 Transformer 不会直接接收一串已经准备好的浮点矩阵。它面对的是文本、Token ID、变长序列，以及不同任务对信息可见性的约束。

今天要理解的是：**文本怎样变成模型能够处理的向量？模型怎样获得 Token 的位置信息？在生成任务中，模型又怎样避免读取未来信息或无效的 PAD 内容？**

完整数据流：

$$
\text{Text}
\rightarrow \text{Token IDs}
\rightarrow \text{Token Embeddings}
\rightarrow \text{Position-aware Representations}
\rightarrow \text{Masked Self-Attention}.
$$

其中有两条并行通路：

- **表示通路：** Token ID → Embedding → 加入位置信息 → Q、K、V；
- **约束通路：** Token ID → Causal / Padding Mask → 限制哪些 Key 可以被读取。

Mask 不负责生成语义表示，而是规定信息流动的边界。

## 二、从文本到 Token ID：模型最初看到的是什么？

神经网络不能直接对自然语言字符串执行矩阵乘法。文本首先经过 Tokenizer，被切分为 Token，再映射为整数 ID。

例如：

~~~text
I love deep learning
       ↓
[12, 35, 81, 106]
~~~

这些数字只是词表索引。ID 106 不比 ID 12 更重要，也不意味着它们之间存在有意义的数值距离。

Tokenization 与 Embedding 是两个不同步骤：

1. **Tokenization：** 把离散文本切分并映射为整数索引；
2. **Embedding：** 把整数索引映射为连续向量。

Tokenizer 决定模型使用什么离散符号；Embedding 决定模型用什么向量表示这些符号。

## 三、Embedding：离散符号如何进入连续向量空间？

### 3.1 Embedding Table

假设词表大小为 V，模型维度为 D。Embedding Table 是一个可学习矩阵：

$$
W_E\in\mathbb{R}^{V\times D}.
$$

矩阵的每一行对应一个 Token ID 的向量。若某个 Token 的 ID 为 k，就取出第 k 行：

$$
e_k=W_E[k,:].
$$

对于一个 Batch，Token ID 矩阵的形状为：

$$
I\in\mathbb{Z}^{B\times T},
$$

其中 B 是 Batch Size，T 是序列长度。查表后得到：

$$
E\in\mathbb{R}^{B\times T\times D},
\qquad
E_{b,t,:}=W_E[I_{b,t},:].
$$

Embedding Lookup 的本质是**以 Token ID 为行索引，从可学习矩阵中取出对应向量**，不是把 ID 当作连续数值参与普通乘法。

### 3.2 Embedding 与 One-hot 的关系

One-hot 向量可以表示一个 Token 的离散身份。若第 k 个位置为 1，其余位置为 0：

$$
e_k^{\mathrm{onehot}}\in\mathbb{R}^{V}.
$$

理论上，Embedding 也可以写成：

$$
\operatorname{Embedding}(k)
=
(e_k^{\mathrm{onehot}})^\top W_E.
$$

所以查表与 One-hot 乘 Embedding Table 在数学上等价。但实际实现通常直接查表，避免创建长度为 V 的稀疏向量再做矩阵乘法。

### 3.3 Embedding 为什么是可学习的？

Embedding Table 中的向量通常是模型参数。训练时，损失函数产生的梯度会更新被使用到的 Embedding 行；同一个 Token 在不同样本、不同位置出现时，对应行的梯度会累积。

Embedding 不是人工规定的语义坐标，而是模型在训练目标驱动下逐渐学习的向量表示。它的几何结构取决于训练数据、模型结构和优化目标。

## 四、为什么还需要位置信息？

### 4.1 Self-Attention 本身不直接知道顺序

Self-Attention 根据 Query 与 Key 的匹配程度计算权重，再对 Value 加权聚合。若只给它 Token Embedding，而不提供位置线索，它本身没有直接获得“哪个 Token 在前、哪个 Token 在后”的信息。

更严格地说，不带位置编码的 Self-Attention 对输入排列具有置换等变性：若输入 Token 重新排列，输出也会按照相同排列关系变化。模型不会凭空知道原始顺序。

### 4.2 将位置信息加入 Token 表示

Day 10 学过 Sinusoidal Positional Encoding。记 Token Embedding 为 E，位置编码为 P：

$$
E\in\mathbb{R}^{B\times T\times D},
\qquad
P\in\mathbb{R}^{T\times D}.
$$

标准相加方式：

$$
X_{\mathrm{input}}=E+P.
$$

位置编码沿 Batch 维广播：

$$
(B,T,D)+(1,T,D)\rightarrow(B,T,D).
$$

因此，位置编码改变的是向量数值，不改变 Batch、Sequence 和 Feature 三个维度。加入位置编码后，模型输入同时包含 Token 内容表示与位置信息。

## 五、Attention Mask：限制信息可以从哪里流入

### 5.1 Mask 作用在哪里？

回顾 Scaled Dot-Product Attention：

$$
S=\frac{QK^\top}{\sqrt{d_k}},
\qquad
A=\operatorname{Softmax}(S),
\qquad
O=AV.
$$

对单个 Head，分数矩阵的元素 S(i,j) 表示 Query 位置 i 对 Key 位置 j 的匹配分数：

- 行 i：当前 Query 在哪个位置；
- 列 j：它正在考虑哪个 Key；
- 被屏蔽的列位置不能向当前 Query 提供信息。

本系列统一约定：True 表示 blocked，False 表示 visible。

在 Softmax 之前，把被屏蔽位置替换为负无穷：

$$
S'_{i,j}=
\begin{cases}
-\infty,&M_{i,j}=\mathrm{True},\\
S_{i,j},&M_{i,j}=\mathrm{False}.
\end{cases}
$$

然后沿 Key 维度执行 Softmax。因为：

$$
\exp(-\infty)=0,
$$

被屏蔽位置的注意力权重就会变成零。Mask 必须在 Softmax 之前应用；若先 Softmax 再删除权重，就会破坏归一化关系。

### 5.2 Causal Mask：为什么不能读取未来？

自回归语言模型按从左到右的顺序预测下一个 Token。预测位置 i 时，只允许使用当前位置及其之前的 Token，不能读取未来位置。

允许访问的条件是：

$$
j\leq i.
$$

因此 Causal Mask 定义为：

$$
M^{\mathrm{causal}}_{i,j}
=
\begin{cases}
\mathrm{True},&j>i,\\
\mathrm{False},&j\leq i.
\end{cases}
$$

序列长度为 4 时，允许访问的是下三角（含对角线），被屏蔽的是严格上三角：

| Query \ Key | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 允许 | 屏蔽 | 屏蔽 | 屏蔽 |
| 1 | 允许 | 允许 | 屏蔽 | 屏蔽 |
| 2 | 允许 | 允许 | 允许 | 屏蔽 |
| 3 | 允许 | 允许 | 允许 | 允许 |

它的作用不是让模型“更关注过去”，而是从结构上**禁止未来信息进入当前预测位置**。

### 5.3 Padding Mask：为什么 PAD 不能作为有效内容？

为了把不同长度的句子放入同一个 Batch，通常需要补齐较短的序列：

~~~text
序列 A: [I, love, AI, PAD, PAD]
序列 B: [I, study, deep, learning, today]
~~~

PAD 只是批处理对齐符号，不是真实语言内容。如果有效 Query 可以关注 PAD Key，就会把无效位置纳入注意力归一化与信息聚合。

定义：

$$
M^{\mathrm{pad}}_{b,j}
=
\mathbb{1}[I_{b,j}=PAD].
$$

当某个位置是 PAD 时，该位置作为 Key 就应该被屏蔽。

**Padding Mask 主要屏蔽 Key 位置，而不是自动清空 Query 位置。**

### 5.4 两种 Mask 为什么要组合？

两种 Mask 解决不同问题：

| Mask | 解决的问题 | 屏蔽对象 |
|---|---|---|
| Causal Mask | 防止读取未来信息 | 未来 Key |
| Padding Mask | 防止读取无效填充内容 | PAD Key |

对自回归 Decoder Self-Attention，两种约束需要同时满足。只要任意一个条件要求屏蔽，就不允许访问：

$$
M^{\mathrm{combined}}
=
M^{\mathrm{causal}}
\lor
M^{\mathrm{pad}}.
$$

按 Batch 展开：

$$
M^{\mathrm{combined}}_{b,1,i,j}
=
M^{\mathrm{causal}}_{i,j}
\lor
M^{\mathrm{pad}}_{b,j}.
$$

逻辑 OR 的含义是：未来位置要屏蔽，PAD 位置也要屏蔽；只有既不是未来位置、也不是 PAD 的 Key 才可见。

## 六、Mask 的维度为什么需要 Broadcasting？

在 Multi-Head Attention 中，注意力分数形状为：

$$
S\in\mathbb{R}^{B\times H\times T\times T}.
$$

其中 H 是 Head 数量。每个 Head 都有自己的 Query–Key 分数矩阵。

Causal Mask 原本是：

$$
(T,T)\rightarrow(1,1,T,T).
$$

Padding Mask 原本是：

$$
(B,T)\rightarrow(B,1,1,T).
$$

两者组合：

$$
(1,1,T,T)\lor(B,1,1,T)
\rightarrow(B,1,T,T).
$$

最终 Mask 沿 Head 维广播：

$$
(B,1,T,T)\rightarrow(B,H,T,T).
$$

不需要为每个 Head 手动复制一份相同的 Mask。Broadcasting 让同一个可见性约束应用于所有 Head，而各个 Head 的注意力分数仍然可以不同。

## 七、必须分清：PAD Key 与 PAD Query

考虑右侧 Padding 的序列：

~~~text
[I, love, PAD, PAD]
~~~

Padding Mask 屏蔽的是 Key 轴，也就是注意力矩阵的列。这意味着有效 Query 不会读取 PAD Key。但是 PAD Query 对应的行并不会自动全部变成零。

例如，第三个位置虽然是 PAD Query，但因果约束允许它读取前面的有效 Key 0、1，因此它仍可能得到非零注意力输出。

这不是 Padding Mask 失效，而是两种不同约束：

- **Key Mask：** 某个位置能不能被别人读取；
- **Query Mask：** 某个位置是否需要产生有效输出。

如果训练目标不应该包含 PAD 位置，通常还要单独使用 Loss Mask：

$$
M^{\mathrm{loss}}_{b,t}
=
\mathbb{1}[I_{b,t}\neq PAD].
$$

它用于排除 PAD 目标位置的损失，而不是替代 Attention Mask。

此外，如果某一行的所有 Key 都被屏蔽，普通 Softmax 会遇到全负无穷输入并产生 NaN。Day 09 的 Masked Softmax 已经处理这种情况：全遮挡行返回全零。

## 八、Encoder、Decoder 与 Mask 的适用范围

今天讨论的 Causal + Padding Mask 对应的是**自回归 Decoder 的 Self-Attention**。

| 模块 | 能否读取未来位置 | 常见 Mask |
|---|---|---|
| Encoder Self-Attention | 可以读取整个输入序列 | Padding Mask |
| Autoregressive Decoder Self-Attention | 不可以 | Causal + Padding Mask |
| Decoder Cross-Attention | 读取 Encoder 输出 | 按源序列有效位置屏蔽 Padding |

不能把 Causal Mask 理解为所有 Transformer 层都必须使用的掩码。是否需要因果约束，取决于任务是否要求当前预测只能依赖过去信息。

今天聚焦 Decoder Self-Attention 的输入准备，并不是完整讲解 Encoder–Decoder 的所有结构。

## 九、从文本到 Attention 的完整数据流

1. 文本经过 Tokenizer，变成 Token IDs；
2. Token IDs 作为索引，从 Embedding Table 中取出 Token Embeddings；
3. 将 Positional Encoding 加入 Embeddings，得到带位置信息的输入表示；
4. 输入表示经过线性投影，得到 Q、K、V；
5. Q 与 K 计算缩放点积分数；
6. Causal Mask 与 Padding Mask 组合，限制 Key 的可见性；
7. Mask 在 Softmax 前作用于 Attention Scores；
8. Softmax 产生注意力权重，再对 V 加权聚合。

完整核心关系：

$$
X_{\mathrm{input}}=E+P,
\qquad
S=\frac{QK^\top}{\sqrt{d_k}},
$$

$$
A=\operatorname{Softmax}(S+M),
\qquad
O=AV.
$$

这里的 M 是加性 Mask：允许访问的位置为 0，被屏蔽的位置为负无穷。若使用 Boolean Mask，则先按规则将 True 位置替换为负无穷。

这条数据流包含两个不同层面的设计：

- Embedding 与 Positional Encoding 决定模型接收什么表示；
- Attention Mask 决定模型在当前任务中允许读取哪些信息。

**表示内容与信息可见性，是 Transformer 输入阶段的两种不同机制。**

## 十、今天的核心结论

1. Token ID 是离散索引，不是有连续数值意义的特征。
2. Embedding Lookup 把离散索引映射到可学习的连续向量空间。
3. Positional Encoding 为 Token 表示补充顺序信息。
4. Causal Mask 保证自回归预测不读取未来。
5. Padding Mask 防止 PAD Key 参与有效信息聚合。
6. 两种 Mask 使用逻辑 OR 组合，并通过 Broadcasting 作用到各个 Head。
7. PAD Key 被屏蔽，不等于 PAD Query 的输出自动为零。
8. Attention Mask 与 Loss Mask 作用不同，不能混为一谈。

## 十一、Day 11 知识词云

![Day 11 Knowledge Word Cloud](images/day11-knowledge-wordcloud.svg)

核心关键词：

**Tokenization · Token IDs · Embedding Table · One-hot · Positional Encoding · Permutation Equivariance · Causal Mask · Padding Mask · Key Visibility · Broadcasting · Decoder Self-Attention · Loss Mask · Information Flow**

---

**Day 11 理论课程：完成。**

下一步再进入独立的代码练习与实验验证，不把理论学习和 Task A/B/C 混在一起。
