# Day 11｜Decoder Input Pipeline：从 Token IDs 到 Masked Attention

> **项目：** Transformer From Zero · 365 Days  
> **阶段：** Transformer 输入表示与注意力整合 · Day 11  
> **今日主题：** Embedding Lookup、Positional Encoding、Causal + Padding Mask、Input Pipeline  
> **状态：** 实现、专项测试、回归测试与 Demo 均已通过

---

## 一、今天要解决的问题

Day 07–10 已经分别实现了 Self-Attention、Multi-Head Attention、Masked Attention 和 Sinusoidal Positional Encoding。今天把这些模块接成一条真实的数据流：

$$
\text{Token IDs}
\rightarrow \text{Token Embeddings}
\rightarrow \text{Embedding + PE}
\rightarrow \text{Causal + Padding Mask}
\rightarrow \text{Masked Multi-Head Attention}.
$$

今天的重点是理解各模块的输入、输出和张量维度如何衔接。

- **表示通路：** Token ID → Embedding → 加入位置信息 → Q、K、V；
- **约束通路：** Token ID → Causal / Padding Mask → 限制哪些 Key 可以被读取。

Mask 不负责生成语义表示，而是规定信息流动的边界。

## 二、Task A：Embedding Lookup

设词表大小为 V，模型维度为 D，Batch 大小为 B，序列长度为 T：

$$
I\in\mathbb{Z}^{B\times T},
\qquad
W_E\in\mathbb{R}^{V\times D}.
$$

Token ID 为 k 时，取 Embedding Table 的第 k 行：

$$
e_k=W_E[k,:].
$$

对整个 Batch：

$$
E_{b,t,:}=W_E[I_{b,t},:],
\qquad
E\in\mathbb{R}^{B\times T\times D}.
$$

Embedding Lookup 的本质是**以 Token ID 为行索引，从可学习矩阵中取出对应向量**，不是把 ID 当作连续数值参与普通乘法。

One-hot 与查表在数学上等价：

$$
\operatorname{Embedding}(k)
=(e_k^{\mathrm{onehot}})^\top W_E.
$$

实际实现通常直接查表，避免创建长度为 V 的稀疏向量再做矩阵乘法。

## 三、Task B：构造 Causal + Padding Mask

本系列统一约定：True 表示 blocked，False 表示 visible。

### 3.1 Causal Mask

自回归预测位置 i 时，只允许读取当前位置及之前的位置：

$$
M^{\mathrm{causal}}_{i,j}
=
\begin{cases}
\mathrm{True},&j>i,\\
\mathrm{False},&j\leq i.
\end{cases}
$$

因此被屏蔽的是严格上三角，主对角线及其下方允许访问。

### 3.2 Padding Mask

PAD 是为了对齐变长序列而加入的占位符，不是真实语言内容：

$$
M^{\mathrm{pad}}_{b,j}
=\mathbb{1}[I_{b,j}=PAD].
$$

Padding Mask 屏蔽 PAD Key，避免有效 Query 从填充位置读取信息。它不会自动清空 PAD Query 对应的输出行。

### 3.3 组合规则

只要任意一个条件要求屏蔽，就不允许访问：

$$
M^{\mathrm{combined}}_{b,1,i,j}
=
M^{\mathrm{causal}}_{i,j}
\lor
M^{\mathrm{pad}}_{b,j}.
$$

这里必须使用逻辑 OR，而不是 AND。

### 3.4 Broadcasting 维度

Causal Mask：

$$
(T,T)\rightarrow(1,1,T,T).
$$

Padding Mask：

$$
(B,T)\rightarrow(B,1,1,T).
$$

组合后：

$$
(1,1,T,T)\lor(B,1,1,T)
\rightarrow(B,1,T,T).
$$

最终沿 Head 维广播到 Attention Scores 的形状：

$$
(B,H,T,T).
$$

## 四、Task C：组装 Decoder 输入

Token Embedding 与 Positional Encoding 相加：

$$
X_{\mathrm{input}}=E+PE.
$$

其中：

$$
E\in\mathbb{R}^{B\times T\times D},
\qquad
PE\in\mathbb{R}^{T\times D},
$$

位置编码沿 Batch 维广播，输出仍为：

$$
X_{\mathrm{input}}\in\mathbb{R}^{B\times T\times D}.
$$

Task C 复用 Task A、Task B 和 Day 10 的 `add_positional_encoding_numpy`，返回：

- `input_embeddings`：$(B,T,D)$；
- `attention_mask`：$(B,1,T,T)$。

## 五、贯穿示例：逐步跟踪形状

设：

$$
B=2,\quad T=5,\quad V=20,\quad D=8,\quad H=2.
$$

| 变量 | 形状 | 说明 |
|---|---|---|
| token_ids | (2,5) | 两条 Token 序列 |
| embedding_table | (20,8) | 20 个 Token，每个 8 维 |
| token_embeddings | (2,5,8) | 查表结果 |
| positional_encoding | (5,8) | 5 个位置，每个 8 维 |
| input_embeddings | (2,5,8) | 加入位置信息 |
| causal mask | (5,5) | 屏蔽未来 Key |
| padding mask | (2,5) | 标记 PAD Key |
| attention_mask | (2,1,5,5) | 两种约束的组合 |
| attention scores | (2,2,5,5) | 两个 Head 的分数 |
| broadcasted mask | (2,2,5,5) | 广播后作用于每个 Head |

## 六、Attention 中 Mask 的作用位置

Scaled Dot-Product Attention：

$$
S=\frac{QK^\top}{\sqrt{d_k}}.
$$

将被屏蔽位置替换为负无穷：

$$
S'_{i,j}
=
\begin{cases}
-\infty,&M_{i,j}=\mathrm{True},\\
S_{i,j},&M_{i,j}=\mathrm{False}.
\end{cases}
$$

然后沿 Key 维度执行 Softmax：

$$
A=\operatorname{Softmax}(S'),
\qquad
O=AV.
$$

因为 $\exp(-\infty)=0$，被屏蔽位置的注意力权重严格为零。Mask 必须在 Softmax 之前应用。

还要区分：

- **Key Mask：** 某个位置能不能被其他 Query 读取；
- **Query Mask：** 某个位置是否需要产生有效输出；
- **Loss Mask：** 某个位置是否参与损失计算。

三者不能混为一谈。

## 七、实现与验收结果

实现文件：`exercises/week11.py`  
测试文件：`tests/test_week11_exercises.py`  
Demo：`scripts/day11_input_pipeline_demo.py`

专项测试：

```powershell
python -m pytest -q -m exercise tests/test_week11_exercises.py
```

结果：

```text
12 passed
```

全量练习回归测试：

```powershell
python -m pytest -q -m exercise
```

结果：

```text
93 passed, 4 deselected
```

Demo：

```powershell
python -m scripts.day11_input_pipeline_demo
```

关键结果：

| 指标 | 实际结果 |
|---|---:|
| Input Embeddings | (2,5,8) |
| Attention Mask | (2,1,5,5) |
| Attention Weights | (2,2,5,5) |
| Attention Output | (2,5,8) |
| Maximum Blocked Attention Weight | 0.0 |
| Maximum Visible Row Sum Error | $2.220446049250313\times10^{-16}$ |
| All Outputs Finite | true |

本地实验结果保存在 `results/day11/metrics.json`，不默认提交实验结果目录。

## 八、今天的核心结论

1. Token ID 是离散索引，不是有连续数值意义的特征。
2. Embedding Lookup 把离散索引映射到可学习的连续向量空间。
3. Positional Encoding 为 Token 表示补充顺序信息。
4. Causal Mask 保证自回归预测不读取未来。
5. Padding Mask 防止 PAD Key 参与有效信息聚合。
6. 两种 Mask 使用逻辑 OR 组合，并通过 Broadcasting 作用到各个 Head。
7. PAD Key 被屏蔽，不等于 PAD Query 的输出自动为零。
8. Attention Mask 与 Loss Mask 作用不同，不能混为一谈。

**Day 11 完成：从离散 Token IDs 到带位置编码、具备因果与 Padding 可见性约束的 Decoder 输入，并通过 Attention Demo 验证端到端数据流。**
