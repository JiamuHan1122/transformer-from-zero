# Day 11｜Decoder Input Pipeline：从 Token IDs 到 Masked Attention

> **项目：** Transformer From Zero · 365 Days  
> **阶段：** Transformer 输入表示与注意力整合 · Day 11  
> **今日主题：** Embedding Lookup、Positional Encoding、Causal + Padding Mask、Input Pipeline  
> **状态：** 学习脚手架已建立；练习待实现与验收

---

## 一、今天要解决的问题

Day 07–10 已经分别实现了 Self-Attention、Multi-Head Attention、Masked Attention 和 Sinusoidal Positional Encoding。今天不再孤立学习一个新公式，而是把这些模块接成一条真实的数据流：

[
	ext{Token IDs}
ightarrow 	ext{Token Embeddings}
ightarrow 	ext{Embedding + PE}
ightarrow 	ext{Causal + Padding Mask}
ightarrow 	ext{Masked Multi-Head Attention}.
]

今天的重点是理解各模块的输入、输出和张量维度如何衔接。

## 二、Task A：Embedding Lookup

词表中的每个 Token ID 对应 Embedding Table 的一行。设：

[
Iinmathbb{Z}^{B	imes T},qquad
W_Einmathbb{R}^{V	imes D},
]

其中 (B) 是 Batch size，(T) 是序列长度，(V) 是词表大小，(D) 是 Embedding 维度。

查表操作：

[
E_{b,t,:}=W_E[I_{b,t},:].
]

输出形状：

[
(B,T)+(V,D)longrightarrow(B,T,D).
]

实现 `token_embedding_lookup_numpy(token_ids, embedding_table)`，要求：
- 输入必须是 NumPy 数组；
- token_ids 是二维整数数组，拒绝布尔和浮点 ID；
- embedding_table 是二维实数数值数组；
- 所有 ID 必须满足 (0leq I_{b,t}<V)；
- 返回新数组，不修改输入。

## 三、Task B：构造 Causal + Padding Mask

Day 09 约定：True 表示 blocked，False 表示 visible。

因果 Mask：

[
M^{causal}_{i,j}=
egin{cases}
mathrm{True},&j>i,\
mathrm{False},&jleq i.
end{cases}
]

Padding Mask：

[
M^{pad}_{b,j}=
mathbb{1}[I_{b,j}=PAD].
]

组合规则：

[
M^{combined}_{b,1,i,j}
=M^{causal}_{i,j}lor M^{pad}_{b,j}.
]

维度变化：

[
(T,T)ightarrow(1,1,T,T),
]

[
(B,T)ightarrow(B,1,1,T),
]

[
(1,1,T,T)lor(B,1,1,T)
ightarrow(B,1,T,T).
]

最终它会广播到注意力分数的 ((B,H,T,T))。注意：这里屏蔽的是 PAD Key；PAD Query 的输出不会因此自动清零。

实现 `causal_padding_mask_numpy(token_ids, pad_token_id)`，要求校验 token_ids 的形状与整数类型，复用 Day 09 的两个 Mask 函数，不修改输入。

## 四、Task C：组装 Decoder 输入

位置编码沿 Batch 维广播：

[
X_{input}=E+PE,
]

[
Einmathbb{R}^{B	imes T	imes D},quad
PEinmathbb{R}^{T	imes D},quad
X_{input}inmathbb{R}^{B	imes T	imes D}.
]

实现 `prepare_decoder_input_numpy(token_ids, embedding_table, positional_encoding, pad_token_id)`，内部复用 Task A、Task B 和 Day 10 的 `add_positional_encoding_numpy`，返回：
- input_embeddings：((B,T,D))；
- attention_mask：((B,1,T,T))。

## 五、贯穿示例：逐步跟踪形状

设：

[
B=2,quad T=5,quad V=20,quad D=8,quad H=2.
]

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

## 六、实现顺序与测试

先完成 Task A，再完成 Task B，最后完成 Task C。每一步都先运行对应单项测试，再运行全量测试：

```powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week11_exercises.py
```

测试覆盖：查表数值与形状、非法 ID 类型、越界 ID、输入不变性、Mask 方向、PAD Key 屏蔽、Mask 形状、Embedding + PE 整合，以及端到端输出形状。

## 七、Demo 验收

从仓库根目录运行：

```powershell
python -m scripts.day11_input_pipeline_demo
```

Demo 将使用固定随机种子构造小型词表与两条不同长度的序列，打印每一步形状、组合 Mask 和数值检查，并调用 Day 09 Masked Multi-Head Attention。只有在实现完成后才应通过。

本地结果保存到 `results/day11/`，遵循仓库规则，不默认提交实验结果目录。

## 八、完成标准

- [ ] Task A：Embedding Lookup 完成。
- [ ] Task B：Causal + Padding Mask 组合完成。
- [ ] Task C：Decoder 输入组装完成。
- [ ] 全部练习测试通过。
- [ ] Demo 运行成功。
- [ ] 能解释 PAD Key 与 PAD Query 的区别。
- [ ] 能从 Token IDs 独立追踪到 Attention 输入的所有维度。
- [ ] 学习笔记与本系列知识词云完成。

**今天的核心：把前十天积累的模块接成一条可验证的数据流。**
