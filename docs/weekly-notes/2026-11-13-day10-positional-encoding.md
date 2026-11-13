# Day 10 学习笔记｜Positional Encoding：让 Transformer 感知 Token 顺序

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-13  
> **阶段：** Transformer 输入表示 · Day 10  
> **今日主题：** Sinusoidal Positional Encoding、Embedding Addition、Position Information  
> **状态：** 学习框架已建立；代码实现、测试与实验结果待完成  
> **核心实现：** exercises/week10.py  
> **测试：** tests/test_week10_exercises.py  
> **实验脚本：** scripts/day10_positional_encoding_demo.py

---

## 一、今天要解决的问题：Attention 如何知道 Token 的顺序？

Self-Attention 根据 Query 与 Key 的匹配关系聚合 Value。若只使用 Token Embedding，普通 Self-Attention 本身不直接携带 Token 的绝对位置编号。

设输入为 X，P 为 Token 维度上的置换矩阵。没有位置编码时，Self-Attention 满足置换等变关系：

$$
\operatorname{Attention}(PX)
=
P\operatorname{Attention}(X)
$$

这意味着输入 Token 被重新排列时，输出会跟着重新排列。注意力机制本身并没有获得“原来哪个 Token 在前、哪个 Token 在后”的额外标识。

位置编码的作用就是给每个 Token 的表示加入位置线索。

## 二、Sinusoidal Positional Encoding

原始 Transformer 使用不同频率的正弦和余弦函数生成位置向量。

偶数维度：

$$
PE(pos,2i)=
\sin\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

奇数维度：

$$
PE(pos,2i+1)=
\cos\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

其中：

- pos 是 Token 位置，从 0 开始；
- i 是维度对编号；
- d_model 是位置向量的总维度。

每一对相邻维度共享同一个频率：偶数维度使用 Sin，奇数维度使用 Cos。不同维度对的频率不同。

## 三、手算例子：d_model=8

偶数维度为 0、2、4、6，其频率分母为：

$$
10000^{0/8}=1,\quad
10000^{2/8}=10,\quad
10000^{4/8}=100,\quad
10000^{6/8}=1000
$$

当 pos=1 时：

$$
PE(1,:)=[
\sin(1),\cos(1),
\sin(0.1),\cos(0.1),
\sin(0.01),\cos(0.01),
\sin(0.001),\cos(0.001)
]
$$

近似为：

$$
[0.84147,0.54030,0.09983,0.99500,
0.01000,0.99995,0.00100,1.00000]
$$

当 pos=0 时：

$$
PE(0,:)=[0,1,0,1,0,1,0,1]
$$

这个例子帮助区分两个索引：pos 选择序列中的 Token 位置，i 选择 Embedding 中的频率组。

## 四、位置编码与 Token Embedding 相加

设 Token Embedding 的形状为 (B,T,D)，位置编码矩阵的形状为 (T,D)：

$$
E\in\mathbb{R}^{B\times T\times D}
$$

$$
PE\in\mathbb{R}^{T\times D}
$$

最终输入为：

$$
\boxed{X_{input}=E+PE}
$$

在 NumPy 中，可以将位置编码增加一个 Batch 轴：

~~~python
x = token_embeddings + positional_encoding[None, :, :]
~~~

维度变化：

$$
(B,T,D)+(1,T,D)\rightarrow(B,T,D)
$$

因此，位置编码不会改变序列长度或 Embedding 维度，只会改变输入表示的数值。

## 五、今天的代码任务

### Task A：生成位置编码

实现 sinusoidal_positional_encoding_numpy(seq_len, d_model)，返回 float64、形状为 (T,D) 的数组。需要支持奇数 d_model，并拒绝非法维度。

### Task B：加入 Token Embedding

实现 add_positional_encoding_numpy(token_embeddings, positional_encoding)，检查两个输入的秩和 T、D 维度，使用 Broadcasting 相加，并保持输入不变。

## 六、测试与实验

测试命令：

~~~powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week10_exercises.py
~~~

Demo 命令：

~~~powershell
python -m scripts.day10_positional_encoding_demo
~~~

Demo 预定观察：

1. 不同位置、不同维度的编码值如何变化；
2. 不同维度对的变化频率有何差异；
3. 位置编码加入 Embedding 后，张量形状是否保持不变。

**实际测试结果与实验数值将在完成实现后补充，不预先填写未验证结果。**

## 七、Day 10 完成记录

- [ ] Task A：Sinusoidal Positional Encoding
- [ ] Task B：Embedding Addition
- [ ] 练习测试全部通过
- [ ] Demo 运行成功
- [ ] 生成实验图与指标
- [ ] 完成统一系列风格知识词云
- [ ] 更新最终笔记并提交 GitHub

**Day 10 进行中。**
