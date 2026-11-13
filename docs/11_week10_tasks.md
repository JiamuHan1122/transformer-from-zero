# Day 10｜Positional Encoding：给 Token 注入位置信息

> **项目：** Transformer From Zero · 365 Days  
> **日期：** 2026-11-13  
> **阶段：** Transformer 输入表示 · Day 10  
> **今日主题：** Sinusoidal Positional Encoding、Embedding Addition、Position Information  
> **状态：** 核心实现完成；22 项练习测试通过；Demo 验证成功

---

## 一、今天为什么需要位置编码？

Day 07–09 已经完成了 Self-Attention、Multi-Head Attention 和 Masked Attention。但普通 Self-Attention 本身不直接编码 Token 的绝对顺序。

如果输入 Token 顺序发生置换，Self-Attention 的输出也会发生相应置换。这种性质称为置换等变性（Permutation Equivariance）。模型因此需要额外的位置表示，才能利用“第几个 Token”这一信息。

今天使用原始 Transformer 中的 Sinusoidal Positional Encoding（正弦位置编码）。

## 二、Task A：Sinusoidal Positional Encoding

设序列长度为 T，Embedding 维度为 D，位置从 0 开始。

偶数维度：

$$
PE(pos,2i)=
\sin\left(\frac{pos}{10000^{2i/D}}\right)
$$

奇数维度：

$$
PE(pos,2i+1)=
\cos\left(\frac{pos}{10000^{2i/D}}\right)
$$

其中：

- pos：Token 在序列中的位置；
- i：维度对的编号；
- D：Embedding 总维度，即 d_model。

注意：i 不是 Token 位置，而是维度对的索引。每一对偶数/奇数维度使用相同频率，分别计算 Sin 和 Cos。

### 2.1 具体数值例子

令 D=8，偶数维度为 0、2、4、6，对应的分母分别为：

$$
10000^{0/8}=1,\quad
10000^{2/8}=10,\quad
10000^{4/8}=100,\quad
10000^{6/8}=1000
$$

位置 pos=1 的编码为：

$$
PE(1,:)=[
\sin(1),\cos(1),
\sin(0.1),\cos(0.1),
\sin(0.01),\cos(0.01),
\sin(0.001),\cos(0.001)
]
$$

约为：

$$
[0.84147,0.54030,0.09983,0.99500,
0.01000,0.99995,0.00100,1.00000]
$$

位置 pos=0 时：

$$
PE(0,:)=[0,1,0,1,0,1,0,1]
$$

### 2.2 实现要求

函数：**sinusoidal_positional_encoding_numpy(seq_len, d_model)**

- 输入：正整数 seq_len 与 d_model；
- 输出：形状为 (T,D) 的 NumPy 数组；
- 输出 dtype 为 float64；
- 偶数维度写入 Sin，奇数维度写入 Cos；
- 支持奇数 d_model；
- 拒绝零、负数、浮点数、布尔值及其他非法维度；
- 不使用 PyTorch，不修改调用者数据。

建议中间变量：

| 变量 | 形状 | 含义 |
|---|---|---|
| positions | (T,1) | 位置编号 |
| dimensions | (1,D) | 维度编号 |
| angle_rates | (T,D) | 每个位置、每个维度对应的角度 |
| positional_encoding | (T,D) | 最终位置编码矩阵 |

## 三、Task B：将位置编码加入 Token Embedding

位置编码不是替代 Token Embedding，而是与 Token Embedding 相加：

$$
X_{input}=E+PE
$$

其中：

$$
E\in\mathbb{R}^{B\times T\times D}
$$

$$
PE\in\mathbb{R}^{T\times D}
$$

通过在 Batch 维增加一个轴，利用 NumPy Broadcasting：

$$
(B,T,D)+(1,T,D)\rightarrow(B,T,D)
$$

函数：**add_positional_encoding_numpy(token_embeddings, positional_encoding)**

实现要求：

- token_embeddings 必须是三维数组 (B,T,D)；
- positional_encoding 必须是二维数组 (T,D)；
- 两者的序列长度和特征维度必须匹配；
- 返回形状仍为 (B,T,D)；
- 不修改任意输入数组；
- 不使用显式循环逐 Token 相加。

## 四、完整维度例子

设置：

$$
B=2,\quad T=5,\quad D=8
$$

| 中间变量 | 形状 |
|---|---|
| Token Embedding | (2,5,8) |
| Positional Encoding | (5,8) |
| 扩展后的 Positional Encoding | (1,5,8) |
| 相加后的输入 | (2,5,8) |

位置编码沿 Batch 维广播，因此同一序列位置在不同样本中使用相同的位置向量；每个样本的 Token Embedding 内容仍然不同。

## 五、测试与运行命令

先运行单项测试，再运行全部测试：

~~~powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week10_exercises.py
~~~

本次实际结果：

~~~text
......................
22 passed in 1.08s
~~~

测试覆盖：

1. 输出形状与 float64 类型；
2. 位置 0 的 Sin/Cos 边界值；
3. D=8 时位置 1 的公式数值；
4. 奇数 d_model；
5. 非法序列长度和模型维度；
6. 输出不存在 NaN / Inf；
7. Embedding 与位置编码的 Batch Broadcasting；
8. 维度不匹配时拒绝输入；
9. 不修改调用者传入的数组。

## 六、Demo 实验

完成实现后，从仓库根目录运行：

~~~powershell
python -m scripts.day10_positional_encoding_demo
~~~

Demo 将生成：

- 位置编码矩阵热图；
- 多个维度通道随位置变化的曲线；
- Embedding 相加前后的形状与数值指标。

实验结果保存在本地 results/day10/，遵循项目既有规则，不默认提交结果目录。

## 七、完成标准

- [x] Task A：Sinusoidal Positional Encoding 实现完成。
- [x] Task B：Embedding Addition 实现完成。
- [x] 全部练习测试通过。
- [x] Demo 运行成功并生成图像与指标。
- [x] 能解释 pos、i、d_model 的区别。
- [x] 能解释为什么位置编码不改变输入张量形状。
- [x] 完成正式学习笔记与 Day 08/09 同系列风格知识词云。
- [x] 最终代码、测试、笔记统一提交 GitHub。

**Day 10 完成：22 项测试全部通过，Demo 验证成功。**
