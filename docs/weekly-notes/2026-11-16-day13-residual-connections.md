# Day 13 学习笔记｜Residual Connections：恒等映射与梯度传播

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-16  
> **阶段：** Transformer Block 核心结构 · Day 13  
> **今日主题：** Residual Connection、Identity Mapping、Gradient Flow、Jacobian、Pre-LN / Post-LN  
> **状态：** 理论学习、实现、梯度验证与 Demo 全部完成  
> **核心实现：** exercises/week13.py  
> **测试：** tests/test_week13_exercises.py  
> **实验脚本：** scripts/day13_residual_demo.py

---

## 一、今天为什么学习 Residual Connection？

Day 12 学习了 Layer Normalization。现在我们开始把 Transformer Block 内部的组件组织起来。除了 Attention、FFN 和 LayerNorm，另一个关键结构就是 Residual Connection（残差连接）。

先看一个直观问题：

**如果一个网络有很多层，反向传播时的梯度必须一层一层经过这些变换。如果某些层不断缩小梯度，传到前面时会发生什么？**

梯度可能越来越小，前面的层就很难获得有效的学习信号。这就是深层网络中的梯度消失问题之一。

残差连接的思路不是让网络放弃学习变换，而是在学习变换之外，保留一条从输入直接通向输出的路径。

## 二、普通映射与残差映射

普通网络层可以写成：

$$
Y=F(X).
$$

残差结构则写成：

$$
\boxed{Y=X+F(X)}.
$$

其中：

- $X$：进入子层之前的输入；
- $F(X)$：Attention、FFN 或其他子层计算出的变换结果；
- $X$：通过恒等映射直接传递的输入；
- $Y$：两条路径逐元素相加后的结果。

残差连接不是拼接，也不是求平均，而是逐元素相加。因此标准恒等残差要求两个分支的形状相同：

$$
X,F(X),Y\in\mathbb{R}^{B\times T\times D}.
$$

一个很有用的边界情况是：当 $F(X)=0$ 时，

$$
Y=X.
$$

也就是说，即使子层暂时没有学到有用的变化，输入信息仍然可以直接传递。

## 三、为什么这条路径能帮助梯度传播？

令损失函数为 $L$，并记输出端传回来的上游梯度为：

$$
G=\frac{\partial L}{\partial Y}.
$$

因为：

$$
Y=X+F(X),
$$

所以输入 $X$ 有两条影响输出的路径：

1. 一条直接经过恒等映射 $X$；
2. 另一条经过变换函数 $F(X)$。

根据链式法则：

$$
\boxed{
\frac{\partial L}{\partial X}
=
G+
J_F(X)^\top G
}
$$

这里 $J_F(X)$ 是 $F$ 对输入 $X$ 的 Jacobian。

关键在于第一项：

$$
G.
$$

它来自恒等映射路径。无论 $F$ 内部有多少运算，这条路径都不需要再乘上 $F$ 的局部导数。

### 用大白话理解

普通路径就像接力传递：梯度必须经过每一层。如果每一层只传下去 10%，经过很多层后，剩下的信号就非常微弱。

残差路径相当于增加了一条直接通道：梯度除了经过原来的变换分支，还能沿着恒等映射直接往前走。

它并不是把所有梯度强行设为 1，而是让输入梯度多出一个直接的贡献项。

## 四、用一个简单数字看清楚梯度消失

今天的 Demo 使用一个刻意简化的标量映射进行示意。

普通映射：

$$
y=-0.1x.
$$

它的局部导数为：

$$
\frac{dy}{dx}=-0.1.
$$

如果连续经过 $L$ 次相同的映射，梯度会乘上：

$$
(-0.1)^L.
$$

残差映射：

$$
y=x-0.1x=0.9x.
$$

它的局部导数为：

$$
\frac{dy}{dx}=0.9.
$$

连续经过 $L$ 次后，梯度乘上：

$$
0.9^L.
$$

两者对比：

| 重复映射次数 | 普通路径的梯度因子绝对值 | 残差路径的梯度因子 |
|---:|---:|---:|
| 1 | $10^{-1}$ | $0.9$ |
| 5 | $10^{-5}$ | $0.59049$ |
| 10 | $10^{-10}$ | $0.34868$ |
| 15 | $10^{-15}$ | $0.20589$ |
| 20 | $10^{-20}$ | $0.12158$ |

这就是今天实验的直观核心：在这个示意例子中，普通路径的梯度快速衰减，而残差路径的梯度衰减慢得多。

**注意：这不是说残差连接能保证梯度永远不消失。** 如果残差分支的 Jacobian 使两条路径发生抵消，或者整个网络的局部映射持续缩小梯度，仍然可能出现梯度消失；反过来也可能出现梯度放大。残差连接提供的是一条直接的梯度通路，而不是对所有网络都成立的稳定性保证。

## 五、加法节点的反向传播：两条路径都收到梯度

先暂时不考虑 $F(X)$ 内部的细节，只看：

$$
Y=X+Z.
$$

令：

$$
G=\frac{\partial L}{\partial Y}.
$$

那么：

$$
\boxed{
\frac{\partial L}{\partial X}=G,
\qquad
\frac{\partial L}{\partial Z}=G
}
$$

为什么两边相同？

因为加法对两个输入的局部导数都是 1。上游传来的梯度会分别传给两个输入。

这也是 Task A2 的核心：把同一份上游梯度分别传给 skip 分支和 residual 分支。代码中还要求返回两个独立数组，避免它们与输入或彼此共享可写内存。

## 六、手动推导一个线性残差块

为了把前向和反向传播真正写成代码，今天使用一个简化的线性残差块：

$$
Y=X+XW+b.
$$

其中：

$$
X\in\mathbb{R}^{B\times T\times D},
\quad
W\in\mathbb{R}^{D\times D},
\quad
b\in\mathbb{R}^{D}.
$$

令：

$$
G=\frac{\partial L}{\partial Y}.
$$

### 6.1 对输入 X 求梯度

$X$ 同时出现在恒等分支和线性分支中，因此：

$$
\boxed{
\frac{\partial L}{\partial X}
=
G+GW^\top
}
$$

第一项来自恒等映射，第二项来自线性变换。

### 6.2 对权重 W 求梯度

$$
\boxed{
\frac{\partial L}{\partial W}
=
\sum_{b,t}
X_{b,t,:}^{\top}G_{b,t,:}
}
$$

这里需要对 Batch 维和 Token 维的贡献进行累积，最终得到 $(D,D)$ 的权重梯度。

### 6.3 对偏置 b 求梯度

$$
\boxed{
\frac{\partial L}{\partial b}
=
\sum_{b,t}G_{b,t,:}
}
$$

偏置对所有 Batch 和 Token 共享，因此同样需要累积。

这个线性残差块是用于理解梯度流的教学示例，不是完整的 Transformer FFN。

## 七、Pre-LN 与 Post-LN：残差连接在 Transformer 中的位置

结合 Day 12 的 LayerNorm，我们可以进一步理解两种常见结构。

### Post-LN

$$
Y=\operatorname{LN}(X+F(X)).
$$

先执行子层变换和残差相加，再进行 LayerNorm。

### Pre-LN

$$
Y=X+F(\operatorname{LN}(X)).
$$

先对输入进行 LayerNorm，再执行子层变换，最后与原始输入相加。

两种结构都包含残差连接，但 LayerNorm 所处的位置不同。

对于 Pre-LN，恒等路径绕过当前子层及其前置 LayerNorm；对于 Post-LN，相加后的结果还要经过后置 LayerNorm。因此，两者的梯度路径并不完全相同。

## 八、今天的代码实现

实现文件：**exercises/week13.py**

### Task A：Residual Addition

实现：

- residual_add_forward_numpy
- residual_add_backward_numpy

前向：

$$
Y=X+Z.
$$

反向：

$$
G_X=G_Z=G.
$$

### Task B：Linear Residual Block

实现：

- residual_linear_forward_numpy
- residual_linear_backward_numpy

前向：

$$
Y=X+XW+b.
$$

反向需要计算：

$$
G_X,\quad G_W,\quad G_b.
$$

核心实现全部使用 NumPy。PyTorch 只在测试中作为独立参考，用于对照前向结果和 Autograd 梯度。

## 九、今天的验收结果

### 9.1 专项测试

运行：

~~~powershell
python -m pytest -q -m exercise tests/test_week13_exercises.py
~~~

实际结果：

~~~text
12 passed in 12.83s
~~~

### 9.2 全项目回归测试

运行：

~~~powershell
python -m pytest -q -m exercise
~~~

实际结果：

~~~text
122 passed, 4 deselected in 2.96s
~~~

### 9.3 Demo

运行：

~~~powershell
python -m scripts.day13_residual_demo
~~~

实际关键结果：

~~~json
{
  "input_shape": [2, 4, 6],
  "residual_add_output_shape": [2, 4, 6],
  "linear_residual_output_shape": [2, 4, 6],
  "linear_residual_grad_x_shape": [2, 4, 6],
  "linear_residual_grad_weight_shape": [6, 6],
  "linear_residual_grad_bias_shape": [6],
  "addition_forward_matches_reference": true,
  "both_branches_receive_same_upstream_gradient": true,
  "linear_residual_gradient_error": 1.1102230246251565e-16,
  "all_outputs_and_gradients_finite": true,
  "inputs_unchanged": true
}
~~~

### 9.4 梯度流示意实验

实际使用的局部梯度因子：

$$
a_{\mathrm{plain}}=-0.1,
\qquad
a_{\mathrm{residual}}=0.9.
$$

深度为 20 时，Demo 的梯度范数为：

| 实验路径 | 梯度范数 |
|---|---:|
| Plain path | $6.888605802790179\times10^{-20}$ |
| Residual path | $0.8374936482964124$ |

这组数值来自固定随机种子的示意实验，不是训练模型的性能指标。

## 十、实验产物

本地生成：

~~~text
results/
└── day13/
    ├── metrics.json
    └── gradient_flow_comparison.png
~~~

这些结果用于本地实验记录，不默认提交到 GitHub。

## 十一、Day 13 核心结论

1. 残差连接的基本形式是 $Y=X+F(X)$。
2. 恒等映射为输入信息提供直接传递路径。
3. 加法节点向两条输入分支传递相同的上游梯度。
4. 输入梯度由恒等路径和变换分支共同贡献。
5. 残差连接能缓解某些深层网络中的梯度传播困难，但不保证梯度永远稳定。
6. Pre-LN 与 Post-LN 都有残差连接，但 LayerNorm 的位置不同。
7. 权重和偏置梯度需要沿共享参数对应的 Batch、Token 维度累积。

今天真正需要记住的一句话：

**残差连接不是让网络不再学习变化，而是让网络在学习变化的同时，不必丢掉原来的信息和梯度通路。**

## 十二、Day 13 知识词云

![Day 13 Knowledge Word Cloud](images/day13-knowledge-wordcloud.svg)

核心关键词：

**Residual Connection · Identity Mapping · Skip Connection · Gradient Flow · Jacobian · Chain Rule · Gradient Accumulation · Element-wise Addition · Pre-LN · Post-LN · Linear Residual Block · NumPy**

---

## Day 13 完成记录

- [x] Task A1：Residual Addition Forward
- [x] Task A2：Residual Addition Backward
- [x] Task B1：Linear Residual Forward
- [x] Task B2：Linear Residual Backward
- [x] 12 项专项测试全部通过
- [x] 122 项全项目 Exercise 测试通过
- [x] PyTorch 前向与梯度对照通过
- [x] 有限差分梯度验证通过
- [x] Demo 运行成功
- [x] 梯度流对比图生成成功
- [x] 输入不变性与数值有限性检查通过

**Day 13：理论、实现、梯度验证与可视化实验全部完成。**
