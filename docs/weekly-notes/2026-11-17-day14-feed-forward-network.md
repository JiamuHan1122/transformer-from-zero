# Day 14 学习笔记｜Position-wise Feed-Forward Network：逐 Token 的非线性特征变换

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-17  
> **阶段：** Transformer Block 核心结构 · Day 14  
> **今日主题：** Feed-Forward Network、GELU、特征升维、Position-wise、Manual Backpropagation  
> **状态：** 理论学习与练习脚手架已发布；实现及实验待完成  
> **核心实现：** exercises/week14.py  
> **测试：** tests/test_week14_exercises.py  
> **实验脚本：** scripts/day14_ffn_demo.py

---

## 一、今天为什么学 FFN？

Day 13 的 Residual Connection 提供了恒等映射路径；Day 12 的 LayerNorm 调整每个 Token 的特征尺度。现在要掌握 Transformer Block 中与 Attention 并列的另一个核心子层：Position-wise Feed-Forward Network（FFN）。

**Attention 负责跨 Token 汇聚信息；FFN 负责独立处理每个 Token 内部的特征通道。**

## 二、FFN 的前向传播

设：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

本次选择两层线性变换，中间使用 Day 05 的 GELU tanh 近似：

$$
Z=XW_1+b_1,
$$

$$
A=\operatorname{GELU}_{\mathrm{tanh}}(Z),
$$

$$
\boxed{Y=AW_2+b_2}.
$$

合并为：

$$
\boxed{
\operatorname{FFN}(X)
=
\operatorname{GELU}_{\mathrm{tanh}}(XW_1+b_1)W_2+b_2
}.
$$

原始 Transformer 使用 ReLU；本课程为了复用 Day 05 的实现，采用 GELU tanh 近似。两者不可混淆。

## 三、维度一次看清楚

取：

$$
B=2,\quad T=5,\quad D=8,\quad H=D_{\mathrm{ff}}=32.
$$

| 变量 | 形状 | 作用 |
|---|---|---|
| X | (2,5,8) | 每个 Token 有 8 个特征 |
| W1 | (8,32) | 升维投影 |
| b1 | (32,) | 第一层偏置 |
| Z | (2,5,32) | GELU 前 |
| A | (2,5,32) | GELU 后 |
| W2 | (32,8) | 降维投影 |
| b2 | (8,) | 第二层偏置 |
| Y | (2,5,8) | FFN 输出 |

整个 FFN 不改变 B 或 T，只改变中间的通道宽度。

参数量为：

$$
N_{\mathrm{params}}=2DH+H+D.
$$

对于 \(D=8,H=32\)，共有 552 个参数。

## 四、为什么 GELU 是必要的？

如果去掉激活函数：

$$
Y=(XW_1+b_1)W_2+b_2
=
X(W_1W_2)+(b_1W_2+b_2).
$$

它仍然只是一个线性/仿射映射。GELU 引入非线性，使模型能够学习更加丰富的特征变换。

Day 05 已有：

- \`gelu_tanh_numpy\`；
- \`gelu_tanh_backward_numpy\`。

今天必须复用它们，不能把 GELU 的实现复制一遍。

## 五、FFN 为什么不混合 Token？

对于每个 Token 位置 \(t\)：

$$
Y_{b,t,:}
=
\operatorname{GELU}_{\mathrm{tanh}}
(X_{b,t,:}W_1+b_1)W_2+b_2.
$$

所有位置共用 W1、b1、W2、b2，但每个位置只读取自己原有的输入特征。因此仅扰动 Token 2 时，其他 Token 的 FFN 输出应保持不变。

## 六、Backward：从输出依次传回输入

令：

$$
G=\frac{\partial L}{\partial Y}.
$$

第二层：

$$
\frac{\partial L}{\partial W_2}
=
\sum_{b,t}A_{b,t,:}^{\top}G_{b,t,:},
\qquad
\frac{\partial L}{\partial b_2}
=
\sum_{b,t}G_{b,t,:}.
$$

传入激活函数：

$$
G_A=GW_2^{\top},
\qquad
G_Z=G_A\odot\operatorname{GELU}_{\mathrm{tanh}}'(Z).
$$

第一层：

$$
\frac{\partial L}{\partial W_1}
=
\sum_{b,t}X_{b,t,:}^{\top}G_{Z,b,t,:},
\qquad
\frac{\partial L}{\partial b_1}
=
\sum_{b,t}G_{Z,b,t,:}.
$$

输入梯度：

$$
\boxed{
\frac{\partial L}{\partial X}
=
G_ZW_1^{\top}
}.
$$

五组梯度输出顺序：

\`grad_x, grad_w1, grad_b1, grad_w2, grad_b2\`

形状分别为：

\`(B,T,D), (D,H), (H,), (H,D), (D,)\`

## 七、代码任务

**Task A：FFN Forward**

\`ffn_forward_numpy(x, w1, b1, w2, b2)\`

**Task B：FFN Backward**

\`ffn_backward_numpy(grad_output, x, w1, b1, w2)\`

核心实现只用 NumPy。测试中用 PyTorch Autograd 和 Day 03 的数值梯度做独立校验。

## 八、验收命令

~~~powershell
python -m pytest -q -m exercise tests/test_week14_exercises.py
~~~

~~~powershell
python -m pytest -q -m exercise
~~~

~~~powershell
python -m scripts.day14_ffn_demo
~~~

验证内容：前向数值、手写五组梯度、有限差分、形状、Token 独立性、输入不变性、数值有限性。

本地实验结果目录：

~~~text
results/day14/
├── metrics.json
├── ffn_feature_heatmaps.png
└── token_perturbation_impact.png
~~~

**请等实际运行后再填写测试通过数量与实验结果，不提前标记完成。**

## 九、Day 14 核心结论

1. FFN 在 Token 内部混合特征通道，不直接交换 Token 信息。
2. 第一层一般扩大特征宽度，第二层回到模型维度。
3. GELU 引入非线性；没有激活的两层线性变换可以合并。
4. 参数在所有 Token 位置共享，所以权重梯度和偏置梯度必须沿 B、T 累积。
5. 反向传播从 W2/b2 开始，经 GELU 返回 W1/b1，再得到输入梯度。
6. FFN、Attention、Residual 和 LayerNorm 是不同模块，后续组合时需要明确各自责任。

## 十、Day 14 知识词云

![Day 14 Knowledge Word Cloud](images/day14-knowledge-wordcloud.svg)

核心关键词：

**Feed-Forward Network · FFN · GELU · Position-wise · Feature Expansion · Two Linear Layers · Broadcasting · Gradient Check · NumPy · Backpropagation · Parameter Sharing**

---

**Day 14：理论与练习材料已准备，等待完成代码与验收。**
