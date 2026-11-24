# Day 18 学习笔记｜Encoder Block Backward：从独立梯度到完整计算图

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-24  
> **阶段：** Transformer Encoder · 手写反向传播  
> **主题：** Pre-LN / Post-LN、Residual Gradient Paths、Block-level Jacobian、13 Gradients  
> **状态：** Day 18 实现与本地验收全部完成；专项 51 passed，全项目 264 passed（4 deselected），Demo SUCCESS  
> **实现：** exercises/week18.py  
> **测试：** tests/test_week18_exercises.py  
> **Demo：** scripts/day18_encoder_block_backward_demo.py

---

## 一、今天比 Day 17 多了什么？

Day 17 解决了一个 Attention 模块的五组梯度；今天要完成完整 Encoder Block 的 13 组梯度（输入 X 加 12 组可学习参数）。已经实现的四种 NumPy Backward 是今天要复用的工具：Attention（Day 17）、LayerNorm（Day 12）、Residual（Day 13）、FFN（Day 14）。

**最大的难点不是矩阵求导，而是不同路径的梯度能否按正确顺序汇合。**

![Day 18 Backward gradient paths](images/day18-gradient-paths.svg)

## 二、链式法则和残差求导

复合函数按前向逆序求导。对于：

$$
Y=X+F(X),
$$

输入梯度包含两项：

$$
\boxed{G_X=G_Y+J_F^\top G_Y}.
$$

即使函数路径产生零梯度，恒等路径仍可能存在。每个 Encoder Block 有两个残差节点，因此反向必须执行两次正确的分叉和汇合。

## 三、Pre-LN 计算图

Forward：

$$
N_1=\operatorname{LN}_1(X),\quad
A=\operatorname{MHA}(N_1),\quad H=X+A,
$$

$$
N_2=\operatorname{LN}_2(H),\quad
F=\operatorname{FFN}(N_2),\quad Y=H+F.
$$

Backward 首先拆分末尾残差节点：

$$
G_{H,\mathrm{skip}}=G_Y,\qquad G_F=G_Y.
$$

FFN -> LN2：

$$
G_{N_2}=\operatorname{FFNBackward}(G_F;N_2),
\quad
G_{H,\mathrm{LN2}}=\operatorname{LNBackward}(G_{N_2};H).
$$

H 上的梯度汇合：

$$
\boxed{G_H=G_Y+G_{H,\mathrm{LN2}}}.
$$

再拆分第一层残差，Attention -> LN1：

$$
G_{N_1}=\operatorname{MHABackward}(G_H;N_1),
\quad
G_{X,\mathrm{LN1}}=\operatorname{LNBackward}(G_{N_1};X).
$$

最终：

$$
\boxed{G_X=G_H+G_{X,\mathrm{LN1}}}.
$$

注意 $G_H$ 已包含 FFN 路径的梯度贡献，不等于原始 $G_Y$。

## 四、Post-LN 计算图

Forward：

$$
A=\operatorname{MHA}(X),\quad
H_{\mathrm{raw}}=X+A,\quad
H=\operatorname{LN}_1(H_{\mathrm{raw}}),
$$

$$
F=\operatorname{FFN}(H),\quad
Y_{\mathrm{raw}}=H+F,\quad
Y=\operatorname{LN}_2(Y_{\mathrm{raw}}).
$$

Backward 必须从 LN2 开始：

$$
G_{Y_{\mathrm{raw}}}
=\operatorname{LNBackward}(G_Y;Y_{\mathrm{raw}}).
$$

第二个残差 + FFN：

$$
G_H=G_{Y_{\mathrm{raw}}}
+\operatorname{FFNBackward}(G_{Y_{\mathrm{raw}}};H).
$$

经过 LN1：

$$
G_{H_{\mathrm{raw}}}
=\operatorname{LNBackward}(G_H;H_{\mathrm{raw}}).
$$

第一个残差 + MHA：

$$
\boxed{
G_X=G_{H_{\mathrm{raw}}}
+\operatorname{MHABackward}(G_{H_{\mathrm{raw}}};X).
}
$$

Post-LN 的直接残差梯度仍然经过 LN。即使 Attention、FFN 两分支都不对输入产生梯度，也不能直接推出 $G_X=G_Y$。

## 五、一个标量残差手算例子

设：

$$
h=x+2x=3x,\qquad
y=h+3h=4h=12x.
$$

若上游 $G_y=5$，反向先经过第二个残差：

$$
G_h=5+3\times5=20.
$$

再经过第一个残差：

$$
G_x=20+2\times20=60.
$$

直接求导 $dy/dx=12$，得到 $G_x=5\times12=60$，一致。

如果两次残差只各取一条路径，结果就会错。这是 Day 18 代码审查最重要的直觉检查。

## 六、形状与参数顺序

今天使用的示例形状：$B=2,T=5,D=8,H=2,D_{\mathrm{ff}}=32$。

| 对象 | 形状 |
|---|---|
| X、Y、G_X、G_Y | (2,5,8) |
| Wq/Wk/Wv/Wo 梯度 | (8,8) |
| gamma_attn/beta_attn 梯度 | (8,) |
| W1 梯度 | (8,32) |
| b1 梯度 | (32,) |
| W2 梯度 | (32,8) |
| b2 梯度 | (8,) |
| gamma_ffn/beta_ffn 梯度 | (8,) |

**返回顺序：**

1. grad_x
2. grad_w_q / grad_w_k / grad_w_v / grad_w_o
3. grad_gamma_attn / grad_beta_attn
4. grad_w1 / grad_b1 / grad_w2 / grad_b2
5. grad_gamma_ffn / grad_beta_ffn

一共 13 项，严格对应 Day 15 Forward 的数组参数。

## 七、今天真正要实现的部分

在 exercises/week18.py 完成函数：

~~~python
transformer_encoder_block_backward_numpy(
    grad_output, x, w_q, w_k, w_v, w_o, num_heads,
    gamma_attn, beta_attn,
    w1, b1, w2, b2,
    gamma_ffn, beta_ffn,
    eps=1e-5, norm_first=True
)
~~~

注意：

- 沿用 Day 15 的输入 Validator；
- 将全部数组转换为 float64；
- 重算正确的前向中间量；
- 复用 Day 12/13/14/17 的 Backward；
- Pre-LN 与 Post-LN 不共享错误的中间值；
- 每个参数梯度保持原始参数 Shape；
- 不修改输入。

## 八、验收设计

目标标量：

$$
\mathcal L=\sum_{b,t,d}Y_{b,t,d}G_{Y,b,t,d}.
$$

通过三个层次验证：

- 13 个梯度的返回顺序、形状、dtype；
- 与独立 PyTorch Autograd 计算图对照，Pre-LN/Post-LN 全部比较；
- 对 13 个输入或参数数组的选定元素做中心有限差分；

附加结构不变量：Pre-LN 把 Attention 与 FFN 分支设置为零，反向输入应满足 $G_X=G_Y$。Post-LN 则仍需要经过两个 LayerNorm。

运行：

~~~powershell
python -m pytest -q -m exercise tests/test_week18_exercises.py
python -m pytest -q -m exercise
python -m scripts.day18_encoder_block_backward_demo
~~~

本地输出目录：

~~~text
results/day18/
├── metrics.json
├── gradient_error_comparison.png
├── pre_post_gradient_norms.png
└── input_gradient_pre_post.png
~~~

**已实际完成本地验收；详细数值结果见第十节。**

## 九、知识词云

![Day 18 Knowledge Word Cloud](images/day18-knowledge-wordcloud.svg)

**Encoder Backward · Pre-LN · Post-LN · Residual Paths · Gradient Accumulation · Computation Graph · Chain Rule · LayerNorm · Attention · FFN · Jacobian · Finite Difference**

---

**Day 18 目标：真正写出完整 Encoder Block 的手动反向传播，全部 13 个梯度可解释、可验证；下一阶段才讨论多层 Stack Backward。**


## 十、Day 18 实验结果与最终复盘（2026-11-24）

**实际运行状态：完成。** 对照本地 `tfz` 运行日志：

- 专项测试：**51 passed in 17.13s**。
- 全项目 Exercise 回归：**264 passed, 4 deselected in 3.92s**。
- Demo：**SUCCESS: Day 18 Encoder Block Backward checks passed**。
- 全部 13 个梯度（输入 X + 12 组参数）均经 Pre-LN 与 Post-LN 对照。
- 梯度数值有限（两种模式均 true），调用者数组未被修改（true）。
- Pre-LN 零分支恒等梯度误差：**0.0**。

### 10.1 手写梯度与 PyTorch Autograd 的一致性

本次输入形状 (2,5,8)，Head 数 2。Pre-LN 13 组梯度的最大绝对误差为 **4.440892098500626e-15**；Post-LN 为 **3.219646771412954e-15**，均处于 float64 数值误差量级。相对 L2 误差各组约 1e-16 到 1e-15，结果支持计算图求导正确。

Post-LN 的 beta_ffn 梯度绝对误差为 **0.0**；误差图中为了对数轴绘制采用 1e-18 的视觉下限，不能误读为实际误差。

### 10.2 有限差分独立验证

| 模式 | 元素 | 数值梯度 | 手写梯度 | 绝对误差 |
|---|---|---:|---:|---:|
| Pre-LN | X [0,2,3] | 1.7353097492289749 | 1.7353097487137328 | 5.152420712306593e-10 |
| Pre-LN | gamma_attn [1] | 1.5795732677403862 | 1.579573268725748 | 9.853617921606883e-10 |
| Post-LN | X [0,2,3] | 0.2661047373564429 | 0.26610473678547675 | 5.709661632380403e-10 |
| Post-LN | gamma_attn [1] | 2.061923691121237 | 2.061923690936588 | 1.8464874074197724e-10 |

### 10.3 梯度范数与数学解释

输入 X 的梯度 L2 范数 Pre-LN 为 **14.272661784487477**，Post-LN 为 **12.32516104807154**；Wq 为 16.058930885934796 vs 11.103237236403857；Wo 为 18.025090920961443 vs 11.603729547696704。

但 LN2 参数的方向恰好有所不同：gamma_ffn 为 **2.7754918107402773 vs 7.320416467536417**；beta_ffn 为 **4.842326531009327 vs 10.79141214995242**。

这组差异说明改变 LayerNorm 的放置位置会改变同一组参数与输入下的梯度传播；**不能将单个随机样例下的范数差异归纳为训练稳定性或架构优劣的结论**。

### 10.4 最重要的编码经验

第一，反向必须严格反转 Day 15 的 Pre-LN / Post-LN 前向执行顺序；第二，Residual Backward 不只是分发梯度，还需要在 H、X 两个汇合点求和；第三，调用 LayerNorm Backward 必须提供它在真实前向见到的输入，而不是另一个形状相同的张量。

### 10.5 文件归档及限制

- 代码：`exercises/week18.py`，完整 Pre-LN / Post-LN 数值梯度实现。
- 测试：`tests/test_week18_exercises.py`，51 项专项通过。
- Demo：`scripts/day18_encoder_block_backward_demo.py`。
- 详细全部 13 组误差与梯度范数：`docs/19_week18_tasks.md` 的“Day 18 最终验收与成果归档”。
- 本地结果：`results/day18/metrics.json`、`gradient_error_comparison.png`、`pre_post_gradient_norms.png`、`input_gradient_pre_post.png`。由于 `.gitignore` 忽略 `results/*`，这些文件**保留在本地而不在本次 GitHub 提交内**。

**Day 18 已完成并归档；接下来可以进入多层 Encoder Stack 的手写反向传播。**
