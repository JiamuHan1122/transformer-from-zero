# Day 13｜Residual Connections：恒等映射与梯度传播

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-16  
> **阶段：** Transformer Block 核心结构 · Day 13  
> **今日主题：** Residual Connection、Identity Mapping、Gradient Flow、Jacobian、Pre-LN / Post-LN  
> **状态：** 学习材料与测试脚手架已发布；实现和实验待完成  
> **核心实现：** `exercises/week13.py`  
> **测试：** `tests/test_week13_exercises.py`  
> **实验脚本：** `scripts/day13_residual_demo.py`

---


## 一、今天要回答的问题

1. 为什么残差层写成 $Y=X+F(X)$，而不只写成 $Y=F(X)$？
2. 为什么标准残差相加要求两条分支形状一致？
3. 加法节点反向传播时，梯度如何分配？
4. 恒等映射路径为什么能提供直接梯度通路？
5. 残差连接与 LayerNorm 如何组成 Pre-LN / Post-LN？
6. 为什么残差连接有帮助，但不保证梯度永远稳定？

## 二、数学定义

普通子层：

$$
Y=F(X).
$$

残差子层：

$$
\boxed{Y=X+F(X)}.
$$

其中 $X$ 是恒等映射分支，$F(X)$ 是 Attention、FFN 或其他子层的输出。残差是逐元素相加，不是拼接或取平均。

若：

$$
X\in\mathbb{R}^{B\times T\times D},
$$

则标准残差要求：

$$
F(X)\in\mathbb{R}^{B\times T\times D},
\quad
(B,T,D)+(B,T,D)\rightarrow(B,T,D).
$$

如果两个分支维度不同，不能直接相加；可使用投影捷径 $Y=W_sX+F(X)$。今天只实现形状一致的恒等映射残差。

## 三、反向传播推导

令：

$$
Y=X+F(X),\qquad G=\frac{\partial L}{\partial Y}.
$$

加法节点对两条输入分支的局部导数都是 1：

$$
\frac{\partial L}{\partial X}\bigg|_{\mathrm{skip}}=G,
\qquad
\frac{\partial L}{\partial F(X)}=G.
$$

第二条梯度还要继续经过 $F$。最终：

$$
\boxed{\nabla_X L=G+J_F(X)^\top G}
$$

第一项来自恒等路径，第二项来自残差分支。残差结构提供直接梯度通路，但不能保证梯度永远不消失或放大；两条路径也可能发生抵消。

一个标量例子：令 $F(x)=-0.1x$。普通映射的局部导数是 $-0.1$，残差映射 $y=x-0.1x$ 的局部导数是 $0.9$。这是用于观察恒等路径作用的示例，不是普遍性能结论。

## 四、Pre-LN 与 Post-LN

Post-LN：

$$
X_1=\operatorname{LN}(X+\operatorname{MHA}(X)),
\quad
Y=\operatorname{LN}(X_1+\operatorname{FFN}(X_1)).
$$

Pre-LN：

$$
X_1=X+\operatorname{MHA}(\operatorname{LN}(X)),
\quad
Y=X_1+\operatorname{FFN}(\operatorname{LN}(X_1)).
$$

两者都使用残差连接，但 LayerNorm 位置不同。Pre-LN 的恒等路径绕过当前子层及其前置 LayerNorm；Post-LN 的相加结果还要经过后置 LayerNorm。

## 五、Task A：Residual Add Forward / Backward

### A1 Forward

函数：**residual_add_forward_numpy(x, branch_output)**

输入均为 (B,T,D)，形状必须完全一致。实现：

$$
Y=X+\text{branch\_output}.
$$

返回 float64，形状不变，不修改输入。

### A2 Backward

函数：**residual_add_backward_numpy(grad_output)**

已知 $G=\partial L/\partial Y$，返回 **grad_skip, grad_branch**：

$$
\boxed{\operatorname{grad\_skip}=G,\quad
\operatorname{grad\_branch}=G}
$$

两个返回值必须是独立的 float64 数组。

## 六、Task B：Linear Residual Block

定义最小线性残差分支：

$$
F(X)=XW+b,\qquad Y=X+XW+b.
$$

形状：

- $X:(B,T,D)$
- $W:(D,D)$
- $b:(D,)$
- $Y:(B,T,D)$

这只是研究残差机制的线性示例，不是完整 Transformer FFN。

### B1 Forward

函数：**residual_linear_forward_numpy(x, weight, bias)**

实现 $Y=X+XW+b$，返回 float64，不修改输入。

### B2 Backward

函数：**residual_linear_backward_numpy(grad_output, x, weight)**

令 $G=\partial L/\partial Y$，返回：

**grad_x, grad_weight, grad_bias**

$$
\boxed{\frac{\partial L}{\partial X}=G+GW^\top}
$$

$$
\boxed{\frac{\partial L}{\partial W}
=\sum_{b,t}X_{b,t,:}^{\top}G_{b,t,:}}
$$

$$
\boxed{\frac{\partial L}{\partial b}
=\sum_{b,t}G_{b,t,:}}
$$

注意：grad_x 的第一项 $G$ 就是恒等映射路径带来的梯度。核心实现只用 NumPy；PyTorch 仅在测试中作为参考。

## 七、测试与验收

专项测试：

~~~powershell
python -m pytest -q -m exercise tests/test_week13_exercises.py
~~~

全项目回归：

~~~powershell
python -m pytest -q -m exercise
~~~

测试覆盖：加法前向、零分支恒等性质、非法输入、加法反向、有限差分、线性残差前向、PyTorch Autograd 对照、三组有限差分、零权重恒等梯度、输入完整性。

## 八、实验 Demo

运行：

~~~powershell
python -m scripts.day13_residual_demo
~~~

本地输出：

~~~text
results/day13/
├── metrics.json
└── gradient_flow_comparison.png
~~~

Demo 使用明确标注的标量 Jacobian 示例，对比重复映射后的梯度范数。它展示一种可能的梯度传播改善机制，不作为普遍性能结论。

## 九、思考题

1. 如果 $F(X)=0$，残差模块退化成什么映射？
2. 如果 $W=0$，Linear Residual Block 的输入梯度是什么？
3. 为什么 $X+F(X)$ 不能直接替换成 concat？
4. Pre-LN 的恒等路径为什么能绕过当前子层内部的 LayerNorm？
5. 如果 $J_F(X)\approx-I$，两条梯度路径可能发生什么？

**Day 13 的目标：从前向结构独立推导反向梯度，并解释恒等映射为什么重要。**
