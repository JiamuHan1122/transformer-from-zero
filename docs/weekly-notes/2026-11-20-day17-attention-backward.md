# Day 17 学习笔记｜Multi-Head Attention Backward：Softmax Jacobian 与梯度分支汇合

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-20（实验验收：2026-11-23）  
> **阶段：** Transformer 手写反向传播 · Day 17  
> **今日主题：** Softmax Backward、Attention Gradients、Q/K/V Branches、Matrix Calculus、Autograd Cross-check  
> **状态：** NumPy 手写实现、29 项专项测试、213 项全项目 Exercise 测试及 Demo 均已通过  
> **核心实现：** exercises/week17.py  
> **测试：** tests/test_week17_exercises.py  
> **实验脚本：** scripts/day17_attention_backward_demo.py

---

## 一、从前向 Encoder 转向完整反向传播

Day 16 已经完成三层 Encoder Stack 的 NumPy 前向。Day 17 补齐 Day 08 Multi-Head Attention 的手写反向传播，目标是从上游梯度计算：

$$
G_X,\quad G_{W_Q},\quad G_{W_K},\quad G_{W_V},\quad G_{W_O}.
$$

![Attention backward map](images/day17-attention-backward.svg)

## 二、前向计算图

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

拆分 Head 后：

$$
S=Q_hK_h^\top/\sqrt{d_k},\quad
A=\operatorname{Softmax}(S),\quad
C=AV_h.
$$

合并多个 Head：

$$
M=\operatorname{MergeHeads}(C),\qquad Y=MW_O.
$$

本练习沿用 Day 08 的无 Mask、无投影 Bias 实现，并支持 $d_k\ne d_v$。

## 三、Softmax Jacobian

逐行 Softmax：

$$
a_i=\frac{e^{s_i}}{\sum_j e^{s_j}}.
$$

Jacobian：

$$
\frac{\partial a_i}{\partial s_j}=a_i(\delta_{ij}-a_j),
\qquad J=\operatorname{diag}(a)-aa^\top.
$$

对上游梯度 $g_a$：

$$
\boxed{g_s=a\odot\left(g_a-\sum_j a_jg_{a,j}\right).}
$$

手算例子：$a=[0.2,0.3,0.5]$，$g_a=[1,2,-1]$，加权点积为 0.3，因此：

$$
g_s=[0.14,0.51,-0.65],\qquad\sum_jg_{s,j}=0.
$$

行和为零反映 Softmax 对同一行 Score 整体平移的不变性。

## 四、Multi-Head Attention 的五组梯度

从上游梯度 $G_Y$ 开始。

输出投影：

$$
G_M=G_YW_O^\top,\quad
G_{W_O}=\sum_{b,t}M_{b,t,:}^\top G_{Y,b,t,:}.
$$

对 $C=AV_h$：

$$
G_A=G_CV_h^\top,\quad G_{V_h}=A^\top G_C.
$$

Softmax：

$$
G_S=A\odot(G_A-\sum_jA_jG_{A,j}).
$$

缩放点积：

$$
G_{Q_h}=G_SK_h/\sqrt{d_k},\quad
G_{K_h}=G_S^\top Q_h/\sqrt{d_k}.
$$

合并各 Head 后得到 $G_Q,G_K,G_V$，参数梯度为：

$$
G_{W_Q}=\sum_{b,t}X_{b,t,:}^\top G_{Q,b,t,:},
$$
$$
G_{W_K}=\sum_{b,t}X_{b,t,:}^\top G_{K,b,t,:},
$$
$$
G_{W_V}=\sum_{b,t}X_{b,t,:}^\top G_{V,b,t,:}.
$$

Q、K、V 三个投影共享同一个输入 $X$，所以输入梯度必须汇总三条路径：

$$
\boxed{G_X=G_QW_Q^\top+G_KW_K^\top+G_VW_V^\top.}
$$

不能求平均，也不能遗漏任意一支。

## 五、维度追踪

示例 $B=2,T=4,D=6,H=3,d_k=2,d_v=4,D_{\mathrm{out}}=5$：

| 张量 / 梯度 | 形状 |
|---|---|
| X / G_X | (2,4,6) |
| W_Q、W_K / 梯度 | (6,6) |
| W_V / 梯度 | (6,12) |
| W_O / 梯度 | (12,5) |
| Qh、Kh | (2,3,4,2) |
| Vh | (2,3,4,4) |
| A、G_A、G_S | (2,3,4,4) |
| Y、G_Y | (2,4,5) |

## 六、代码任务与验收方式

实现两个 NumPy 函数：

1. `softmax_backward_numpy`：实现 Softmax Jacobian-vector product。
2. `multi_head_attention_backward_numpy`：沿输出投影、Value 聚合、Softmax、Q/K 点积和输入投影逆序传播。

核心函数统一使用 float64，不修改调用者输入。PyTorch Autograd 只作为独立数值参考，不参与核心函数实现。

对照目标：

$$
\mathcal L=\sum_{b,t,d}Y_{b,t,d}G_{Y,b,t,d}.
$$

有限差分：

$$
\frac{\partial\mathcal L}{\partial z}
\approx\frac{\mathcal L(z+\epsilon)-\mathcal L(z-\epsilon)}{2\epsilon}.
$$

运行命令：

~~~powershell
python -m pytest -q -m exercise tests/test_week17_exercises.py
python -m pytest -q -m exercise
python -m scripts.day17_attention_backward_demo
~~~

本地生成 `results/day17/metrics.json`、`gradient_error_by_tensor.png` 和 `input_gradient_heatmaps.png`。仓库 `.gitignore` 将 `results/` 作为本地实验产物排除在 Git 提交之外，因此本笔记归档关键数值，不把结果图和临时文件混入源码提交。

## 七、能力边界与下一步

本次只覆盖 Day 08 的**无 Mask、无投影 Bias** Multi-Head Self-Attention Backward，不等于已经完成带 Padding/Causal Mask、Dropout、完整 Encoder Block 和 Encoder Stack 的手写训练流程。

下一阶段将把 MHA、LayerNorm、Residual 和 FFN 的反向链路组合起来，实现 Day 15–16 的 Block / Stack Backward。

## 八、Day 17 知识词云

![Day 17 knowledge word cloud](images/day17-knowledge-wordcloud.svg)

**Attention Backward · Jacobian · Softmax Gradient · Q/K/V · Scaled Dot Product · Chain Rule · Gradient Branches · Finite Differences · Autograd Reference · dL/dX**

---

## 九、Day 17 最终验收记录（2026-11-23）

### 9.1 测试结果

| 验收项 | 结果 |
|---|---:|
| Day 17 专项测试 | **29 passed in 16.03s** |
| 全项目 Exercise 回归 | **213 passed, 4 deselected in 3.73s** |
| Demo | **SUCCESS** |
| 所有梯度有限 | `true` |

4 deselected 是本次 `-m exercise` 筛选下未参与运行的测试，不是失败。

### 9.2 Demo 配置与梯度形状

本次配置为 $B=2,T=5,D=8,H=2,d_k=4,d_v=3,D_{\mathrm{out}}=8$，特意令 $d_k\ne d_v$。

| 梯度 | 形状 |
|---|---|
| $G_X$ | (2,5,8) |
| $G_{W_Q}$ | (8,8) |
| $G_{W_K}$ | (8,8) |
| $G_{W_V}$ | (8,6) |
| $G_{W_O}$ | (6,8) |

标量目标值：-0.97109374704345。

### 9.3 PyTorch Autograd 对照

| 梯度 | 最大绝对误差 | 相对 L2 误差 |
|---|---:|---:|
| X | $1.1102\times10^{-16}$ | $2.8465\times10^{-16}$ |
| W_Q | $6.9389\times10^{-17}$ | $4.6207\times10^{-16}$ |
| W_K | $5.5511\times10^{-17}$ | $2.5339\times10^{-16}$ |
| W_V | $4.4409\times10^{-16}$ | $2.2248\times10^{-16}$ |
| W_O | $4.4409\times10^{-16}$ | $1.7632\times10^{-16}$ |

最大绝对误差处于 float64 双精度浮点运算的舍入误差量级。

### 9.4 有限差分与结构性不变量

| 输入坐标 | 数值梯度与解析梯度绝对误差 |
|---|---:|
| (0,0,0) | $1.5155\times10^{-10}$ |
| (0,2,3) | $1.0976\times10^{-11}$ |
| (1,4,7) | $2.6874\times10^{-10}$ |

- Softmax Score 梯度行和最大误差：$2.4980\times10^{-16}$。
- Attention 权重行和最大误差：$2.2204\times10^{-16}$。

专项测试还覆盖多个 Head 配置、$d_k\ne d_v$、参数与输入的有限差分、输入不被修改和非法输入检查。

### 9.5 梯度尺度

| 梯度 | L2 范数 |
|---|---:|
| X | 0.988725 |
| W_Q | 0.354197 |
| W_K | 0.669067 |
| W_V | 3.649251 |
| W_O | 5.455883 |

本地误差柱状图显示五组梯度最大绝对误差均约为 $10^{-16}$；梯度热力图的“Manual minus Autograd”色标同样只有 $10^{-16}$ 量级，反映浮点舍入误差，而非实质性梯度偏差。

### 9.6 Warning 清理

首次运行出现的 PyTorch Warning 来自 Demo 将带 `requires_grad=True` 的 Tensor 直接转换为 Python 标量：`float(loss)`。它不在手写梯度计算路径中，没有影响本次测试和数值对照。

最终同步的 Demo 已改为 `float(loss.detach().item())`，先脱离计算图再读取标量。该修改只清理日志转换警告，不改变计算或梯度公式。拉取后建议重新运行 Demo，确认本地不再出现该 Warning。

### 9.7 最终结论

Day 17 已完成从 Softmax Jacobian 到 Multi-Head Attention 五组梯度的手写反向传播。专项测试、全项目 Exercise 回归、PyTorch Autograd 对照、有限差分和关键数值不变量均通过。

当前能力边界仍是**无 Mask、无投影 Bias 的 Multi-Head Self-Attention Backward**；完整 Encoder Block / Stack Backward 尚未完成。
