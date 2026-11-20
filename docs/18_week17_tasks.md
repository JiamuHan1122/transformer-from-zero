# Day 17｜Multi-Head Attention Backward：手写梯度与 Softmax Jacobian

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-20  
> **阶段：** Transformer 手写反向传播 · Day 17  
> **今日主题：** Softmax Backward、Q/K/V 分支、Scaled Dot-Product Gradients、Finite Differences  
> **状态：** 理论、NumPy 练习、PyTorch 测试和 Demo 脚手架已发布；本地验收待完成  
> **核心实现：** exercises/week17.py  
> **测试：** tests/test_week17_exercises.py  
> **实验：** scripts/day17_attention_backward_demo.py

---

## 一、今日科学问题

Day 08 实现了 Multi-Head Attention 的前向；Day 12–14 已完成 LayerNorm、Residual 和 FFN 的手写反向；Day 15–16 完成了完整 Encoder Block 和 Encoder Stack 的前向组装。

Attention 的反向是当前缺口。今天必须从上游梯度出发，计算：

$$
\boxed{G_X,\quad G_{W_Q},\quad G_{W_K},\quad G_{W_V},\quad G_{W_O}}
$$

![Day 17 Attention Backward](weekly-notes/images/day17-attention-backward.svg)

## 二、复盘前向与具体形状

Day 08 采用行向量约定、无投影 Bias、无 Attention Mask：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
$$

各 Head：

$$
S=\frac{Q_hK_h^\top}{\sqrt{d_k}},
\quad A=\operatorname{Softmax}(S),
\quad C=AV_h.
$$

合并 Head：

$$
M=\operatorname{MergeHeads}(C),\quad Y=MW_O.
$$

特别注意：原有 Day 08 支持 $d_k\ne d_v$，今天不能退化为只支持两者相等。

示例 $B=2,T=4,D=6,H=3,d_k=2,d_v=4,D_{\rm out}=5$：

| 数据 | 形状 |
|---|---|
| X | (2,4,6) |
| Qh、Kh | (2,3,4,2) |
| Vh | (2,3,4,4) |
| Scores、Attention | (2,3,4,4) |
| Context | (2,3,4,4) |
| Merged | (2,4,12) |
| Y、上游梯度 | (2,4,5) |

## 三、Task A 的数学：Softmax 不是独立的逐元素激活

一行 Softmax：

$$
a_i=\frac{e^{s_i}}{\sum_j e^{s_j}}.
$$

Jacobian：

$$
\frac{\partial a_i}{\partial s_j}
=a_i(\delta_{ij}-a_j),
\qquad J=\operatorname{diag}(a)-aa^\top.
$$

若上游梯度为 $g_a$，则：

$$
\boxed{g_s=a\odot(g_a-\sum_j a_jg_{a,j}).}
$$

**手算例子：**

$$
a=[0.2,0.3,0.5],\qquad g_a=[1,2,-1].
$$

加权点积：

$$
c=0.2(1)+0.3(2)+0.5(-1)=0.3.
$$

反向结果：

$$
g_s=[0.14,0.51,-0.65],
\qquad\sum_jg_{s,j}=0.
$$

原因是 Softmax 对整行分数加相同常数不敏感：

$$
\operatorname{Softmax}(s+c\mathbf 1)
=\operatorname{Softmax}(s).
$$

因此沿全 1 方向的梯度为零。写 NumPy 代码时必须沿最后一维求和，并保留可广播维度（keepdims=True）。

## 四、Task B 的完整计算图逆序推导

给定上游梯度 $G_Y$。

**Step 1：输出投影。**

$$
G_M=G_YW_O^\top,
\qquad
G_{W_O}=\sum_{b,t}M_{b,t,:}^\top G_{Y,b,t,:}.
$$

将 $G_M$ 拆分 Head 得到 $G_C$。

**Step 2：Value 聚合的两个分支。**

$$
C=AV_h,\quad
\boxed{G_A=G_CV_h^\top},\quad
\boxed{G_{V_h}=A^\top G_C}.
$$

**Step 3：Softmax。**

$$
G_S=A\odot(G_A-\sum_j A_jG_{A,j}).
$$

**Step 4：缩放点积。**

$$
S=Q_hK_h^\top/\sqrt{d_k},
$$

$$
\boxed{G_{Q_h}=\frac{G_SK_h}{\sqrt{d_k}}},
\qquad
\boxed{G_{K_h}=\frac{G_S^\top Q_h}{\sqrt{d_k}}}.
$$

**Step 5：合并各 Head 的三个梯度。**

得到 $G_Q,G_K,G_V$，再由输入投影计算三个参数梯度：

$$
G_{W_Q}=\sum_{b,t}X_{b,t,:}^\top G_{Q,b,t,:},
$$

$$
G_{W_K}=\sum_{b,t}X_{b,t,:}^\top G_{K,b,t,:},
$$

$$
G_{W_V}=\sum_{b,t}X_{b,t,:}^\top G_{V,b,t,:}.
$$

同一个输入 X 被 Q、K、V 三个投影复用，因此输入梯度必须**累加三支**：

$$
\boxed{
G_X=G_QW_Q^\top+G_KW_K^\top+G_VW_V^\top.
}
$$

这不是求平均，也不能只保留一支。参数梯度需要对 Batch 和 Token 贡献求和；输入梯度保留每个位置自己的 (B,T,D) 形状。

## 五、今天的两个编程任务

### Task A：Softmax Backward

实现 softmax_backward_numpy(weights, grad_weights)，输出与输入相同形状的 float64 Score 梯度。检查 Softmax 行和梯度约为零、输入不被修改。

### Task B：Multi-Head Attention Backward

实现 multi_head_attention_backward_numpy(grad_output, x, w_q, w_k, w_v, w_o, num_heads)，返回五组梯度（顺序为 grad_x、grad_w_q、grad_w_k、grad_w_v、grad_w_o）。

实现顺序：

1. 验证参数类型与维度，先将输入转成 float64。
2. 重算前向 Q/K/V、Scores、Attention、Context 和 Merged。
3. 反向经过输出 W_O 投影。
4. 拆分 Context 梯度，计算 G_A 与 G_V。
5. 通过 Task A 计算 Softmax Score 梯度。
6. 通过点积计算 G_Q、G_K。
7. 合并各 Head 的梯度。
8. 累计四组参数梯度，并将 Q/K/V 三支贡献合并成 G_X。
9. 返回五个 float64 张量，不修改原始数组。

## 六、数值梯度为何需要双重检查？

PyTorch 自动求导对照的目标为：

$$
\mathcal L=\sum_{b,t,d}Y_{b,t,d}G_{Y,b,t,d}.
$$

这里上游梯度视为固定常数。PyTorch Autograd 得到的五组梯度应与手写 NumPy 一致。

另外对每一个抽样元素做中心有限差分：

$$
g_{\mathrm{num}}
\approx
\frac{\mathcal L(z+\epsilon)-\mathcal L(z-\epsilon)}
{2\epsilon}.
$$

二者检查方向不同：Autograd 检查计算图整体，有限差分通过扰动目标函数独立检查导数值。还需要测试单 Head、多 Head、$d_k\ne d_v$、错误维度、dtype 和数组不被修改。

## 七、正式验收和输出文件

~~~powershell
python -m pytest -q -m exercise tests/test_week17_exercises.py
python -m pytest -q -m exercise
python -m scripts.day17_attention_backward_demo
~~~

运行成功后在本地生成：

~~~text
results/day17/
├── metrics.json
├── gradient_error_by_tensor.png
└── input_gradient_heatmaps.png
~~~

检查五组梯度最大绝对误差、相对误差、Softmax 反向的行和、有限差分结果以及所有数值是否有限。尚未实际运行，不能预先记录通过数量或声称图像已生成。

## 八、今天的边界和下一步

本次只针对 Day 08 的**无 Mask、无投影 Bias** Self-Attention Backward；不包含 Day 09 的 Mask、Dropout 或完整 Encoder Block 的反向。

完成后，才有条件把 MHA、LayerNorm、Residual 和 FFN 的反向链路组合起来，继续实现 Day 15–16 的 Block/Stack Backward。

## 九、思考题

1. 为什么 Softmax 反向不能仅用逐元素乘法？
2. 为什么 Score 梯度各行求和为零？
3. Q/K 梯度公式为什么分别有不同的转置位置？
4. 为什么原始 X 的梯度包含三支相加？
5. 为什么参数梯度对 Batch、Token 求和，输入梯度却不求和？
6. 为什么 Day 08 的 $d_k\ne d_v$ 不会阻止完成反向？
7. 为什么正确形状不等于正确梯度？

---

**Day 17 完成标准：真正推导并实现 Softmax Backward 与完整 MHA Backward，在 PyTorch 和有限差分两套独立检查中通过后再归档最终代码。**
