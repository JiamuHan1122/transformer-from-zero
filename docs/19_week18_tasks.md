# Day 18｜完整 Transformer Encoder Block 的手写反向传播

> **项目：** Transformer From Zero · 365 Days  
> **日期：** 2026-11-24  
> **阶段：** Transformer Backward · Encoder Block  
> **主题：** Pre-LN / Post-LN、Reverse-Mode Autodiff、Residual Gradient Accumulation、MHA + LayerNorm + FFN  
> **状态：** Pre-LN / Post-LN 手写反向实现已完成；本地专项 51 passed、全项目 264 passed（另 4 deselected）、Demo SUCCESS；已归档 GitHub  
> **实现：** exercises/week18.py  
> **测试：** tests/test_week18_exercises.py  
> **Demo：** scripts/day18_encoder_block_backward_demo.py

---

## 一、Day 18 的核心问题

Day 17 已验证无掩码 Multi-Head Attention Backward；Day 12/13/14 完成了 LayerNorm、Residual、FFN 的手写反向。今天需要以 Day 15 的 Forward 为准，组合成完整 Encoder Block Backward。

多个模块不仅串联，还通过残差结构产生梯度的**分叉与汇合**。反向传播必须保持逆序，不能漏掉任一恒等分支。

![Encoder Block Backward](weekly-notes/images/day18-gradient-paths.svg)

## 二、固定真实接口与参数顺序

输入、输出：

$$
X,Y\in\mathbb R^{B\times T\times D}.
$$

注意力参数 $W_Q,W_K,W_V,W_O$ 均为 (D,D)，Head 数 H 整除 D。LN1 的 gamma/beta 与 LN2 的 gamma/beta 均为 (D,)；FFN 参数分别为 W1 (D,Dff)、b1 (Dff,)、W2 (Dff,D)、b2 (D,)。

加上 X，需输出**13 个梯度**，顺序必须和 Day 15 的参数一致（忽略非可微 Head 数）：

1. grad_x
2. grad_w_q
3. grad_w_k
4. grad_w_v
5. grad_w_o
6. grad_gamma_attn
7. grad_beta_attn
8. grad_w1
9. grad_b1
10. grad_w2
11. grad_b2
12. grad_gamma_ffn
13. grad_beta_ffn

## 三、两个基本法则

复合函数 $Y=F_3(F_2(F_1(X)))$ 的反向按 F3、F2、F1 逆序计算。

残差函数：

$$
Y=X+F(X).
$$

其输入梯度必须汇合两条路径：

$$
\boxed{G_X=G_Y+J_F^\top G_Y}.
$$

例如 $y=x+2x=3x$，若上游梯度为 4，则 $G_x=4+8=12$。

## 四、Pre-LN 的完整 Forward 与 Backward

### 4.1 Forward

$$
N_1=\operatorname{LN}_1(X),\qquad A=\operatorname{MHA}(N_1),
$$

$$
H=X+A,
$$

$$
N_2=\operatorname{LN}_2(H),\qquad F=\operatorname{FFN}(N_2),
$$

$$
\boxed{Y=H+F}.
$$

### 4.2 Backward：第一次残差汇合

从 $G_Y$ 开始，经第二个残差节点：

$$
G_{H,\mathrm{skip}}=G_Y,\qquad G_F=G_Y.
$$

然后经 FFN 和 LN2：

$$
G_{N_2}=\operatorname{FFNBackward}(G_F;N_2),
$$

$$
G_{H,\mathrm{LN2}}=\operatorname{LNBackward}(G_{N_2};H).
$$

在 H 汇合：

$$
\boxed{G_H=G_{H,\mathrm{skip}}+G_{H,\mathrm{LN2}}}.
$$

### 4.3 Backward：第二次残差汇合

经过第一个残差节点，将 $G_H$ 分给 X 恒等路径和 Attention：

$$
G_{X,\mathrm{skip}}=G_H,\qquad G_A=G_H.
$$

反向经过 Attention 再经过 LN1：

$$
G_{N_1}=\operatorname{MHABackward}(G_A;N_1),
$$

$$
G_{X,\mathrm{LN1}}=\operatorname{LNBackward}(G_{N_1};X).
$$

最终：

$$
\boxed{G_X=G_H+G_{X,\mathrm{LN1}}}.
$$

Pre-LN 需在 H 与 X **分别进行一次梯度合并**。

### 4.4 零分支不变量

当 Wv=Wo=0 且 W2=b2=0 时，两个输出分支都变为零，Pre-LN 变成 $Y=X$，因此：

$$
\boxed{G_X=G_Y}.
$$

这是额外的数学结构检查，不能用 Shape 检查替代。

## 五、Post-LN 的完整 Forward 与 Backward

### 5.1 Forward

$$
A=\operatorname{MHA}(X),\qquad H_{\mathrm{raw}}=X+A,
$$

$$
H=\operatorname{LN}_1(H_{\mathrm{raw}}),
$$

$$
F=\operatorname{FFN}(H),\qquad Y_{\mathrm{raw}}=H+F,
$$

$$
\boxed{Y=\operatorname{LN}_2(Y_{\mathrm{raw}})}.
$$

### 5.2 Backward

Post-LN 的第一个反向步骤必须是 LN2：

$$
G_{Y_{\mathrm{raw}}}
=\operatorname{LNBackward}(G_Y;Y_{\mathrm{raw}}).
$$

第二个残差拆分，FFN 反向并在 H 合并：

$$
G_H=G_{Y_{\mathrm{raw}}}+
\operatorname{FFNBackward}(G_{Y_{\mathrm{raw}}};H).
$$

随后经 LN1：

$$
G_{H_{\mathrm{raw}}}
=\operatorname{LNBackward}(G_H;H_{\mathrm{raw}}).
$$

第一个残差拆分，经过 MHA 并合并：

$$
\boxed{
G_X=G_{H_{\mathrm{raw}}}+
\operatorname{MHABackward}(G_{H_{\mathrm{raw}}};X).
}
$$

与 Pre-LN 不同，Post-LN 即使输出分支为零，梯度仍需穿过两次 LN，不应默认 $G_X=G_Y$。

## 六、复用 Day 12/13/14/17 的函数

已验证的接口：

~~~python
grad_skip, grad_branch = residual_add_backward_numpy(grad_output)

grad_input, grad_gamma, grad_beta = layer_norm_backward_numpy(
    grad_output, ln_input, gamma, eps
)

grad_input, grad_w1, grad_b1, grad_w2, grad_b2 = ffn_backward_numpy(
    grad_output, ffn_input, w1, b1, w2
)

grad_input, grad_wq, grad_wk, grad_wv, grad_wo = (
    multi_head_attention_backward_numpy(
        grad_output, attention_input, w_q, w_k, w_v, w_o, num_heads
    )
)
~~~

必须传入 **该模块在前向真正使用的输入**，特别是 LN1 的 Pre-LN 输入是 X，而 Post-LN 输入为 H_raw。形状相同不意味着梯度正确。

## 七、今日代码要求

在 exercises/week18.py 中完成 transformer_encoder_block_backward_numpy。

函数以 grad_output 为第一个参数，随后严格沿用 Day 15 Forward 的参数顺序，最后 eps=1e-5、norm_first=True。

约束：

- 核心部分只使用 NumPy，不调用 Torch/Autograd；
- 使用给定输入 Validator，先转换全部 14 个 NumPy 数组到 float64；
- 重算完整前向的中间状态，按相反顺序调用现有四类 Backward；
- 两个残差节点都必须拆分并合并；
- 参数梯度不按 Batch 或 Token 平均；
- 返回 13 个梯度，不修改任何原数组；
- 不添加 Mask、Dropout、分类头或 Stack-level LayerNorm。

## 八、今天为什么要使用三重验收？

1. **形状测试：** 每组梯度 Shape、顺序、dtype 正确；
2. **独立 PyTorch Autograd：** Pre-LN、Post-LN 两种模式下 13 组梯度一致；
3. **中心有限差分：** 逐一抽查 X、Attention 权重、LayerNorm gamma/beta、FFN 权重与 Bias 的导数。

目标函数：

$$
\mathcal L=\sum_{b,t,d}Y_{b,t,d}G_{Y,b,t,d}.
$$

有限差分：

$$
g_{\mathrm{num}}\approx
\frac{\mathcal L(z+\epsilon)-\mathcal L(z-\epsilon)}{2\epsilon}.
$$

其他边界测试涵盖 float32 输入转 float64、无输入变异、零输出分支、非法形状、非全 1 gamma 与非零 beta。

## 九、正式运行与预期产物

~~~powershell
python -m pytest -q -m exercise tests/test_week18_exercises.py
python -m pytest -q -m exercise
python -m scripts.day18_encoder_block_backward_demo
~~~

运行成功后，在本地产生：

~~~text
results/day18/
├── metrics.json
├── gradient_error_comparison.png
├── pre_post_gradient_norms.png
└── input_gradient_pre_post.png
~~~

这些图是随机参数下的梯度数值校验，不证明 Pre-LN 或 Post-LN 谁更适合训练或任务。实际验收数据见第十一节。

## 十、思考题

1. Pre-LN 为什么需要在 H 与 X 两个位置分别相加梯度？
2. 为什么 Pre-LN 中 MHA 收到的上游梯度不是原始 G_Y？
3. 为什么 Post-LN 反向第一步必须是 LN2？
4. LN Backward 为什么必须使用它真正的前向输入？
5. 为什么 Wq 的梯度要对 Batch 与 Token 贡献求和，而 X 的梯度不能？
6. 为什么 Pre-LN 零分支时 G_X=G_Y，而 Post-LN 不一定？
7. 完成 Block Backward 是否就意味着多层 Encoder Stack Backward 已完成？为什么？

---

**Day 18 目标：独立实现 Pre-LN/Post-LN Encoder Block 的完整反向传播，并通过 PyTorch Autograd、有限差分和结构不变量验证全部 13 组梯度。**


## 十一、Day 18 最终验收与成果归档（2026-11-24）

### 11.1 实际运行结果

本次先完成完整实现，再在本地 `tfz` 环境下执行：

~~~powershell
python -m pytest -q -m exercise tests/test_week18_exercises.py
python -m pytest -q -m exercise
python -m scripts.day18_encoder_block_backward_demo
~~~

| 验收项目 | 实测结果 |
|---|---|
| Day 18 专项测试 | **51 passed in 17.13s** |
| 全项目 Exercise 回归 | **264 passed, 4 deselected in 3.92s** |
| Demo | **SUCCESS: Day 18 Encoder Block Backward checks passed** |
| Pre-LN / Post-LN 梯度数量 | 各 13 组 |
| 输入形状 | (2,5,8) |
| Head 数 | 2 |
| 所有梯度有限 | 两种架构均 true |
| 输入数组未被修改 | true |
| Pre-LN 零分支恒等输入梯度误差 | **0.0** |

`4 deselected` 表示这次运行的 `-m exercise` 标记筛选未执行的测试，不是错误。以上结果来自 2026-11-24 提供的本地终端运行日志；GitHub 归档并不等同于远端 CI 独立重跑。

### 11.2 PyTorch Autograd 对照

| 梯度 | Pre-LN 最大绝对误差 | Post-LN 最大绝对误差 |
|---|---:|---:|
| X | 1.3322676295501878e-15 | 2.220446049250313e-15 |
| Wq | 3.552713678800501e-15 | 1.7763568394002505e-15 |
| Wk | 1.942890293094024e-15 | 1.7763568394002505e-15 |
| Wv | 3.9968028886505635e-15 | 3.219646771412954e-15 |
| Wo | 2.6645352591003757e-15 | 2.220446049250313e-15 |
| gamma_attn | 4.440892098500626e-15 | 2.4424906541753444e-15 |
| beta_attn | 2.886579864025407e-15 | 1.5543122344752192e-15 |
| W1 | 1.1102230246251565e-15 | 1.7763568394002505e-15 |
| b1 | 6.661338147750939e-16 | 8.881784197001252e-16 |
| W2 | 1.7763568394002505e-15 | 7.771561172376096e-16 |
| b2 | 8.881784197001252e-16 | 2.220446049250313e-15 |
| gamma_ffn | 1.2628786905111156e-15 | 9.992007221626409e-16 |
| beta_ffn | 2.220446049250313e-16 | 0.0 |

两种架构的全部 13 组梯度与 Autograd 的偏差均为 float64 浮点舍入误差量级。**全局最大绝对误差：Pre-LN 4.440892098500626e-15；Post-LN 3.219646771412954e-15。**

注意 Demo 柱图在对数纵轴上把 0 误差夹到 1e-18 作为绘图下界；Post-LN 的 beta_ffn 实际误差是 0，不是 1e-18。

### 11.3 中心有限差分核查

| 模式 | 变量/坐标 | 数值梯度 | 解析梯度 | 绝对误差 |
|---|---|---:|---:|---:|
| Pre-LN | X (0,2,3) | 1.7353097492289749 | 1.7353097487137328 | 5.152420712306593e-10 |
| Pre-LN | gamma_attn (1,) | 1.5795732677403862 | 1.579573268725748 | 9.853617921606883e-10 |
| Post-LN | X (0,2,3) | 0.2661047373564429 | 0.26610473678547675 | 5.709661632380403e-10 |
| Post-LN | gamma_attn (1,) | 2.061923691121237 | 2.061923690936588 | 1.8464874074197724e-10 |

均通过 Demo 数值验收阈值。

### 11.4 Pre-LN / Post-LN 梯度范数差异

| 梯度 L2 范数 | Pre-LN | Post-LN |
|---|---:|---:|
| X | 14.272661784487477 | 12.32516104807154 |
| Wq | 16.058930885934796 | 11.103237236403857 |
| Wk | 9.72711748279988 | 7.918325549645487 |
| Wv | 16.927609641129475 | 11.30190108564354 |
| Wo | 18.025090920961443 | 11.603729547696704 |
| gamma_attn | 5.3377128939037775 | 4.577811327660807 |
| beta_attn | 10.310188428013285 | 4.456386563293685 |
| W1 | 16.673848857733777 | 16.77664842341995 |
| b1 | 7.189045292234508 | 4.678742652689804 |
| W2 | 12.987099642276497 | 12.307019584350856 |
| b2 | 10.79141214995242 | 6.102809419559137 |
| gamma_ffn | 2.7754918107402773 | 7.320416467536417 |
| beta_ffn | 4.842326531009327 | 10.79141214995242 |

本次实验中，Pre-LN 的 Attention 权重梯度范数较大，而 Post-LN 的 gamma_ffn 与 beta_ffn 梯度范数较大；这**只反映当前随机参数与上游梯度的局部数值情形**，不能直接推出泛化或训练稳定性孰优孰劣。

### 11.5 图像输出与本地保留策略

本次 Demo 生成并检查：

~~~text
results/day18/
├── metrics.json
├── gradient_error_comparison.png
├── pre_post_gradient_norms.png
└── input_gradient_pre_post.png
~~~

三张图分别展示 13 组梯度误差、两种架构梯度范数、以及输入梯度热力图。仓库当前的 `.gitignore` 规定 `results/*` 为本地实验产物，本次保留本地结果目录，使用 GitHub Markdown 归档关键数值，**不宣称结果 PNG 和 metrics.json 已经上传到 GitHub**。

### 11.6 验收结论与后续边界

当前已完成 Day 15 完整单层 Encoder Block 的 Pre-LN/Post-LN NumPy Backward，且复用了 Day 12、13、14、17 的模块求导；精确返回 X 与 12 组可学习参数的 13 个梯度。**多层 Encoder Stack 的反向传播、Mask、Dropout、训练循环仍不在 Day 18 范围内。**
