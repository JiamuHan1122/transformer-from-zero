# Day 14｜Position-wise Feed-Forward Network：从 Token 特征到手写反向传播

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-17  
> **阶段：** Transformer Block 核心结构 · Day 14  
> **今日主题：** FFN、两次线性变换、GELU、特征升维、共享参数、手动反向传播  
> **状态：** 本地 15 项专项测试、137 项回归测试及 FFN Demo 已通过；最终代码待推送 GitHub  
> **核心实现：** exercises/week14.py  
> **测试：** tests/test_week14_exercises.py  
> **实验脚本：** scripts/day14_ffn_demo.py

---

## 一、今天为什么学习 FFN？

我们已经分别实现：

- Day 08：Multi-Head Self-Attention，允许 Token 之间读取和聚合信息；
- Day 12：LayerNorm，对每个 Token 的特征进行归一化；
- Day 13：Residual Connection，让子层在输入基础上学习变化。

一个常见的 Transformer Block 还包含 **Position-wise Feed-Forward Network（FFN）**。它负责处理每个 Token 内部的特征，而不是直接在不同 Token 之间交换信息。

今天的科学问题很具体：

1. 为什么 Attention 后面还要有 FFN？
2. 为什么 FFN 常常先将特征维度升高，再投影回原维度？
3. 为什么它不会直接将 Token 0 的信息混合到 Token 1？
4. GELU 在 FFN 中为什么不可省略？
5. 如何不用 Autograd，手算两层线性变换与 GELU 的五组梯度？

## 二、FFN 的数学定义

设输入序列张量为：

$$
X\in\mathbb{R}^{B\times T\times D}.
$$

今天采用两层线性变换，中间使用 **Day 05 实现过的 GELU tanh 近似**：

$$
Z=XW_1+b_1,
$$

$$
A=\operatorname{GELU}_{\mathrm{tanh}}(Z),
$$

$$
\boxed{Y=AW_2+b_2}.
$$

合并书写为：

$$
\boxed{
\operatorname{FFN}(X)
=
\operatorname{GELU}_{\mathrm{tanh}}(XW_1+b_1)W_2+b_2
}.
$$

注意：经典的原版 Transformer FFN 使用 ReLU。本课程选择 GELU tanh 近似，是为了连接 Day 05 的激活函数与手写梯度训练；两者都是合法的 FFN 激活选择，但不能混称为同一个公式。

## 三、维度：升维并不增加 Token 个数

记 \(H=D_{\mathrm{ff}}\)，通常选 \(H>D\)。今天输入与输出的最后一维同为 \(D\)，以方便后续加入残差连接。

| 变量 | 形状 | 意义 |
|---|---|---|
| X | (B,T,D) | 输入序列 |
| W1 | (D,H) | 第一层权重 |
| b1 | (H,) | 第一层偏置 |
| Z | (B,T,H) | 升维后的预激活 |
| A | (B,T,H) | GELU 激活 |
| W2 | (H,D) | 第二层权重 |
| b2 | (D,) | 第二层偏置 |
| Y | (B,T,D) | 输出序列 |

### 具体例子

假设：

$$
B=2,\qquad T=5,\qquad D=8,\qquad H=32.
$$

逐步对应：

1. 输入 X：(2,5,8)。
2. 第一层矩阵乘法 (2,5,8) @ (8,32)，再加 b1 (32,)，得到 Z：(2,5,32)。
3. GELU 逐元素激活，得到 A：(2,5,32)，**不改变形状**。
4. 第二层矩阵乘法 (2,5,32) @ (32,8)，再加 b2 (8,)，得到 Y：(2,5,8)。

整个过程只有每个 Token 的通道宽度先变大再变小，Batch 和 Token 轴从头到尾没有改变。

### 参数量

两层 FFN 的参数总数为：

$$
N_{\mathrm{params}}
=
DH+H+HD+D
=
2DH+H+D.
$$

对 \(D=8,H=32\)：

$$
N_{\mathrm{params}}=2\times8\times32+32+8=552.
$$

## 四、为什么要加入 GELU？

如果去掉激活函数：

$$
Y=(XW_1+b_1)W_2+b_2,
$$

则可以改写为：

$$
Y=X(W_1W_2)+(b_1W_2+b_2).
$$

也就是说：两次纯线性变换可以合并为一次线性变换。中间的非线性 GELU 使 FFN 能学习更加复杂的特征变换。

我们复用 Day 05 的近似 GELU：

$$
\operatorname{GELU}_{\mathrm{tanh}}(z)
=
\frac{z}{2}
\left[
1+\tanh\left(
\sqrt{\frac{2}{\pi}}
(z+0.044715z^3)
\right)
\right].
$$

本课程不在 Day 14 重新实现 GELU，而是从 \`exercises.week05\` 导入：

- \`gelu_tanh_numpy\`
- \`gelu_tanh_backward_numpy\`

## 五、为什么叫 Position-wise？

这里的 Position 指序列中的 Token 位置，不是模型的特征通道。

对于每个位置 \(t\)，FFN 都使用相同的参数：

$$
Y_{b,t,:}
=
\operatorname{GELU}_{\mathrm{tanh}}
(X_{b,t,:}W_1+b_1)W_2+b_2.
$$

因此：

- 变换会混合**同一个 Token 的不同特征通道**；
- 变换不直接读取**其他 Token 的表示**；
- 每个 Token 的 FFN 权重完全共享，而不是每个位置单独学习一套。

如果只修改某个输入 Token，标准 FFN 只有该位置的输出会变化。

这与 Day 08 的 Attention 形成对照：Attention 通过注意力矩阵跨 Token 聚合 Value；FFN 则独立地变换各个 Token 的通道表示。

## 六、Task A：FFN Forward

函数：

\`ffn_forward_numpy(x, w1, b1, w2, b2)\`

实现流程：

1. 校验输入类型、维度和相容形状（辅助函数已提供）。
2. 将计算输入转为 float64。
3. 用 \`@\` 执行第一层矩阵乘法，再添加 b1。
4. **复用 Day 05** 的 \`gelu_tanh_numpy\`。
5. 用 \`@\` 执行第二层矩阵乘法，再添加 b2。
6. 返回 (B,T,D)，不得修改原始数组。

今天的 FFN 函数不包含 Residual Addition 或 LayerNorm；这些是不同模块。后续组合 Transformer Block 时再连接。

## 七、Task B：FFN Backward

这是今天最重要的部分。先定义：

$$
G=\frac{\partial L}{\partial Y},
\qquad
G\in\mathbb{R}^{B\times T\times D}.
$$

从最后一层倒着推：

### 7.1 第二层权重与偏置

$$
\boxed{
\frac{\partial L}{\partial W_2}
=
\sum_{b,t} A_{b,t,:}^{\top}G_{b,t,:}
}
$$

$$
\boxed{
\frac{\partial L}{\partial b_2}
=
\sum_{b,t} G_{b,t,:}
}
$$

形状分别为 (H,D) 与 (D,)。

### 7.2 传回 GELU

$$
G_A=GW_2^{\top},
\qquad
G_A:(B,T,H).
$$

再使用 Day 05 的反向函数：

$$
G_Z=G_A\odot\operatorname{GELU}_{\mathrm{tanh}}'(Z).
$$

\`gelu_tanh_backward_numpy(Z, G_A)\` 已经实现了这条链式法则。

### 7.3 第一层权重与偏置

$$
\boxed{
\frac{\partial L}{\partial W_1}
=
\sum_{b,t} X_{b,t,:}^{\top}G_{Z,b,t,:}
}
$$

$$
\boxed{
\frac{\partial L}{\partial b_1}
=
\sum_{b,t}G_{Z,b,t,:}
}
$$

形状分别为 (D,H) 与 (H,)。

### 7.4 最后传回输入

$$
\boxed{
\frac{\partial L}{\partial X}
=
G_ZW_1^{\top}
}.
$$

形状重新回到 (B,T,D)。

Backward 函数需要按固定顺序返回：

\`grad_x, grad_w1, grad_b1, grad_w2, grad_b2\`

**注意：** 反向函数没有传入 b2 的数值，因为 \(\partial Y/\partial b_2=I\)，且它不会影响前面各层的反向计算。虽然没有传入 b2，仍然需要返回 \`grad_b2\`。

## 八、五组梯度的维度表

| 梯度 | 输出形状 | 应沿哪些维度求和 |
|---|---|---|
| grad_x | (B,T,D) | 不进行 Batch/Token 累积 |
| grad_w1 | (D,H) | B、T |
| grad_b1 | (H,) | B、T |
| grad_w2 | (H,D) | B、T |
| grad_b2 | (D,) | B、T |

和 Day 06、Day 12、Day 13 一样：因为参数对所有 Token 共享，所以权重梯度与偏置梯度需要累积每个 Token 的贡献。

## 九、验收与可视化

专项测试：

~~~powershell
python -m pytest -q -m exercise tests/test_week14_exercises.py
~~~

完成后进行全项目回归测试：

~~~powershell
python -m pytest -q -m exercise
~~~

测试覆盖：与 PyTorch GELU-tanh 前向对照、Day 05 函数复用、Token 独立性、float64 与输入不变性、非法形状、PyTorch Autograd 梯度对照、Day 03 有限差分梯度检查、零输出权重下的梯度与上游梯度合法性。

最后运行 Demo：

~~~powershell
python -m scripts.day14_ffn_demo
~~~

本地生成：

~~~text
results/day14/
├── metrics.json
├── ffn_feature_heatmaps.png
└── token_perturbation_impact.png
~~~

这些实验产物默认不提交到 GitHub；待实际执行后再填写验收数值。

## 十、今天与接下来的 Transformer Block

我们已经掌握：

1. Attention：跨 Token 信息聚合；
2. LayerNorm：逐 Token 的特征归一化；
3. Residual：保留恒等映射与直接梯度路径；
4. FFN：逐 Token 的非线性特征变换。

今天专注 FFN 自身。下一步才将它们组合成完整的 Encoder 或 Decoder Block。

## 十一、思考题

1. 为什么不加 GELU 的两层 FFN 可以合并成一层？
2. FFN 明明处理全部 Token，为什么却不发生跨 Token 信息交换？
3. \(H>D\) 为什么通常叫“特征升维”，而不是“序列变长”？
4. 为什么 \`grad_w1\` 与 \`grad_w2\` 需要沿 Batch 和 Token 两个轴求和？
5. 在当前 FFN 中，将一个位置的输入修改后，其他位置的输出是否应该改变？

**Day 14 目标：独立实现 FFN 的前向与五组梯度，验证它如何进行逐 Token 的非线性特征变换。**

---

## 十二、Day 14 本地验收记录（2026-11-17）

- 专项测试：`15 passed in 13.42s`。
- 全项目回归：`137 passed, 4 deselected in 2.46s`。
- Demo：`SUCCESS: Day 14 FFN checks passed.`。
- 关键现象：Token 2 的输出 L2 变化约为 `1.3879174060563375`，其他 Token 的最大变化为 `0.0`，证明当前 Position-wise FFN 不直接混合 Token。
- 输入 `(2,5,8)` → 隐藏层 `(2,5,32)` → 输出 `(2,5,8)`，参数总数 `552`，各输出和梯度均有限。
- 这些结果来自用户本地实际运行；`exercises/week14.py` 的最终代码尚需推送 GitHub。
