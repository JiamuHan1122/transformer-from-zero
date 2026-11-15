# Day 12 学习笔记｜Layer Normalization：稳定每个 Token 的特征尺度

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-15  
> **阶段：** Transformer 核心组件 · Day 12  
> **今日主题：** LayerNorm、Feature-wise Statistics、Affine Parameters、Manual Backpropagation  
> **状态：** 理论学习与练习脚手架已发布；实现和实验待验收  
> **核心实现：** exercises/week12.py  
> **测试：** tests/test_week12_exercises.py  
> **实验脚本：** scripts/day12_layer_norm_demo.py

---

## 一、今天为什么学习 LayerNorm？

Day 11 已经把 Token IDs、Embedding、Positional Encoding 和 Masked Attention 输入连接起来。接下来要补上 Transformer 子层中的关键组件：Layer Normalization（层归一化）。

Attention 负责在 Token 之间交换信息；LayerNorm 则负责对每个 Token 的特征表示进行归一化。它们承担不同的职责。

今天重点不是记住“均值为零、方差为一”，而是弄清楚：

- 均值和方差究竟在哪个轴上计算；
- 为什么每个 Token 独立归一化；
- gamma 和 beta 为什么仍然是可学习参数；
- 输入梯度为什么需要考虑均值和方差两条依赖路径；
- 为什么 gamma、beta 的梯度要跨所有 Token 累积。

## 二、先从一个 Token 开始

假设某个 Token 的特征向量为：

$$
x=[1,2,3,4].
$$

均值：

$$
\\mu=\\frac{1+2+3+4}{4}=2.5.
$$

总体方差：

$$
\\sigma^2
=
\\frac{(1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2}{4}
=1.25.
$$

标准化：

$$
\\widehat{x}
=
\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}.
$$

当 epsilon 很小时：

$$
\\widehat{x}
\\approx[-1.3416,-0.4472,0.4472,1.3416].
$$

这一步把该 Token 的特征重新居中，并缩放到接近单位方差的尺度。

但标准化并不是最终输出。LayerNorm 还包含可学习的仿射变换：

$$
y=\\gamma\\odot\\widehat{x}+\\beta.
$$

当 gamma 为全 1、beta 为全 0 时，输出等于标准化结果；训练过程中 gamma 和 beta 可以学习更合适的尺度与偏移。

## 三、从单个 Token 扩展到 (B,T,D)

Transformer 的序列表示：

$$
X\\in\\mathbb{R}^{B\\times T\\times D}.
$$

- B：Batch 中的序列数量；
- T：每条序列的 Token 数量；
- D：每个 Token 的特征维度。

LayerNorm 对每一个 (b,t) 单独计算：

$$
\\mu_{b,t}
=
\\frac{1}{D}\\sum_{d=1}^{D}X_{b,t,d}.
$$

$$
\\sigma^2_{b,t}
=
\\frac{1}{D}\\sum_{d=1}^{D}
(X_{b,t,d}-\\mu_{b,t})^2.
$$

标准化：

$$
\\widehat{X}_{b,t,d}
=
\\frac{X_{b,t,d}-\\mu_{b,t}}
{\\sqrt{\\sigma^2_{b,t}+\\epsilon}}.
$$

仿射变换：

$$
Y_{b,t,d}
=
\\gamma_d\\widehat{X}_{b,t,d}+\\beta_d.
$$

最终：

$$
Y\\in\\mathbb{R}^{B\\times T\\times D}.
$$

### 维度一次看清楚

| 变量 | 形状 | 说明 |
|---|---|---|
| X | (B,T,D) | 输入 |
| mean | (B,T,1) | 每个 Token 一个均值 |
| variance | (B,T,1) | 每个 Token 一个方差 |
| X_hat | (B,T,D) | 标准化结果 |
| gamma | (D,) | 每个特征一组缩放参数 |
| beta | (D,) | 每个特征一组平移参数 |
| Y | (B,T,D) | 输出 |

Forward 只在 D 维上归约，gamma 和 beta 则沿 B、T 维广播。

## 四、为什么不是跨 Batch 或 Token 计算？

LayerNorm 的统计量对每个 Token 独立计算。对于 (B,T,D)，它只沿 axis=-1 计算均值与方差。

这带来一个直接性质：

**如果只修改一个 Token 的输入特征，其他 Token 的 LayerNorm 输出不会变化。**

这与 Attention 不同。Attention 会根据 Q/K/V 在 Token 之间交换信息；LayerNorm 本身不进行跨 Token 信息交互。

与 BatchNorm 的区别：

| | LayerNorm | BatchNorm |
|---|---|---|
| 统计范围 | 单个 Token 的 D 个特征 | 通常跨 Batch 统计通道 |
| 是否依赖 Batch 其他样本 | 否 | 训练时依赖 |
| 训练/推理统计量 | 当前输入自身计算 | 通常涉及运行均值、方差 |
| Transformer 中的常见选择 | 广泛使用 | 不是原始 Transformer 的标准选择 |

## 五、epsilon 放在哪里？

标准形式：

$$
\\widehat{x}
=
\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}.
$$

epsilon 是为了避免方差极小时分母为零或过小。

注意以下表达并不等价：

$$
\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}
\\neq
\\frac{x-\\mu}{\\sqrt{\\sigma^2}+\\epsilon}.
$$

实现必须把 epsilon 加在方差内部，再开平方。

本项目使用总体方差（ddof=0），与 PyTorch LayerNorm 的方差约定一致。

## 六、Backward：三个梯度从哪里来？

令上游梯度：

$$
G=\\frac{\\partial L}{\\partial Y}.
$$

### 6.1 beta 梯度

因为：

$$
Y=\\gamma\\odot\\widehat{X}+\\beta,
$$

所以：

$$
\\boxed{
\\frac{\\partial L}{\\partial\\beta_d}
=
\\sum_{b,t}G_{b,t,d}
}
$$

### 6.2 gamma 梯度

$$
\\boxed{
\\frac{\\partial L}{\\partial\\gamma_d}
=
\\sum_{b,t}G_{b,t,d}\\widehat{X}_{b,t,d}
}
$$

gamma 和 beta 对所有 Token 共享，所以参数梯度必须累积所有 Batch 和 Token 的贡献。这与 Day 06 中共享线性层的权重梯度累积是相似的思想。

### 6.3 输入梯度

先定义：

$$
Q=G\\odot\\gamma,
\\qquad
r=\\frac{1}{\\sqrt{\\sigma^2+\\epsilon}}.
$$

输入梯度的简洁形式：

$$
\\boxed{
\\frac{\\partial L}{\\partial X}
=
r\\odot
\\left[
Q
-\\operatorname{Mean}_{D}(Q)
-\\widehat{X}\\odot
\\operatorname{Mean}_{D}(Q\\odot\\widehat{X})
\\right]
}
$$

两个 Mean 都只沿最后一个特征维度计算，并保留维度用于广播。

这条表达已经合并了均值和方差对输入的影响，不需要构造 D×D 的 Jacobian。

## 七、LayerNorm 在 Transformer Block 中的位置

LayerNorm 本身不等于 Residual Connection。两种常见结构：

Post-LN：

$$
Y=\\operatorname{LayerNorm}(X+\\operatorname{Sublayer}(X)).
$$

Pre-LN：

$$
Y=X+\\operatorname{Sublayer}(\\operatorname{LayerNorm}(X)).
$$

原始 Transformer 使用 Post-LN；许多现代深层 Transformer 使用 Pre-LN 或其变体。今天先把 LayerNorm 自身的数学与梯度掌握好，后续再把它与 Residual、Attention、FFN 组合成完整 Block。

## 八、今天的实现任务

实现文件：exercises/week12.py

- Task A：layer_norm_forward_numpy
- Task B：layer_norm_backward_numpy

测试文件：tests/test_week12_exercises.py

~~~powershell
python -m pytest -q -m exercise tests/test_week12_exercises.py
~~~

验收重点：

1. 与 PyTorch LayerNorm 前向结果一致；
2. 每个 Token 的均值接近 0、方差接近 1；
3. gamma/beta 正确沿特征维广播；
4. 修改一个 Token 不影响其他 Token；
5. 手写的 grad_x、grad_gamma、grad_beta 与 PyTorch Autograd 一致；
6. 三组梯度再由 Day 03 有限差分独立检查；
7. 输入数组不被修改。

完成后运行：

~~~powershell
python -m scripts.day12_layer_norm_demo
~~~

本地实验产物：

~~~text
results/day12/
├── metrics.json
├── layer_norm_heatmap.png
└── layer_norm_statistics.png
~~~

这些结果不默认提交 GitHub。

## 九、Day 12 核心结论

1. LayerNorm 对每个 Token 独立地沿特征维度 D 计算均值和方差。
2. 使用总体方差，epsilon 放在平方根内部。
3. gamma 和 beta 是可学习的逐特征仿射参数。
4. LayerNorm 不依赖 Batch 中其他样本，也不混合不同 Token 的统计量。
5. 输入梯度同时考虑均值与方差对输入的依赖。
6. gamma、beta 梯度需要累积所有 Batch 和 Token 的贡献。
7. 输出形状保持 (B,T,D)。
8. LayerNorm 是 Transformer 子层的组件，不是完整 Transformer Block。

**Day 12 理论学习目标：从归一化公式一直理解到三个手写梯度。**
