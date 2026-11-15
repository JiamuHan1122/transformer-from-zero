# Day 12｜Layer Normalization：稳定每个 Token 的特征尺度

> **项目：** Transformer From Zero · 365 Days  
> **日期：** 2026-11-15  
> **阶段：** Transformer 核心组件 · Day 12  
> **今日主题：** LayerNorm、Feature-wise Statistics、Affine Parameters、Manual Backpropagation  
> **状态：** 理论材料与练习脚手架已发布；实现待完成  
> **核心实现：** exercises/week12.py  
> **测试：** tests/test_week12_exercises.py  
> **实验脚本：** scripts/day12_layer_norm_demo.py

---

## 一、今天要解决的问题

Day 11 已经把 Token IDs、Embedding、Positional Encoding 和 Masked Attention 输入连接起来。现在需要补上 Transformer 子层中的另一个关键组件：**Layer Normalization（层归一化）**。

今天要回答：

1. 对于一个形状为 (B,T,D) 的序列张量，LayerNorm 究竟沿哪个维度计算均值和方差？
2. 为什么 LayerNorm 是逐 Token 归一化，而不是跨 Batch 归一化？
3. 可学习参数 gamma 和 beta 分别做什么？
4. LayerNorm 的反向传播如何得到输入梯度，以及 gamma、beta 的梯度？
5. 为什么 gamma、beta 的梯度需要累积所有 Batch 和 Token 的贡献？

## 二、LayerNorm 的数学定义

输入：

$$
X\\in\\mathbb{R}^{B\\times T\\times D}.
$$

LayerNorm 对每个 Batch 中的每一个 Token，独立地沿最后一个特征维度 D 计算统计量。

均值：

$$
\\mu_{b,t}
=
\\frac{1}{D}\\sum_{d=1}^{D}X_{b,t,d}.
$$

方差使用总体方差（population variance，ddof=0）：

$$
\\sigma^2_{b,t}
=
\\frac{1}{D}
\\sum_{d=1}^{D}
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

其中：

- D：每个 Token 的特征维度；
- epsilon：数值稳定项，放在方差内部、平方根之前；
- gamma：可学习缩放参数；
- beta：可学习平移参数。

gamma 和 beta 都是 (D,) 向量，在 Batch 和 Token 两个轴上广播。

## 三、为什么只沿最后一个维度归一化？

输入形状：

$$
X:(B,T,D).
$$

对每个 (b,t) 单独计算：

$$
\\mu,\\sigma^2:(B,T,1).
$$

因此统计量的形状变化为：

$$
(B,T,D)\\xrightarrow{\\mathrm{mean/var\\ over\\ }D}(B,T,1).
$$

归一化后：

$$
\\widehat{X}:(B,T,D).
$$

再与 gamma、beta 广播：

$$
(B,T,D)\\odot(D,)+(D,)
\\rightarrow(B,T,D).
$$

输出形状不变：

$$
Y\\in\\mathbb{R}^{B\\times T\\times D}.
$$

**关键：** LayerNorm 不会把不同 Token 的特征混在一起计算统计量，也不依赖同一个 Batch 中还有哪些样本。

## 四、一个具体数值例子

假设一个 Token 的特征向量为：

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

标准差：

$$
\\sqrt{1.25+\\epsilon}.
$$

当 epsilon 很小时，标准化结果近似：

$$
\\widehat{x}
\\approx[-1.3416,-0.4472,0.4472,1.3416].
$$

如果：

$$
\\gamma=[1,1,1,1],\\qquad
\\beta=[0,0,0,0],
$$

输出就是标准化结果本身。

如果 gamma 和 beta 不是单位缩放和零平移，输出就不再严格具有零均值、单位方差。这正是仿射参数的作用：模型可以在标准化之后学习合适的特征尺度与偏移。

## 五、LayerNorm 与 BatchNorm 的区别

| 对比项 | LayerNorm | BatchNorm |
|---|---|---|
| 统计范围 | 单个样本/Token 的特征维度 | 通常跨 Batch 统计每个通道 |
| 对 (B,T,D) 的典型处理 | 每个 (b,t) 沿 D 归一化 | 具体轴取决于数据布局 |
| 是否依赖 Batch 中其他样本 | 不依赖 | 训练时依赖 |
| 小 Batch 行为 | 统计定义不受 Batch 大小影响 | 统计量可能不稳定 |
| Transformer 中的常见使用 | 广泛使用 | 不是原始 Transformer 的标准选择 |

LayerNorm 的重要性质是：对同一个 Token 的归一化结果，不会因为 Batch 中其他样本或序列中其他 Token 改变而改变。

## 六、Task A：手写 LayerNorm Forward

函数：

**layer_norm_forward_numpy(x, gamma, beta, eps=1e-5)**

输入：

| 变量 | 形状 | 说明 |
|---|---|---|
| x | (B,T,D) | 输入序列 |
| gamma | (D,) | 可学习缩放 |
| beta | (D,) | 可学习平移 |
| eps | 标量 | 正数，默认 1e-5 |

推荐实现顺序：

1. 沿 axis=-1 计算均值，保留维度；
2. 计算总体方差，使用 ddof=0；
3. 减均值并除以 sqrt(var + eps)；
4. 使用 gamma 和 beta 做逐特征仿射变换；
5. 返回与 x 同形状的 float64 数组。

核心公式：

$$
\\boxed{
Y=
\\gamma\\odot
\\frac{X-\\mu}
{\\sqrt{\\sigma^2+\\epsilon}}
+\\beta
}
$$

注意：不要把 epsilon 放到平方根外面；不要沿 Batch 轴或 Token 轴计算均值和方差；不要修改输入数组。

## 七、Task B：手写 LayerNorm Backward

这是今天更重要的部分。令上游梯度为：

$$
G=\\frac{\\partial L}{\\partial Y}.
$$

### 7.1 gamma 与 beta 的梯度

由于：

$$
Y=\\gamma\\odot\\widehat{X}+\\beta,
$$

所以：

$$
\\boxed{
\\frac{\\partial L}{\\partial\\beta_d}
=
\\sum_{b=1}^{B}\\sum_{t=1}^{T}G_{b,t,d}
}
$$

$$
\\boxed{
\\frac{\\partial L}{\\partial\\gamma_d}
=
\\sum_{b=1}^{B}\\sum_{t=1}^{T}
G_{b,t,d}\\widehat{X}_{b,t,d}
}
$$

为什么要对 B、T 求和？因为同一组 gamma、beta 被所有 Token 共享，每个 Token 对参数的梯度都要累积起来。

### 7.2 输入梯度

先定义：

$$
Q=G\\odot\\gamma,
\\qquad
r=\\frac{1}{\\sqrt{\\sigma^2+\\epsilon}}.
$$

则 LayerNorm 对输入的梯度可以整理成：

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

这里两个 Mean 都只沿最后一个特征维度计算，并且保留维度以便广播。

这条公式已经把均值和方差两条依赖路径合并在一起，不需要在代码中手工构造一个 D×D Jacobian。

### 7.3 返回形状

$$
\\frac{\\partial L}{\\partial X}:(B,T,D)
$$

$$
\\frac{\\partial L}{\\partial\\gamma}:(D,),
\\qquad
\\frac{\\partial L}{\\partial\\beta}:(D,).
$$

函数：

**layer_norm_backward_numpy(grad_output, x, gamma, eps=1e-5)**

返回：

**grad_x, grad_gamma, grad_beta**

为了让本次练习接口直观，Backward 会从 x 重新计算均值、方差和标准化结果；实际训练框架通常会缓存部分前向中间量，以减少重复计算。

## 八、完整维度例子

设置：

$$
B=2,\\quad T=3,\\quad D=4.
$$

| 中间变量 | 形状 |
|---|---|
| x | (2,3,4) |
| mean | (2,3,1) |
| variance | (2,3,1) |
| x_hat | (2,3,4) |
| gamma | (4,) |
| beta | (4,) |
| output | (2,3,4) |
| grad_output | (2,3,4) |
| grad_x | (2,3,4) |
| grad_gamma | (4,) |
| grad_beta | (4,) |

注意两个不同的归约方向：

- Forward 的 mean/variance：只沿 D 计算；
- Backward 的 gamma/beta 梯度：沿 B 和 T 累积。

这是 LayerNorm 最容易弄混的维度问题。

## 九、LayerNorm 在 Transformer Block 中的位置

LayerNorm 是子层周围的重要组件，但不能把 LayerNorm 本身与 Residual Connection 混为一谈。

两种常见结构：

**Post-LN：**

$$
Y=\\operatorname{LayerNorm}(X+\\operatorname{Sublayer}(X)).
$$

**Pre-LN：**

$$
Y=X+\\operatorname{Sublayer}(\\operatorname{LayerNorm}(X)).
$$

原始 Transformer 采用的是 Post-LN 形式；许多现代深层 Transformer 使用 Pre-LN 或其变体。

今天只实现 LayerNorm 本身，不把 Residual、Attention、FFN 混入同一个函数。后续再组合完整 Block。

## 十、测试与验收

测试文件：tests/test_week12_exercises.py

专项测试：

~~~powershell
python -m pytest -q -m exercise tests/test_week12_exercises.py
~~~

测试覆盖：

1. Forward 与 PyTorch LayerNorm 数值一致；
2. 每个 Token 的特征均值和方差正确；
3. gamma、beta 沿特征维广播；
4. 修改一个 Token 不影响其他 Token；
5. 不修改调用者输入；
6. 非法形状、dtype、epsilon 被拒绝；
7. Backward 的输入、gamma、beta 梯度与 PyTorch Autograd 一致；
8. 三组梯度再由 Day 03 的有限差分独立核对。

全部测试通过后运行：

~~~powershell
python -m scripts.day12_layer_norm_demo
~~~

Demo 将保存：

~~~text
results/day12/
├── metrics.json
├── layer_norm_heatmap.png
└── layer_norm_statistics.png
~~~

实验结果只保存在本地，不默认提交 results 目录。

## 十一、今天需要真正掌握的结论

1. LayerNorm 对每个 Token 独立地沿特征维度 D 计算均值和方差。
2. 使用总体方差，且 epsilon 放在 sqrt 内部。
3. gamma 和 beta 是可学习的逐特征仿射参数。
4. LayerNorm 不依赖 Batch 中其他样本，也不混合不同 Token 的统计量。
5. 输入梯度需要同时考虑均值与方差对输入的依赖。
6. gamma、beta 的梯度要累积所有 Batch 和 Token 的贡献。
7. LayerNorm 输出形状仍为 (B,T,D)。
8. LayerNorm 本身不等于完整 Transformer Block；Residual 与 FFN 是独立组件。

**Day 12 的目标：不仅能写出 LayerNorm 前向，还要能独立推导并实现三个梯度。**
