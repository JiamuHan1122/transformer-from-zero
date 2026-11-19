# Day 16｜Deep Transformer Encoder：多层堆叠、参数独立与逐层传播

> **项目：** [Transformer From Zero · 365 Days](https://github.com/JiamuHan1122/transformer-from-zero)  
> **日期：** 2026-11-19  
> **阶段：** Transformer Encoder · 多层组合与表示诊断  
> **今日主题：** Encoder Stack、Depth、Weight Tying、Hidden-State Trajectory、Cross-Token Influence、Pre-LN / Post-LN  
> **状态：** 理论材料、练习脚手架、测试和 Demo 已准备；待本地完成与验收  
> **核心实现：** exercises/week16.py  
> **测试：** tests/test_week16_exercises.py  
> **实验脚本：** scripts/day16_encoder_stack_demo.py

---

## 一、为什么今天不只是增加一个循环？

Day 15 已实现一个 Encoder Block：

$$
X^{(1)}=F_1(X^{(0)};\theta_1).
$$

今天需要 L 个 Encoder Block：

$$
\boxed{
X^{(\ell)}=F_\ell(X^{(\ell-1)};\theta_\ell),
\qquad \ell=1,\ldots,L.
}
$$

最终为：

$$
X^{(L)}=F_L\circ F_{L-1}\circ\cdots\circ F_1(X^{(0)}).
$$

每层有自己的 Attention、FFN 和两组 LayerNorm 参数；每层接收的隐藏状态也不同。

今天的五个重要问题：

1. 为什么模型深度可能带来更复杂的表示组合，却不保证准确率必然提高？
2. 相同模块结构是否意味着层与层之间共享权重？
3. 为什么形状始终为 (B,T,D)，但 Token 表示仍能改变？
4. 一个 Token 的扰动如何随着网络深度扩散到其他位置？
5. Pre-LN / Post-LN 的不同归一化位置如何改变梯度路径？

## 二、输入、层间状态和结构

设：

$$
X^{(0)}\in\mathbb{R}^{B\times T\times D}.
$$

每层满足：

$$
F_\ell:\mathbb{R}^{B\times T\times D}
\rightarrow\mathbb{R}^{B\times T\times D}.
$$

所以各层可以串联：

$$
X^{(0)}\rightarrow F_1\rightarrow X^{(1)}
\rightarrow F_2\rightarrow X^{(2)}
\rightarrow\cdots\rightarrow F_L\rightarrow X^{(L)}.
$$

不变的是张量形状，不是特征数值。定义第 \(\ell\) 层的状态更新：

$$
\Delta^{(\ell)}=X^{(\ell)}-X^{(\ell-1)}.
$$

今天需要完整记录所有状态和每层的 Attention，而不仅保存最后输出。

## 三、独立参数和参数绑定有什么区别？

普通独立参数 Encoder 使用：

$$
\theta_1,\theta_2,\ldots,\theta_L,
$$

其中不同层即使具有相同形状，权重数值和内存对象也可以完全不同。

如果重复使用同一个参数字典，则是显式绑定：

$$
X^{(2)}=F(F(X^{(0)};\theta);\theta).
$$

这不属于编程错误，但它改变了参数化方式。为了让每层有独立的可学习变换能力，今天 Demo 默认创建 L 组不同的参数。练习测试也将检查明确绑定参数的情况。

### 3.1 模型参数量

沿用 Day 15 的具体实现：Q/K/V/O 无 Bias，FFN 有两组 Bias，两组 LayerNorm 各有 gamma 和 beta。

$$
N_{\mathrm{block}}
=
4D^2+2DD_{\mathrm{ff}}+D_{\mathrm{ff}}+5D.
$$

对于 \(D=8,D_{\mathrm{ff}}=32\)：

$$
N_{\mathrm{block}}=840.
$$

相同宽度、独立参数的 L 层 Stack：

$$
\boxed{N_{\mathrm{stack}}=L\,N_{\mathrm{block}}}.
$$

今天 \(L=3\) 时：

$$
N_{\mathrm{stack}}=3\times840=2520.
$$

显式绑定参数时，独立参数数量不应重复计数，即使每层仍然执行一次计算。

## 四、Pre-LN 与 Post-LN 的层间递推

### 4.1 Pre-LN

$$
H^{(\ell)}
=
X^{(\ell-1)}
+
\operatorname{MHA}_{\ell}
\big(
\operatorname{LN}_{\ell,a}(X^{(\ell-1)})
\big),
$$

$$
X^{(\ell)}
=
H^{(\ell)}
+
\operatorname{FFN}_{\ell}
\big(
\operatorname{LN}_{\ell,f}(H^{(\ell)})
\big).
$$

每个子层的残差路径可以绕开当前子层和前置 LayerNorm。

### 4.2 Post-LN

$$
H^{(\ell)}
=
\operatorname{LN}_{\ell,a}
\big(
X^{(\ell-1)}
+
\operatorname{MHA}_{\ell}(X^{(\ell-1)})
\big),
$$

$$
X^{(\ell)}
=
\operatorname{LN}_{\ell,f}
\big(
H^{(\ell)}
+
\operatorname{FFN}_{\ell}(H^{(\ell)})
\big).
$$

两者具有相同输出形状，但计算过程、表示尺度及梯度路径可能不同。不能仅凭未训练随机网络的数值大小判断哪一种架构更准确。

## 五、从双 Token 玩具例子理解跨层信息传播

注意下面是一个不含 Q/K、FFN、Residual 或 LayerNorm 的简化固定权重线性模型，用于直观理解多步聚合；**它不是完整 Transformer 的等价公式**。

定义两层 Token 聚合权重：

$$
A^{(1)}
=
\begin{bmatrix}0.8&0.2\\0.3&0.7\end{bmatrix},
\qquad
A^{(2)}
=
\begin{bmatrix}0.6&0.4\\0.1&0.9\end{bmatrix}.
$$

如果：

$$
X^{(1)}=A^{(1)}X^{(0)},\quad
X^{(2)}=A^{(2)}X^{(1)},
$$

那么：

$$
A^{(2)}A^{(1)}
=
\begin{bmatrix}
0.60&0.40\\
0.35&0.65
\end{bmatrix}.
$$

第一层 Token 0 对 Token 1 的权重为 0.2；在这个固定权重简化例子中，经过第二层后有效比例变为 0.4。

真实 Transformer 的 Q/K 权重随隐藏状态变化，还存在 V 投影、残差和非线性变换。因此不能简单将每层 Attention 矩阵相乘并称为真实模型的 Token 因果贡献。

## 六、逐层观测的四类指标

### 6.1 每层平均 Token 范数

$$
r_\ell=\frac{1}{BT}\sum_{b,t}
\|X^{(\ell)}_{b,t,:}\|_2.
$$

观察特征尺度随深度的变化。范数大不代表表示质量高。

### 6.2 每层平均更新幅度

$$
m_\ell=\frac{1}{BT}\sum_{b,t}
\|X^{(\ell)}_{b,t,:}-X^{(\ell-1)}_{b,t,:}\|_2.
$$

判断各层对当前输入做了多大数值更新。更新量小不能直接判断该层冗余。

### 6.3 Attention Entropy

$$
\mathcal H_{\ell,h,i}
=
-\sum_{j=1}^{T}
A_{\ell,h,i,j}\log A_{\ell,h,i,j}.
$$

没有 Mask 时：

$$
0\le\mathcal H_{\ell,h,i}\le\ln T.
$$

熵越高表示权重越分散，但它不是语义正确性或实际任务价值的直接度量。

### 6.4 单个 Token 扰动响应

仅修改输入位置 k：

$$
\widetilde X^{(0)}_{:,k,:}
=
X^{(0)}_{:,k,:}+\delta.
$$

在第 0 个 Batch 中定义：

$$
R_{\ell,i}
=
\left\|
\widetilde X^{(\ell)}_{0,i,:}
-
X^{(\ell)}_{0,i,:}
\right\|_2.
$$

在层 0，除位置 k 外其余位置应严格为 0；在后续包含 Self-Attention 的层中，其他位置可能出现非零影响。影响不一定随深度单调增强，也不一定有益于任务。

## 七、深度模型的反向传播预备知识

今天只实现完整 Stack 的 NumPy Forward。但梯度的数学结构应提前理解。

$$
\frac{\partial\mathcal L}{\partial X^{(0)}}
=
J_{F_1}^{\top}J_{F_2}^{\top}\cdots J_{F_L}^{\top}
\frac{\partial\mathcal L}{\partial X^{(L)}}.
$$

对残差映射：

$$
F_\ell(X)=X+G_\ell(X),
\qquad
J_{F_\ell}=I+J_{G_\ell}.
$$

恒等项为梯度提供直接通道，但不保证深层训练中绝不会发生梯度消失或爆炸。Pre-LN 和 Post-LN 因 LayerNorm 位置不同，其 Jacobian 结构也会不同。

Day 08 的 Multi-Head Attention 目前没有完整手写 Backward，因此我们不能把现有 Encoder Stack 称为已经实现了完整 NumPy 反向训练。

## 八、Day 16 代码任务

核心函数：

encoder_stack_forward_numpy(x, layers, num_heads, norm_first=True, eps=1e-5)

其中 layers 为非空列表或元组，每层参数字典包含以下名称：

w_q、w_k、w_v、w_o、gamma_attn、beta_attn、w1、b1、w2、b2、gamma_ffn、beta_ffn。

每一层使用自己的字典，调用 Day 15 的 transformer_encoder_block_forward_numpy。

必须返回三个值：

1. final_output，形状 (B,T,D)；
2. all_attentions，形状 (L,B,num_heads,T,T)；
3. all_hidden_states，形状 (L+1,B,T,D)，第 0 项为初始输入。

### 具体维度

$$
B=2,\quad T=5,\quad D=8,\quad h=2,\quad
L=3,\quad D_{\mathrm{ff}}=32.
$$

| 对象 | 形状 |
|---|---|
| 输入 X | (2,5,8) |
| 单层输出 | (2,5,8) |
| 全部 Hidden States | (4,2,5,8) |
| 单层 Attention | (2,2,5,5) |
| 全部 Attention 历史 | (3,2,2,5,5) |
| 最终输出 | (2,5,8) |

隐藏状态历史是 L+1，因为需要保留输入层 \(X^{(0)}\)。

### 实现约束

- 不复制 Day 15 Block 的内部代码；
- 每次循环传入上一层输出，而不是反复使用原始 X；
- 不把不同层的 Attention 取平均；
- 输入及参数均不可原地修改；
- 运算及返回使用 float64；
- 支持 Pre-LN 与 Post-LN；
- 不添加 Mask、Dropout、额外 Final LN 或分类头。

## 九、测试为什么必须包含 PyTorch 参考？

即使输出形状正确，以下错误也可能没有被普通 Shape 检查发现：循环未更新隐藏状态、只调用第一层权重、记录错 Attention 的 Layer 轴、误用 LN 顺序。

专项测试应检查：

- 单层结果严格匹配 Day 15；
- 三层 Pre-LN、Post-LN 分别匹配 PyTorch 参考；
- Attention 概率行和约等于 1；
- L+1 个隐藏状态的顺序正确；
- 更改第 2 层参数，不应改变原输入和第 1 层输出；
- 独立参数与显式 Weight Tying 的区别；
- 各层可以具有不同 FFN 隐藏宽度；
- Pre-LN 零分支恒等行为；
- 输入和参数不被修改；
- 非法参数应拒绝。

## 十、运行验收

专项：

~~~powershell
python -m pytest -q -m exercise tests/test_week16_exercises.py
~~~

全项目：

~~~powershell
python -m pytest -q -m exercise
~~~

Demo：

~~~powershell
python -m scripts.day16_encoder_stack_demo
~~~

预期本地输出：

~~~text
results/day16/
├── metrics.json
├── representation_norms.png
├── attention_entropy_by_layer.png
└── token_perturbation_by_depth.png
~~~

这三类可视化分别研究表示尺度、Attention 熵、扰动影响随深度的变化。固定随机种子的 Demo 是机制检验，不能当作训练后语义能力的证据。

## 十一、Day 16 思考题

1. 为什么重复调用相同的 Python 函数不等于共享同一套参数？
2. 为什么 all_hidden_states 需要 L+1 个元素？
3. 为什么不能把所有 Attention 的 Layer 轴丢掉？
4. 如果某层的更新范数很小，可以直接判断它没有价值吗？
5. 如果某个 Token 的变化在更深层传播得更广，是否必然有益？
6. Pre-LN 的恒等残差路径如何出现在梯度 Jacobian 中？
7. 为什么我们仍然不能说 NumPy Encoder Stack 已经实现了完整反向训练？

---

**Day 16 完成标准：手写多层 Encoder Forward，保留全部逐层特征与注意力，并通过独立 PyTorch 参考和深度诊断验证其正确性。**
