# Day 05｜Sigmoid、Tanh、GELU 与梯度传播

> Transformer From Zero — 365 Days  
> 日期：2026-11-08  
> 主题：非线性激活、数值稳定性、链式法则、梯度消失与 Transformer FFN

## 一、为什么今天要比较激活函数？

Day 04 我们实现了 ReLU MLP，比较了单层线性模型和非线性网络对 $y=x^2$ 的拟合效果。

今天进一步回答：**为什么不同激活函数会改变梯度的大小、分布，进而影响优化难度？**

本日不重复实现 Day 04 的 MLP，而是系统实现并验证三个新的激活函数。最后将它们与 ReLU 的导数放在同一坐标系中观察。

## 二、Task A：Sigmoid

### A1 数值稳定的前向传播

$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

如果直接对 $x=-1000$ 计算 $e^{-x}=e^{1000}$，可能发生溢出。应当使用等价分段式：

$$
\sigma(x)=\begin{cases}
1/(1+e^{-x}),& x\geq0,\\
e^x/(1+e^x),& x<0.
\end{cases}
$$

**容易踩的坑：** NumPy 的 `np.where(condition, true_expression, false_expression)` 会先计算两个候选表达式，因此直接写入两种 `np.exp` 仍可能触发溢出。建议使用布尔索引掩码，仅在相应分支上的元素上计算指数。

要求：支持任意形状的数组；保留输入形状；不修改输入；对 ±1000 不产生 overflow warning。

对应函数：`sigmoid_numpy`。

### A2 Sigmoid 的反向传播

$$
\sigma'(x)=\sigma(x)(1-\sigma(x)),
$$

$$
G_x=G_{out}\odot\sigma(x)(1-\sigma(x)).
$$

因为 $0<\sigma(x)<1$（在精确实数运算下），其导数满足 $0<\sigma'(x)\le0.25$，最大值出现在 $x=0$。远离零点时导数趋于 0，这就是饱和区的梯度衰减现象。

对应函数：`sigmoid_backward_numpy`。

## 三、Task B：Tanh

### B1 前向

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}.
$$

输出属于 $(-1,1)$，且以 0 为中心。本任务直接使用 NumPy 已实现数值稳定处理的 `np.tanh`，而不是自己利用指数函数展开。

对应函数：`tanh_numpy`。

### B2 反向

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x).
$$

$$
G_x=G_{out}\odot(1-\tanh^2(x)).
$$

在 $x=0$ 处导数为 1；远离零点时同样会饱和并趋近于 0。

对应函数：`tanh_backward_numpy`。

## 四、Task C：GELU（Tanh 近似）

### C1 前向

Transformer 常用 GELU 激活。今天实现 **tanh 近似版本**：

$$
\operatorname{GELU}_{tanh}(x)=
\frac{x}{2}\left[1+\tanh\left(\sqrt{\frac2\pi}(x+0.044715x^3)\right)\right].
$$

请特别注意：这不是基于误差函数 `erf` 的精确 GELU，而是 PyTorch 中 `gelu(..., approximate='tanh')` 对应的常用近似。

对应函数：`gelu_tanh_numpy`。

### C2 手动反向

令：

$$
a=\sqrt{2/\pi},\quad u=a(x+0.044715x^3),\quad t=\tanh(u).
$$

其导数：

$$
\frac{du}{dx}=a(1+3\cdot0.044715x^2).
$$

根据乘积法则与链式法则：

$$
\frac{d\,\operatorname{GELU}_{tanh}(x)}{dx}
=\frac12(1+t)+\frac12 x(1-t^2)\frac{du}{dx}.
$$

最终还需乘以上游梯度。**GELU 的导数并非始终在 0 与 1 之间；不要错误地将它裁剪为 0～1。**

对应函数：`gelu_tanh_backward_numpy`。

## 五、Task D：梯度曲线与比较

返回以下 4 组梯度数组：

| 字典 key | 计算方式 | 在 x=0 时的梯度 |
| --- | --- | ---: |
| sigmoid | 复用 Sigmoid 反向传播，上游梯度全为 1 | 0.25 |
| tanh | 复用 Tanh 反向传播，上游梯度全为 1 | 1 |
| relu | 复用 Day 04 ReLU 反向传播，上游梯度全为 1 | 0（本项目约定） |
| gelu | 复用 GELU 近似反向传播，上游梯度全为 1 | 0.5 |

对应函数：`activation_gradient_profiles`。

思考：为什么 Sigmoid/Tanh 在饱和区更容易造成梯度衰减？为什么 ReLU 的正半轴梯度不衰减，但负半轴却可能出现零梯度？GELU 的平滑变化有什么不同？

**严谨提醒：** 一个激活函数的导数性质只是影响训练的因素之一；初始化、残差连接、归一化、网络深度、学习率等也会影响实际梯度传播。不能仅凭单点导数断言哪一种激活函数总是最好。

## 六、今天的 8 个测试

| 测试 | 对应任务 |
| --- | --- |
| test_sigmoid_numpy_stable_values_and_shape | A1 |
| test_sigmoid_backward_numpy_against_torch | A2 |
| test_tanh_numpy_forward_against_torch | B1 |
| test_tanh_backward_numpy_against_torch | B2 |
| test_gelu_tanh_numpy_forward_against_torch | C1 |
| test_gelu_tanh_backward_numpy_against_torch | C2 |
| test_activation_gradient_profiles_values_and_reuse | D |
| test_gelu_backward_matches_day03_numerical_gradients | D / 综合梯度验证 |

从项目根目录运行：

~~~powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_sigmoid_numpy_stable_values_and_shape
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_sigmoid_backward_numpy_against_torch
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_tanh_numpy_forward_against_torch
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_tanh_backward_numpy_against_torch
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_gelu_tanh_numpy_forward_against_torch
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_gelu_tanh_backward_numpy_against_torch
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_activation_gradient_profiles_values_and_reuse
python -m pytest -q -m exercise tests/test_week05_exercises.py::test_gelu_backward_matches_day03_numerical_gradients
python -m pytest -q -m exercise tests/test_week05_exercises.py
~~~

初始的 `NotImplementedError` 是刻意保留的练习占位符；不要修改测试来绕过。完成后期望 **8 passed**。

## 七、今日最终展示

完成四组任务后，我们将使用同一个输入区间绘制四种激活函数及其导数曲线，并比较饱和区域、非饱和区域与负数区域的梯度行为。

如有必要，再在同一数据集、尽量一致的初始化和训练预算下比较不同激活的 MLP 训练曲线。最终实验的真实数值、图和正式 Day 05 笔记，**必须依据用户实际运行结果**整理，不能事先编造。

## 八、Git 提交

完成代码后：

~~~powershell
git add exercises/week05.py
git commit -m "feat: implement day05 activation functions and gradients"
git -c http.proxy=http://127.0.0.1:7897 push origin main
~~~

**今天从 Task A1：数值稳定的 Sigmoid 开始。**