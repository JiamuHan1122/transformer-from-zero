# Day 04｜ReLU、非线性与两层 MLP

> Transformer From Zero — 365 Days  
> 日期：2026-11-07  
> 核心目标：理解非线性激活，并手写两层神经网络的前向与反向传播。

## 一、为什么需要 ReLU

如果没有非线性激活：

$$
Y=(XW_1+b_1)W_2+b_2=X(W_1W_2)+(b_1W_2+b_2).
$$

也就是说，连续两层仿射变换仍可合并成一层仿射变换。加入 ReLU 后：

$$
Z_1=XW_1+b_1,\quad H=\operatorname{ReLU}(Z_1),
\quad\hat Y=HW_2+b_2.
$$

这样就能表达不容易由单个仿射层描述的非线性关系。

## 二、Task A：ReLU 与其导数

前向：

$$
\operatorname{ReLU}(z)=\max(0,z)
$$

例如，输入 [-3,0,2,5]，输出 [0,0,2,5]。

反向：

$$
\frac{\partial L}{\partial z}
=\frac{\partial L}{\partial\operatorname{ReLU}(z)}
\odot\mathbf 1[z>0]
$$

其中 \(\odot\) 是逐元素相乘。本练习**约定 ReLU 在零点的梯度为 0**，虽然 ReLU 在该点严格不可导。正数梯度为 1，负数为 0。

例：pre_activation=[-2,0,3]，grad_output=[5,7,-2]，最终 grad_input=[0,0,-2]。

函数名：relu_numpy、relu_backward_numpy。

## 三、Task B：两层 MLP 前向传播

| 符号 | 形状 |
| --- | --- |
| X | (N,D) |
| W1 | (D,H) |
| b1 | (H,) |
| Z1、H1 | (N,H) |
| W2 | (H,C) |
| b2 | (C,) |
| Ŷ | (N,C) |

$$
Z_1=XW_1+b_1,\qquad H_1=\operatorname{ReLU}(Z_1),\qquad \hat Y=H_1W_2+b_2.
$$

函数名：two_layer_mlp_numpy。

## 四、Task C：手写两层 MLP 反向传播

沿用前两天的损失定义：

$$
E=\hat Y-Y,\quad L=\frac1{2N}\sum_{i=1}^{N}\sum_{j=1}^{C}E_{ij}^2.
$$

先对输出层求导：

$$
G_2=\frac{\partial L}{\partial \hat Y}=\frac E N.
$$

$$
\nabla_{W_2}L=H_1^\top G_2,\quad
\nabla_{b_2}L=\sum_i G_{2,i}.
$$

传回隐藏层：

$$
G_H=G_2W_2^\top,\quad
G_1=G_H\odot \mathbf 1[Z_1>0].
$$

第一层梯度：

$$
\nabla_{W_1}L=X^\top G_1,\quad
\nabla_{b_1}L=\sum_i G_{1,i}.
$$

注意：G2 已经除过 N，后续不要重复除 N。

返回值顺序：

1. loss（Python float）
2. grad_w1
3. grad_b1
4. grad_w2
5. grad_b2

函数名：two_layer_mlp_mse_gradients。

## 五、五个测试

从项目根目录运行：

~~~powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week04_exercises.py::test_relu_numpy_forward_values_and_shape
python -m pytest -q -m exercise tests/test_week04_exercises.py::test_relu_backward_numpy_chain_rule_and_zero
python -m pytest -q -m exercise tests/test_week04_exercises.py::test_two_layer_mlp_forward_against_torch
python -m pytest -q -m exercise tests/test_week04_exercises.py::test_two_layer_mlp_backprop_against_torch_autograd
python -m pytest -q -m exercise tests/test_week04_exercises.py::test_mlp_gradient_one_step_reduces_loss
python -m pytest -q -m exercise tests/test_week04_exercises.py
~~~

初始练习文件保留 NotImplementedError，测试暂时失败是预期现象，完成后目标为 **5 passed**。

其中两个测试用 PyTorch 自动微分做参考，手写实现不得直接调用 PyTorch 的 autograd 或 optimizer。

## 六、理解题

1. 为什么连续叠加线性层而不加非线性，在表达能力上等价于一层？
2. ReLU 在零点不可导，为什么实际训练仍然可以规定其梯度？
3. 为什么从输出层反向传播到第一层时要乘以 W2 的转置？
4. 为什么得到 G2=E/N 后，不应再对 W2、W1 的梯度除 N？

## 七、今天如何推进

建议按 Task A → Task B → Task C 的顺序完成。五个测试通过之后，进一步用非线性合成数据做训练演示，检查 MLP 的实际表达能力，而不是只通过单元测试判断训练效果。

和 Transformer 的联系：今天学习的两层网络是 Transformer Feed Forward Network（FFN）的基础形态；实际 Transformer 也可能使用 GELU、SiLU 或门控激活。

## 八、Git 提交

完成以后执行：

~~~powershell
git add exercises/week04.py
git commit -m "feat: implement day04 relu and two-layer mlp"
git -c http.proxy=http://127.0.0.1:7897 push origin main
~~~

当天训练与测试完成后再整理 Day 04 正式笔记、例题和简洁知识词云。
