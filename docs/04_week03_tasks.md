# Day 03｜数值梯度检查与完整训练循环

Transformer From Zero — 365 Days  
日期：2026-11-06

## 任务

- Task A：finite_difference_gradient —— 中心差分数值求导（支持任意形状数组）。
- Task B：gradient_check_linear_mse —— 对比解析梯度和数值梯度。
- Task C：train_linear_regression_numpy —— 从初始参数出发完成多轮训练，拟合 y=3x+2。

## 1. 中心差分

$$
\frac{\partial f}{\partial \theta_i}
\approx\frac{f(\theta+\epsilon e_i)-f(\theta-\epsilon e_i)}{2\epsilon}
$$

每次只扰动一个参数；对所有参数逐一估计梯度；保留原数组不变；先使用 float64，epsilon=1e-5。epsilon 太大产生截断误差，太小会产生舍入误差。

### 已讲解例题

模型：预测值 = wx+b，损失 = 0.5*(预测值-y)^2。
x=2, w=1, b=0, y=5，固定 b，仅改变 w。

$$
L(w)=\frac12(2w-5)^2
$$

取 epsilon=0.001：

$$
L(1.001)=4.494002,\quad L(0.999)=4.506002
$$

中心差分：(4.494002-4.506002)/0.002 = -6，与 Day 2 的解析梯度相同。

## 2. 梯度检查

本轮仍使用 Day 2 的原始损失定义：

$$
L=\frac1{2N}\sum_{i=1}^{N}\sum_{j=1}^{C}E_{ij}^2
$$

$$
\nabla_W L=\frac{X^\top E}{N},
\qquad
\nabla_b L=\frac1N\sum_i E_i
$$

相对误差计算：

$$
r=\frac{\|g_{\rm analytic}-g_{\rm numeric}\|_2}
{\max(10^{-12},\|g_{\rm analytic}\|_2+\|g_{\rm numeric}\|_2)}
$$

数值梯度必须通过独立的有限差分计算，不能直接复制解析梯度。对于本次 float64 测试数据，目标相对误差小于 1e-7。这不是所有模型通用的误差阈值。

## 3. 完整训练循环

目标：拟合 y=3x+2。

每次迭代：
1. 前向计算预测值。
2. 调用 Week 02 的 batch_linear_mse_gradients 得到损失、权重梯度、偏置梯度。
3. 将权重和偏置沿负梯度方向更新。
4. 重新计算 Loss 并记录。

本轮是 Full-Batch Gradient Descent：每次使用全部训练样本，并非随机抽取样本的 SGD。

返回参数与 loss_history，后者应包含初始 Loss 及每次更新后的 Loss，因此长度为 steps+1。不得原地修改初始参数数组。

## 4. 文件与测试命令

练习：exercises/week03.py  
测试：tests/test_week03_exercises.py  
前置依赖：exercises/week02.py

~~~powershell
conda activate tfz
python -m pytest -q -m exercise tests/test_week03_exercises.py::test_finite_difference_gradient_quadratic
python -m pytest -q -m exercise tests/test_week03_exercises.py::test_finite_difference_gradient_matrix
python -m pytest -q -m exercise tests/test_week03_exercises.py::test_gradient_check_linear_mse
python -m pytest -q -m exercise tests/test_week03_exercises.py::test_numpy_training_learns_y_equals_3x_plus_2
python -m pytest -q -m exercise tests/test_week03_exercises.py
~~~

全部完成后预期 4 passed。起初 NotImplementedError 属于正常的练习失败；不要修改测试来绕过。

完成后：

~~~powershell
git add exercises/week03.py
git commit -m "feat: implement day03 gradient checks and training loop"
git push
~~~

## 5. 思考题

1. 中心差分为什么能独立检查手写反向传播？
2. 为什么梯度为负时，执行梯度下降可能让参数增大？
3. 为什么需要记录 steps+1 个损失数值？
4. 为什么数百万参数的模型不使用有限差分代替反向传播？

完成后按项目规范整理 Day 03 学习笔记，附简洁知识词云。
