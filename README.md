# GPT2
Building GPT-2 from Scratch
I followed KJ's tutorial to try training a GPT-2 model from scratch.

# Micrograd

Micrograd is a lightweight automatic differentiation engine, based on Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

Micrograd 是一个轻量级的自动微分引擎，基于 Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd) 实现。
---

## **Features / 项目特点**
- **Automatic Differentiation / 自动微分**: Supports automatic differentiation for scalar values based on backpropagation.

  支持标量值的自动微分，基于反向传播算法。

---
# code / 代码
micrograd.py / 微梯度核心实现
English:
micrograd.py is the core implementation of the automatic differentiation engine. It supports scalar computation and backpropagation. Key features include:

Scalar Values : Stores scalar values and their gradients.
Backpropagation : Implements the chain rule for gradient computation.
Operations : Supports basic arithmetic operations (addition, multiplication, power, etc.) and activation functions (ReLU, Tanh).
中文:
micrograd.py 是自动微分引擎的核心实现。它支持标量计算和反向传播。主要特点包括：

标量值 ：存储标量值及其梯度。
反向传播 ：实现梯度计算的链式法则。
操作 ：支持基本算术操作（加法、乘法、幂运算等）和激活函数（ReLU、Tanh）。
nn.py / 神经网络模块实现
English:
nn.py implements neural network modules, similar to the MLP (Multi-Layer Perceptron) in the original Micrograd. Key features include:

Linear Layer : A fully connected layer with weights and biases.
Activation Functions : Implements ReLU and Tanh activation functions.
Modular Design : Allows users to easily build and train neural networks by combining Linear layers and activation functions.
中文:
nn.py 实现了神经网络模块，类似于原始 Micrograd 中的 MLP（多层感知机）。主要特点包括：

线性层 ：一个带有权重和偏置的全连接层。
激活函数 ：实现了 ReLU 和 Tanh 激活函数。
模块化设计 ：用户可以通过组合线性层和激活函数轻松构建和训练神经网络。