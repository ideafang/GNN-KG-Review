# Pytorch 学习
## numpy_nn 用numpy实现两层全连接网络

一个全连接ReLU神经网络，一个隐藏层，没有bias。用来从x预测y，使用L2 Loss.

- $h = W_1x + b_1$
- $a = max(0, h)$
- $y_{hat} = W_2a + b_2$

这一实现完全使用numpy来计算前向神经网络，loss，和反向传播

- forward pass
- loss
- backward pass

numpy ndarray 是一个普通的n维array，没有任何关于深度学习和梯度（gtadient）的辅助方法，也没有计算图（computation graph）方法，只是一种用来计算数学运算的数据结构。

```python
import numpy as np

'''
N     输入数据
D_in  输入数据维度
H     隐层参数量
D_out 输出维度
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for it in range(500):
    # Forward pass
    h = x.dot(w1) # N * H
    h_relu = np.maximum(h, 0) # N * H 
    y_pred = h_relu.dot(w2) # N * D_out

    # compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)

    # Backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_w1 = x.T.dot(grad_h)

    #update weights of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

## Pytorch_nn 用Pytorch实现两层全连接网络

### 手动求导

替换上文中的numpy:

- `numpy.dot()`  => `torch.mm(input, mat2, out=None) -> Tensor`实现torch.tensor矩阵相乘(matrix multiplication)，返回值为tensor。
- `numpy.T` => `torch.t()`矩阵转置
- `numpy.maximum(h, 0)` => `torch.clamp(min = 0)` 构建ReLU激活函数
- `numpy.copy()` => `torch.clone()` 复制

调用gpu:

```python
x = torch.randn(N, D_in).to("cuda:0")
y = torch.randn(N, D_out).to("cuda:0")
w1 = torch.randn(D_in, H).to("cuda:0")
w2 = torch.randn(H, D_out).to("cuda:0")
```

完整code：

```python
import torch

'''
N     输入数据
D_in  输入数据维度
H     隐层参数量
D_out 输出维度
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建训练数据
x = torch.randn(N, D_in).to("cuda:0")
y = torch.randn(N, D_out).to("cuda:0")

w1 = torch.randn(D_in, H).to("cuda:0")
w2 = torch.randn(H, D_out).to("cuda:0")

learning_rate = 1e-6
for it in range(500):
    # Forward pass
    h = x.mm(w1) # N * H
    h_relu = h.clamp(min = 0) # N * H
    y_pred = h_relu.mm(w2) # N * D_out

    # compute loss
    loss = (y_pred - y).pow(2).sum().item()
    print(it, loss)

    # Backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    #update weights of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
```

### Pytorch自动求导（autograd）

将`w1`，`w2`的`requires_grad`设为True

```python
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)
```

精简Forward pass：

```python
# Forward pass
y_pred = x.mm(w1).clamp(min = 0).mm(w2)
```

使用`backward()`自动求导

```python
# Backward pass
loss.backward()
```

更新梯度

```python
# update weights of w1 and w2
with torch.no_grad():
    w1 -= learning_rate * w1.grad
    w2 -= learning_rate * w2.grad
    w1.grad.zero_()
    w2.grad.zero_()
```

完整code

```python
import torch

'''
N     输入数据
D_in  输入数据维度
H     隐层参数量
D_out 输出维度
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for it in range(500):
    # Forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compute loss
    loss = (y_pred - y).pow(2).sum() #computation graph
    print(it, loss.item())

    # Backward pass
    # compute the gradient
    loss.backward()

    # update weights of w1 and w2
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
```

## Pytorch：nn

使用Pytorch中的nn库构建网络，用autograd构建计算图和计算gradients

全连接神经网络： `torch.nn.Linear(in_features, out_features, bias=True)`

构建网络

```python
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=False),
).cuda()
```

完整code

```python
import torch

'''
N     输入数据
D_in  输入数据维度
H     隐层参数量
D_out 输出维度
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建训练数据
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=False),
).cuda()

torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)

loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
for it in range(500):
    # Forward pass
    y_pred = model(x) # model.forward()

    # compute loss
    loss = loss_fn(y_pred, y) # computation graph
    print(it, loss.item())

    # Backward pass
    # compute the gradient
    loss.backward()

    # update weights of w1 and w2
    with torch.no_grad():
        for param in model.parameters(): # param(tensor, grad)
            param -= learning_rate * param.grad

    model.zero_grad()
```

## Pytorch：optim

使用optim包来自动更新模型的weight。optim包提供了多种模型的优化方法，包括SGD+momentum, RMSProp, Adam等等

```python
import torch

'''
N     输入数据
D_in  输入数据维度
H     隐层参数量
D_out 输出维度
'''
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机创建训练数据
x = torch.randn(N, D_in).cuda()
y = torch.randn(N, D_out).cuda()

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=False),
).cuda()

# torch.nn.init.normal_(model[0].weight)
# torch.nn.init.normal_(model[2].weight)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for it in range(500):
    # Forward pass
    y_pred = model(x) # 等效于 model.forward(x)

    # compute loss
    loss = loss_fn(y_pred, y) # computation graph
    print(it, loss.item())

    optimizer.zero_grad()

    # Backward pass
    # compute the gradient
    loss.backward()

    # update model parameters
    optimizer.step()
```

## Pytorch：nn.Modules

通过nn.Module类可以定义一个比Sequential模型更加复杂的模型。

将model的定义更改为：

```python
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        # define the model architecture
        self.linear1 = torch.nn.Linear(D_in, H, bias=False)
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min = 0))
        return y_pred

model = TwoLayerNet(D_in, H, D_out).cuda()
```