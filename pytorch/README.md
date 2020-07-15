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

