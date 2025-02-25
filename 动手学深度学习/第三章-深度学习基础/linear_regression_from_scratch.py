"""自己从头实现的线性回归，使用了自动求导"""

import random
import torch


def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b   # 这里是二维张量和向量，也就是一维张量相乘，结果是一维张量，如：[1000, 2] @ [2] ---> [1000]
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # 这里将y的形状从1维变成2维

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def data_iter(batch_size, features, labels):
    example_num = len(features)
    indices = list(range(example_num))
    random.shuffle(indices)
    for i in range(0, example_num, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, example_num)])
        yield features[batch_indices], labels[batch_indices]

# 初始化模型参数
# w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
w = torch.zeros((2, 1), requires_grad=True) # 初始化为0，对结果没有影响
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 # 这里用reshape是为了避免形状不匹配，比如(3, 1) 和 (3,) 就要用到reshape

# 定义优化算法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= param.grad * lr / batch_size
            param.grad.zero_() # 使用完后梯度清零

lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss
batch_size = 10


for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_loss = loss(net(features, w, b), labels)
        print(f"Epoch:{epoch + 1}", f', loss:{float(train_loss.mean()):.5f}')

print(f"w的估计误差:{true_w - w.reshape(true_w.shape)}")
print(f"b的估计误差:{true_b - b}")





















