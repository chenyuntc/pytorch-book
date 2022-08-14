import torch
from torch.autograd import Function
from torch.nn import Module
import mysigmoid  # 导入我们的扩展
import time
import math


# 将扩展的前向传播和反向传播封装为一个Function对象
class MySigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        fx = mysigmoid.forward(x)
        vars = [fx]
        ctx.save_for_backward(*vars)
        return fx

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = mysigmoid.backward(*ctx.saved_tensors, grad_out)
        return grad_x


class Test(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MySigmoid.apply(x)


def main():
    # checkResult()
    compare_pytorch_with_cpp()

def checkResult():
    x1 = torch.arange(2, 12).view(2, 5).float().requires_grad_()
    model = Test()
    fx1 = model(x1)
    fx1.sum().backward()

    x2 = torch.arange(2, 12).view(2, 5).float().requires_grad_()
    fx2 = torch.exp(-x2) / (1. - torch.exp(-x2))
    fx2.sum().backward()

    assert fx1.equal(fx2)
    assert x1.grad.allclose(x2.grad)

# 测试C++扩展的性能
def compare_pytorch_with_cpp():
    torch.set_num_threads(1)
    # 调用model时必须保证是二维数据
    x1 = torch.randn((1280, 1280)).requires_grad_()
    model = Test()
    starttime = time.time()
    fx1 = model(x1)
    print("cpp extension forward time:", time.time() - starttime)
    starttime = time.time()
    fx1.sum().backward()
    print("cpp extension backward time:", time.time()-starttime)


    # 使用相同大小的数据测试PyTorch循环
    x2 = torch.randn((1280, 1280)).float().requires_grad_()
    fx2 = torch.zeros_like(x2)
    starttime = time.time()
    for i in range(x2.size(0)):
        for j in range(x2.size(1)):
            fx2[i][j] = torch.exp(-x2[i][j]) / (1. - torch.exp(-x2[i][j]))
    print("PyTorch for_loop forward time:", time.time() - starttime)
    starttime = time.time()
    fx2.sum().backward()
    print("PyTorch for_loop backward time:", time.time() - starttime)


    # 使用相同大小的数据测试PyTorch向量化运算
    x3 = torch.randn((1280, 1280)).float().requires_grad_()
    starttime = time.time()
    fx3 = torch.exp(-x3) / (1. - torch.exp(-x3))
    print("pytorch forward time:", time.time() - starttime)
    starttime = time.time()
    fx3.sum().backward()
    print("pytorch backward time:", time.time() - starttime)



if __name__ == '__main__':
    main()
