import torch
from torch.autograd import Function
from torch.nn import Module
import mysigmoid2  # 导入我们的扩展
import time


# 将扩展的前向传播和反向传播封装为一个Function对象
class MySigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        fx = mysigmoid2.forward(x)
        vars = [fx]
        ctx.save_for_backward(*vars)
        return fx

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out.contiguous()
        grad_x = mysigmoid2.backward(*ctx.saved_tensors, grad_out)
        return grad_x


class Test(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return MySigmoid.apply(x)

def main():
    x2 = torch.randn(4).requires_grad_()
    x1 = x2.clone().detach().cuda().requires_grad_()
    model = Test().cuda()
    fx1 = model(x1)
    fx1.sum().backward()

    fx2 = torch.exp(-x2) / (1. + torch.exp(-x2))
    fx2.sum().backward()

    assert fx1.data.cpu().allclose(fx2)
    assert x1.grad.data.cpu().allclose(x2.grad)


def compare_pytorch_with_cuda():
    # 调用model时必须保证是二维数据
    x1 = torch.randn((1280, 1280)).cuda().requires_grad_()
    model = Test().cuda()
    starttime = time.time()
    fx1 = model(x1)
    print("CUDA extension forward time:", time.time() - starttime)
    starttime = time.time()
    fx1.sum().backward()
    print("CUDA extension backward time:", time.time() - starttime)

    x2 = torch.randn((1280, 1280)).float().requires_grad_()
    starttime = time.time()
    fx2 = torch.exp(-x2) / (1. - torch.exp(-x2))
    print("pytorch forward time:", time.time() - starttime)
    starttime = time.time()
    fx2.sum().backward()
    print("pytorch backward time:", time.time() - starttime)


if __name__ == '__main__':
    main()
    compare_pytorch_with_cuda()