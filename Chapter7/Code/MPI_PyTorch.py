# MPI_PyTorch.py
import torch
import torchvision as tv
import mpi4py.MPI as MPI
import torch.nn as nn

## 第一步：环境初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 这样tensor.cuda()会默认使用第rank个gpu
torch.cuda.set_device(rank)

## 第二步：构建数据
dataset = tv.datasets.CIFAR10(root="./", download=True, transform=tv.transforms.ToTensor())
# 为每一个进程划分不同的data
# X[rank::size]的意思是：从第<rank>个元素开始，每隔<size>个元素取一个
dataset.data = dataset.data[rank::size]
dataset.targets = dataset.targets[rank::size]
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512)

## 第三步：构建模型
model = tv.models.resnet18(pretrained=False).cuda()
# 将随机初始化的参数同步，确保每一个进程都有与rank0相同的模型参数
for name, param in model.named_parameters():
    param_from_rank_0 = comm.bcast(param.detach(), root=0)
    param.data.copy_(param_from_rank_0)

lr = 0.001
loss_fn = torch.nn.CrossEntropyLoss().cuda()

## 第四步：训练
for ii, (data, target) in enumerate(dataloader):
    data = data.cuda()
    output = model(data)
    print("data",data.shape)
    print("output",output.shape)
    print("target",target.shape)
    loss = loss_fn(output, target.cuda())
    # 反向传播，每个进程都会各自计算梯度
    loss.backward()
    # 重点！计算所有进程的平均梯度，更新模型参数
    for name, param in model.named_parameters():
        grad_sum = comm.allreduce(param.grad.detach().cpu(), op=MPI.SUM)
        grad_mean = grad_sum/(grad_sum * size)
        param.data -= lr * grad_mean.cuda()  # 梯度下降-更新模型参数

# 只在rank-0打印和保存模型参数
if rank == 0:
    print('training finished, saving data')
    torch.save(model.state_dict(), "./000.ckpt")