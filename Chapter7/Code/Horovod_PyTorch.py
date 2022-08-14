# Horovod_PyTorch.py
import horovod.torch as hvd
import torch
import torchvision as tv

## 第一步：初始化
hvd.init()
size = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()
# 这样tensor.cuda()会默认使用第local_rank个gpu
torch.cuda.set_device(local_rank)

## 第二步：构建数据
dataset = tv.datasets.CIFAR10(root="./", download=True, transform=tv.transforms.ToTensor())
# 为每一个进程分别划分不同的data
# DistributedSampler可以实现为每个进程分配不同的数据
sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=size, rank=rank)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, sampler=sampler)

## 第三步：构建模型
model = tv.models.resnet18(pretrained=False).cuda()
# 将初始化的参数同步，确保每一个进程都有与rank0相同的模型参数
hvd.broadcast_parameters(model.state_dict(), root_rank=0)

loss_fn = torch.nn.CrossEntropyLoss().cuda()

## 第四步：训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
for ii, (data, target) in enumerate(dataloader):
    # 确保将数据打乱（DistributedSampler使用epoch作为随机种子）
    sampler.set_epoch(ii)
    optimizer.zero_grad()
    output = model(data.cuda())
    loss = loss_fn(output, target.cuda())
    # 反向传播，每隔进程都会各自计算梯度
    loss.backward()
    # 计算所有进程的平均梯度，并更新模型参数
    optimizer.step()

# 只在rank0打印和保存模型参数
if rank == 0:
    print("training finished, saving data")
    torch.save(model.state_dict(), "./000.ckpt")