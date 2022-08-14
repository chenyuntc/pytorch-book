# distributed_PyTorch.py
import torch
import torch.distributed as dist
import torchvision as tv

## 第一步：初始化
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
# 这样tensor.cuda()会默认使用第local_rank个gpu
torch.cuda.set_device(local_rank)

## 第二步：构建数据
dataset = tv.datasets.CIFAR10(root="./", download=True, transform=tv.transforms.ToTensor())
# 为每一个进程分别划分不同的data
# DistributedSampler可以实现为每个进程分配不同的数据
sampler = torch.utils.data.DistributedSampler(dataset)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, sampler=sampler)

## 第三步：构建模型
model = tv.models.resnet18(pretrained=False).cuda()
# 使用DistributedDataParallel封装模型，实现分布式训练
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
loss_fn = torch.nn.CrossEntropyLoss().cuda()

## 第四步：训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for ii, (data, target) in enumerate(dataloader):
    # 确保将数据打乱（DistributedSampler使用epoch作为随机种子）
    sampler.set_epoch(ii)
    optimizer.zero_grad()
    output = model(data.cuda())
    loss = loss_fn(output, target.cuda())
    # 反向传播，每个进程都会各自计算梯度
    loss.backward()
    # 计算所有进程的平均梯度，并更新模型参数
    optimizer.step()
    
# 只在rank-0打印和保存模型参数
if local_rank == 0:
    print("training finished, saving data")
    torch.save(model.state_dict(), "./000.ckpt")