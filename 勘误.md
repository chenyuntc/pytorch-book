## 第二章
##　第三章
### 3.1.2节
torch.FloatTensor(ndarray in float32) --- share memory
torch.FloatTensor(ndarray in float64) --- allocate new memory
### 3.2.2节
saved_variables 在v0.3 版本中被删除

### 5.1.1节
- torchvision Scale -> Resize
- torchvision RandomSizedCrop -> RandomResizedCrop
- dataset collate_fn 一节
  ```
  dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1,shuffle=True)
  ```
### 5.4节
module.cuda(device = 1) # device_id 也变了


### 8.2 节
P 225
其网络结构如图 8-5 所示。图中 (a) 是网络的总体结构，左边 (d) 是一个残差单元的结构图，右边 (b) 和 (c) 分别是下采样和上采样单元的结构图。

### 第四章
https://github.com/chenyuntc/pytorch-book/pull/76
