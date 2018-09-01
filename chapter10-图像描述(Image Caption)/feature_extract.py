# coding:utf8
"""
利用resnet50提取图片的语义信息
并保存层results.pth
"""
from config import Config
import tqdm
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torch.utils import data
import os
from PIL import Image
import numpy as np

t.set_grad_enabled(False)
opt = Config()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class CaptionDataset(data.Dataset):

    def __init__(self, caption_data_path):
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(256),
            tv.transforms.ToTensor(),
            normalize
        ])

        data = t.load(caption_data_path)
        self.ix2id = data['ix2id']
        # 所有图片的路径
        self.imgs = [os.path.join(opt.img_path, self.ix2id[_]) \
                     for _ in range(len(self.ix2id))]

    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transforms(img)
        return img, index

    def __len__(self):
        return len(self.imgs)


def get_dataloader(opt):
    dataset = CaptionDataset(opt.caption_data_path)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 )
    return dataloader


# 数据
opt.batch_size = 256
dataloader = get_dataloader(opt)
results = t.Tensor(len(dataloader.dataset), 2048).fill_(0)
batch_size = opt.batch_size

# 模型
resnet50 = tv.models.resnet50(pretrained=True)
del resnet50.fc
resnet50.fc = lambda x: x
resnet50.cuda()

# 前向传播，计算分数
for ii, (imgs, indexs) in tqdm.tqdm(enumerate(dataloader)):
    # 确保序号没有对应错
    assert indexs[0] == batch_size * ii
    imgs = imgs.cuda()
    features = resnet50(imgs)
    results[ii * batch_size:(ii + 1) * batch_size] = features.data.cpu()

# 200000*2048 20万张图片，每张图片2048维的feature
t.save(results, 'results.pth')
