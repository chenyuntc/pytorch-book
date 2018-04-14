# coding:utf8
import torch as t
from torch.utils import data
import os
from PIL import Image
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# - 区分训练集和验证集
# - 不是随机返回每句话，而是根据index%5
# - 

# def create_collate_fn():
#     def collate_fn():
#         pass
#     return collate_fn

def create_collate_fn(padding, eos, max_length=50):
    def collate_fn(img_cap):
        """
        将多个样本拼接在一起成一个batch
        输入： list of data，形如
        [(img1, cap1, index1), (img2, cap2, index2) ....]
        
        拼接策略如下：
        - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
          选取长度最长的句子，将所有句子pad成一样长
        - 长度不够的用</PAD>在结尾PAD
        - 没有START标识符
        - 如果长度刚好和词一样，那么就没有</EOS>
        
        返回：
        - imgs(Tensor): batch_sie*2048
        - cap_tensor(Tensor): batch_size*max_length
        - lengths(list of int): 长度为batch_size
        - index(list of int): 长度为batch_size
        """
        img_cap.sort(key=lambda p: len(p[1]), reverse=True)
        imgs, caps, indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths), indexs)

    return collate_fn


class CaptionDataset(data.Dataset):

    def __init__(self, opt):
        """
        Attributes:
            _data (dict): 预处理之后的数据，包括所有图片的文件名，以及处理过后的描述
            all_imgs (tensor): 利用resnet50提取的图片特征，形状（200000，2048）
            caption(list): 长度为20万的list，包括每张图片的文字描述
            ix2id(dict): 指定序号的图片对应的文件名
            start_(int): 起始序号，训练集的起始序号是0，验证集的起始序号是190000，即
                前190000张图片是训练集，剩下的10000张图片是验证集
            len_(init): 数据集大小，如果是训练集，长度就是190000，验证集长度为10000
            traininig(bool): 是训练集(True),还是验证集(False)
        """
        self.opt = opt
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        self.all_imgs = t.load(opt.img_feature_path)

    def __getitem__(self, index):
        """
        返回：
        - img: 图像features 2048的向量
        - caption: 描述，形如LongTensor([1,3,5,2]),长度取决于描述长度
        - index: 下标，图像的序号，可以通过ix2id[index]获取对应图片文件名
        """
        img = self.all_imgs[index]

        caption = self.captions[index]
        # 5句描述随机选一句
        rdn_index = np.random.choice(len(caption), 1)[0]
        caption = caption[rdn_index]
        return img, t.LongTensor(caption), index

    def __len__(self):
        return len(self.ix2id)


def get_dataloader(opt):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=opt.shuffle,
                                 num_workers=opt.num_workers,
                                 collate_fn=create_collate_fn(dataset.padding, dataset.end))
    return dataloader


if __name__ == '__main__':
    from config import Config

    opt = Config()
    dataloader = get_dataloader(opt)
    for ii, data in enumerate(dataloader):
        print(ii, data)
        break
