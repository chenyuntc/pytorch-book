# coding:utf8

import torch as t
import torchvision as tv
import torchnet as tnt

from torch.utils import data
from model import Model
import utils
from torch.nn import functional as F
import tqdm
import os
import ipdb
from PIL import Image

IMAGENET_MEAN=[0.485,0.456,0.406]
IMAGENET_STD=[0.229,0.224,0.225]
mean=t.Tensor(IMAGENET_MEAN).reshape(-1, 1, 1)
std=t.Tensor(IMAGENET_STD).reshape(-1, 1, 1)
class Config(object):
    use_gpu = True
    model_path = None # 预训练模型的路径（用于继续训练/测试）
    
    # 训练用参数
    image_size = 256 # 图片大小
    batch_size = 16  # 一个batch的大小
    content_data_root = '/mnt/sda1/COCO/train' # 内容图片数据集路径
    style_data_root = '/mnt/sda1/Style' # 风格图片数据集路径
    num_workers = 4 # 多线程加载数据
    lr = 5e-5 # 学习率
    epoches = 40 # 训练epoch

    env = 'neural-style' # visdom env
    plot_every = 20 # 每20个batch可视化一次

    debug_file = '/tmp/debugnn' # 进入调试模式

    # 测试用参数
    content_path = 'input.png' # 需要进行风格迁移的图片
    style_path = None # 风格图片
    result_path = 'output.png' # 风格迁移结果的保存路径


def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = utils.Visualizer(opt.env)

    # Data loading
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.RandomCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    content_dataset = tv.datasets.ImageFolder(opt.content_data_root, transfroms)
    content_dataloader = data.DataLoader(content_dataset, opt.batch_size,shuffle=True,drop_last=True)
    style_dataset = tv.datasets.ImageFolder(opt.style_data_root, transfroms)
    style_dataloader = data.DataLoader(style_dataset, opt.batch_size,shuffle=True,drop_last=True)
    # style transformer network

    model = Model()
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    model.to(device)

    # Optimizer
    optimizer = t.optim.Adam(model.parameters(), opt.lr)

    # Loss meter
    loss_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        loss_meter.reset()

        for ii, image in tqdm.tqdm(enumerate(zip(content_dataloader,style_dataloader))):
            # Train
            optimizer.zero_grad()
            content = image[0][0].to(device)
            style = image[1][0].to(device)
            loss = model(content,style)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                # visualization
                vis.plot('loss', loss_meter.value()[0])
                vis.img('input_content', (content.cpu()[0]*std+mean).clamp(min=0, max=1))
                vis.img('input_style', (style.cpu()[0]*std+mean).clamp(min=0, max=1))
                vis.img('output', (model.generate(content,style)[0].cpu()*std+mean).clamp(min=0, max=1))

        # save checkpoint
        vis.save([opt.env])
        if (epoch+1)%5==0:
            t.save(model.state_dict(), 'checkpoints/%s_style.pth' % str(epoch+1))

@t.no_grad()
def stylize(**kwargs):
    """
    perform style transfer
    """
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    
    # 图片处理
    transfroms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    content_image = Image.open(opt.content_path)
    content_image = transfroms(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    style_image = Image.open(opt.style_path)
    style_image = transfroms(style_image)
    style_image = style_image.unsqueeze(0).to(device)

    # 加载模型
    model = Model().eval()
    model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    model.to(device)

    # 风格迁移与保存
    output = model.generate(content_image,style_image)
    output_data = output.cpu()*std+mean
    tv.utils.save_image(output_data.clamp(0,1), opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()
