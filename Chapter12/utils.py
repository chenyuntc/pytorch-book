# coding:utf8
from itertools import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

def calc_mean_std(features):
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def AdaIn(x, y):
    x_mean, x_std = calc_mean_std(x)
    y_mean, y_std = calc_mean_std(y)
    normalized_features = y_std * (x - x_mean) / x_std + y_mean
    return normalized_features


class Visualizer():
    """
    wrapper on visdom, but you may still call native visdom by `self.vis.function`
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        
        """
        self.vis = visdom.Visdom(env=env,use_incoming_socket=False,  **kwargs)
        return self

    def plot_many(self, d):
        """
        plot multi values in a time
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        convert batch images to grid of images
        i.e. input（36，64，64） ->  6*6 grid，each grid is an image of size 64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


# def get_style_data(path):
#     """
#     load style image，
#     Return： tensor shape 1*c*h*w, normalized
#     """
#     style_transform = tv.transforms.Compose([
#         tv.transforms.ToTensor(),
#         tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ])

#     style_image = tv.datasets.folder.default_loader(path)
#     style_tensor = style_transform(style_image)
#     return style_tensor.unsqueeze(0)


# def normalize_batch(batch):
#     """
#     Input: b,ch,h,w  0~255
#     Output: b,ch,h,w  -2~2
#     """
#     mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
#     std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
#     mean = (mean.expand_as(batch.data))
#     std = (std.expand_as(batch.data))
#     return (batch / 255.0 - mean) / std
