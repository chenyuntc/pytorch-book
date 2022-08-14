import torch as t
from torch import nn
import numpy as np
from utils import AdaIn,calc_mean_std
from torchvision.models import vgg19
from collections import namedtuple


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = list(vgg19(pretrained=True).features)[:21]
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {1, 6, 11, 20}:
                results.append(x)
        vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        return vgg_outputs(*results)
        
class ConvLayer(nn.Module):
    """
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpSampleLayer(nn.Module):
    def __init__(self, in_channels):
        super(UpSampleLayer,self).__init__()
        self.input = nn.Parameter(t.randn(in_channels))
    
    def forward(self,x):
        return nn.functional.interpolate(x,scale_factor=2)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.decode = nn.Sequential(
            ConvLayer(512,256,kernel_size=3,stride=1),
            nn.ReLU(),
            UpSampleLayer(256),
            ConvLayer(256,256,kernel_size=3,stride=1),
            nn.ReLU(),
            ConvLayer(256,256,kernel_size=3,stride=1),
            nn.ReLU(),
            ConvLayer(256,256,kernel_size=3,stride=1),
            nn.ReLU(),
            ConvLayer(256,128,kernel_size=3,stride=1),
            nn.ReLU(),
            UpSampleLayer(128),
            ConvLayer(128,128,kernel_size=3,stride=1),
            nn.ReLU(),
            ConvLayer(128,64,kernel_size=3,stride=1),
            nn.ReLU(),
            UpSampleLayer(64),
            ConvLayer(64,64,kernel_size=3,stride=1),
            nn.ReLU(),
            ConvLayer(64,3,kernel_size=3,stride=1),
        )
    def forward(self,x):
        return self.decode(x)
        
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.encoder=VGG19()
        self.decoder=Decoder()
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def get_content_loss(self,content,adain):
        return nn.functional.mse_loss(content,adain)

    def get_style_loss(self,content,style):
        loss=0
        for i,j in zip(content,style):
            content_mean,content_std=calc_mean_std(i)
            style_mean,style_std=calc_mean_std(j)
            loss += nn.functional.mse_loss(content_mean,style_mean)+nn.functional.mse_loss(content_std,style_std)
        return loss
    
    def generate(self,content,style):
        content_feature=self.encoder(content).relu4_1
        style_feature=self.encoder(style).relu4_1
        adain = AdaIn(content_feature,style_feature)
        return self.decoder(adain)

    def forward(self,content,style):
        content_feature=self.encoder(content).relu4_1
        style_feature=self.encoder(style).relu4_1
        adain = AdaIn(content_feature,style_feature)
        output = self.decoder(adain)

        output_features = self.encoder(output).relu4_1
        content_mid = self.encoder(output)
        style_mid = self.encoder(style)

        content_loss = self.get_content_loss(output_features,adain)
        style_loss = self.get_style_loss(content_mid,style_mid)
        return content_loss + 10* style_loss