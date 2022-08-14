import torch as t
import torch.nn as nn
import numpy as np
from torchvision.models import resnet18
from utils import get_loss

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvLayer, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2d(x)

class DeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DeConvLayer, self).__init__()
        self.conv2d = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride=stride,padding=1,output_padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv2d(x)

class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
    
    def forward(self,x):
        x_1 = nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        return t.cat([x,x_1,x_2,x_3],dim=1)

class Model(nn.Module):
    def __init__(self,num_classes,topk):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.backbone = resnet18(pretrained=True)
        self.backbone=nn.Sequential(*list(self.backbone.children())[:-2])
        self.smooth = nn.Sequential(
            SPP(),
            ConvLayer(512*4,256,kernel_size=1,padding=0),
            ConvLayer(256,512,kernel_size=3,padding=1)
        )
        self.deconv5 = DeConvLayer(512,256,kernel_size=4,stride=2)
        self.deconv4 = DeConvLayer(256,256,kernel_size=4,stride=2)
        self.deconv3 = DeConvLayer(256,256,kernel_size=4,stride=2)

        self.cls_pred = nn.Sequential(
            ConvLayer(256,64,kernel_size=3,padding=1),
            nn.Conv2d(64,self.num_classes,kernel_size=1)
        )

        self.txty_pred = nn.Sequential(
            ConvLayer(256,64,kernel_size=3,padding=1),
            nn.Conv2d(64,2,kernel_size=1)
        )

        self.twth_pred = nn.Sequential(
            ConvLayer(256,64,kernel_size=3,padding=1),
            nn.Conv2d(64,2,kernel_size=1)
        )

    def decode(self,pred):
        output = t.zeros_like(pred)
        grid_y , grid_x = t.meshgrid([t.arange(128,device=pred.device),t.arange(128,device=pred.device)])
        grid_cell = t.stack([grid_x,grid_y],dim=-1).float().view(1, 128*128, 2)
        pred[:,:,:2] = (t.sigmoid(pred[:,:,:2])+grid_cell)*4
        pred[:,:,2:] = (t.exp(pred[:,:,2:]))*4

        #坐标转换：[cx,cy,w,h] -> [xmin,ymin,xmax,ymax]
        output[:,:,0]=pred[:,:,0]-pred[:,:,2]/2
        output[:,:,1]=pred[:,:,1]-pred[:,:,3]/2
        output[:,:,2]=pred[:,:,0]+pred[:,:,2]/2
        output[:,:,3]=pred[:,:,1]+pred[:,:,3]/2
        return output
    
    def gather_feat(self,feat,ind):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0),ind.size(1),dim)
        return feat.gather(1,ind)

    # 选取topk个满足要求的点
    def get_topk(self,scores):
        B,C,H,W = scores.size()
        topk_scores,topk_inds = t.topk(scores.view(B,C,-1),self.topk)
        topk_inds = topk_inds % (H*W)
        topk_score,topk_ind = t.topk(topk_scores.view(B,-1),self.topk)
        topk_inds = self.gather_feat(topk_inds.view(B,-1,1),topk_ind).view(B,self.topk)
        topk_clses = t.floor_divide(topk_ind,self.topk).int()
        return topk_score,topk_inds,topk_clses
    
    def generate(self,x):
        c5=self.backbone(x)
        B=c5.size(0)
        p5 = self.smooth(c5)
        p4 = self.deconv5(p5)
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)
        cls_pred = self.cls_pred(p2)
        txty_pred = self.txty_pred(p2)
        twth_pred = self.twth_pred(p2)

        cls_pred = t.sigmoid(cls_pred)
        # 寻找8-近邻极大值点，其中keep为hmax极大值点的位置，cls_pred为对应的极大值点
        hmax = nn.functional.max_pool2d(cls_pred,kernel_size=5, padding=2,stride=1)
        keep = (hmax==cls_pred).float()
        cls_pred *= keep

        txtytwth_pred = t.cat([txty_pred,twth_pred],dim=1).permute(0,2,3,1).contiguous().view(B,-1,4)

        scale = np.array([[[512,512,512,512]]])
        scale_t = t.tensor(scale.copy(),device=txtytwth_pred.device).float()
        bbox_pred = t.clamp((self.decode(txtytwth_pred)/scale_t)[0],0.,1.)

        # 得到topk取值，topk_score：置信度，topk_ind：index，topk_clses：类别
        topk_score,topk_ind,topk_clses = self.get_topk(cls_pred)
        topk_bbox_pred = bbox_pred[topk_ind[0]]
        return topk_bbox_pred.cpu().numpy(),topk_score[0].cpu().numpy(),topk_clses[0].cpu().numpy()


    def forward(self,x,target):
        c5=self.backbone(x)
        B=c5.size(0)
        p5 = self.smooth(c5)
        p4 = self.deconv5(p5)
        p3 = self.deconv4(p4)
        p2 = self.deconv3(p3)

        cls_pred = self.cls_pred(p2)
        txty_pred = self.txty_pred(p2)
        twth_pred = self.twth_pred(p2)

        # 热力图：[B, H*W, num_classes]
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
        # 中心点偏移：[B, H*W, 2]
        txty_pred = txty_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        # 物体尺度：[B, H*W, 2]
        twth_pred = twth_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)

        # 计算损失函数
        total_loss = get_loss(pred_cls=cls_pred, pred_txty=txty_pred, pred_twth=twth_pred, label=target, num_classes=self.num_classes)

        return total_loss    
