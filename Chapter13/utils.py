import numpy as np
import torch as t
import torch.nn as nn
import json
import tempfile
from pycocotools.cocoeval import COCOeval
from dataset import COCODataset
from torch.utils.data import DataLoader
import cv2

# 确定高斯圆最小半径r，保留IOU与GT大于阈值0.7的预测框
def gaussian_radius(det_size, min_overlap=0.7):
    box_h, box_h  = det_size
    a3 = 1
    b3 = (box_h + box_h)
    c3 = box_h * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2 #(2*a3)

    a2 = 4
    b2 = 2 * (box_h + box_h)
    c2 = (1 - min_overlap) * box_h * box_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2 #(2*a2)

    a1 = 4 * min_overlap
    b1 = -2 * min_overlap * (box_h + box_h)
    c1 = (min_overlap - 1) * box_h * box_h
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2 #(2*a1)

    return min(r1, r2, r3)

# 将原始bbox标注映射到feature map上
def generate_txtytwth(gt_label, w, h, s):
    '''
    中心点坐标：(grid_x, grid_y)
    偏移量：(tx, ty)
    尺度：(tw, th)
    二维高斯函数方差：sigma_w, sigma_h
    '''
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算中心及高度宽度
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    box_w_s = box_w / s
    box_h_s = box_h / s

    r = gaussian_radius([box_w_s, box_h_s])
    sigma_w = sigma_h = r / 3


    if box_w < 1e-28 or box_h < 1e-28:
        return False    

    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)

    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w_s)
    th = np.log(box_h_s)

    return grid_x, grid_y, tx, ty, tw, th, sigma_w, sigma_h

# 创建高斯热力图，生成可用的标注信息
def gt_creator(input_size, stride, num_classes, label_lists=[]):
    batch_size = len(label_lists)
    w = input_size
    h = input_size
    
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, num_classes+4+1])
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_cls = gt_label[-1]
            result = generate_txtytwth(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, sigma_w, sigma_h = result

                gt_tensor[batch_index, grid_y, grid_x, int(gt_cls)] = 1.0
                gt_tensor[batch_index, grid_y, grid_x, num_classes:num_classes + 4] = np.array([tx, ty, tw, th])
                gt_tensor[batch_index, grid_y, grid_x, num_classes + 4] = 1.0

                # 创建高斯热力图
                for i in range(grid_x - 3*int(sigma_w), grid_x + 3*int(sigma_w) + 1):
                    for j in range(grid_y - 3*int(sigma_h), grid_y + 3*int(sigma_h) + 1):
                        if i < ws and j < hs:
                            v = np.exp(- (i - grid_x)**2 / (2*sigma_w**2) - (j - grid_y)**2 / (2*sigma_h**2))
                            pre_v = gt_tensor[batch_index, j, i, int(gt_cls)]
                            gt_tensor[batch_index, j, i, int(gt_cls)] = max(v, pre_v)

    gt_tensor = gt_tensor.reshape(batch_size, -1, num_classes+4+1)

    return gt_tensor


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(t.FloatTensor(sample[1]))
    return t.stack(imgs, 0), targets

# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    def forward(self, inputs, targets):
        inputs = t.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**2 * t.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**4 * (inputs)**2 * t.log(1.0 - inputs + 1e-14)
        return center_loss + other_loss


def get_loss(pred_cls, pred_txty, pred_twth, label, num_classes):
    cls_loss_function = FocalLoss()
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')

    # 获取标记框gt    
    gt_cls = label[:, :, :num_classes].float()
    gt_txtytwth = label[:, :, num_classes:-1].float()
    gt_box_scale_weight = label[:, :, -1]

    # 中心点热力图损失L_k
    batch_size = pred_cls.size(0)
    cls_loss = t.sum(cls_loss_function(pred_cls, gt_cls)) / batch_size
        
    # 中心点偏移量损失L_off
    txty_loss = t.sum(t.sum(txty_loss_function(pred_txty, gt_txtytwth[:, :, :2]), 2) * gt_box_scale_weight) / batch_size

    # 物体尺度损失L_size
    twth_loss = t.sum(t.sum(twth_loss_function(pred_twth, gt_txtytwth[:, :, 2:]), 2) * gt_box_scale_weight) / batch_size

    # 总损失
    total_loss = cls_loss + txty_loss + twth_loss

    return total_loss