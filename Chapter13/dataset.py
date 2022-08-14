import os
import numpy as np

import torch as t
import torchvision as tv
from torch.utils.data import Dataset
import cv2
import numpy as np
import types
from pycocotools.coco import COCO

class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, data_dir='COCO', json_file='instances_train2017.json',
                 name='train2017', img_size=416,
                 transform=None, min_size=1):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            min_size (int): bounding boxes smaller than this are ignored
        """
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.min_size = min_size
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)

        # 读取图像，做预处理
        img_file = os.path.join(self.data_dir, self.name,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)
        
        assert img is not None

        height, width, channels = img.shape
        
        # 预处理，将bbox原始标注[xmin,ymin,w,h]转化为[xmin,ymin,xmax,ymax,label]
        target = []
        for anno in annotations:
            if 'bbox' in anno and anno['area'] > 0:   
                xmin = np.max((0, anno['bbox'][0]))
                ymin = np.max((0, anno['bbox'][1]))
                xmax = np.min((width - 1, xmin + np.max((0, anno['bbox'][2] - 1))))
                ymax = np.min((height - 1, ymin + np.max((0, anno['bbox'][3] - 1))))
                if xmax > xmin and ymax > ymin:
                    label_ind = anno['category_id']
                    cls_id = self.class_ids.index(label_ind)
                    xmin /= width
                    ymin /= height
                    xmax /= width
                    ymax /= height

                    target.append([xmin, ymin, xmax, ymax, cls_id]) 
            else:
                print('No bbox !!!')

        if len(target) == 0:
            target = np.zeros([1, 5])
        else:
            target = np.array(target)

        img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
        # to rgb
        img = img[:, :, (2, 1, 0)]
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return t.from_numpy(img).permute(2, 0, 1), target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 255.
        image -= self.mean
        image /= self.std

        return image, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, boxes, labels

class Augmentation(object):
    def __init__(self, size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.size = size
        self.std = std
        self.augment = Compose([
            Resize(self.size),
            Normalize(self.mean, self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)