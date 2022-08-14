# coding:utf8
import os
import ipdb
import torch as t
import torchvision as tv
import tqdm
from model import Model
from torch.utils.data import DataLoader
from dataset import COCODataset,Augmentation
from utils import *
from torchnet.meter import AverageValueMeter
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

class_color = [(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for _ in range(80)]

class Config(object):
    data_path = '/mnt/sda1/COCO/'  # 数据集存放路径
    num_workers = 4  # 多进程加载数据所用的进程数
    batch_size = 32
    max_epoch = 100
    num_classes = 80
    topk = 100
    lr = 1e-3  # 学习率
    gpu = True  # 是否使用GPU
    model_path = None

    vis = True  # 是否使用visdom可视化
    env = 'CenterNet'  # visdom的env
    plot_every = 20  # 每间隔20 batch，visdom画图一次

    debug_file = '/tmp/debugcenternet'  # 存在该文件则进入debug模式
    save_every = 10  # 每10个epoch保存一次模型

    # 测试时所用参数
    test_img_path = 'test_img/'  # 待测试图片保存路径
    test_save_path = 'test_result/'

opt = Config()


def train(**kwargs):
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device=t.device('cuda') if opt.gpu else t.device('cpu')
    if opt.vis:
        from visualize import Visualizer
        vis = Visualizer(opt.env)
    
    dataset = COCODataset(data_dir=opt.data_path,img_size=512,transform=Augmentation())
    dataloader = DataLoader(dataset, opt.batch_size,shuffle=True,collate_fn=detection_collate,num_workers=opt.num_workers)

    # 网络
    model = Model(num_classes=opt.num_classes,topk=opt.topk)
    map_location = lambda storage, loc: storage
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location=map_location))
    model.to(device)


    # 定义优化器和损失
    optimizer = t.optim.Adam(model.parameters(), opt.lr)

    loss_meter = AverageValueMeter()

    for epoch in range(opt.max_epoch):
        loss_meter.reset()

        for ii,(image,target) in tqdm.tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            image = image.to(device)
            target = [label.tolist() for label in target]

            # 生成可用的标注信息
            target = gt_creator(512,4,opt.num_classes,target)
            target = t.tensor(target).float().to(device)
            total_loss = model(image,target)
            total_loss.backward()
            optimizer.step()
            loss_meter.add(total_loss.item())
            
            if (ii+1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                vis.plot('total_loss',loss_meter.value()[0])
        
        vis.save([opt.env])
        if (epoch+1)%opt.save_every == 0:
            t.save(model.state_dict(),'checkpoints/centernet_%s.pth'% str(epoch+1))

@t.no_grad()
def test(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device=t.device('cuda') if opt.gpu else t.device('cpu')
    
    # 加载模型
    model = Model(num_classes=opt.num_classes,topk=opt.topk).eval()
    model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    model.to(device)
    transform = Augmentation()

    for index,file in enumerate(os.listdir(opt.test_img_path)):
        img = cv2.imread(opt.test_img_path+'/'+file,cv2.IMREAD_COLOR)
        x = t.from_numpy(transform(img,boxes=None,labels=None)[0][:,:,(2,1,0)]).permute(2,0,1)
        x = x.unsqueeze(0).to(device)
        bbox_pred,score,cls_ind = model.generate(x)
        bbox_pred = bbox_pred * np.array([[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]])
        for i,box in enumerate(bbox_pred):
            if score[i]>0.35:
                cls_indx = cls_ind[i]
                cls_id = coco_class_index[int(cls_indx)]
                cls_name = coco_class_labels[cls_id]
                label = '%s:%.3f' % (cls_name,score[i])
                xmin,ymin,xmax,ymax = box
                cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),class_color[int(cls_indx)],2)
                cv2.rectangle(img,(int(xmin),int(abs(ymin)-15)),(int(xmin+int(xmax-xmin)*0.55),int(ymin)),class_color[int(cls_indx)],-1)
                cv2.putText(img,label,(int(xmin),int(ymin)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        cv2.imwrite(os.path.join(opt.test_save_path,str(index).zfill(3)+'.jpg'),img)


if __name__ == '__main__':
    import fire
    fire.Fire()
