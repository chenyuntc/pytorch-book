#coding:utf8
import os#,ipdb
import torch as t
from torch.autograd import Variable
import torchvision as tv
from torchnet import meter
import tqdm

from torch.nn.utils.rnn import pack_padded_sequence
from model import CaptionModel
from config import Config
from utils import Visualizer
from data import get_dataloader
from PIL import Image

def generate(**kwargs):
    opt = Config()
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    # 数据预处理
    data = t.load(opt.caption_data_path,map_location=lambda s,l:s)
    word2ix,ix2word = data['word2ix'],data['ix2word']

    IMAGENET_MEAN =  [0.485, 0.456, 0.406]
    IMAGENET_STD =  [0.229, 0.224, 0.225]
    normalize =  tv.transforms.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)
    transforms = tv.transforms.Compose([
                tv.transforms.Scale(opt.scale_size),
                tv.transforms.CenterCrop(opt.img_size),
                tv.transforms.ToTensor(),
                normalize
        ])
    img = Image.open(opt.test_img)
    img = transforms(img).unsqueeze(0)

    # 用resnet50来提取图片特征
    resnet50 = tv.models.resnet50(True).eval()
    del resnet50.fc
    resnet50.fc = lambda x:x
    if opt.use_gpu:
        resnet50.cuda() 
        img = img.cuda()
    img_feats = resnet50(Variable(img,volatile=True))

    # Caption模型
    model = CaptionModel(opt,word2ix,ix2word)
    model = model.load(opt.model_ckpt).eval()
    if opt.use_gpu:
         model.cuda()

    results = model.generate(img_feats.data[0])
    print('\r\n'.join(results))


def  train(**kwargs):
    opt = Config()    
    opt.caption_data_path = 'caption.pth' # 原始数据
    opt.test_img = '' # 输入图片
    #opt.model_ckpt='caption_0914_1947' # 预训练的模型

    # 数据
    vis = Visualizer(env = opt.env)
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix,ix2word = _data['word2ix'],_data['ix2word']

    # 模型
    model = CaptionModel(opt,word2ix,ix2word)
    if opt.model_ckpt:
        model.load(opt.model_ckpt)
    optimizer = model.get_optimizer(opt.lr)
    criterion = t.nn.CrossEntropyLoss()
    if opt.use_gpu:
        model.cuda()
        criterion.cuda()

    # 统计
    loss_meter = meter.AverageValueMeter()

    for epoch in range(opt.epoch):        
        loss_meter.reset()
        for ii,(imgs, (captions, lengths),indexes)  in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            optimizer.zero_grad()
            input_captions = captions[:-1]
            if opt.use_gpu:
                imgs = imgs.cuda()
                captions = captions.cuda()
            imgs = Variable(imgs)
            captions = Variable(captions)
            input_captions = captions[:-1]
            target_captions = pack_padded_sequence(captions,lengths)[0]
            score,_ = model(imgs,input_captions,lengths)
            loss = criterion(score,target_captions)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data[0])

            # 可视化
            if (ii+1)%opt.plot_every ==0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss',loss_meter.value()[0])

                # 可视化原始图片 + 可视化人工的描述语句
                raw_img = _data['ix2id'][indexes[0]]
                img_path=opt.img_path+raw_img
                raw_img = Image.open(img_path).convert('RGB')
                raw_img = tv.transforms.ToTensor()(raw_img)

                raw_caption = captions.data[:,0]
                raw_caption = ''.join([_data['ix2word'][ii] for ii in raw_caption])
                vis.text(raw_caption,u'raw_caption')
                vis.img('raw',raw_img,caption=raw_caption)

                # 可视化网络生成的描述语句
                results = model.generate(imgs.data[0])
                vis.text('</br>'.join(results),u'caption')
        model.save()

if __name__=='__main__':
    import fire
    fire.Fire()

