## PyTorch实战指南

[TOC]

在学习某个深度学习框架时，掌握其基本知识和接口固然重要，但如何合理组织代码，使得代码具有良好的可读性和可扩展性也必不可少。本文不会深入讲解过多知识性的东西，更多的则是传授一些经验，这些内容可能有些争议，因其受我个人喜好和coding风格影响较大，**你可以将这部分当成是一种参考或提议，而不是作为必须遵循的准则**。归根到底，都是希望你能以一种更为合理的方式组织自己的程序。

在做深度学习实验或项目时，为了得到最优的模型结果，中间往往需要很多次的尝试和修改。而合理的文件组织结构，以及一些小技巧可以极大地提高代码的易读易用性。根据我的个人经验，在从事大多数深度学习研究时，程序都需要实现以下几个功能：

- 模型定义
- 数据处理和加载
- 训练模型（Train&Validate）
- 训练过程的可视化
- 测试（Test/Inference）

另外程序还应该满足以下几个要求：

- 模型需具有高度可配置性，便于修改参数、修改模型，反复实验
- 代码应具有良好的组织结构，使人一目了然
- 代码应具有良好的说明，使其他人能够理解

在本文我将应用这些内容，并结合实际的例子，来讲解如何用PyTorch完成Kaggle上的经典比赛：Dogs vs. Cats[^1]。本文所有示例程序均在github上开源 https://github.com/chenyuntc/pytorch-best-practice 。

[^1]: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

### 6.1.1 比赛介绍

Dogs vs. Cats是一个传统的二分类问题，其训练集包含25000张图片，均放置在同一文件夹下，命名格式为`<category>.<num>.jpg`, 如`cat.10000.jpg`、`dog.100.jpg`，测试集包含12500张图片，命名为`<num>.jpg`，如`1000.jpg`。参赛者需根据训练集的图片训练模型，并在测试集上进行预测，输出它是狗的概率。最后提交的csv文件如下，第一列是图片的`<num>`，第二列是图片为狗的概率。

```
id,label
10001,0.889
10002,0.01
...
```

### 6.1.2 文件组织架构

前面提到过，程序主要包含以下功能：

- 模型定义
- 数据加载
- 训练和测试

首先来看程序文件的组织结构：

```
├── checkpoints/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── get_data.sh
├── models/
│   ├── __init__.py
│   ├── AlexNet.py
│   ├── BasicModule.py
│   └── ResNet34.py
└── utils/
│   ├── __init__.py
│   └── visualize.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
```

其中：

- `checkpoints/`： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练
- `data/`：数据相关操作，包括数据预处理、dataset实现等
- `models/`：模型定义，可以有多个模型，例如上面的AlexNet和ResNet34，一个模型对应一个文件
- `utils/`：可能用到的工具函数，在本次实验中主要是封装了可视化工具
- `config.py`：配置文件，所有可配置的变量都集中在此，并提供默认值
- `main.py`：主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数
- `requirements.txt`：程序依赖的第三方库
- `README.md`：提供程序的必要说明

### 6.1.3 关于`__init__.py`

可以看到，几乎每个文件夹下都有`__init__.py`，一个目录如果包含了`__init__.py` 文件，那么它就变成了一个包（package）。`__init__.py`可以为空，也可以定义包的属性和方法，但其必须存在，其它程序才能从这个目录中导入相应的模块或函数。例如在`data/`文件夹下有`__init__.py`，则在`main.py` 中就可以`from data.dataset import DogCat`。而如果在`__init__.py`中写入`from .dataset import DogCat`，则在main.py中就可以直接写为：`from data import DogCat`，或者`import data; dataset = data.DogCat`，相比于`from data.dataset import DogCat`更加便捷。

### 6.1.4 数据加载

数据的相关处理主要保存在`data/dataset.py`中。关于数据加载的相关操作，在上一章中我们已经提到过，其基本原理就是使用`Dataset`提供数据集的封装，再使用`Dataloader`实现数据并行加载。Kaggle提供的数据包括训练集和测试集，而我们在实际使用中，还需专门从训练集中取出一部分作为验证集。对于这三类数据集，其相应操作也不太一样，而如果专门写三个`Dataset`，则稍显复杂和冗余，因此这里通过加一些判断来区分。对于训练集，我们希望做一些数据增强处理，如随机裁剪、随机翻转、加噪声等，而验证集和测试集则不需要。下面看`dataset.py`的代码：

```python
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    
    def __init__(self, root, transforms=None, train=True, test=False):
        """
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        """
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)] 

        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg 
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
            
        imgs_num = len(imgs)
        
        # 划分训练、验证集，验证:训练 = 3:7
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else :
            self.imgs = imgs[int(0.7*imgs_num):]            
    
        if transforms is None:
        
            # 数据转换操作，测试验证和训练的数据转换有所区别
	        
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])

            # 测试集和验证集
            if self.test or not train: 
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ]) 
            # 训练集
            else :
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomReSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ]) 
                
        
    def __getitem__(self, index):
        """
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        """
        img_path = self.imgs[index]
        if self.test: 
             label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else: 
             label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label
    
    def __len__(self):
        """
        返回数据集中所有图片的个数
        """
        return len(self.imgs)
```

关于数据集使用的注意事项，在上一章中已经提到，将文件读取等费时操作放在`__getitem__`函数中，利用多进程加速。避免一次性将所有图片都读进内存，不仅费时也会占用较大内存，而且不易进行数据增强等操作。另外在这里，我们将训练集中的30%作为验证集，可用来检查模型的训练效果，避免过拟合。在使用时，我们可通过dataloader加载数据。

```
train_dataset = DogCat(opt.train_data_root, train=True)
trainloader = DataLoader(train_dataset,
                        batch_size = opt.batch_size,
                        shuffle = True,
                        num_workers = opt.num_workers)
                  
for ii, (data, label) in enumerate(trainloader):
	train()
```

### 6.1.5 模型定义

模型的定义主要保存在`models/`目录下，其中`BasicModule`是对`nn.Module`的简易封装，提供快速加载和保存模型的接口。

```python
class BasicModule(t.nn.Module):
    """
    封装了nn.Module，主要提供save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 模型的默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
```

在实际使用中，直接调用`model.save()`及`model.load(opt.load_path)`即可。

其它自定义模型一般继承`BasicModule`，然后实现自己的模型。其中`AlexNet.py`实现了AlexNet，`ResNet34`实现了ResNet34。在`models/__init__py`中，代码如下：

```python
from .AlexNet import AlexNet
from .ResNet34 import ResNet34
```

这样在主函数中就可以写成：

```python
from models import AlexNet
或
import models
model = models.AlexNet()
或
import models
model = getattr('models', 'AlexNet')()
```

其中最后一种写法最为关键，这意味着我们可以通过字符串直接指定使用的模型，而不必使用判断语句，也不必在每次新增加模型后都修改代码。新增模型后只需要在`models/__init__.py`中加上`from .new_module import new_module`即可。

其它关于模型定义的注意事项，在上一章中已详细讲解，这里就不再赘述，总结起来就是：

- 尽量使用`nn.Sequential`（比如AlexNet）
- 将经常使用的结构封装成子Module（比如GoogLeNet的Inception结构，ResNet的Residual Block结构）
- 将重复且有规律性的结构，用函数生成（比如VGG的多种变体，ResNet多种变体都是由多个重复卷积层组成）

### 6.1.6 工具函数

在项目中，我们可能会用到一些helper方法，这些方法可以统一放在`utils/`文件夹下，需要使用时再引入。在本例中主要是封装了可视化工具visdom的一些操作，其代码如下，在本次实验中只会用到`plot`方法，用来统计损失信息。

```python
#coding:utf8
import visdom
import time
import numpy as np

class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如 
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # 画的第几个数，相当于横坐标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)

        !!! don't ~~self.img('input_imgs', t.Tensor(100, 64, 64), nrows=10)~~ !!!
        """
        self.vis.images(img_.cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs
                       )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'),\
                            info=info)) 
        self.vis.text(self.log_text, win)   

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        """
        return getattr(self.vis, name)
```

### 6.1.6 配置文件

在模型定义、数据处理和训练等过程都有很多变量，这些变量应提供默认值，并统一放置在配置文件中，这样在后期调试、修改代码或迁移程序时会比较方便，在这里我们将所有可配置项放在`config.py`中。

```python
class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'AlexNet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = './data/train/' # 训练集存放路径
    test_data_root = './data/test1' # 测试集存放路径
    load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128 # batch size
    use_gpu = True # use GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 10
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
```

可配置的参数主要包括：

- 数据集参数（文件路径、batch_size等）
- 训练参数（学习率、训练epoch等）
- 模型参数

这样我们在程序中就可以这样使用：

```
import models
from config import DefaultConfig

opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root)
```

这些都只是默认参数，在这里还提供了更新函数，根据字典更新配置参数。

```
def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                # 警告还是报错，取决于你个人的喜好
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self, k, v)
            
        # 打印配置信息	
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
```

这样我们在实际使用时，并不需要每次都修改`config.py`，只需要通过命令行传入所需参数，覆盖默认配置即可。

例如：

```
opt = DefaultConfig()
new_config = {'lr':0.1,'use_gpu':False}
opt.parse(new_config)
opt.lr == 0.1
```

### 6.1.7 main.py

在讲解主程序`main.py`之前，我们先来看看2017年3月谷歌开源的一个命令行工具`fire`[^3] ，通过`pip install fire`即可安装。下面来看看`fire`的基础用法，假设`example.py`文件内容如下：

```python
import fire

def add(x, y):
  return x + y
  
def mul(**kwargs):
    a = kwargs['a']
    b = kwargs['b']
    return a * b

if __name__ == '__main__':
  fire.Fire()
```

那么我们可以使用：

```bash
python example.py add 1 2 # 执行add(1, 2)
python example.py mul --a=1 --b=2 # 执行mul(a=1, b=2), kwargs={'a':1, 'b':2}
python example.py add --x=1 --y==2 # 执行add(x=1, y=2)
```

可见，只要在程序中运行`fire.Fire()`，即可使用命令行参数`python file <function> [args,] {--kwargs,}`。fire还支持更多的高级功能，具体请参考官方指南[^4] 。

[^3]: https://github.com/google/python-fire
[^4]: https://github.com/google/python-fire/blob/master/doc/guide.md

在主程序`main.py`中，主要包含四个函数，其中三个需要命令行执行，`main.py`的代码组织结构如下：

```python
def train(**kwargs):
    """
    训练
    """
    pass
	 
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    """
    pass

def test(**kwargs):
    """
    测试（inference）
    """
    pass

def help():
    """
    打印帮助的信息 
    """
    print('help')

if __name__=='__main__':
    import fire
    fire.Fire()
```

根据fire的使用方法，可通过`python main.py <function> --args=xx`的方式来执行训练或者测试。

#### 训练

训练的主要步骤如下：

- 定义网络
- 定义数据
- 定义损失函数和优化器
- 计算重要指标
- 开始训练
  - 训练网络
  - 可视化各种指标
  - 计算在验证集上的指标

训练函数的代码如下：

```python
def train(**kwargs):
    
    # 根据命令行参数更新配置
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    
    # step1: 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: 数据
    train_data = DogCat(opt.train_data_root,train=True)
    val_data = DogCat(opt.train_data_root,train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,
                        num_workers=opt.num_workers)
    
    # step3: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),
                            lr = lr,
                            weight_decay = opt.weight_decay)
        
    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):
        
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in enumerate(train_dataloader):

            # 训练模型参数 
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()
            
            # 更新统计指标以及可视化
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data, target.data)

            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss', loss_meter.value()[0])
                
                # 如果需要的话，进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        model.save()

        # 计算验证集上的指标及可视化
        val_cm,val_accuracy = val(model,val_dataloader)
        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
        .format(
                    epoch = epoch,
                    loss = loss_meter.value()[0],
                    val_cm = str(val_cm.value()),
                    train_cm=str(confusion_matrix.value()),
                    lr=lr))
        
        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        previous_loss = loss_meter.value()[0]
```

这里用到了PyTorchNet[^5]里面的一个工具: meter。meter提供了一些轻量级的工具，用于帮助用户快速统计训练过程中的一些指标。`AverageValueMeter`能够计算所有数的平均值和标准差，这里用来统计一个epoch中损失的平均值。`confusionmeter`用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标。例如对于表格6-1，共有50张狗的图片，其中有35张被正确分类成了狗，还有15张被误判成猫；共有100张猫的图片，其中有91张被正确判为了猫，剩下9张被误判成狗。相比于准确率等统计信息，混淆矩阵更能体现分类的结果，尤其是在样本比例不均衡的情况下。

表6-1 混淆矩阵

| 样本   | 判为狗  | 判为猫  |
| ---- | ---- | ---- |
| 实际是狗 | 35   | 15   |
| 实际是猫 | 9    | 91   |

PyTorchNet从TorchNet[^6]迁移而来，提供了很多有用的工具，但其目前开发和文档都还不是很完善，本书不做过多的讲解。

[^5]: https://github.com/pytorch/tnt
[^6]: https://github.com/torchnet/torchnet

#### 验证

验证相对来说比较简单，但要注意需将模型置于验证模式(`model.eval()`)，验证完成后还需要将其置回为训练模式(`model.train()`)，这两句代码会影响`BatchNorm`和`Dropout`等层的运行模式。验证模型准确率的代码如下。

```python
def val(model,dataloader):
    """
    计算模型在验证集上的准确率等信息
    """

    # 把模型设为验证模式
    model.eval()
    
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in enumerate(dataloader):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.long(), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.long())

    # 把模型恢复为训练模式
    model.train()
    
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) /\
			     (cm_value.sum())
    return confusion_matrix, accuracy
```

#### 测试

测试时，需要计算每个样本属于狗的概率，并将结果保存成csv文件。测试的代码与验证比较相似，但需要自己加载模型和数据。

```python
def test(**kwargs):
    opt.parse(kwargs)
    
    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 数据
    train_data = DogCat(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,\
							    batch_size=opt.batch_size,\
							    shuffle=False,\
							    num_workers=opt.num_workers)
    
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax\
	        (score)[:,1].data.tolist()      
        batch_results = [(path_,probability_) \
	        for path_,probability_ in zip(path,probability) ]
        results += batch_results
    write_csv(results,opt.result_file)
    return results
```

#### 帮助函数

为了方便他人使用, 程序中还应当提供一个帮助函数，用于说明函数是如何使用。程序的命令行接口中有众多参数，如果手动用字符串表示不仅复杂，而且后期修改config文件时，还需要修改对应的帮助信息，十分不便。这里使用了Python标准库中的inspect方法，可以自动获取config的源代码。help的代码如下:

```python
def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)
```

当用户执行`python main.py help`的时候，会打印如下帮助信息：

```bash
    usage : python main.py <function> [--args=value,]
    <function> := train | test | help
    example: 
            python main.py train --env='env0701' --lr=0.01
            python main.py test --dataset='path/to/dataset/'
            python main.py help
    avaiable args:
class DefaultConfig(object):
    env = 'default' # visdom 环境
    model = 'AlexNet' # 使用的模型
    
    train_data_root = './data/train/' # 训练集存放路径
    test_data_root = './data/test1' # 测试集存放路径
    load_model_path = 'checkpoints/model.pth' # 加载预训练的模型

    batch_size = 128 # batch size
    use_gpu = True # user GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' 
    result_file = 'result.csv' # 结果文件
      
    max_epoch = 10
    lr = 0.1 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # 损失函数
```

### 6.1.8 使用

正如`help`函数的打印信息所述，可以通过命令行参数指定变量名.下面是三个使用例子，fire会将包含`-`的命令行参数自动转层下划线`_`，也会将非数值的值转成字符串。所以`--train-data-root=data/train`和`--train_data_root='data/train'`是等价的。

```
# 训练模型
python main.py train 
        --train-data-root=data/train/ 
        --lr=0.005 
        --batch-size=32 
        --model='ResNet34'  
        --max-epoch = 20

# 测试模型
python main.py test
       --test-data-root=data/test1 
       --load-model-path='checkpoints/resnet34_00:23:05.pth' 
       --batch-size=128 
       --model='ResNet34' 
       --num-workers=12

# 打印帮助信息
python main.py help
```

### 6.1.9 争议

以上的程序设计规范带有作者强烈的个人喜好，并不想作为一个标准，而是作为一个提议和一种参考。上述设计在很多地方还有待商榷，例如对于训练过程是否应该封装成一个`trainer`对象，或者直接封装到`BaiscModule`的`train`方法之中。对命令行参数的处理也有不少值得讨论之处。因此不要将本文中的观点作为一个必须遵守的规范，而应该看作一个参考。

本章中的设计可能会引起不少争议，其中比较值得商榷的部分主要有以下两个方面：

- 命令行参数的设置。目前大多数程序都是使用Python标准库中的`argparse`来处理命令行参数，也有些使用比较轻量级的`click`。这种处理相对来说对命令行的支持更完备，但根据作者的经验来看，这种做法不够直观，并且代码量相对来说也较多。比如`argparse`，每次增加一个命令行参数，都必须写如下代码：

```python
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
```

在读者眼中，这种实现方式远不如一个专门的`config.py`来的直观和易用。尤其是对于使用Jupyter notebook或IPython等交互式调试的用户来说，`argparse`较难使用。

- 模型训练。有不少人喜欢将模型的训练过程集成于模型的定义之中，代码结构如下所示：

```python
  class MyModel(nn.Module):
  	
      def __init__(self,opt):
          self.dataloader = Dataloader(opt)
          self.optimizer  = optim.Adam(self.parameters(),lr=0.001)
          self.lr = opt.lr
          self.model = make_model()
      
      def forward(self,input):
          pass
      
      def train_(self):
          # 训练模型
          for epoch in range(opt.max_epoch)
          	for ii,data in enumerate(self.dataloader):
              	train_epoch()
              
          	model.save()
  	
      def train_epoch(self):
          pass
```

抑或是专门设计一个`Trainer`对象，形如：

```python
    """
  code simplified from:
  https://github.com/pytorch/pytorch/blob/master/torch/utils/trainer/trainer.py
  """
  import heapq
  from torch.autograd import Variable

  class Trainer(object):

      def __init__(self, model=None, criterion=None, optimizer=None, dataset=None):
          self.model = model
          self.criterion = criterion
          self.optimizer = optimizer
          self.dataset = dataset
          self.iterations = 0

      def run(self, epochs=1):
          for i in range(1, epochs + 1):
              self.train()

      def train(self):
          for i, data in enumerate(self.dataset, self.iterations + 1):
              batch_input, batch_target = data
              self.call_plugins('batch', i, batch_input, batch_target)
              input_var = Variable(batch_input)
              target_var = Variable(batch_target)
    
              plugin_data = [None, None]
    
              def closure():
                  batch_output = self.model(input_var)
                  loss = self.criterion(batch_output, target_var)
                  loss.backward()
                  if plugin_data[0] is None:
                      plugin_data[0] = batch_output.data
                      plugin_data[1] = loss.data
                  return loss
    
              self.optimizer.zero_grad()
              self.optimizer.step(closure)
    
          self.iterations += i
```

还有一些人喜欢模仿keras和scikit-learn的设计，设计一个`fit`接口。对读者来说，这些处理方式很难说哪个更好或更差，找到最适合自己的方法才是最好的。

`BasicModule` 的封装，可多可少。训练过程中的很多操作都可以移到`BasicModule`之中，比如`get_optimizer`方法用来获取优化器，比如`train_step`用来执行单歩训练。对于不同的模型，如果对应的优化器定义不一样，或者是训练方法不一样，可以复写这些函数自定义相应的方法，取决于自己的喜好和项目的实际需求。