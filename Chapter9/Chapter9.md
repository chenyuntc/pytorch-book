# 第9章 PyTorch实战指南

通过前面几章的学习，读者已经掌握了PyTorch的基础知识和部分高级扩展。本章将结合之前所讲的内容，带领读者从头实现一个完整的深度学习项目。本章的重点不在于如何使用PyTorch的接口，而在于如何合理地设计程序的结构，使得程序更具有可读性。

## 9.1 编程实战：猫和狗二分类

在学习某个深度学习框架时，掌握其基本知识和接口固然重要，但如何合理地组织代码，使代码具有良好的可读性和可扩展性也很关键。本章将不再深入讲解过多知识性内容，而是更多地传授一些经验。这些内容可能有些争议，因为这些受笔者个人喜好和代码风格影响较大，所以读者可以将这部分当成是一种参考或提议，而不是作为必须遵循的准则。归根到底，笔者都是希望读者能以一种更为合理的方式组织自己的程序。

在做深度学习实验或项目时，为了得到最优的模型结果，往往需要很多次的尝试和修改。合理的文件组织结构，以及一些小技巧可以极大地提高代码的易读性和易用性。根据经验，在从事大多数深度学习研究时，程序都需要实现以下几个功能。

- 模型定义。
- 数据处理和加载。
- 训练模型（Train&Validate）。
- 训练过程的可视化。
- 测试（Test/Inference）。

另外，程序还应该满足以下几个要求。

- 模型需具有高度可配置性，便于修改参数、修改模型和反复实验。
- 代码应具有良好的组织结构，使人一目了然。
- 代码应具有良好的说明，使其他人能够理解。

之前的章节已经讲解了PyTorch的大部分内容。本章将应用其中最基础、最常见的内容，并结合实例讲解如何使用PyTorch完成Kaggle上的经典比赛：“Dogs vs. Cats”。本章所有示例程序均在本书的配套代码`Chapter9`中。


### 9.1.1 比赛介绍

“Dogs vs. Cats”是一个传统的二分类问题，其训练集包含25000张图片，部分图片如图9-1所示。这些图片均放置在同一文件夹下，命名格式为`<category>.<num>.jpg`。例如，`cat.10000.jpg`、`dog.100.jpg`。测试集包含12500张图片，命名为`<num>.jpg`。例如，`1000.jpg`。参赛者需根据训练集的图片训练模型，并在测试集上进行预测，输出它是狗的概率。最后提交的csv文件如下，第一列是图片的`<num>`，第二列是图片为狗的概率。

```
id,label
10001,0.889
10002,0.01
...
```

![图9-1  猫和狗的数据](imgs/dogsvscats.png)

### 9.1.2 文件组织架构

首先，整个程序文件的组织结构如下所示：

```
├── checkpoints/
├── data/
    ├── __init__.py
    └── dataset.py
├── models/
    ├── __init__.py
    ├── squeezenet.py
    ├── BasicModule.py
    └── resnet34.py
├── utils/
    ├── __init__.py
    └── visualize.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

- `checkpoints/`： 用于保存训练好的模型，使得程序在异常退出后仍能重新载入模型，恢复训练。
- `data/`：数据相关操作，包括数据预处理、dataset实现等。
- `models/`：模型定义，可以有多个模型。例如，SqueezeNet和ResNet34，每一个模型对应一个文件。
- `utils/`：可能用到的工具函数，本次实验中主要封装了可视化工具。
- `config.py`：配置文件，所有可配置的变量都集中在此，并提供默认值。
- `main.py`：主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数。
- `requirements.txt`：程序依赖的第三方库。
- `README.md`：提供程序的必要说明。

### 9.1.3 `__init__.py`

从整个程序文件的组织结构中可以看到，几乎每个文件夹下都有`__init__.py`。一个目录如果包含了`__init__.py` 文件，那么它就变成了一个包（package）。`__init__.py`可以为空，也可以定义包的属性和方法，但它必须存在。只有该目录下存在`__init__.py`文件，其它程序才能从这个目录中导入相应的模块或函数。例如，在`data/`文件夹下有`__init__.py`，在`main.py` 中可以使用命令：`from data.dataset import DogCat`。如果在`__init__.py`中写入`from .dataset import DogCat`，那么在main.py中就可以直接写为：`from data import DogCat`，或者`import data; dataset = data.DogCat`。

### 9.1.4 数据加载

数据处理的相关操作保存在文件`data/dataset.py`中。数据加载的相关操作在第5章中已经提及，其基本原理就是使用`Dataset`封装数据集，然后使用`Dataloader`实现数据的并行加载。Kaggle提供的数据只有训练集和测试集，然而在实际使用中，还需要专门从训练集中取出一部分作为验证集。对于这三种数据集，其相应操作也不太一样，如果专门编写三个`Dataset`，那么会显得复杂和冗余，因此可以加一些判断进行区分。同时，在训练集需要进行一些数据增强处理，例如随机裁剪、随机翻转、加噪声等，在验证集和测试集不做任何处理。下面看`dataset.py`的代码：

```python
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    
    def __init__(self, root, transforms=None, mode=None):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试划分数据
        mode ∈ ["train", "test", "val"]
        '''
        assert mode in ["train", "test", "val"]
        self.mode = mode
        imgs = [os.path.join(root, img) for img in os.listdir(root)] 

        if self.mode == "test":
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))
            
        imgs_num = len(imgs)
        
        # 划分训练、验证集，验证:训练 = 3:7
        if self.mode == "test": self.imgs = imgs
        if self.mode == "train": self.imgs = imgs[:int(0.7 * imgs_num)]
        if self.mode == "val": self.imgs = imgs[int(0.7 * imgs_num):]    
    
        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])

            # 测试集和验证集 不需要数据增强
            if self.mode == "test" or self.mode == "val":
                self.transforms = T.Compose([
                    T.Scale(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ]) 
            # 训练集 需要数据增强
            else :
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ]) 
                
    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集，返回图片id，如1000.jpg返回1000
        '''
        img_path = self.imgs[index]
        if self.mode == "test":
             label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else: 
             label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label
    
    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)
```

在第5章中提到，可以将文件读取等费时操作放在`__getitem__`函数中，利用多进程加速。这里将30%的训练集作为验证集，用来检查模型的训练效果，从而避免过拟合。定义完`Dataset`，就可以使用`Dataloader`加载数据了：

```python
train_dataset = DogCat(opt.train_data_root, mode="train")
trainloader = DataLoader(train_dataset,
                        batch_size = opt.batch_size,
                        shuffle = True,
                        num_workers = opt.num_workers)
                  
for ii, (data, label) in enumerate(trainloader):
	train()
```

如果有些时候数据过于复杂，那么可能需要进行一些数据清洗或者预处理。建议这部分操作专门使用脚本进行处理，而不是放在`Dataset`里面。

### 9.1.5 模型定义

模型的定义主要保存在`models/`目录下，其中`BasicModule`是对`nn.Module`的简易封装，提供快速加载和保存模型的接口。代码如下：

```python
class BasicModule(t.nn.Module):
    '''
    封装了nn.Module，主要提供save和load两个方法
    '''

    def __init__(self):
        super().__init__()
        self.model_name = str(type(self)) # 模型的默认名字

    def load(self, path):
        '''
        可加载指定路径的模型
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名，
        如SqueezeNet_0710_23:57:29.pth
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
```

在实际使用中，直接调用`model.save()`和`model.load(model_path)`即可保存、加载模型。

其它的自定义模型可以继承`BasicModule`，然后实现模型的具体细节。其中`squeezenet.py`实现了SqueezeNet，`resnet34.py`实现了ResNet34。在`models/__init__py`中，代码如下：

```python
from .squeezenet import SqueezeNet
from .resnet34 import ResNet34
```

在主函数中可以写成：

```python
from models import SqueezeNet
或
import models
model = models.SqueezeNet()
或
import models
model = getattr('models', 'SqueezeNet')()
```

其中，最后一种写法最重要，这意味着可以通过字符串直接指定使用的模型，而不必使用判断语句，也不必在每次新增加模型后都修改代码，只需要在`models/__init__.py`中加上`from .new_module import new_module`即可。

其它关于模型定义的注意事项在第4章中已经做了详细讲解，本节就不再赘述，总结如下。

- 尽量使用`nn.Sequential`。

- 将经常使用的结构封装成子module。

- 将重复且有规律性的结构，用函数生成。

### 9.1.6 工具函数

在实际项目中可能会用到一些helper方法，这些方法统一放在`utils/`文件夹下，在需要使用时再引入。下面这个示例封装了可视化工具visdom的一些操作，本次实验只会用到`plot`方法，用来统计损失信息。

```python
import visdom
import time
import numpy as np

class Visualizer(object):
    '''
    封装了visdom的基本操作，但是仍然可以通过`self.vis.function`
    或者`self.function`调用原生的visdom接口
    比如 
    self.text('hello visdom')
    self.histogram(t.randn(1000))
    self.line(t.arange(0, 10),t.arange(1, 11))
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        
        # 画的第几个数，相当于横坐标
        # 保存（'loss',23） 即loss的第23个点
        self.index = {} 
        self.log_text = ''
    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=(name), 
                      opts=dict(title=name), 
                      update=None if x == 0 else 'append',
                      **kwargs )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_img', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.vis.images(img_.cpu().numpy(), win=(name), opts=dict(title=name), **kwargs )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1, 'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(
                            time=time.strftime('%m%d_%H%M%S'), info=info)) 
        self.vis.text(self.log_text, win)   

    def __getattr__(self, name):
        '''
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        '''
        return getattr(self.vis, name)
```

### 9.1.7 配置文件

在模型定义、数据处理和训练等过程中均存在很多变量，应该为这些变量提供默认值，然后统一放置在配置文件中。在后期调试、修改代码或迁移程序时这种做法会比较方便，这里将所有可配置项放在文件`config.py`中。

```python
class DefaultConfig(object):
    env = 'default' 	# visdom 环境
    model = 'SqueezeNet'# 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = './data/train/' 	# 训练集存放路径
    test_data_root = './data/test' 		# 测试集存放路径
    load_model_path = None # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128 	# batch_size的大小
    use_gpu = True 		# 是否使用GPU加速
    num_workers = 4 	# 加载数据时的进程数
    print_freq = 20 	# 打印信息的间隔轮数

    debug_file = '/tmp/debug' 
    result_file = 'result.csv'
      
    max_epoch = 10 	# 训练轮数
    lr = 0.1 		# 初始化学习率
    lr_decay = 0.95 # 学习率衰减, lr = lr×lr_decay
    weight_decay = 1e-4
```

可配置的参数主要包括以下几个方面。

- 数据集参数（文件路径、batch_size等）。
- 训练参数（学习率、训练epoch等）。
- 模型参数。

在程序中，可以使用如下方式使用相应的参数：

```python
import models
from config import DefaultConfig

opt = DefaultConfig()
lr = opt.lr
model = getattr(models, opt.model)
dataset = DogCat(opt.train_data_root, mode="train")
```

以上都只是默认参数，这里还可以使用更新函数，例如根据字典更新配置参数：

```python
def parse(self, kwargs):
    '''
    根据字典kwargs 更新 config参数
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    # 打印配置信息	
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))
```

在实际使用时不需要每次都修改`config.py`文件，只需要通过命令行传入所需参数将默认配置覆盖即可，例如：

```python
opt = DefaultConfig()
new_config = {'lr':0.1, 'use_gpu':False}
opt.parse(new_config)
```

### 9.1.8 main.py

在讲解主程序`main.py`之前，先来看看2017年3月谷歌开源的一个命令行工具`fire` ，读者可通过`pip install fire`安装该工具。下面对`fire`的基础用法进行介绍，`example.py`文件内容如下：

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

```bash
python example.py add 1 2 # 执行add(1, 2)
python example.py mul --a=1 --b=2 # 执行mul(a=1, b=2), kwargs={'a':1, 'b':2}
python example.py add --x=1 --y=2 # 执行add(x=1, y=2)
```

从上面的代码可以看出，只要在程序中运行`fire.Fire()`，即可使用命令行参数`python file <function> [args,] {--kwargs,}`。`fire`还支持更多的高级功能，具体请参考官方文档。

在主程序`main.py`中，主要包含四个函数，其中三个需要命令行执行，`main.py`的代码组织结构如下：

```python
def train(**kwargs):
    '''
    训练
    '''
    pass
	 
def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    '''
    pass

def test(**kwargs):
    '''
    测试（inference）
    '''
    pass

def help():
    '''
    打印帮助的信息 
    '''
    print('help')

if __name__=='__main__':
    import fire
    fire.Fire()
```

根据`fire`的使用方法，可通过`python main.py <function> --args=xx`的方式来执行训练或者测试。

#### 训练

训练的主要步骤如下。

（1）定义网络模型。

（2）数据预处理，加载数据。

（3）定义损失函数和优化器。

（4）计算重要指标。

（5）开始训练：训练网络；可视化各种指标；计算在验证集上的指标。

在本章的场景下，训练函数的代码如下：

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
    train_data = DogCat(opt.train_data_root, mode="train")
    val_data = DogCat(opt.train_data_root, mode="val")
    train_dataloader = DataLoader(train_data, opt.batch_size, 
                                  shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, 
                                shuffle=False, num_workers=opt.num_workers)
    
    # step3: 目标函数和优化器
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr = lr, 
                             weight_decay = opt.weight_decay)
        
    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_dataloader):

            # 训练模型参数 
            input = data.to(opt.device)
            target = label.to(opt.device)
            
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            
            # 更新统计指标以及可视化
            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())

            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                
                # 如果需要的话，进入调试模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()

        model.save()

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch}, lr:{lr}, loss:{loss}, train_cm:{train_cm}, val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), 
            train_cm=str(confusion_matrix.value()), lr=lr) )
        
        # 如果损失不再下降，则降低学习率
        if loss_meter.value()[0] > previous_loss:          
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        previous_loss = loss_meter.value()[0]
```

上面的示例用到了PyTorchNet里面的一个工具：`meter`。`meter`提供了一些轻量级的工具，帮助用户快速统计训练过程中的一些指标。`AverageValueMeter`能够计算所有数的平均值和标准差，统计一个epoch中损失的平均值。`confusionmeter`用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标。如表9-1所示，狗的图片共有50张，其中有35张被正确分类成了狗，还有15张被误判成猫；猫的图片共有100张，其中有91张被正确判为了猫，剩下9张被误判成狗。相比于准确率等统计信息，在样本比例不均衡的情况下，混淆矩阵更能体现分类的结果。

:猫和狗的数据

| 样本     | 判为狗 | 判为猫 |
| -------- | ------ | ------ |
| 实际是狗 | 35     | 15     |
| 实际是猫 | 9      | 91     |

PyTorchNet从TorchNet迁移而来，其中提供了很多有用的工具，感兴趣的读者可以自行查阅相关资料。

#### 验证

验证相对来说比较简单，**注意需要将模型置于验证模式**：`model.eval()`，在验证完成后还需要将其置回为训练模式：`model.train()`。这两句代码会影响`BatchNorm`、`Dropout`等层的运行模式。验证模型准确率的代码如下：

```python
def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    # 把模型设为验证模式
    model.eval()
    
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in enumerate(dataloader):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.long())

    # 把模型恢复为训练模式
    model.train()
    
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy
```

#### 测试

在做测试时，需要计算每个样本属于狗的概率，并将结果保存成csv文件。虽然测试代码与验证代码比较相似，但是需要重新加载模型和数据。

```python
def test(**kwargs):
    opt.parse(kwargs)
    
    # 模型加载
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # 数据加载
    test_data = DogCat(opt.test_data_root, mode="test")
    test_dataloader = DataLoader(train_data, batch_size=opt.batch_size,
							    shuffle=False, num_workers=opt.num_workers)
    
    results = []
    for ii, (data, path) in enumerate(test_dataloader):
        input = data.to(opt.device)
        score = model(input)
        # 计算每个样本属于狗的概率
        probability = t.nn.functional.softmax(score)[:, 1].data.tolist()      
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results
    write_csv(results, opt.result_file)
    return results
```

#### 帮助函数

为了方便他人使用，程序中还应当提供一个帮助函数，用于说明函数是如何使用的。程序的命令行接口中有众多参数，手动用字符串表示不仅复杂，而且后期修改配置文件时还需要修改对应的帮助信息。在这里，笔者推荐使用Python标准库中的`inspect`方法，它可以自动获取config的源代码。帮助函数代码如下:

```python
def help():
    '''
    打印帮助的信息： python file.py help
    '''
    print('''
    usage : python {0} <function> [--args=value,]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

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
    env = 'default' 	# visdom 环境
    model = 'SqueezeNet'# 使用的模型，名字必须与models/__init__.py中的名字一致
    
    train_data_root = './data/train/' 	# 训练集存放路径
    test_data_root = './data/test' 		# 测试集存放路径
    load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128 	# batch_size的大小
    use_gpu = True 		# 是否使用GPU加速
    num_workers = 4 	# 加载数据时的进程数
    print_freq = 20 	# 打印信息的间隔轮数

    debug_file = '/tmp/debug' 
    result_file = 'result.csv'
      
    max_epoch = 10 	# 训练轮数
    lr = 0.1 		# 初始化学习率
    lr_decay = 0.95 # 学习率衰减, lr = lr×lr_decay
    weight_decay = 1e-4
```

### 9.1.9 使用

正如`help`函数的打印信息所述，可以通过命令行参数指定变量名。下面是三个示例，`fire`会将包含`-`的命令行参数自动转层下划线`_`，也会将非数字的值转成字符串，因此`--train-data-root=data/train`和`--train_data_root='data/train'`是等价的。

```bash
# 训练模型
python main.py train 
        --train-data-root=data/train/ 
        --load-model-path='checkpoints/resnet34_16:53:00.pth' 
        --lr=0.005 
        --batch-size=32
        --model='ResNet34'  
        --max-epoch=20

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

###  9.1.10 争议 

以上的程序设计规范带有笔者强烈的个人喜好，并不能作为一个标准。读者无需将本章中的观点作为一个必须遵守的规范，仅作为一个参考即可。

本章中的设计可能会引起不少争议，其中比较值得商榷的部分主要有以下两个方面。

- 命令行参数的设置。目前大多数程序都是使用Python标准库中的`argparse`来处理命令行参数，也有些使用轻量级的`click`。这种处理对命令行的支持更完备，但根据笔者的经验来看，这种做法不够直观，并且代码量相对来说也较多。比如`argparse`，每次增加一个命令行参数，都必须写如下代码：

```python
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
```

在读者眼中，这种实现方式远不如一个专门的`config.py`来的直观和易用。尤其是对于使用Jupyter notebook或IPython等交互式调试的用户来说，`argparse`较难使用。

- 模型训练。不少人喜欢将模型的训练过程集成于模型的定义之中，代码结构如下所示：

```python
class MyModel(nn.Module):

    def __init__(self, opt):
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

亦或是专门设计一个`Trainer`对象，大概结构如下：

```python
import heapq

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
            input_var = batch_input.cuda()
            target_var = batch_target.cuda()
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

还有一些人喜欢模仿`Keras`和`scikit-learn`的设计，设计一个`fit`接口。对读者来说，这些处理方式很难说哪个更好或更差，找到最适合自己的方法才是最好的。


## 9.2 PyTorch 调试指南

### 9.2.1 ipdb 介绍

很多初学者使用`print`或是`log`来调试程序，这在小规模程序下很方便。更好的调试方法应是一边运行程序，一边检查里面的变量和函数。`pdb`是一个交互式的调试工具，集成于Python标准库之中，凭借其强大的功能，`pdb`广泛应用于Python环境中。`pdb`能够根据用户的需求跳转到任意的Python代码断点、查看任意变量、单歩执行代码，甚至还能修改变量的值，不必重启程序。`ipdb`是一个增强版的`pdb`，读者可通过命令`pip install ipdb`安装。`ipdb`提供了调试模式下的代码自动补全，还具有更好的语法高亮、代码溯源和更好的内省功能。同时，它与`pdb`接口完全兼容。

本书第2章中曾粗略地提到过`ipdb`的基本使用，本节将继续介绍如何结合PyTorch和`ipdb`进行调试。首先看一个例子，如果需要使用`ipdb`，那么只用在想要进行调试的地方插入`ipdb.set_trace()`，当代码运行到此处时，就会自动进入交互式调试模式。

```python
try:
    import ipdb
except:
    import pdb as ipdb

def sum(x):
    r = 0
    for ii in x:
        r += ii
    return r

def mul(x):
    r = 1
    for ii in x:
        r *= ii
    return r

ipdb.set_trace()
x = [1, 2, 3, 4, 5]
r = sum(x)
r = mul(x)
```

当程序运行至`ipdb.set_trace()`，会自动进入调试模式，在该模式中，用户可以使用调试命令。例如`next`或缩写`n`单歩执行，查看Python变量，运行Python代码等等。如果Python变量名和调试命令冲突，那么需要在变量名前加`!`，这样`ipdb`即会执行对应的Python命令，而不是调试命令。下面举例说明`ipdb`的调试，重点讲解ipdb的两大功能。

- 查看：在函数调用堆栈中自由跳动，并查看函数的局部变量。

- 修改：修改程序中的变量，并能以此影响程序运行结果。

```python
> /tmp/mem2/debug.py(16)<module>()
     15 ipdb.set_trace()
---> 16 x = [1,2,3,4,5]
     17 r = sum(x)

ipdb> l 1,18 # list 1,18 的缩写，查看第1行到第18行的代码
             # 光标所指的这一行尚未运行
      1 import ipdb
      2 
      3 def sum(x):
      4     r = 0
      5     for ii in x:
      6         r += ii
      7     return r
      8 
      9 def mul(x):
     10     r = 1
     11     for ii in x:
     12         r *= ii
     13     return r
     14 
     15 ipdb.set_trace()
---> 16 x = [1,2,3,4,5]
     17 r = sum(x)
     18 r = mul(x)

ipdb> n # next的缩写，执行下一步
> /tmp/mem2/debug.py(17)<module>()
     16 x = [1,2,3,4,5]
---> 17 r = sum(x)
     18 r = mul(x)

ipdb> s # step的缩写，进入sum函数内部
--Call--
> /tmp/mem2/debug.py(3)sum()
      2 
----> 3 def sum(x):
      4     r = 0

ipdb> n # next 单歩执行
> /tmp/mem2/debug.py(4)sum()
      3 def sum(x):
----> 4     r = 0
      5     for ii in x:

ipdb> n #单歩执行
> /tmp/mem2/debug.py(5)sum()
      4     r = 0
----> 5     for ii in x:
      6         r += ii

ipdb> n #单歩执行
> /tmp/mem2/debug.py(6)sum()
      5     for ii in x:
----> 6         r += ii
      7     return r

ipdb> u # up的缩写，跳回上一层的调用
> /tmp/mem2/debug.py(17)<module>()
     16 x = [1,2,3,4,5]
---> 17 r = sum(x)
     18 r = mul(x)

ipdb> d # down的缩写，跳到调用的下一层
> /tmp/mem2/debug.py(6)sum()
      5     for ii in x:
----> 6         r += ii
      7     return r

ipdb> !r # 查看变量r的值,该变量名与调试命令`r(eturn)`冲突
0
ipdb> return # 继续运行直到函数返回
--Return--
15
> /tmp/mem2/debug.py(7)sum()
      6         r += ii
----> 7     return r
      8 

ipdb> n #下一步
> /tmp/mem2/debug.py(18)<module>()
     17 r = sum(x)
---> 18 r = mul(x)
     19 

ipdb> x # 查看变量x
[1, 2, 3, 4, 5]
ipdb> x[0] = 10000 # 修改变量x
ipdb> b 10 # break的缩写，在第10行设置断点
Breakpoint 1 at /tmp/mem2/debug.py:10

ipdb> c # continue的缩写，继续运行，直到遇到断点
> /tmp/mem2/debug.py(10)mul()
      9 def mul(x):
1--> 10     r = 1
     11     for ii in x:

ipdb> return # 可见计算的是修改之后的x的乘积
--Return--
1200000
> /tmp/mem2/debug.py(13)mul()
     12         r *= ii
---> 13     return r
     14 
     
ipdb> q # 退出调试
```

关于`ipdb`的使用还有一些技巧。

- `<tab>`键能够自动补齐，补齐用法与IPython类似。
- `j(ump) <lineno>` 能够跳过中间某些行代码的执行。
- `ipdb`可以直接修改变量的值。
- `h(elp)`能够查看调试命令的用法，比如`h h`可以查看`h(elp)`命令的用法，`h jump`能够查看`j(ump)`命令的用法。

### 9.2.2 在PyTorch中调试

PyTorch作为一个动态图框架，和`ipdb`结合使用能为调试过程带来很多便捷。对于TensorFlow等静态图框架，其使用Python接口定义计算图，然后使用C++代码执行底层运算。因为在定义图的时候不进行任何计算，所以在计算的时候无法使用`pdb`进行调试，`pdb`调试只能调试Python代码。与TensorFlow不同，PyTorch可以在执行计算的同时定义计算图，这些计算定义过程是使用Python完成的。虽然其底层的计算也是使用C/C++完成的，但是用户能够查看Python定义部分的变量值，意味着可以使用`pdb`进行调试。下面将例举以下三种情形。

- 如何在PyTorch中查看神经网络各个层的输出。
- 如何在PyTorch中分析各个参数的梯度。
- 如何动态修改PyTorch训练流程。

首先，运行9.1.8节所给的示例程序：

```bash
python main.py train 
			--train-data-root=data/train/
		    --lr=0.005 
		    --batch-size=8 
		    --model='SqueezeNet'  
		    --load-model-path=None
		    --debug-file='/tmp/debug'
```

待程序运行一段时间后，读者可通过`touch /tmp/debug`创建debug标识文件，当程序检测到这个文件存在时，会自动进入调试模式。

```bash
/home/admin/PyTorch/Chapter9/main.py(82)train()
     81 
---> 82         for ii,(data,label) in tqdm(enumerate(train_dataloader)):
     83             # train model
     
ipdb> l 90
     85             target = label.to(opt.device)
     86 
     87             optimizer.zero_grad()
     88             score = model(input)
     89             loss = criterion(score,target)
     90             loss.backward()
     91             optimizer.step()
     92 
     93 
     94             # meters update and visualize
     95             loss_meter.add(loss.item())

ipdb> break 88 # 在第88行设置断点，当程序运行到此处进入调试模式
Breakpoint 1 at /home/admin/PyTorch/Chapter9/main.py:88

ipdb> opt.lr # 查看学习率
0.005

ipdb> opt.lr = 0.001 # 修改学习率
ipdb> for p in optimizer.param_groups:\    
			p['lr']=opt.lr

ipdb> model.save() # 保存模型
'checkpoints/squeezenet_0824_17_12_48.pth'

ipdb> c # 继续运行，直至第88行暂停
/home/admin/PyTorch/Chapter9/main.py(88)train()
     87             optimizer.zero_grad()
2--> 88             score = model(input)
     89             loss = criterion(score,target)
     
ipdb> s # 进入model(input)内部
> torch/nn/modules/module.py(710)_call_impl()
    709 
--> 710     def _call_impl(self, *input, **kwargs):
    711         for hook in itertools.chain(
    
ipdb> n # 下一步
> torch/nn/modules/module.py(711)_call_impl()
    710     def _call_impl(self, *input, **kwargs):
--> 711         for hook in itertools.chain(
    712                 _global_forward_pre_hooks.values(),

# 重复几次下一步后，直至看到下面的结果
ipdb> n
> torch/nn/modules/module.py(722)_call_impl()
    721         else:
--> 722             result = self.forward(*input, **kwargs)
    723         for hook in itertools.chain(

ipdb> s # 进入forward函数内部
--Call--
> /home/admin/PyTorch/Chapter9/models/squeezenet.py(20)forward()
     19 
---> 20     def forward(self,x):
     21         return self.model(x)
     
ipdb> n # 下一步
> /home/admin/PyTorch/Chapter9/models/squeezenet.py(21)forward()
     20     def forward(self,x):
---> 21         return self.model(x)
     22 

ipdb> x.data.mean(), x.data.std() # 查看x的均值和方差，读者还可以继续调试查看每一层的输出
(tensor(0.1930, device='cuda:0'), tensor(0.9645, device='cuda:0'))

ipdb> u # 跳回上一层
> torch/nn/modules/module.py(722)_call_impl()
    721         else:
--> 722             result = self.forward(*input, **kwargs)
    723         for hook in itertools.chain(

ipdb> clear # 清除所有断点
Clear all breaks? y
Deleted breakpoint 2 at /home/admin/PyTorch/Chapter9/main.py:88

ipdb> c # 继续运行，记得先删除'/tmp/debug'，否则很快又会进入调试模式
```

如果想要进入调试模式修改程序中某些参数值，或者想分析程序时，那么可以通过命令`touch /tmp/debug`创建debug标识文件，随时进入调试模式。调试完成之后输入命令`rm /tmp/debug`，然后在`ipdb`调试接口输入`c`继续运行程序。如果想要退出程序，也可以使用这种方法，首先输入命令`touch /tmp/debug`进入调试模式，然后输入`quit`而不是`continue` ，这样程序会退出而不是继续运行。这种退出程序的方法，相比于使用`Ctrl+C`的方法更加安全，这能保证数据加载的多进程程序也能正确地退出，并释放内存显存等资源。

PyTorch和`ipdb`结合能完成很多其它框架所不能完成或很难实现的功能，总结如下。

- 通过调试暂停程序。当程序进入调试模式之后，将不再执行GPU和CPU运算，但是内存和显存以及相应的堆栈空间不会释放。

- 通过调试分析程序，查看每个层的输出，查看网络的参数情况。通过`u(p)`、`d(own)`、`s(tep)`等命令，能够进入到指定的代码，通过`n(ext)`可以单歩执行，从而看到每一层的运算结果，便于分析网络的数值分布等信息。

- 作为动态图框架，PyTorch拥有Python动态语言解释执行的优点。在运行程序时，能够通过`ipdb`修改某些变量的值或属性，这些修改能够立即生效。例如，可以在训练开始不久后，根据损失函数调整学习率，而不必重启程序。

- 如果你在IPython中通过`%run`魔法方法运行程序，那么在程序异常退出的时候，可以使用`%debug`命令，直接进入调试模式，通过`u(p)`和`d(own)`跳到报错的地方，查看对应的变量。找出报错原因，直接修改相应的代码。有些时候模型训练了好几个小时，却在将要保存模型之前，因为一个小小的拼写错误异常退出。此时，如果修改错误再重新运行程序又要花费好几个小时。因此，最好的方法就是利用`%debug`进入调试模式，在调试模式中直接运行`model.save()`保存模型。在IPython中，`%pdb`魔术方法能够使得程序出现问题后，不用手动输入`%debug`而自动进入调试模式。

PyTorch调用cuDNN报错时，报错信息诸如：`CUDNN_STATUS_BAD_PARAM`，从这些报错内容中很难得到有用的帮助信息，此时最好先利用CPU运行代码，一般会得到相对友好的报错信息。比如，在`ipdb`中执行`model.cpu()(input.cpu())`，PyTorch底层的TH库会给出相对比较详细的信息。

常见的错误主要有以下几种。

- 类型不匹配问题。比如`CrossEntropyLoss`的输入target应该是一个`LongTensor`，很多人输入为`FloatTensor`。

- 部分数据忘记从CPU转移到GPU上。比如当`model`存放于GPU时，输入`input`也需要转移到GPU，才能输入到`model`中。还有可能就是把多个module存放于一个list对象，而在执行`model.cuda()`的时候，这个list中的对象是不会被转移到CUDA上的，正确用法是用`ModuleList`代替。

- Tensor形状不匹配。此类问题一般是输入数据形状不对，或是网络结构设计有问题，一般通过`u(p)`跳到指定代码，查看输入和模型参数的形状即可得知。

此外，读者还可能会经常遇到程序正常运行，没有报错，但是模型无法收敛的情况。例如，一个二分类问题，交叉熵损失一直徘徊在0.69（ln 2）附近，或者是数值出现溢出等问题。此时可以进入调试模式，用单歩执行看看每一层输出的均值和方差，从而观察从哪一层的输出开始出现数值异常。此外，还可以查看每个参数梯度的均值和方差，看看是否出现梯度消失或者梯度爆炸等问题。一般来说通过在激活函数之前增加`BatchNorm`层、进行合理地参数初始化、使用恰当的优化器、设置一个较小的学习率，基本就能确保模型在一定程度收敛。

## 9.3 小结

本章带领读者从头完成了一个Kaggle上的经典竞赛，其中重点讲解了如何合理地组织安排程序，同时介绍了一些在PyTorch中调试的技巧。

