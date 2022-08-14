# PyTorch 实战指南 

本文是本书第九章：PyTorch实践指南配套代码的说明，读者可参考[对应的markdown文件](实战指南.md)以更好地了解文件组织和代码细节。


## 数据下载
- 从[kaggle比赛官网](https://www.kaggle.com/c/dogs-vs-cats/data) 下载所需的数据；或者直接从此下载[训练集](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/data/dogcat/train.zip)和[测试集](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/data/dogcat/test1.zip)；
- 解压并把训练集和测试集分别放在一个文件夹data中


## 安装
- PyTorch : 可按照[PyTorch官网](http://pytorch.org)的指南，根据自己的平台安装指定的版本
- 安装指定依赖：

```
pip install -r requirements.txt
```

## 训练
必须首先启动visdom：

```
python -m visdom.server
```

然后使用如下命令启动训练：

```
# 在gpu0上训练,并把可视化结果保存在visdom 的classifier env上
python main.py train --train-data-root=./data/train --use-gpu --env=classifier
```


详细的使用命令 可使用
```
python main.py help
```

## 测试

```
python main.py test --data-root=./data/test  --batch-size=256 --load-path='checkpoints/squeezenet.pth'
```
