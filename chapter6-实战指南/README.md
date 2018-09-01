# PyTorch 实践指南 



本文是文章[PyTorch实践指南](https://zhuanlan.zhihu.com/p/29024978)配套代码，请参照[知乎专栏原文](https://zhuanlan.zhihu.com/p/29024978)或者[对应的markdown文件](PyTorch实战指南.md)更好的了解而文件组织和代码细节。


## 数据下载
- 从[kaggle比赛官网](https://www.kaggle.com/c/dogs-vs-cats/data) 下载所需的数据；或者直接从此下载[训练集](http://pytorch-1252820389.file.myqcloud.com/data/dogcat/train.zip)和[测试集](http://pytorch-1252820389.file.myqcloud.com/data/dogcat/test1.zip)
- 解压并把训练集和测试集分别放在一个文件夹中


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
