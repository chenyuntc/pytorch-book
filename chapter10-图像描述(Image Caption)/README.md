这是电子工业出版社的《深度学习框架PyTorch：入门与实践》第十章的配套代码，利用PyTorch实现了图像描述Image Caption。

## 1 下载数据

本次实验的数据来自于[AI Challenger图像描述](https://challenger.ai/competition/caption/)。 请下载对应的训练数据(ai_challenger_caption_train_20170902.zip)。如果你只是想测试看看效果，可以跳过这一步。 读者还可以从[MEGA](https://mega.nz/#!fP4TSJ6I!mgG_HSXqi1Kgg5gvwYArUnuRNgcDqpd8qoj09e0Yg10)下载数据。

## 2 环境配置

- 安装[PyTorch](http://pytorch.org)
- 安装第三方依赖

```Python
pip install -r requirements.txt
```
- 启动visodm
```Bash
 python -m visdom.server
```

## 3 数据预处理
### 3.1 标记文件的预处理（对人工描述的预处理）
可以跳过这一步，直接下载预处理好的[caption.pth](http://pytorch-1252820389.cosbj.myqcloud.com/caption.pth)

当然，你也可以自行进行处理，运行 
```Bash
python data_preprocess.py process --annotation-file=/data/annotation.json --max-words=5000
```
最后会生成`caption.pth`

### 3.2 提取图片特征
```Bash
python feature_extract.py
```

**注意修改`config.py`中的`img_path`**，一般是`ai_challenger_caption_train_20170902/caption_train_images_20170902/`

这里使用的是resnet50，提取图片特征。提取完成之后，会在当前文件夹生成`results.pth`, `results.pth`保存着一个tensor数组，形如（21w X 2048），保存着21w张图片的特征信息。


## 4 训练
训练的命令如下：

```Bash
python main.py train 
```

注意修改`config.py`中的文件名路径，比如
```Bash
python main.py train 
    --img_path= 'ai_challenger_caption_train/caption_train_images_20170902/'\
    --img_features_path='../results.pth'
```
 

完整的命令行选项：
```Python
    caption_data_path='caption.pth'# 经过预处理后的人工描述信息
    img_path='/home/cy/caption_data/' # 图片保存的原始文件夹
    # img_path='/mnt/ht/aichallenger/raw/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
    img_feature_path = 'results.pth' # 所有图片的features,20w*2048的向量
    scale_size = 300
    img_size = 224
    batch_size=8
    shuffle = True
    num_workers = 4
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights=False
    prefix='checkpoints/caption'#模型保存前缀
    env = 'caption'
    plot_every = 10
    debug_file = '/tmp/debugc'

    model_ckpt = None # 模型断点保存路径
    lr=1e-3
    use_gpu=True
    epoch = 1
    test_img = 'img/example.jpeg' 

```

### 测试&Demo
下载[预训练好的模型](http://pytorch-1252820389.file.myqcloud.com/caption_0914_1947), 或者使用你自己训练好的模型

参照 [demo.ipynb](demo.ipynb),查看效果。


部分效果图

![img](img/caption-results.png)

### 兼容性测试

train 
- [x] GPU  
- [x] CPU  
- [x] Python2
- [x] Python3

tested: 

- [x] GPU
- [x] CPU
- [x] Python2
- [x] Python3

