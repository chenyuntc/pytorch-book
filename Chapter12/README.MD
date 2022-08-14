# AI艺术家：神经网络风格迁移

这是本书第十二章任意风格快速迁移的配套代码。


## 1 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖


##2  数据准备

本实验使用的内容图片数据集来自[COCO](http://cocodataset.org/#download)，点[此](http://images.cocodataset.org/zips/train2017.zip)下载。使用的风格图片数据集来自[Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers/data)。你也可以使用其它数据，比如ImageNet。请尽量保证数据的多样性，**不建议**使用单一种类的数据集，比如LSUN或者人脸识别数据集之类的。

请确保所有的图片保存于`data/coco/`文件夹下,形如：
```Bash
    ├──data/
    │  ├──content_data/ # 用来保存用于训练的内容图片数据集
    │  └──style_data/ 	# 用来保存用于训练的风格图片数据集
```

## 用法
如果想要使用visdom可视化，请先运行`python -m visdom.server`启动visdom服务

- 训练
```bash
python main.py train \
		--use-gpu \ 
		--content-data-root='data/content_data' \ 
		--style-data-root='data/style_data' \
        --batch-size=16
```

- 生成图片
读者可以自行寻找内容图片`input.png`和风格图片`style.png`，用来生成风格迁移后的图像。

```bash
python main.py stylize \ 
		--use-gpu=False \
		--model-path='checkpoints/20_style.pth' \
		--content-path='input.png' \
		--style-path='style.png' \ 
		--result-path='output.png'
```
完整的选项及默认值
```python
    use_gpu = True
    model_path = None # 预训练模型的路径（用于继续训练/测试）
    
    # 训练用参数
    image_size = 256 # 图片大小
    batch_size = 16  # 一个batch的大小
    content_data_root = '/mnt/sda1/COCO/train' # 内容图片数据集路径
    style_data_root = '/mnt/sda1/Style' # 风格图片数据集路径
    num_workers = 4 # 多线程加载数据
    lr = 5e-5 # 学习率
    epoches = 40 # 训练epoch

    env = 'neural-style' # visdom env
    plot_every = 20 # 每20个batch可视化一次

    debug_file = '/tmp/debugnn' # 进入调试模式

    # 测试用参数
    content_path = 'input.png' # 需要进行风格迁移的图片
    style_path = None # 风格图片
    result_path = 'output.png' # 风格迁移结果的保存路径
   
```
部分风格迁移的结果：
![imgs](imgs/Style_result.png)


如果需要生成其它风格的图片，只需要修改`--style-path`对应的风格图片。
