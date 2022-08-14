# AI诗人：用Transformer写诗

这是本书第十一章用Transformer写诗的对应代码。

## 环境准备

- 本程序需要安装PyTorch；

- 还需要通过`pip install -r requirements.txt` 安装其它依赖；

## 数据准备

我们已经完成了诗词数据的预处理工作，并提供了一个NumPy的压缩包`data/tang.npz` ，里面包含三个对象。

- data：形为(57598, 125)的NumPy数组，总共有57598首诗词，每首诗词长度为125字符（不足125的补空格，超过125的丢弃）。
- word2ix：每个字和它对应的序号，例如”春“这个字对应的序号是1000。
- ix2word：每个序号和它对应的字，例如序号1000对应着“春”这个字。

## 用法

如果想要使用Visdom进行可视化，请先运行`python -m visidom.server`启动Visdom服务。

- 数据准备

  - 诗词数据`tang.npz`的下载链接：https://pan.baidu.com/s/1-96Cj-jtC5cYvCDF2VYZzw?pwd=usuv 你需要将它放置在`data`目录下。
  - （Optional）预训练好的自动写诗模型`tang_200.pth`的下载链接：链接: https://pan.baidu.com/s/1FNiTEjED11W4DP2oNO0Epg?pwd=qyhu 你需要将它放置在`checkpoints`目录下。

- 训练

    ```bash
    python main.py train --batch-size=128
                     --pickle-path='tang.npz'
                     --lr=1e-3 
                     --epoch=50
    ```

- 续写诗词

    ```bash
    python predict.py gen  --model-path='checkpoints/tang_200.pth' 
					   --pickle-path='tang.npz' 
					   --start-words='海内存知己'
    ```

- 生成藏头诗

    ```bash
    python predict.py gen_acrostic  --model-path='checkpoints/tang_200.pth' 
					 			--pickle-path='tang.npz' 
					 			--start-words='深度学习' # 藏头诗
    ```

- 完整的选项及默认值

  ```python
    data_path = 'data'         # 诗歌的文本文件存放路径
    pickle_path = 'data/tang.npz'    # 预处理好的二进制文件
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 200
    env = 'poetry1'             # visdom env
    batch_size = 128
    maxlen = 125                # 超过这个长度之后的字被丢弃，小于这个长度的在前面补空格
    max_gen_len = 200           # 生成诗歌最长长度
    model_path = None           # 预训练模型路径
    start_words = '深度学习'     # 诗歌开始
    model_prefix = 'checkpoints/tang'  # 模型保存路径
    plot_every = 20             # 每20个batch 可视化一次
    debug_file = '/tmp/debugp/'
  ```

- 写出来的部分诗词：

    > 江流天地外，风景属清明。白日无人见，青山有鹤迎。水寒鱼自跃，云暗鸟难惊。独有南归路，悠悠去住情。
    >
    > 蜀道难为宰，江湖易为舟。两乡三月夜，万里一星秋。
    >
    > 同是天涯沦落人，相逢相识未相亲。一杯酒熟君应醉，万里山川我未春。
    >
    > **白**日照秋色，清光动远林。**色**连三径合，香满四邻深。**风**送宜新草，花开爱旧林。**车**轮不可驻，日暮欲归心。
    >
    > **烟**霞何处去，万里一帆飞。**花**发南陵岸，春生北国衣。**易**穷经世乱，难得到乡稀。**冷**淡蒹葭雨，空蒙雨雪归。



