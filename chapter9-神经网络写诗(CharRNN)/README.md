这是电子工业出版社的《深度学习框架PyTorch：入门与实践》第九章的配套代码，利用PyTorch实现了CharRNN用以写唐诗。

本次实验的数据来自于[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)。但是作者已经进行处理成二进制文件`tang.npz`，可以直接使用。读者可以[点此](http://pytorch-1252820389.cosbj.myqcloud.com/tang.npz)下载`tang.npz`

## 环境配置
- 安装[PyTorch](http://pytorch.org)
- 安装第三方依赖

```Python
pip install -r requirements.txt
```
- 启动visodm
```Bash
 python -m visdom.server
```
或者
```Bash
nohup python -m visdom.server &
``` 
## 训练
训练的命令如下：

```Bash
python main.py train --plot-every=150\
					 --batch-size=128\
                     --pickle-path='tang.npz'\
                     --lr=1e-3 \
                     --env='poetry3' \
                     --epoch=50
```

命令行选项：
```Python
    data_path = 'data/' # 诗歌的文本文件存放路径
    pickle_path= 'tang.npz' # 预处理好的二进制文件 
    author = None # 只学习某位作者的诗歌
    constrain = None # 长度限制
    category = 'poet.tang' # 类别，唐诗还是宋诗歌(poet.song)
    lr = 1e-3 
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20  
    batch_size = 128
    maxlen = 125 # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20 # 每20个batch 可视化一次
    # use_env = True # 是否使用visodm
    env='poetry' # visdom env
    max_gen_len = 200 # 生成诗歌最长长度
    debug_file='/tmp/debugp'
    model_path=None # 预训练模型路径
    prefix_words = '细雨鱼儿出,微风燕子斜。' # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words='闲云潭影日悠悠' # 诗歌开始
    acrostic = False # 是否是藏头诗
    model_prefix = 'checkpoints/tang' # 模型保存路径

```
## 生成诗歌
作者提供了预训练好的模型，可以[点此](http://pytorch-1252820389.cosbj.myqcloud.com/tang_199.pth)下载`tang_199.pth`，用以生成诗歌

生成藏头诗的命令如下：

```Bash
python  main.py gen  --model-path='checkpoints/tang_199.pth' \
       --pickle-path='tang.npz' \
       --start-words='深度学习' \
       --prefix-words='江流天地外，山色有无中。' \
       --acrostic=True\
       --nouse-gpu  # 或者 --use-gpu=False
深居不可见，浩荡心亦同。度年一何远，宛转三千雄。学立万里外，诸夫四十功。习习非吾仕，所贵在其功。
```

生成其它诗歌的命令如下：

```Bash
python2 main.py gen  --model-path='model.pth' 
					 --pickle-path='tang.npz' 
					 --start-words='江流天地外，' # 诗歌的开头
					 --prefix-words='郡邑浮前浦，波澜动远空。' 
江流天地外，风日水边东。稍稍愁蝴蝶，心摧苎范蓬。云飞随海远，心似汉阳培。按俗朝廷上，分军朔雁通。封疆朝照地，赐劒豫章中。畴昔分曹籍，高名翰墨场。翰林推国器，儒冠见忠贞。臯宙非无事，姦邪亦此中。渥仪非贵盛，儒实不由锋。几度沦亡阻，千年垒数重。宁知天地外，长恐海西东。邦测期戎逼，箫韶故国通。蜃楼瞻凤篆，云辂接旌幢。別有三山里，来随万里同。烟霞临海路，山色落云中。渥泽三千里，青山万古通。何言陪宴侣，复使
```

### 兼容性测试
train 
- [x] GPU  
- [x] CPU  
- [x] Python2
- [x] Python3

test: 

- [x] GPU
- [x] CPU
- [x] Python2
- [x] Python3


## 举例

- 藏头诗
```Bash
 python3  main.py gen  --model-path='checkpoints/tang_199.pth' \
                                     --pickle-path='tang.npz' \
                                     --start-words="深度学习" \
                                     --prefix-words="江流天地外，山色有无中。" \
                                     --acrostic=True\
                                     --nouse-gpu
深井松杉下，前山云汉东。度山横北极，飞雪凌苍穹。学稼落羽化，潺湲浸天空。习习时更惬，俯视空林濛。
```

- 深度学习开头，七言
```Bash
python2  main.py gen    --model-path='checkpoints/tang_199.pth' \
                        --pickle-path='tang.npz' \
                        --start-words="深度学习" \
                        --prefix-words="庄生晓梦迷蝴蝶，望帝春心托杜鹃。" \
                        --acrostic=False\
                        --nouse-gpu
深度学习书不怪，今朝月下不胜悲。芒砀月殿春光晓，宋玉堂中夜月升。玉徽美，玉股洁。心似镜，澈圆珠，金炉烟额红芙蕖。红缕金钿舞凤管，夜妆妆妓。歌中有女子孙子，嫁得新年花下埽。君不见金沟里，裴回春日丛。歌舞一声声断，一语中肠千万里。罗帐前传，娉婷花月春，一歌一曲声声。可怜眼，芙蓉露。妾心明，颜色暗相思，主人愁，万重金。红粉，冉冉，芙蓉帐前飞。鸳鸯鬬鸭，绣衣罗帐，鹦鹉抹。凰翠忽，菱管。音舞，行路，蹙罗金钿
```