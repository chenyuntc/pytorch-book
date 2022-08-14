# coding:utf8
import sys, os
import torch as t
from torch.utils.data import DataLoader
from data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb


class Config(object):
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

opt = Config()

def gen(**kwargs):
    """
    给定几个词，根据这几个词接着生成一首完成的诗词
    例如，start_words为'海内存知己'，可以生成
    海内存知己，天涯尚未安。
    故人归旧国，新月到新安。
    海气生边岛，江声入夜滩。
    明朝不可问，应与故人看。
    """

    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix))
    model.load_state_dict(t.load(opt.model_path))
    model.to(device)
    model.eval()

    src = [word2ix[word] for word in opt.start_words]
    res = src = [word2ix['<START>']] + src
    max_len = 100

    for _ in range(max_len):
        src = t.tensor(res).to(device)[:, None]
        src_mask = generate_square_subsequent_mask(src.shape[0])
        src_pad_mask = src == len(word2ix) - 1
        src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
        memory, logits = model(src, src_mask.cuda(), src_pad_mask.cuda())
        next_word = logits[-1, 0].argmax().item()
        if next_word == word2ix['<EOP>']:
            break
        res.append(next_word)
    res = [ix2word[_] for _ in res[1:]]
    print(''.join(res))

def gen_acrostic(**kwargs):
    """
    生成藏头诗
    start_words为'深度学习'
    生成：
	深山高不极，望望极悠悠。
	度日登楼望，看云上砌秋。
	学吟多野寺，吟想到江楼。
	习静多时选，忘机尽处求。
    """
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix))
    model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    model.eval()

    start_word_len = len(opt.start_words)
    index = 0  # 用来指示已经生成了多少句藏头诗
    src_base = [word2ix[word] for word in opt.start_words]
    res = [word2ix['<START>']] + [src_base[index]]
    index += 1
    max_len = 100

    for _ in range(max_len):
        src = t.tensor(res).to(device)[:, None]
        src_mask = generate_square_subsequent_mask(src.shape[0])
        src_pad_mask = src == len(word2ix) - 1
        src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
        memory, logits = model(src, src_mask.cuda(), src_pad_mask.cuda())

        next_word = logits[-1, 0].argmax().item()
        # 如果遇到句号感叹号等，把藏头的词作为下一个句的输入
        if next_word in {word2ix[u'。'],word2ix[u'！'],word2ix['<START>']}:
            # 如果生成的诗歌已经包含全部藏头的词，则结束
            if index == start_word_len:
                res.append(next_word)
                break
            # 把藏头的词作为输入，预测下一个词
            res.append(next_word)
            res.append(src_base[index])
            index += 1
        else:
            res.append(next_word)

    res = [ix2word[_] for _ in res[1:]]
    print(''.join(res))


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = Visualizer(env = opt.env)

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix))
    optimizer = t.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=len(word2ix)-1)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            src_mask = generate_square_subsequent_mask(input_.shape[0])
            src_pad_mask = input_ == len(word2ix) - 1
            src_pad_mask = src_pad_mask.permute(1,0).contiguous()
            
            memory, logit = model(input_, src_mask.to(device), src_pad_mask.to(device))

            mask = target != word2ix['</s>']
            target = target[mask] # 去掉前缀的空格
            logit = logit.flatten(0, 1)[mask.view(-1)]

            loss = criterion(logit, target)
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())
        
            # 可视化
            if (1 + ii) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])

                # 诗歌原文
                poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                            for _iii in range(data_.shape[1])][:16]
                vis.text('</br>'.join([''.join(poetry) for poetry in poetrys]), win=u'origin_poem')

                gen_poetries = []
                # 分别以这几个字作为诗歌的第一个字，生成8首诗
                for word in list(u'春江花月夜凉如水'):
                    gen_poetry = ''.join(generate(model, word, word2ix, ix2word))
                    gen_poetries.append(gen_poetry)

                # gen_poetries = generate(model, u'春江花月夜凉如水', ix2word, word2ix)

                vis.text('</br>'.join([''.join(poetry) for poetry in gen_poetries]), win=u'gen_poem')

        if (epoch+1) % opt.plot_every == 0:
            t.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch+1))

@t.no_grad()
def generate(model, start_words, word2ix, ix2word, max_len=100):
    model.eval()
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    src = [word2ix[word] for word in start_words]
    res = src = [word2ix['<START>']] + src 

    for _ in range(max_len):
        src = t.tensor(res).to(device)[:, None]
        src_mask = generate_square_subsequent_mask(src.shape[0])
        src_pad_mask = src == len(word2ix)-1
        src_pad_mask = src_pad_mask.permute(1,0).contiguous()
        memory,logits = model(src, src_mask.cuda(), src_pad_mask.cuda())
        
        next_word =  logits[-1,0].argmax().item()
        if next_word == word2ix['<EOP>']:
            break
        res.append(next_word)
        
        if next_word == word2ix['<EOP>']:
            break
    res = [ix2word[_] for _ in res]
    model.train()
    return res


def generate_square_subsequent_mask(sz):
    mask = (t.triu(t.ones(sz, sz)) == 1).transpose(0, 1) # 生成下三角矩阵（下三角全为True，其余位False）
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

 
if __name__ == '__main__':
    import fire
    fire.Fire()
