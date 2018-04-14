#encoding=utf-8

#本文件继承构建了Dataset类和DataLoader类，用来处理音频和标签文件
#转化为网络可输入的格式

import os
import torch
import scipy.signal
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import parse_audio, process_label_file

windows = {'hamming':scipy.signal.hamming, 'hann':scipy.signal.hann, 'blackman':scipy.signal.blackman,
            'bartlett':scipy.signal.bartlett}
audio_conf = {"sample_rate":16000, 'window_size':0.025, 'window_stride':0.01, 'window': 'hamming'}
int2char = ["_", "'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
            "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]

class SpeechDataset(Dataset):
    def __init__(self, data_dir, data_set='train', normalize=True):
        self.data_set = data_set
        self.normalize = normalize
        self.char2int = {}
        self.n_feats = int(audio_conf['sample_rate']*audio_conf['window_size']/2+1)
        for i in range(len(int2char)):
            self.char2int[int2char[i]] = i
        
        wav_path = os.path.join(data_dir, data_set+'_wav.scp')
        label_file = os.path.join(data_dir, data_set+'.text')
        self.process_audio(wav_path, label_file)
        
    def process_audio(self, wav_path, label_file):
        #read the label file
        self.label = process_label_file(label_file, self.char2int)
        
        #read the path file
        self.path  = []
        with open(wav_path, 'r') as f:
            for line in f.readlines():
                utt, path = line.strip().split()
                self.path.append(path)
        
        #ensure the same samples of input and label
        assert len(self.label) == len(self.path)

    def __getitem__(self, idx):
        return parse_audio(self.path[idx], audio_conf, windows, normalize=self.normalize), self.label[idx]

    def __len__(self):
        return len(self.path) 

def collate_fn(batch):
    #将输入和标签转化为可输入网络的batch
    #batch :     batch_size * (seq_len * nfeats, target_length)
    def func(p):
        return p[0].size(0)
    
    #sort batch according to the frame nums
    batch = sorted(batch, reverse=True, key=func)
    longest_sample = batch[0][0]
    feat_size = longest_sample.size(1)
    max_length = longest_sample.size(0)
    batch_size = len(batch)
    
    inputs = torch.zeros(batch_size, max_length, feat_size)   #网络输入,相当于长度不等的补0
    input_sizes = torch.IntTensor(batch_size)                 #输入每个样本的序列长度，即帧数
    target_sizes = torch.IntTensor(batch_size)                #每句标签的长度
    targets = []
    input_size_list = []
    
    for x in range(batch_size):
        sample = batch[x]
        feature = sample[0]
        label = sample[1]
        seq_length = feature.size(0)
        inputs[x].narrow(0, 0, seq_length).copy_(feature)
        input_sizes[x] = seq_length
        input_size_list.append(seq_length)
        target_sizes[x] = len(label)
        targets.extend(label)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_sizes, input_size_list, target_sizes

"""
class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, 
                                        sampler=None, batch_sampler=None, num_workers=0, 
                                        collate_fn=<function default_collate>, 
                                        pin_memory=False, drop_last=False)
"""
class SpeechDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(SpeechDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = collate_fn


