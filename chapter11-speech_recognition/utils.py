#encoding=utf-8

#本文件提供载入音频文件的函数,提取音频对数幅度谱的函数以及处理文本标签的函数
#语音的对数频谱作为网络的输入

import torch
import librosa
import torchaudio

def load_audio(path):
    """使用torchaudio读取音频
    Args:
        path(string)            : 音频的路径
    Returns:
        sound(numpy.ndarray)    : 单声道音频数据，如果是多声道进行平均(Samples * 1 channel)
    """
    sound, _ = torchaudio.load(path)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis = 1)
    return sound

def parse_audio(path, audio_conf, windows, normalize=False):
    """使用librosa计算音频的对数幅度谱
    Args:
        path(string)       : 音频的路径
        audio_conf(dict)   : 求频谱的参数
        windows(dict)      : 加窗类型
    Returns:
        spect(FloatTensor) : 音频的对数幅度谱(numFrames * nFeatures)
                             nFeatures = n_fft / 2 + 1
    """
    y = load_audio(path)
    n_fft = int(audio_conf['sample_rate']*audio_conf["window_size"])
    win_length = n_fft
    hop_length = int(audio_conf['sample_rate']*audio_conf['window_stride'])
    window = windows[audio_conf['window']]
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length, window=window)
    spect, phase = librosa.magphase(D) 
    spect = torch.FloatTensor(spect)
    spect = spect.log1p()
    
    #每句话自己做归一化
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)  
    return spect.transpose(0,1)

def process_label_file(label_file, char2int):
    """将文本标签处理为数字，转化为numpy类型是为了存储为h5py文件
    Args:
        label_file(string)  :  标签文件路径
        char2int(dict)      :  标签到数字的映射关系 "_'abcdefghijklmnopqrstuvwxyz"
    Output:
        label_dict(list)    :  所有句子的标签，每个句子是list类型
    """
    label_all = []
    with open(label_file, 'r') as f:
        for label in f.readlines():
            label = label.strip()
            label_list = []
            utt = label.split('\t', 1)[0]
            label = label.split('\t', 1)[1]
            for i in range(len(label)):
                if label[i].lower() in char2int:
                    label_list.append(char2int[label[i].lower()])
                else:
                    print("%s not in the label map list" % label[i].lower())
            label_all.append(label_list)
    return label_all

