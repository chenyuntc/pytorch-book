#encoding=utf-8

#本文件为数据集测试文件
#解码类型在run.sh中定义

import time
import torch
import argparse
import ConfigParser
import torch.nn as nn
from torch.autograd import Variable

from model import *
from decoder import GreedyDecoder, BeamDecoder
from data  import int2char, SpeechDataset, SpeechDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--conf', help='conf file for training')
def test():
    args = parser.parse_args()
    cf = ConfigParser.ConfigParser()
    cf.read(args.conf)
    USE_CUDA = cf.getboolean('Training', 'USE_CUDA')
    model_path = cf.get('Model', 'model_file')
    data_dir = cf.get('Data', 'data_dir')
    beam_width = cf.getint('Decode', 'beam_width')
    package = torch.load(model_path)
    
    rnn_param = package["rnn_param"]
    num_class = package["num_class"]
    n_feats = package['epoch']['n_feats']
    drop_out = package['_drop_out']

    decoder_type =  cf.get('Decode', 'decoder_type')
    data_set = cf.get('Decode', 'eval_dataset')

    test_dataset = SpeechDataset(data_dir, data_set=data_set)
    
    model = CTC_Model(rnn_param=rnn_param, num_class=num_class, drop_out=drop_out)
        
    test_loader = SpeechDataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=False)
    
    model.load_state_dict(package['state_dict'])
    model.eval()
    
    if USE_CUDA:
        model = model.cuda()

    if decoder_type == 'Greedy':
        decoder  = GreedyDecoder(int2char, space_idx=len(int2char) - 1, blank_index = 0)
    else:
        decoder = BeamDecoder(int2char, beam_width=beam_width, blank_index = 0, space_idx = len(int2char) - 1)    

    total_wer = 0
    total_cer = 0
    start = time.time()
    for data in test_loader:
        inputs, target, input_sizes, input_size_list, target_sizes = data 
        inputs = inputs.transpose(0,1)
        inputs = Variable(inputs, volatile=True, requires_grad=False)
        
        if USE_CUDA:
            inputs = inputs.cuda()
        
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_size_list)
        probs = model(inputs)

        probs = probs.data.cpu()
        decoded = decoder.decode(probs, input_size_list)
        targets = decoder._unflatten_targets(target, target_sizes)
        labels = decoder._process_strings(decoder._convert_to_strings(targets))

        for x in range(len(labels)):
            print("origin : " + labels[x])
            print("decoded: " + decoded[x])
        cer = 0
        wer = 0
        for x in range(len(labels)):
            cer += decoder.cer(decoded[x], labels[x])
            wer += decoder.wer(decoded[x], labels[x])
            decoder.num_word += len(labels[x].split())
            decoder.num_char += len(labels[x])
        total_cer += cer
        total_wer += wer
    CER = (1 - float(total_cer) / decoder.num_char)*100
    WER = (1 - float(total_wer) / decoder.num_word)*100
    print("Character error rate on test set: %.4f" % CER)
    print("Word error rate on test set: %.4f" % WER)
    end = time.time()
    time_used = (end - start) / 60.0
    print("time used for decode %d sentences: %.4f minutes." % (len(test_dataset), time_used))

if __name__ == "__main__":
    test()
