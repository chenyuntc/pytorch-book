#encoding=utf-8

#模型文件

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SequenceWise(nn.Module):
    """调整输入满足module的需求，因为多次使用，所以模块化构建一个类
    适用于将LSTM的输出通过batchnorm或者Linear层
    """
    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        """
        Args:
            x :    PackedSequence
        """
        x, batch_size_len = x.data, x.batch_sizes
        #x.data:    sum(x_len) * num_features
        x = self.module(x)
        x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr

class BatchSoftmax(nn.Module):
    """
    The layer to add softmax for a sequence, which is the output of rnn
    Which state use its own softmax, and concat the result
    """
    def forward(self, x):
        #x: seq_len * batch_size * num
        if not self.training:
            seq_len = x.size()[0]
            return torch.stack([F.softmax(x[i], dim=1) for i in range(seq_len)], 0)
        else:
            return x

class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, 
                    bidirectional=False, batch_norm=True, dropout=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                                bidirectional=bidirectional, dropout = dropout, bias=False)
        
    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x, _ = self.rnn(x)
        #self.rnn.flatten_parameters()
        return x

class CTC_Model(nn.Module):
    def __init__(self, rnn_param=None, num_class=48, drop_out=0.1):
        """
        rnn_param(dict)  :  the dict of rnn parameters
                            rnn_param = {"rnn_input_size":201, "rnn_hidden_size":256, ....}
        num_class(int)   :  the number of units, add one for blank to be the classes to classify
        drop_out(float)  :  drop_out paramteter for all place where need drop_out
        """
        super(CTC_Model, self).__init__()
        if rnn_param is None or type(rnn_param) != dict:
            raise ValueError("rnn_param need to be a dict to contain all params of rnn!")
        self.rnn_param = rnn_param
        self.num_class = num_class
        self.num_directions = 2 if rnn_param["bidirectional"] else 1
        self._drop_out = drop_out
        
        rnn_input_size = rnn_param["rnn_input_size"]
        rnns = []
        
        rnn_hidden_size = rnn_param["rnn_hidden_size"]
        rnn_type = rnn_param["rnn_type"]
        rnn_layers = rnn_param["rnn_layers"]
        bidirectional = rnn_param["bidirectional"]
        batch_norm = rnn_param["batch_norm"]
        
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, 
                        rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out,
                        batch_norm=False)
        
        rnns.append(('0', rnn))
        #堆叠RNN,除了第一次不使用batchnorm，其他层RNN都加入BachNorm
        for i in range(rnn_layers - 1):
            rnn = BatchRNN(input_size=self.num_directions*rnn_hidden_size, 
                            hidden_size=rnn_hidden_size, rnn_type=rnn_type, 
                            bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i+1), rnn))

        self.rnns = nn.Sequential(OrderedDict(rnns))

        if batch_norm:
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions*rnn_hidden_size),
                                nn.Linear(self.num_directions*rnn_hidden_size, num_class+1, bias=False),)
        else:
            fc = nn.Linear(self.num_directions*rnn_hidden_size, num_class+1, bias=False)
        
        self.fc = SequenceWise(fc)
        self.inference_softmax = BatchSoftmax()
    
    def forward(self, x, dev=False):
        x = self.rnns(x)
        x = self.fc(x)
        x, batch_seq = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
            
        out = self.inference_softmax(x)
        if dev:
            return x, out         #如果是验证集，需要同时返回x计算loss和out进行wer的计算
        return out

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None, dev_cer_results=None):
        package = {
                'rnn_param': model.rnn_param,
                'num_class': model.num_class,
                '_drop_out': model._drop_out,
                'state_dict': model.state_dict()
                }
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_cer_results'] = dev_cer_results
        return package

