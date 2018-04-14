#encoding=utf-8

#greedy decoder and beamsearch decoder for ctc

import torch

class Decoder(object):
    "解码器基类定义，作用是将模型的输出转化为文本使其能够与标签计算正确率"
    def __init__(self, int2char, space_idx = 28, blank_index = 0):
        """
        int2char     :     将类别转化为字符标签
        space_idx    :     空格符号的索引，如果为为-1，表示空格不是一个类别
        blank_index  :     空白类的索引，默认设置为0
        """
        self.int_to_char = int2char
        self.space_idx = space_idx
        self.blank_index = blank_index
        self.num_word = 0
        self.num_char = 0

    def decode(self):
        "解码函数，在GreedyDecoder和BeamDecoder继承类中实现"
        raise NotImplementedError;

    def phone_word_error(self, prob_tensor, frame_seq_len, targets, target_sizes):
        """计算词错率和字符错误率
        Args:
            prob_tensor     :   模型的输出
            frame_seq_len   :   每个样本的帧长
            targets         :   样本标签
            target_sizes    :   每个样本标签的长度
        Returns:
            wer             :   词错率，以space为间隔分开作为词
            cer             :   字符错误率
        """
        strings = self.decode(prob_tensor, frame_seq_len)
        targets = self._unflatten_targets(targets, target_sizes)
        target_strings = self._process_strings(self._convert_to_strings(targets))
        
        cer = 0
        wer = 0
        for x in range(len(target_strings)):
            cer += self.cer(strings[x], target_strings[x])
            wer += self.wer(strings[x], target_strings[x])
            self.num_word += len(target_strings[x].split())
            self.num_char += len(target_strings[x])
        return cer, wer

    def _unflatten_targets(self, targets, target_sizes):
        """将标签按照每个样本的标签长度进行分割
        Args:
            targets        :    数字表示的标签
            target_sizes   :    每个样本标签的长度
        Returns:
            split_targets  :    得到的分割后的标签
        """
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset : offset + size])
            offset += size
        return split_targets

    def _process_strings(self, seqs, remove_rep = False):
        """处理转化后的字符序列，包括去重复等，将list转化为string
        Args:
            seqs       :   待处理序列
            remove_rep :   是否去重复
        Returns:
            processed_strings  :  处理后的字符序列
        """
        processed_strings = []
        for seq in seqs:
            string = self._process_string(seq, remove_rep)
            processed_strings.append(string)
        return processed_strings
   
    def _process_string(self, seq, remove_rep = False):
        string = ''
        for i, char in enumerate(seq):
            if char != self.int_to_char[self.blank_index]:
                if remove_rep and i != 0 and char == seq[i - 1]: #remove dumplicates
                    pass
                elif self.space_idx == -1:
                    string = string + ' '+ char
                elif char == self.int_to_char[self.space_idx]:
                    string += ' '
                else:
                    string = string + char
        return string

    def _convert_to_strings(self, seq, sizes=None):
        """将数字序列的输出转化为字符序列
        Args:
            seqs       :   待转化序列
            sizes      :   每个样本序列的长度
        Returns:
            strings  :  转化后的字符序列
        """
        strings = []
        for x in range(len(seq)):
            seq_len = sizes[x] if sizes is not None else len(seq[x])
            string = self._convert_to_string(seq[x], seq_len)
            strings.append(string)
        return strings

    def _convert_to_string(self, seq, sizes):
        result = []
        for i in range(sizes):
            result.append(self.int_to_char[seq[i]])
        if self.space_idx == -1:
            return result
        else:
            return ''.join(result)
 
    def wer(self, s1, s2):
        "将空格作为分割计算词错误率"
        b = set(s1.split() + s2.split())
        word2int = dict(zip(b, range(len(b))))

        w1 = [word2int[w] for w in s1.split()]
        w2 = [word2int[w] for w in s2.split()]
        return self._edit_distance(w1, w2)

    def cer(self, s1, s2):
        "计算字符错误率"
        return self._edit_distance(s1, s2)
    
    def _edit_distance(self, src_seq, tgt_seq):
        "计算两个序列的编辑距离，用来计算字符错误率"
        L1, L2 = len(src_seq), len(tgt_seq)
        if L1 == 0: return L2
        if L2 == 0: return L1
        # construct matrix of size (L1 + 1, L2 + 1)
        dist = [[0] * (L2 + 1) for i in range(L1 + 1)]
        for i in range(1, L2 + 1):
            dist[0][i] = dist[0][i-1] + 1
        for i in range(1, L1 + 1):
            dist[i][0] = dist[i-1][0] + 1
        for i in range(1, L1 + 1):
            for j in range(1, L2 + 1):
                if src_seq[i - 1] == tgt_seq[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[i][j] = min(dist[i][j-1] + 1, dist[i-1][j] + 1, dist[i-1][j-1] + cost)
        return dist[L1][L2]


class GreedyDecoder(Decoder):
    "直接解码，把每一帧的输出概率最大的值作为输出值，而不是整个序列概率最大的值"
    def decode(self, prob_tensor, frame_seq_len):
        """解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            解码得到的string，即识别结果
        """
        prob_tensor = prob_tensor.transpose(0,1)
        _, decoded = torch.max(prob_tensor, 2)
        decoded = decoded.view(decoded.size(0), decoded.size(1))
        decoded = self._convert_to_strings(decoded, frame_seq_len)     # convert digit idx to chars
        return self._process_strings(decoded, remove_rep=True)


class BeamDecoder(Decoder):
    "Beam search 解码。解码结果为整个序列概率的最大值"
    def __init__(self, int2char, beam_width = 200, blank_index = 0, space_idx = 28):
        self.beam_width = beam_width
        super(BeamDecoder, self).__init__(int2char, space_idx=space_idx, blank_index=blank_index)

        import BeamSearch
        self._decoder = BeamSearch.ctcBeamSearch(int2char, beam_width, blank_index = blank_index)

    def decode(self, prob_tensor, frame_seq_len=None):
        """解码函数
        Args:
            prob_tensor   :   网络模型输出
            frame_seq_len :   每一样本的帧数
        Returns:
            res           :   解码得到的string，即识别结果
        """
        probs = prob_tensor.transpose(0, 1)
        res = self._decoder.decode(probs, frame_seq_len)
        return res

