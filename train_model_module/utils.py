# -- coding: utf-8 --
# @Time : 2020/3/10 下午8:32
# @Author : Gao Shang
# @File : utils.py
# @Software : PyCharm

import collections
import torch


class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            # self.dict[char] = i + 1
            self.dict[char] = i

    def encode(self, text, depth=0):
        """Support batch or single str."""
        if isinstance(text, str):
            text = [self.dict.get(char, 0) for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(text)]
            text = ''.join(str(v) for v in text)
            text, _ = self.encode(text)
        if depth:
            return text, len(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


class averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        self.n_count += v.data.numel()
        # NOTE: not `+= v.sum()`, which will add a node in the compute graph,
        # which lead to memory leak
        self.sum += v.data.sum()

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def loadData(v, data):
    v.resize_(data.size()).copy_(data)
