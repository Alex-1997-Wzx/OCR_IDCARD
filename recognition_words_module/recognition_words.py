# -- coding: utf-8 --
# @Time : 2020/3/20 下午5:38
# @Author : Gao Shang
# @File : recognition_words.py
# @Software : PyCharm


import cv2
import torch
import numpy as np
from recognition_words_module import networks
from torch.autograd import Variable
import torch.nn.functional as F
from recognition_words_module.alphabet import alphabet

# with open('../data/alphabet.txt', 'r') as f:
#     alphabet = f.read().replace('\n', '')
alphabet = [alphabet, '0123456789X-.长期']
model = [networks.chsNet(1, len(alphabet[0]) + 1), networks.digitsNet(1, len(alphabet[1]) + 1)]
if torch.cuda.is_available():
    # 中文模型
    model[0] = model[0].cuda()
    # 数字模型
    model[1] = model[1].cuda()
model[0].load_state_dict({k.replace('module.', ''): v for k, v in torch.load('recognition_words_module/data/chs.pth').items()})
model[1].load_state_dict({k.replace('module.', ''): v for k, v in torch.load('recognition_words_module/data/number.pth').items()})


def index_str(preds, alphabet):
    channels = preds.argmax(dim=1)
    char_list = []
    for i in range(channels.size(0)):
        if channels[i] != 0 and (not (i > 0 and channels[i - 1] == channels[i])):
            char_list.append(alphabet[channels[i] - 1])
    return ''.join(char_list)


def networkOutput(image, model):
    imgH = 22
    if image.shape[0] != imgH:
        image = cv2.resize(image, (max(int(imgH * image.shape[1] / image.shape[0]), imgH), imgH),
                           cv2.INTER_LINEAR)

    image = torch.from_numpy(image.astype(np.float32))
    if torch.cuda.is_available():
        image = image.cuda()
    image = Variable(image.view(1, 1, *image.size()))
    model.eval()
    preds = model(image)
    preds = preds.view(preds.size(0), -1)
    preds = F.softmax(preds, dim=1)

    return preds.data


def getWordsResult(images, beam_decode=False):
    # 无字典修正
    if not beam_decode:
        name = index_str(networkOutput(images[0], model[0]), alphabet[0])
        sex = index_str(networkOutput(images[1], model[0]), alphabet[0])
        nation = index_str(networkOutput(images[2], model[0]), alphabet[0])
        year = index_str(networkOutput(images[3], model[1]), alphabet[1])
        month = index_str(networkOutput(images[4], model[1]), alphabet[1])
        day = index_str(networkOutput(images[5], model[1]), alphabet[1])
        address = index_str(networkOutput(images[6], model[0]), alphabet[0])
        idnumber = index_str(networkOutput(images[7], model[1]), alphabet[1]) + \
                   index_str(networkOutput(images[8], model[1]), alphabet[1]) + \
                   index_str(networkOutput(images[9], model[1]), alphabet[1]) + \
                   index_str(networkOutput(images[10], model[1]), alphabet[1]) + \
                   index_str(networkOutput(images[11], model[1]), alphabet[1])

        agency = index_str(networkOutput(images[12], model[0]), alphabet[0])
        date = index_str(networkOutput(images[13], model[1]), alphabet[1]) + '.' + \
               index_str(networkOutput(images[14], model[1]), alphabet[1]) + '.'
               # index_str(networkOutput(images[15], model[1]), alphabet[1]) + '-'
        if len(images) == 19:
            date += index_str(networkOutput(images[16], model[1]), alphabet[1]) + '.' + \
                    index_str(networkOutput(images[17], model[1]), alphabet[1]) + '.' + \
                    index_str(networkOutput(images[18], model[1]), alphabet[1])
        else:
            date += '长期'

        return [name, sex, nation, year, month, day, address, idnumber, agency, date]
