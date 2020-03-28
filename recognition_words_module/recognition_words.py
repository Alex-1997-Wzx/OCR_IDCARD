# -- coding: utf-8 --
# @Time : 2020/3/20 下午5:38
# @Author : Gao Shang
# @File : recognition_words.py
# @Software : PyCharm


import os
import cv2
import torch
import numpy as np
from recognition_words_module import model
from recognition_words_module import decode
from torch.autograd import Variable
import torch.nn.functional as F
from recognition_words_module.alphabet import alphabet

absolute_path = os.path.dirname(__file__)

# with open('../data/alphabet.txt', 'r') as f:
#     alphabet = f.read().replace('\n', '')
# alphabet = [alphabet, '0123456789X-.长期']

# model = [model.chsNet(1, len(alphabet[0]) + 1), model.digitsNet(1, len(alphabet[1]) + 1)]
# if torch.cuda.is_available():
#     # 中文模型
#     model[0] = model[0].cuda()
#     # 数字模型
#     model[1] = model[1].cuda()
# model[0].load_state_dict({k.replace('module.', ''): v for k, v in torch.load(absolute_path + '/data/chs.pth').items()})
# model[1].load_state_dict({k.replace('module.', ''): v for k, v in torch.load(absolute_path + '/data/number.pth').items()})
#

# 定义词典，存储地区码和地址的对应关系
lex_sex = ['男', '女']
lex_nation = ['仡佬', '高山', '藏', '珞巴', '景颇', '门巴', '仫佬', '柯尔克孜',
              '畲', '维吾尔', '阿昌', '瑶', '裕固', '撒拉', '土', '塔塔尔',
              '侗', '傈僳', '傣', '崩龙', '苗', '达斡尔', '羌', '怒',
              '水', '哈尼', '乌孜别克', '鄂温克', '回', '汉', '赫哲', '壮',
              '黎', '布依', '保安', '土家', '鄂伦春', '佤', '哈萨克', '塔吉克',
              '毛难', '俄罗斯', '蒙古', '纳西', '独龙', '东乡', '布朗', '拉祜',
              '普米', '京', '彝', '朝鲜', '满', '白', '基诺', '锡伯']
lex_year = [str(i) for i in range(1958, 2009)]
lex_month = [str(i) for i in range(1, 13)]
lex_day = [str(i) for i in range(1, 32)]
lex_month_02d = ['%02d' % i for i in range(1, 13)]
lex_day_02d = ['%02d' % i for i in range(1, 32)]
lex_year_start = [str(i) for i in range(2009, 2020)]
lex_year_end = [str(i) for i in range(2014, 2040)]

lex_region = []  # 存储市级名称
lex_code = []  # 存储市级代码编号
region_code = {}
code_region = {}  # 代码对应行政区字典
for line in open(absolute_path + '/data/code_region.txt', encoding='gbk'):
    seg = line.strip().split(' ')
    if len(seg[0]) != 6:
        continue
    code_region[seg[0]] = seg[1]
    # 市级行政代码编号
    if seg[0][2:] == '0000':
        r1 = seg[1]
        lex_region.append(r1)
    elif seg[0][4:] == '00':
        r2 = seg[1]
        lex_region.append(r1 + r2)
    else:
        if lex_code[-1][4:] == '00':
            lex_code.pop()
            lex_region.pop()
            if lex_code[-1][2:] == '0000':
                lex_code.pop()
                lex_region.pop()
        lex_region.append(r1 + r2 + seg[1])
    lex_code.append(seg[0])
for i in range(len(lex_code)):
    region_code[lex_region[i]] = lex_code[i]

lexicon = [decode.Lexicon(lex_sex, alphabet[0]), decode.Lexicon(lex_nation, alphabet[0]),
           decode.Lexicon(lex_year, alphabet[1]), decode.Lexicon(lex_month, alphabet[1]),
           decode.Lexicon(lex_day, alphabet[1]),
           decode.Lexicon(lex_month_02d, alphabet[1]), decode.Lexicon(lex_day_02d, alphabet[1]),
           decode.Lexicon(lex_region, alphabet[0]), decode.Lexicon(lex_code, alphabet[1]),
           decode.Lexicon(lex_year_start, alphabet[1]), decode.Lexicon(lex_year_end, alphabet[1])]


def getRegionByCode(code):
    if code not in code_region:
        return None
    cs = code[:2] + '0000'
    rs = code_region[cs]
    if cs == code:
        return rs
    cs = code[:4] + '00'
    if len(code_region[cs]) > 1:
        rs += code_region[cs]
    if cs == code:
        return rs
    return rs + code_region[code]


def getBureauByCode(code):
    if code not in code_region:
        return None
    cs = code[:4] + '00'
    if len(code_region[cs]) > 1:
        rs = code_region[cs]
    else:
        rs = ''
    return rs + code_region[code] + '公安局'


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


def getWordsResult(images, model):

    name = decode.index_str(networkOutput(images[0], model[0]), alphabet[0])

    sex, _ = decode.wordBeamSearch(networkOutput(images[1], model[0]), lexicon[0])

    nation, _ = decode.wordBeamSearch(networkOutput(images[2], model[0]), lexicon[1])

    year = decode.index_str(networkOutput(images[3], model[1]), alphabet[1])
    month = decode.index_str(networkOutput(images[4], model[1]), alphabet[1])
    day = decode.index_str(networkOutput(images[5], model[1]), alphabet[1])

    addr_output = networkOutput(images[6], model[0])
    region, score, t = decode.prefixBeamSearch(addr_output, lexicon[7])
    code_output = networkOutput(images[7], model[1])
    id_code, id_conf = decode.wordBeamSearch(code_output, lexicon[8])
    id_tail = decode.index_str(networkOutput(images[11], model[1]), alphabet[1])
    if region not in region_code:
        region = getRegionByCode(id_code)
        t, _ = decode.prefixMatch(addr_output, alphabet[0], region)
    elif region_code[region] != id_code:
        _, id_conf1 = decode.prefixMatch(code_output, alphabet[1], region_code[region])
        t1, conf1 = decode.prefixMatch(addr_output, alphabet[0], getRegionByCode(id_code))
        if score * id_conf1 < conf1 * id_conf:  # code is more confident
            t = t1
            region = getRegionByCode(id_code)
        else:  # region is more confident
            id_code = region_code[region]
    else:  # just wanna align the region
        t, _ = decode.prefixMatch(addr_output, alphabet[0], region)

    address = region + decode.index_str(addr_output[t + 1:], alphabet[0])
    idnumber = id_code + year + month + day + id_tail
    agency = getBureauByCode(id_code)

    if len(images) == 16:
        # 长期
        valid_year, _ = decode.wordBeamSearch(networkOutput(images[13], model[1]), lexicon[9])
        valid_month, _ = decode.wordBeamSearch(networkOutput(images[14], model[1]), lexicon[5])
        valid_day, _ = decode.wordBeamSearch(networkOutput(images[15], model[1]), lexicon[6])
        valid_date = valid_year + '.' + valid_month + '.' + valid_day + '-长期'
    else:
        valid_year_start = networkOutput(images[13], model[1])
        valid_year, score = decode.wordBeamSearch(valid_year_start, lexicon[9])

        valid_year_end = networkOutput(images[16], model[1])
        valid_year1, score1 = decode.wordBeamSearch(valid_year_end, lexicon[10])
        if int(valid_year1) - int(valid_year) not in [5, 10, 20]:
            if score1 > score:
                score = 0
                for i in [5, 10, 20]:
                    year = str(int(valid_year1) - i)
                    _, year_score = decode.prefixMatch(valid_year_start, alphabet[1], year)
                    if year_score > score:
                        valid_year = year
                        score = year_score
            else:
                score1 = 0
                for i in [5, 10, 20]:
                    year = str(int(valid_year) + i)
                    _, year_score = decode.prefixMatch(valid_year_end, alphabet[1], year)
                    if year_score > score1:
                        valid_year1 = year
                        score1 = year_score
        valid_month, score = decode.wordBeamSearch(networkOutput(images[14], model[1]), lexicon[5])
        valid_month1, score1 = decode.wordBeamSearch(networkOutput(images[17], model[1]), lexicon[5])
        if valid_month != valid_month1 and score1 > score:
            valid_month = valid_month1

        valid_day, score = decode.wordBeamSearch(networkOutput(images[15], model[1]), lexicon[6])
        valid_day1, score1 = decode.wordBeamSearch(networkOutput(images[18], model[1]), lexicon[6])
        if valid_day != valid_day1 and score1 > score:
            valid_day = valid_day1

        valid_date = '.' + valid_month + '.' + valid_day
        valid_date = valid_year + valid_date + '-' + valid_year1 + valid_date

    return [name, sex, nation, year, month, day, address, idnumber, agency, valid_date]
