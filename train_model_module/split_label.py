# -- coding: utf-8 --
# @Time : 2020/5/9 下午5:57
# @Author : Gao Shang
# @File : split_label.py
# @Software : PyCharm

import os
import csv

# 将CSV文件中的类别分别保存
train_path = '/media/alton/Data/Documents/DataSet/Synthetic Chinese String/train/label/'
# valid_path = '/media/alton/Data/Documents/DataSet/DataFountain/valid/sex/'
file_list = os.listdir(train_path)
fo = open('/media/alton/Data/Documents/DataSet/DataFountain/valid/lmdb/sex/path_label.txt', 'w')

with open('/media/alton/Data/Documents/DataSet/DataFountain/label/valid_1000.csv', 'r') as f:
    result = list(csv.reader(f))
for i in range(len(result)):
    string = str(valid_path + result[i][0] + '.jpg' + ' ' + result[i][3]) + '\n'
    fo.write(string)

fo.close()
print('ok')
