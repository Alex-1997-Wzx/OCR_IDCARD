# -- coding: utf-8 --
# @Time : 2020/4/17 下午9:58
# @Author : Gao Shang
# @File : split_dataset.py
# @Software : PyCharm
"""
  将标签文本切分为9:1的训练集和验证集
"""

import os
import csv

# 将训练集和验证集中的图片名称转化到列表
train_path = '/media/alton/Data/Documents/DataSet/Synthetic Chinese String/train/images/'
train_list = os.listdir(train_path)
valid_path = '/media/alton/Data/Documents/DataSet/Synthetic Chinese String/valid/images/'
valid_list = os.listdir(valid_path)

# 创建训练集和验证集的文件对象
fo_train = open('/media/alton/Data/Documents/DataSet/Synthetic Chinese String/train/label/train_983883.csv', 'w')
csv_writer_train = csv.writer(fo_train)
fo_valid = open('/media/alton/Data/Documents/DataSet/Synthetic Chinese String/valid/label/valid_109320.csv', 'w')
csv_writer_valid = csv.writer(fo_valid)

# 打开总结果集，用于匹配图片名称
with open('/media/alton/Data/Documents/DataSet/Synthetic Chinese String/label/Train_Labels_Synthetic.txt', 'r') as f:
    result = list(f.readlines())
# 进行训练集匹配
for i in range(len(train_list)):
    for j in range(len(result)):
        if train_list[i] == result[j].split(' ')[0]:
            csv_writer_train.writerow(result[j])
            break

# 进行验证集匹配
for i in range(len(valid_list)):
    for j in range(len(result)):
        if valid_list[i] == result[j].split(' ')[0]:
            csv_writer_valid.writerow(result[j])
            break
