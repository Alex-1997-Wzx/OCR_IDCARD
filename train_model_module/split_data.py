# -- coding: utf-8 --
# @Time : 2020/4/10 下午10:43
# @Author : Gao Shang
# @File : split_data.py
# @Software : PyCharm


# 将一个文件夹下图片按比例分在两个文件夹下，比例改0.7这个值即可

import os
import random
import shutil
from shutil import copy2
# trainfiles = os.listdir('/media/alton/Data/Documents/graduate_Python_OCR/data/train/')
trainfiles = os.listdir('/media/alton/Windows_10/first/1/')
num_train = len(trainfiles)
print("num_train: " + str(num_train))
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)
num = 0
# trainDir = '/media/alton/Data/Documents/DataSet/DataFountain/train/images/'
trainDir = '/media/alton/Data/Documents/DataSet/Synthetic Chinese String/train/images/'
# validDir = '/media/alton/Data/Documents/DataSet/DataFountain/valid/images/'
validDir = '/media/alton/Data/Documents/DataSet/Synthetic Chinese String/valid/images/'
for i in index_list:
    fileName = os.path.join('/media/alton/Windows_10/first/1/', trainfiles[i])
    if num < num_train*0.9:
        print(str(fileName))
        copy2(fileName, trainDir)
    else:
        copy2(fileName, validDir)
    num += 1
