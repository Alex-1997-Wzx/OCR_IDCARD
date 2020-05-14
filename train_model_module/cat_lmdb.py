# -- coding: utf-8 --
# @Time : 2020/5/9 下午6:48
# @Author : Gao Shang
# @File : cat_lmdb.py
# @Software : PyCharm

import lmdb

outputPath = '/media/alton/Data/Documents/DataSet/DataFountain/train/lmdb/address/'
env = lmdb.open(outputPath)
txn = env.begin(write=False)
for key, value in txn.cursor():
    print(key, value)

env.close()
