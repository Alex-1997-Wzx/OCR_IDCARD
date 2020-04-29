# -- coding: utf-8 --
# @Time : 2020/3/20 下午6:34
# @Author : Gao Shang
# @File : validation.py
# @Software : PyCharm

import cv2
import os
import csv
from detection_card_module import detection_card
from detection_text_module import detection_text
from recognition_words_module import recognition_words


def recognition(path):
    """
    调用各模块进行识别文字
    :param path: 图片目录路径
    :return: 识别结果和花费时间
    """
    files = [file for file in os.listdir(path)]
    with open('/media/alton/Data/Documents/DataSet/DataFountain/label/valid_1000.csv', 'r') as f:
        result = list(csv.reader(f))
    count = 0
    for name in files:
        try:

            image = cv2.imread(os.path.join(path, name), 0)
            points = detection_card.getCardPoint(image)
            text = detection_text.getTextLine(image, points)
            words = recognition_words.getWordsResult(text)

            for i in range(len(result)):
                if name.split('.jpg')[0] == result[i][0]:
                    for j in range(10):
                        if words[j] == result[i][j+1]:
                            count += 1
        except (RuntimeError, IndexError) as e:
            continue

        print(count / (len(files)*10.0))


if __name__ == '__main__':
    image_path = '/media/alton/Data/Documents/DataSet/DataFountain/valid/images/'
    recognition(image_path)
