# -- coding: utf-8 --
# @Time : 2020/3/20 下午6:34
# @Author : Gao Shang
# @File : main.py
# @Software : PyCharm


import cv2
import os
from detection_card_module import detection_card
from detection_text_module import detection_text
from recognition_words_module import recognition_words


def recognition(path):
    files = [file for file in os.listdir(path)]
    for name in files:
        image = cv2.imread(os.path.join(path, name), 0)

        points = detection_card.getCardPoint(image)
        text = detection_text.getTextLine(image, points)
        words = recognition_words.getWordsResult(text)

        # 输出结果，测试使用
        print('姓名；{name}\n性别：{sex}\n民族：{nation}\n'
          '出生日期：{brith_year}年{brith_month}月{brith_day}日\n'
          '住址：{address}\n身份证号码：''{number}\n签证机关：{organization}\n有效期限：{date}'.format(
        name=words[0], sex=words[1], nation=words[2], brith_year=words[3], brith_month=words[4],
        brith_day=words[5], address=words[6], number=words[7], organization=words[8], date=words[9]))
        print('----------------------')
    # return words


if __name__ == '__main__':
    test_path = './test_images'
    recognition(test_path)