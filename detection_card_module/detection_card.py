# -- coding: utf-8 --
# @Time : 2020/3/18 下午9:03
# @Author : Gao Shang
# @File : detection_card.py
# @Software : PyCharm

import cv2
import numpy as np
import os

"""

    Description: 检测OCR扫描件中身份证的顶点坐标
    输入参数: 待处理的OCR扫描件
    输出结果: 身份证正反面的四对顶点坐标,顺序为: 左上　右上　右下　左下

"""


def image_filter(image, image_name='1.jpg', save_path='./save_path'):
    """
    对图片进行灰度和滤波操作
    :param image: 待处理的图片
    :param image_name: 图片名称，测试使用
    :param save_path: 结果保存路径，测试使用
    :return: 滤波后的图片
    """
    # 转换为灰度图
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 将结果保存到文件，测试使用
    # cv2.imwrite(os.path.join(save_path, image_name.split('.')[0] + '_gray.jpg'), image_gray)

    # 对图像进行滤波操作，采用锐化内核，使图像更清晰
    img_filter = cv2.filter2D(image, -1,
                              kernel=np.array([[0, -1, 0],  # 锐化内核
                                               [-1, 5, -1],
                                               [0, -1, 0]], np.float32))
    # 二次滤波，效果更好
    img_filter = cv2.filter2D(img_filter, -1,
                              kernel=np.array([[0, -1, 0],
                                               [-1, 5, -1],
                                               [0, -1, 0]], np.float32))

    # 将结果保存到文件，测试使用
    # cv2.imwrite(os.path.join(save_path, image_name.split('.')[0] + '_filter.jpg'), img_filter)

    return img_filter


def image_binary(image, image_name='1.jpg', save_path='./save_path'):
    """
    对滤波后的图片进行二值化
    :param image: 滤波后的图片
    :param image_name: 图片名称，测试使用
    :param save_path: 保存路径，测试使用
    :return: 二值图
    """
    # 计算X方向梯度
    grad_X = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0)
    # 计算Y方向梯度
    grad_Y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1)

    # 矩阵相减并取绝对值
    img_gradient = cv2.subtract(grad_X, grad_Y)
    img_gradient = cv2.convertScaleAbs(img_gradient)

    # 对图片进行二值化操作
    img_binary = cv2.adaptiveThreshold(img_gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -3)

    # 将结果保存到文件，测试使用
    # cv2.imwrite(os.path.join(save_path, image_name.split('.')[0] + '_binary.jpg'), img_binary)

    # 图像形态学处理
    # 获取十字形结构的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 闭运算，去除小黑洞区域
    img_closed = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)
    # 开运算，消除小黑点区域,具有平滑较大物体边界时并不明显改变面积的特点
    img_closed = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel)
    # 腐蚀操作，变瘦
    img_closed = cv2.erode(img_closed, None, iterations=9)
    # 膨胀操作，变胖
    img_closed = cv2.dilate(img_closed, None, iterations=9)

    return img_closed


def point_distance(p1, p2):
    """
    计算两点间距离
    :param p1: 坐标1
    :param p2: 坐标2
    :return: 两点间距离的平方
    """
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    result = (dx * dx + dy * dy) ** 0.5
    return result


def point_sort(center, card_point):
    """
    对身份证顶点坐标进行排序
    :param center: 最小矩形的中心坐标
    :param card_point: 矩形的顶点
    :return: 排序后的矩形顶点：左上，右上，右下，左下
    """
    left_point = []
    right_point = []
    for i in range(4):
        # X坐标比中心坐标大，则在右边，反之则在左边
        if card_point[i][0] > center[0]:
            right_point.append(card_point[i])
        else:
            left_point.append(card_point[i])

    # Y坐标大则在右下，反之则在右上
    if right_point[0][1] < right_point[1][1]:
        right_down = right_point[1]
        right_up = right_point[0]
    else:
        right_down = right_point[0]
        right_up = right_point[1]

    # Y坐标大则在左下，反之则在左上
    if left_point[0][1] < left_point[1][1]:
        left_down = left_point[1]
        left_up = left_point[0]
    else:
        left_down = left_point[0]
        left_up = left_point[1]

    points = np.array([left_up, right_up, right_down, left_down], dtype=np.float32)

    # 对矩形的顶点微调为445*280
    # 水平调整
    for i, j in [(0, 1), (2, 3)]:
        dist = point_distance(points[i], points[j])
        delta = (points[i] - points[j]) * (445 / dist - 1) / 2
        points[i] += delta
        points[j] += delta
    # 垂直调整
    for i, j in [(0, 3), (1, 2)]:
        dist = point_distance(points[i], points[j])
        delta = (points[i] - points[j]) * (280 / dist - 1) / 2
        points[i] += delta
        points[j] += delta

    return points


def getCardPoint(image):
    # 对图片进行锐化操作
    img_filter = image_filter(image)

    # 对图片进行二值化操作
    img_binary = image_binary(img_filter)

    # 检测身份证区域
    contours, _ = cv2.findContours(img_binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 存储顶点的列表
    res_point = []
    for i in range(0, len(contours)):
        # 面积占比在0.05~0.5的即为身份证区域
        card_area = cv2.contourArea(contours[i])
        if (card_area <= 0.5 * image.shape[0] * image.shape[1]) and (
                card_area >= 0.05 * image.shape[0] * image.shape[1]):
            # 获取最小外接矩形
            rect = cv2.minAreaRect(contours[i])
            # 返回最小外接矩形的中心坐标
            card_point = cv2.boxPoints(rect)
            # 对顶点坐标进行排序
            point = point_sort((int(rect[0][0]), int(rect[0][1])), card_point)
            res_point.append(point)

    # # 将检测到的区域在原图上标示，测试使用
    # print(res_point)
    # for point in res_point:
    #     cv2.line(image, tuple(point[0]), tuple(point[1]), 255)
    #     print(point[0], point[1])
    #     cv2.line(image, tuple(point[1]), tuple(point[2]), 200)
    #     print(point[1], point[2])
    #     cv2.line(image, tuple(point[2]), tuple(point[3]), 155)
    #     print(point[2], point[3])
    #     cv2.line(image, tuple(point[3]), tuple(point[0]), 100)
    #     print(point[3], point[0])
    # cv2.imshow('card_point', image)
    # cv2.waitKey(0)
    return res_point


if __name__ == '__main__':
    image_path = './test_images/'
    image = cv2.imread(os.path.join(image_path, '1.jpg'), 0)
    getCardPoint(image)
