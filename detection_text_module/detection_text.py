# -- coding: utf-8 --
# @Time : 2020/3/20 下午3:45
# @Author : Gao Shang
# @File : detection_text.py
# @Software : PyCharm


import cv2
import numpy as np
from scipy import signal


def removeWatermark(image, laplace):
    # 设置身份证的宽高为445*280
    card_size = (445, 280)
    # 水印laplace模板
    template_wm = [np.load('detection_text_module/template/watermark1_laplace.npy'), np.load('detection_text_module/template/watermark2_laplace.npy')]
    # 二值图模板用于生成蒙版以进行后续去水印操作
    img_wm = [np.load('detection_text_module/template/watermark1_solid.npy'), np.load('detection_text_module/template/watermark2_solid.npy') ]
    img_shape = [img.shape for img in img_wm]
    # 锐化内核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)
    mask = cv2.filter2D(laplace/255, -1, kernel=template_wm[0])
    irange, jrange = np.where(mask == np.max(mask))
    # 利用mask获取水印蒙版
    mask = np.zeros(mask.shape, np.uint8)
    for i in irange:
        for j in jrange:
            bi1 = i - (img_shape[0][0] // 2)
            bi2 = bi1 + img_shape[0][0]

            bj1 = j - (img_shape[0][1] // 2)
            bj2 = bj1 + img_shape[0][1]

            wi1 = 0
            wi2 = img_shape[0][0]

            wj1 = 0
            wj2 = img_shape[0][1]
            if bi1 < 0:
                wi1 = -bi1
                bi1 = 0
            elif bi2 > card_size[1]:
                wi2 -= bi2 - card_size[1]
                bi2 = card_size[1]

            if bj1 < 0:
                wj1 = -bj1
                bj1 = 0
            elif bj2 > card_size[0]:
                wj2 -= bj2-card_size[0]
                bj2 = card_size[0]
            mask[bi1:bi2, bj1:bj2] += img_wm[0][wi1:wi2, wj1:wj2]
    # 假设水印的灰度值为mc，以不透明度alpha叠加
    # 原图的灰度值为b，叠加后的灰度值a可由下式计算
    # a = mc * alpha * transparency + b * (1 - alpha * transparency)
    # transparency是水印自带的透明通道信息，边缘处透明渐变
    transparency = mask / np.max(mask)
    # 获取水印部分和身份证全图上面积最大的灰度值
    mask[mask < 100] = 0
    # 水印部分的最大灰度值
    hist_cv = cv2.calcHist([image], [0], mask, [256], [0, 256])
    a = np.where(hist_cv == np.max(hist_cv))[0][0]
    # 身份证全图最大灰度值
    hist_cv = cv2.calcHist([image], [0], None, [256], [0, 256])
    b = np.where(hist_cv == np.max(hist_cv))[0][0]
    mc = 60
    alpha = (b-a)/(b-mc)
    # 若水印完全不透明，则无法进行线性变换，设置上限
    if alpha >= 0.95:
        alpha = 0.95
    # 高度
    for i in range(mask.shape[0]):
        # 宽度
        for j in range(mask.shape[1]):
            if transparency[i, j]:
                trans = transparency[i, j] * alpha
                # 线性变换
                image[i, j] = max(min((image[i, j] - trans*mc) / (1-trans), b), 20)

    # 图像去模糊，laplace锐化
    image = cv2.filter2D(image, -1, kernel=kernel)
    # 为反变换后的灰度值设置合理上线
    image[image > b] = b

    return image


def getTextLine(image, points):
    # 设置身份证的宽高为445*280
    card_size = (445, 280)
    # "仅限BDCI比赛使用"水印出现在左上角，用于旋转矫正
    template_logo = np.load('detection_text_module/template/left-top_logo.npy')
    # 正反面Laplace模板，用于边缘检测误差造成的平移矫正
    # 用拉普拉斯而非二值图做模板，因为边缘信息更加丰富，定位更准确
    template_front = np.load('detection_text_module/template/front_laplace.npy')
    template_back = np.load('detection_text_module/template/back_laplace.npy')
    # 反面有小区第二项，判断是否长期
    template_validity = [np.load('detection_text_module/template/validity_longterm.npy'), np.load(
        'detection_text_module/template/validity_date.npy')]
    # 文本行候选框，rect: [top, bottom, left, right]
    front_text = np.load('detection_text_module/template/front_text_rect.npy')
    back_text = np.load('detection_text_module/template/back_text_rect.npy')

    # (0,0) (445, 0) (445, 280) (0, 280)
    target_pt = np.array([[0, 0], [card_size[0] - 1, 0],
                          [card_size[0] - 1, card_size[1] - 1],
                          [0, card_size[1] - 1]], np.float32)
    # 存储不同类型的图像
    card_image = []
    card_binary = []
    card_laplace = []

    for i in range(2):
        # 获取投射变化矩阵
        matrix = cv2.getPerspectiveTransform(points[i], target_pt)
        # 进行透视变换
        img_warp = cv2.warpPerspective(image, matrix, card_size)
        # 二值化
        _, img_binary = cv2.threshold(img_warp, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 使用二值图的左上区域进行旋转矫正，计算相关系数
        topleft_conv = signal.correlate2d(template_logo, img_binary[4:26, 2:152], mode='valid')
        bottomright_conv = signal.correlate2d(template_logo, img_binary[254:276, 293:443], mode='valid')
        if np.max(bottomright_conv) > np.max(topleft_conv):
            cv2.flip(img_binary, -1, img_binary)
            cv2.flip(img_warp, -1, img_warp)
        # 将结果保存到文件，测试使用
        # cv2.imwrite('./save_path' + str(i) + '_flip.jpg', img_warp)

        # 去图像水印
        laplace = cv2.Laplacian(img_warp, -1, ksize=5)
        # 将结果保存到文件，测试使用
        # cv2.imwrite('./save_path' + str(i) + '_image.jpg', img_warp)
        # cv2.imwrite('./save_path' + str(i) + '_image.jpg', laplace)
        img_warp = removeWatermark(img_warp, laplace)
        # 将结果保存到文件，测试使用
        # cv2.imwrite('./save_path' + str(i) + '_removewater.jpg'), img_warp

        # 使用去去水印后的图片重新生成二值图
        _, img_binary = cv2.threshold(img_warp, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # 去水印锐化后的身份证正反面：用于截取文本行
        # 去水印后的二值图：用于缩紧文本行候选框的右边界
        # Laplace图像：用于正反面判断及平移矫正
        card_image.append(img_warp)
        card_binary.append(img_binary)
        card_laplace.append(laplace)

    # 0为正面，1为反面
    front_conv = signal.correlate2d(template_front, card_laplace[0], mode='valid')
    back_conv = signal.correlate2d(template_back, card_laplace[1], mode='valid')
    sum = np.max(front_conv) + np.max(back_conv)
    # 1为正面，0为反面
    front_conv1 = signal.correlate2d(template_front, card_laplace[1], mode='valid')
    back_conv1 = signal.correlate2d(back_conv, card_laplace[0], mode='valid')
    sum1 = np.max(front_conv1) + np.max(back_conv1)
    if sum1 > sum:
        card_binary.reverse()
        card_image.reverse()
        front_conv = front_conv1
        back_conv = back_conv1

    text_imgs = []

    # 身份证正面
    i, j = np.where(front_conv == np.max(front_conv))
    i = i[0]
    j = j[0]
    text_rect = front_text.copy()
    k = 0
    while k < 14:
        # 调用存储的各要素位置
        loc = text_rect[k]
        # 调整偏移量
        loc[0] -= i
        loc[1] -= i
        loc[2] -= j
        loc[3] -= j
        if k in [0, 2, 6, 7, 8]:
            # 不定长文本，调整右边界
            hist = card_binary[0][loc[0]:loc[1], loc[2]:loc[3]].sum(axis=0) / 255
            if k in [7, 8] and len(np.where(hist[4:24] >= 3)[0]) < 5:
                k = 9
                continue
            right = len(hist) - 6
            while right >= 0:
                if hist[right] < 3:
                    right -= 1
                elif len(np.where(hist[max(0, right-20):right] >= 3)[0]) < 5:
                    right -= 20
                else:
                    break
            if right < 0:
                k += 1
                continue
            right = len(hist) - 6 -right
            loc[3] -= right
        if k in [7, 8]:
            text_imgs[-1] = cv2.hconcat((text_imgs[-1], card_image[0][loc[0]:loc[1], loc[2]:loc[3]]))
        else:
            text_imgs.append(card_image[0][loc[0]:loc[1], loc[2]:loc[3]])

        if k in [6, 7] and right > 20:
            k = 9
            continue
        k += 1

    # 身份证背面
    i, j = np.where(back_conv == np.max(back_conv))
    i = i[0]
    j = j[0]
    text_rect = back_text.copy()
    k = 0
    while k < 9:
        # 调用存储的各要素位置
        loc = text_rect[k]
        # 调整偏移量
        loc[0] -= i
        loc[1] -= i
        loc[2] -= j
        loc[3] -= j
        if k in [0, 1]:
            hist = card_binary[1][loc[0]:loc[1]-6, loc[2]:loc[3]].sum(axis=0) / 255
            if k == 1 and len(np.where(hist[4:24] >= 3)[0]) < 5:
                k =2
                continue
            right = len(hist) - 6
            while right >= 0:
                if hist[right] < 3:
                    right -= 1
                elif len(np.where(hist[max(0, right-20):right] >= 3)[0]) < 5:
                    right -= 20
                else:
                    break
            if right < 0:
                k += 1
                continue
            right = len(hist) -6 -right
            loc[3] -= right
        elif k == 5:
            date_img = card_binary[1][loc[0]:loc[1], loc[2]:loc[3]]
            if signal.correlate2d(template_validity[0], date_img, mode='valid')[0][0] >\
                signal.correlate2d(template_validity[1], date_img, mode='valid')[0][0]:
                break
            else:
                k = 6
                continue

        if k == 1:
            text_imgs[-1] = cv2.hconcat(((text_imgs[-1], card_image[1][loc[0]:loc[1], loc[2]:loc[3]])))
        else:
            text_imgs.append(card_image[1][loc[0]:loc[1], loc[2]:loc[3]])

        if k == 0 and right> 20:
            k = 2
            continue

        k += 1

    # 将结果保存到文件，测试使用
    # for i in range(len(text_imgs)):
        # cv2.imshow('text' + i, text_imgs[i])
    # cv2.waitKey(0)

    return text_imgs

