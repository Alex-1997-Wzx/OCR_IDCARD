import os
import cv2
import lmdb  # install lmdb by "pip install lmdb"
import numpy as np
import glob
# from genLineText import GenTextImage


# 删除
def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
    """
    # print (len(imagePathList) , len(labelList))
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    print('开始创建LMDB文件')
    env = lmdb.open(outputPath, map_size=1099511627776)

    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i]).encode()
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


# 删除
def read_text(path):
    with open(path) as f:
        text = f.read()
    text = text.strip()

    return text


if __name__ == '__main__':

    # # lmdb 输出目录
    # outputPath = '../data/lmdb/test/train'
    # txt_path = '../data/dataline/mini_train/name/train_names_856.txt'
    #
    # with open(txt_path, 'r') as f:
    #     path_and_label = f.readlines()
    # print('------------', '图片总数：', len(path_and_label), '------------')
    # # 路径　path_and_label[0].split()[0] 标签　path_and_label[0].split()[1]
    # imgPaths = []
    # txtLists = []
    # for i in range(len(path_and_label)):
    #     imgPaths.append(path_and_label[i].split()[0]) # 图片路径
    #     txtLists.append(path_and_label[i].split()[1]) # 标签内容
    #
    # createDataset(outputPath, imgPaths, txtLists, lexiconList=None)


    # 源版本
    ##lmdb 输出目录
    outputPath = '../data/lmdb/train'

    path = '../data/dataline/*.jpg'
    imagePathList = glob.glob(path)
    print('------------', len(imagePathList), '------------')
    imgLabelLists = []
    for p in imagePathList:
        try:
            imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    ##sort by lebelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, txtLists, lexiconList=None)