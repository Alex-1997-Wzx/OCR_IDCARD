import os
import lmdb


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
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
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    # lmdb 输出目录
    outputPath = '/media/alton/Data/Documents/DataSet/DataFountain/valid/lmdb/agency/'
    txt_path = '/media/alton/Data/Documents/DataSet/DataFountain/valid/lmdb/agency/path_label.txt'

    with open(txt_path, 'r') as f:
        path_and_label = f.readlines()
    # 路径　path_and_label[0].split()[0] 标签　path_and_label[0].split()[1]
    imgPaths = []
    txtLists = []
    for i in range(len(path_and_label)):
        imgPaths.append(path_and_label[i].split()[0])  # 图片路径
        txtLists.append(path_and_label[i].split()[1])  # 标签内容

    createDataset(outputPath, imgPaths, txtLists)
