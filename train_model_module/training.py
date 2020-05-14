# -- coding: utf-8 --
# @Time : 2020/3/10 下午8:30
# @Author : Gao Shang
# @File : training.py
# @Software : PyCharm

from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from train_model_module.model.networks import chsNet
from train_model_module import utils
from train_model_module import lmdb_dataset
from train_model_module.alphabet_chinese import alphabet
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

train_path = '/media/alton/Data/Documents/DataSet/DataFountain/train/lmdb/name'
valid_path = '/media/alton/Data/Documents/DataSet/DataFountain/valid/lmdb/name'
cnn_data = '/media/alton/Data/Documents/Alton_OCR_Project/recognition_words_module/data/chs.pth'
model_save = './save_model'
workers = 4
batchSize = 2
imgH = 22
imgW = 220
niter = 1000000
lr = 0.0005
display_loss = 10
display_accuray = 20
save_pth = 300

os.system('mkdir {0}'.format(model_save))

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

train_dataset = lmdb_dataset.lmdbDataset(root=train_path)
test_dataset = lmdb_dataset.lmdbDataset(root=valid_path)


assert train_dataset
sampler = lmdb_dataset.randomSequentialSampler(train_dataset, batchSize)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=False,
                                           sampler=sampler, num_workers=int(workers),
                                           collate_fn=lmdb_dataset.alignCollate(keep_ratio=True))

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# 初始化权值
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 创建网络模型
cnn = chsNet(1, len(alphabet) + 1)
cnn.apply(weights_init)
if cnn_data != '':
    print('loading pretrained model from %s' % cnn_data)

cnn.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cnn_data).items()})

image = torch.FloatTensor(batchSize, 1, imgH, imgW)  # 3
text = torch.IntTensor(batchSize * 5)
length = torch.IntTensor(batchSize)

if torch.cuda.is_available():
    cnn = cnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

optimizer = optim.RMSprop(cnn.parameters(), lr=lr)


def val(net, test_dataset, criterion, max_iter=2):
    print('Start val')

    for p in cnn.parameters():
        p.requires_grad = False

    net.eval()
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(cnn_data).items()})
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=batchSize,
        num_workers=int(workers),
        collate_fn=lmdb_dataset.alignCollate(keep_ratio=True))

    val_iter = iter(val_loader)

    n_correct = 0
    loss_avg = utils.averager()

    image = torch.FloatTensor(batchSize, 1, imgH, imgW)
    max_iter = min(max_iter, len(val_loader))
    for i in range(max_iter):
        data = val_iter.next()
        # i += 1
        cpu_images, cpu_texts = data
        # 输入的图片数
        batch_size = cpu_images.size(0)
        # print('cpu images', cpu_images, 'shape', cpu_images.size())
        utils.loadData(image, cpu_images)
        cpu_texts = [clean_txt(tx.encode('utf-8').decode('utf-8')) for tx in cpu_texts]
        t, l = converter.encode(cpu_texts)
        # 重新匹配尺寸
        utils.loadData(text, t)  # 文字索引
        utils.loadData(length, l)  # 文字

        image = cpu_images * 255
        image = image.cuda()
        preds = cnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        # 返回最大值和索引
        _, preds = preds.max(2)
        # 返回最大值的索引
        # print('max preds', preds)
        # preds = preds.squeeze(1)
        # 将tensor内存变为连续
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)

        print('preds', sim_preds, 'target', cpu_texts)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred.strip() == target.strip():
                n_correct += 1

    accuracy = n_correct / float(max_iter * batchSize)
    testLoss = loss_avg.val()
    # print('Test loss: %f, accuray: %f' % (testLoss, accuracy))
    return testLoss, accuracy


def clean_txt(txt):
    """
    删掉不在字符集中的汉字
    """
    newTxt = u''
    for t in txt:
        if t in alphabet:
            newTxt += t
        else:
            newTxt += u' '
    return newTxt


def trainBatch(net, criterion, optimizer, flage=False):
    data = train_iter.next()
    cpu_images, cpu_texts = data  # decode utf-8 to unicode
    cpu_texts = [clean_txt(tx) for tx in cpu_texts]
    batch_size = cpu_images.size(0)
    # utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    image = cpu_images * 255
    image = image.cuda()
    preds = cnn(image)

    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds / 255, text, preds_size, length) / batch_size
    cnn.zero_grad()
    cost.backward()
    if flage:
        lr = 0.0001
        optimizer = optim.Adadelta(cnn.parameters(), lr=lr)
    optimizer.step()
    return cost


for epoch in range(niter):
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        print('The step{} ........\n'.format(i))
        for p in cnn.parameters():
            p.requires_grad = True

        cnn.train()
        cost = trainBatch(cnn, criterion, optimizer)
        loss_avg.add(cost)

        if i % display_loss == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % display_accuray == 0:
            testLoss, accuracy = val(cnn, test_dataset, criterion)
            print('Test loss: %f, accuray: %f' % (testLoss, accuracy))
            # print("epoch:{},step:{},Test loss:{},accuracy:{},train loss:{}".
            #       format(epoch, num, testLoss, accuracy, loss_avg.val()))
            # loss_avg.reset()

        if i % save_pth == 0:
            print('Save model to:', model_save)
            torch.save(cnn.state_dict(), '{}/netCNN.pth'.format(model_save))
