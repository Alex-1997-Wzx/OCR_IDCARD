# -- coding: utf-8 --
# @Time : 2020/3/10 下午8:30
# @Author : Gao Shang
# @File : cnn_main.py
# @Software : PyCharm

from __future__ import print_function
import argparse
import os
import random
from networks import chsNet
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from networks_train import utils
from networks_train import dataset
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from networks_train.chinese import alphabet
from torchvision.transforms import transforms

parser = argparse.ArgumentParser()
parser.add_argument(
    '--trainroot', help='path to dataset', default='./data/lmdb/test/train')
parser.add_argument(
    '--valroot', help='path to dataset', default='./data/lmdb/test/val')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--batchSize', type=int, default=2, help='input batch size')  # 128
parser.add_argument(
    '--imgH',
    type=int,
    default=22,  # 32
    help='the height of the input image to network')
parser.add_argument(
    '--imgW',
    type=int,
    default=220,  # 256
    help='the width of the input image to network')
parser.add_argument(
    '--niter', type=int, default=1, help='训练的epoch次数')  # 1000000
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,  # 0.00005
    help='learning rate for Critic, default=0.00005')
parser.add_argument(
    '--crnn',
    help="path to crnn (to continue training)",
    default=
    '/home/alton/桌面/CCF-OCR-master/networks_train/model_data/model_chs.pth')

# parser.add_argument('--crnn', help="path to crnn (to continue training)", default='')
parser.add_argument('--alphabet', default=alphabet)
parser.add_argument(
    '--experiment',
    help='Where to store samples and models',
    default='./save_model')
parser.add_argument(
    '--displayInterval', type=int, default=10, help='执行指定次数后显示Loss值')
parser.add_argument(
    '--valInterval', type=int, default=20, help='执行指定次数后计算损失值和正确率')  # 100
parser.add_argument(
    '--keep_ratio',
    action='store_true',
    help='whether to keep ratio for image resize')
parser.add_argument(
    '--random_sample',
    action='store_true',
    help='whether to sample the dataset with random sampler')
opt = parser.parse_args()

ifUnicode = True
if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
test_dataset = dataset.lmdbDataset(root=opt.valroot)

# transform=dataset.resizeNormalize((220, 22))transform=transforms.Grayscale())

assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(keep_ratio=True))


alphabet = opt.alphabet
nclass = len(alphabet) + 1
nc = 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 创建网络模型
crnn = chsNet(nc, len(alphabet) + 1)
# crnn.apply(weights_init)
# if opt.crnn != '':
#     print('loading pretrained model from %s' % opt.crnn)

crnn.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.crnn).items()})

image = torch.FloatTensor(opt.batchSize, 1, opt.imgH, opt.imgW)  # 3
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if torch.cuda.is_available():
    crnn = crnn.cuda()
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, test_dataset, criterion, max_iter=2):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.crnn).items()})
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        collate_fn=dataset.alignCollate(keep_ratio=True))

    val_iter = iter(val_loader)

    n_correct = 0
    loss_avg = utils.averager()

    image = torch.FloatTensor(opt.batchSize, 1, opt.imgH, opt.imgW)
    max_iter = min(max_iter, len(val_loader))
    for i in range(max_iter):
        data = val_iter.next()
        # i += 1
        cpu_images, cpu_texts = data
        # 输入的图片数
        batch_size = cpu_images.size(0)
        # print('cpu images', cpu_images, 'shape', cpu_images.size())
        utils.loadData(image, cpu_images)
        if ifUnicode:
            cpu_texts = [clean_txt(tx.encode('utf-8').decode('utf-8')) for tx in cpu_texts]
        t, l = converter.encode(cpu_texts)
        # 重新匹配尺寸
        utils.loadData(text, t)  # 文字索引
        utils.loadData(length, l)  # 文字

        image = cpu_images * 255
        image = image.cuda()
        preds = crnn(image)
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

    accuracy = n_correct / float(max_iter * opt.batchSize)
    testLoss = loss_avg.val()
    # print('Test loss: %f, accuray: %f' % (testLoss, accuracy))
    return testLoss, accuracy


def clean_txt(txt):
    """
    filter char where not in alphabet with ' '
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

    if ifUnicode:
        cpu_texts = [clean_txt(tx) for tx in cpu_texts]
    batch_size = cpu_images.size(0)
    # utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    image = cpu_images * 255
    image = image.cuda()
    preds = crnn(image)

    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds / 255, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    if flage:
        lr = 0.0001
        optimizer = optim.Adadelta(crnn.parameters(), lr=lr)
    optimizer.step()
    return cost


for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    for i in range(len(train_loader)):
        print('The step{} ........\n'.format(i))
        for p in crnn.parameters():
            p.requires_grad = True

        crnn.train()
        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            testLoss, accuracy = val(crnn, test_dataset, criterion)
            print('Test loss: %f, accuray: %f' % (testLoss, accuracy))
            # print("epoch:{},step:{},Test loss:{},accuracy:{},train loss:{}".
            #       format(epoch, num, testLoss, accuracy, loss_avg.val()))
            # loss_avg.reset()

            print('Save model to:', opt.experiment)
            torch.save(crnn.state_dict(), '{}/netCNN.pth'.format(opt.experiment))
        # do checkpointing
