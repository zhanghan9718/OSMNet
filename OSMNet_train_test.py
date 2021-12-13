#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
"""
This is OSMNet based on PyTorch

If you use this code, please cite
@article{
 author = {Han Zhang, Lin Lei, Weiping Ni, Tao Tang, Junzheng Wu, Deliang Xiang, Gangyao Kuang},
    title = "{Explore Better Network Framework for High Resolution Optical and SAR Image Matching}",
     year = 2021}
(c) 2021 by Han Zhang
"""

from __future__ import division, print_function

import argparse
import torch
import torch.nn.init
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random

from Utils import str2bool
import torch.nn as nn


import torch.utils.data.dataloader
import lmdb
import pyarrow as pa
import torch.utils.data as data
import os.path as osp

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet_Dense')
# Model options
# ./data/models/gf3_ge1/checkpoint_2.pth
parser.add_argument('--resume', default='./models/checkpoint_osmnet_wo.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--extent-pos', default=1, type=int,
                    help='Extent of positive samples on the ground truth map')
parser.add_argument('--test-only', type=bool, default=True,
                    help='')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)') # 0.01 for sgd
parser.add_argument('--suffix', default='gf3_ge1_att/',
                    help='suffix of trained modal')
parser.add_argument('--dataroot', type=str,
                    default='/DataS/zhanghan_data/lmdb/',
                    help='path to dataset')
parser.add_argument('--name-train', type=str, default='osdataset_train.lmdb',
                    help='name of training dataset')
parser.add_argument('--name-test', type=str, default='osdataset_test.lmdb', # mix7_test.lmdb
                    help='name of testing dataset')
parser.add_argument('--offset-test', default=6, type=int,
                    help='Offset value between test image pairs')
parser.add_argument('--enable-logging', type=str2bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='data/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--search-rad', default=32, type=int,
                    help='Search radius for fft match')
parser.add_argument('--num-workers', default=0, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True,
                    help='')
parser.add_argument('--mean-image', type=float, default=0.4309,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.2236,
                    help='std of train dataset for normalization')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=40, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                    help='input batch size for training (default: 1024)') # 5
parser.add_argument('--test-batch-size', type=int, default=10, metavar='BST',
                    help='input batch size for testing (default: 1024)') # 5
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--use-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = args.use_cuda and torch.cuda.is_available()

print(("NOT " if not args.cuda else "") + "Using cuda")

if args.cuda:
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

# set random seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class DatasetLMDB(data.Dataset):
    def __init__(self, db_path, db_name, transform=None):
        self.full_path = osp.join(db_path, db_name)
        self.transform = transform
        self.env = lmdb.open(self.full_path, subdir=osp.isdir(self.full_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        img_pair = loads_pyarrow(byteflow)

        img_sar_quart = img_pair[0]
        img_opt_quart = img_pair[1]

        img_sar_quart = self.transform(img_sar_quart)
        img_opt_quart = self.transform(img_opt_quart)

        return img_sar_quart, img_opt_quart

    def __len__(self):
        return self.length


def create_loaders():
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((args.mean_image,), (args.std_image,))])

    train_loader = torch.utils.data.DataLoader(
        DatasetLMDB(db_path=args.dataroot,
                    db_name=args.name_train,
                    transform=transform),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        DatasetLMDB(db_path=args.dataroot,
                    db_name=args.name_test,
                    transform=transform),
        batch_size=args.test_batch_size,
        shuffle=False, **kwargs)

    return train_loader, test_loader


def fft_match_batch(feature_sar, feature_opt, search_rad):

    search_rad = int(search_rad)

    b, c, w, h = np.shape(feature_sar)
    nt = search_rad
    # torch.set_default_tensor_type(torch.DoubleTensor)
    T = torch.zeros(np.shape(feature_sar))
    T[:, :, 0:h - 2 * nt, 0:w - 2 * nt] = 1
    fake_imag = torch.zeros(np.shape(feature_sar))
    if args.use_cuda:
        T = T.cuda()
        fake_imag = fake_imag.cuda()

    sen_x = feature_sar ** 2

    tmp1 = torch.fft.fft2(sen_x)
    tmp2 = torch.fft.fft2(T)

    # zhh = tmp1*tmp2

    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)

    ssd_f_1 = torch.fft.ifft2(tmp_sum)



    ssd_fr_1 = torch.real(ssd_f_1)
    # ssd_fr_1 = ssd_f_1[:, :, :, 0]
    ssd_fr_1 = ssd_fr_1[:, 0:2 * nt + 1, 0:2 * nt + 1]

    ref_T = feature_opt[:, :, nt:w - nt, nt:h - nt]
    ref_Tx = torch.zeros(np.shape(feature_opt))
    ref_Tx[:, :, 0:w - 2 * nt, 0:h - 2 * nt] = ref_T

    if args.use_cuda:
        ref_Tx = ref_Tx.cuda()

    tmp1 = torch.fft.fft2(feature_sar)
    tmp2 = torch.fft.fft2(ref_Tx)

    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)
    ssd_f_2 = torch.fft.ifft2(tmp_sum)
    # tmp2x = torch.conj(tmp2)



    ssd_fr_2 = torch.real(ssd_f_2)
    # ssd_fr_2 = ssd_f_2[:, :, :, 0]

    ssd_fr_2 = ssd_fr_2[:, 0:2 * nt + 1, 0:2 * nt + 1]

    ssd_batch = (ssd_fr_1 - 2 * ssd_fr_2) / w / h

    return ssd_batch



def loss_fft_match_batch(out_sar, out_opt, gt_map, search_rad):

    bs = out_sar.shape[0]
    if bs != args.batch_size:
        return 0
    # eps = 1e-10
    out = fft_match_batch(out_sar, out_opt, search_rad)

    out = 1.0-out
    gamma = 32 # 32
    margin = 0.25 # 0.25
    gt_map_neg = 1-gt_map
    gt_mapx = gt_map.type(dtype=torch.bool)
    sp = out[gt_mapx]
    gt_map_negx = gt_map_neg.type(dtype=torch.bool)
    sn = out[gt_map_negx]

    sp = sp.view(out.size()[0],-1)
    sn = sn.view(out.size()[0],-1)

    ap = torch.clamp_min(-sp.detach() + 1 + margin, min=0.)
    an = torch.clamp_min(sn.detach() + margin, min=0.)

    delta_p = 1-margin
    delta_n = margin

    logit_p = -ap * (sp - delta_p) * gamma
    logit_n = an * (sn - delta_n) * gamma

    soft_plus = nn.Softplus()
    loss_circle = soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))

    loss_regu = loss_circle.mean()

    return loss_regu



def train(train_loader, model, gt_maps, optimizer, epoch, logger):
    # switch to train mode
    model.train()
    pbar = tqdm(enumerate(train_loader))

    gt_map_rnd = torch.zeros_like(gt_maps[0])
    if args.use_cuda:
        gt_map_rnd = gt_map_rnd.cuda()

    for batch_idx, data in pbar:
        data_sar, data_opt = data
        data_sar = data_sar[:, :, 24:232, 24:232]

        i = random.randint(-15, 15)
        j = random.randint(-15, 15)

        data_opt = data_opt[:, :, 24 + i:232 + i, 24 + j:232 + j]

        gt_map_rnd.fill_(0)
        gt_map_rnd[:, args.search_rad - args.extent_pos + i:args.search_rad + args.extent_pos + 1 + i,
        args.search_rad - args.extent_pos + j:args.search_rad + args.extent_pos + 1 + j] = 1

        if args.cuda:
            data_sar, data_opt = data_sar.cuda(), data_opt.cuda()

        out_opt, out_sar = model(data_opt, data_sar)

        loss = loss_fft_match_batch(out_sar, out_opt, gt_map_rnd, args.search_rad)
        if loss != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adjust_learning_rate(optimizer)
            if batch_idx % args.log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_sar), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.item()))

    if (args.enable_logging):
        logger.log_value('loss', loss.item()).step()
    suffix = args.suffix
    try:
        os.stat('{}{}'.format(args.model_dir, suffix))
    except:
        os.makedirs('{}{}'.format(args.model_dir, suffix))

    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir, suffix, epoch))


def test(test_loader, model, gt_maps, epoch, logger):
    # switch to evaluate mode
    model.eval()

    results = []
    pos_max = []
    gt_map_vec = torch.reshape(gt_maps[1][0], [1, (2*args.search_rad+1)*(2*args.search_rad+1)])

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_sar, data_opt) in pbar:
            data_sar = data_sar[:, :, 24:232, 24:232]
            data_opt = data_opt[:, :, 24 + args.offset_test:232 + args.offset_test, 24 + args.offset_test:232 + args.offset_test]
            if args.cuda:
                data_sar, data_opt = data_sar.cuda(), data_opt.cuda()

            out_opt, out_sar = model(data_opt, data_sar)

            out = fft_match_batch(out_sar, out_opt, args.search_rad)

            out_vec = torch.reshape(out, [out.shape[0], (2*args.search_rad+1)*(2*args.search_rad+1)])
            out_min = out_vec.min(1)[1]
            for i in range(out.shape[0]):
                rst = gt_map_vec[0, out_min[i]]
                results.append(rst)
                pos_max.append(out_min[i])

    # np.savetxt('pos_max.txt', pos_max, delimiter=' ', fmt='%10.5f')
    correct_rate = np.sum(results) / len(results)
    print('Validation Results: ', correct_rate)
    return correct_rate


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        group['lr'] = args.lr * (
                1.0 - float(group['step']) * float(4*args.batch_size) / (233244*4 * float(args.epochs)))  # 63338 for spring, 233244 for all
    return


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, gt_map, logger, file_logger):
    # print the experiment configuration
    print('\nparsed options:\n{}\n'.format(vars(args)))

    # if (args.enable_logging):
    #    file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))

    if args.cuda:
        model.cuda()

    optimizer1 = create_optimizer(model, args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))

    start = args.start_epoch
    end = start + args.epochs
    for epoch in range(start, end):
        if not args.test_only:
            # iterate over test loaders and test results
            train(train_loader, model, gt_map, optimizer1, epoch, logger)

        test(test_loaders, model, gt_map, epoch, logger)


if __name__ == '__main__':


    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = os.path.join(args.log_dir, args.suffix)

    logger, file_logger = None, None

    from OSMNet import SSLCNetPseudo
    model = SSLCNetPseudo()

    for param in model.parameters():
        print(type(param.data), param.size())

    # if (args.enable_logging):
    #     from Loggers import Logger, FileLogger
    #
    #     logger = Logger(LOG_DIR)
    #     # file_logger = FileLogger(./log/+suffix)

    gt_map = torch.zeros(args.batch_size, 2 * args.search_rad + 1, 2 * args.search_rad + 1)
    gt_map[:, args.search_rad - args.extent_pos:args.search_rad + args.extent_pos + 1,
    args.search_rad - args.extent_pos:args.search_rad + args.extent_pos + 1] = 1

    offset_zh = args.offset_test
    gt_map_shift = torch.zeros(args.batch_size, 2 * args.search_rad + 1, 2 * args.search_rad + 1)
    gt_map_shift[:, args.search_rad - args.extent_pos+offset_zh:args.search_rad + args.extent_pos + 1+offset_zh,
    args.search_rad - args.extent_pos+offset_zh:args.search_rad + args.extent_pos + 1+offset_zh] = 1

    if args.use_cuda:
        gt_map = gt_map.cuda()
        gt_map_shift = gt_map_shift.cuda()
    train_loader, test_loaders = create_loaders()
    gt_maps = [gt_map, gt_map_shift]
    main(train_loader, test_loaders, model, gt_maps, logger, file_logger)


