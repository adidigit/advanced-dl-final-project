#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os
#from selectors import EpollSelector

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
from torch.utils.data import random_split

import matplotlib.pyplot as plt
from utils_c import load_txt
from torch.utils.data import Subset, ConcatDataset

import time

import models
from utils import progress_bar
from dataset import CIFAR10C, CIFAR100C

#%matplotlib inline

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    #parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    #parser.add_argument('--resume', '-r', action='store_true',
    #                    help='resume from checkpoint')
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type (default: ResNet18)')
    # from paper: DenseNet190,RsNet18,
    # we want to check: ResNet101
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--dir', default='cifar10_cifar10C_resnet18', type=str, help='directory of checkpoint file')
    parser.add_argument('--checkpoint', default='ckpt.t7_ResNet18_CIFAR10_CIFAR10C_last', type=str, help='name checkpoint file')
    parser.add_argument('--seed', default=20170922, type=int, help='random seed')
    #parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    #parser.add_argument('--epoch', default=200, type=int,
    #                    help='total epochs to run')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    #parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    #parser.add_argument('--alpha', default=1., type=float,
    #                    help='mixup interpolation coefficient (default: 1)')
    #parser.add_argument('--dataset', default="CIFAR10", type=str, help='data set')
    parser.add_argument('--testset', default="CIFAR10C", type=str, help='data set')
    #parser.add_argument('--testset', default="CIFAR10", type=str, help='data set')
    args = parser.parse_args()


    my_data_root = 'C:/Users/naama-alon/data'

    use_cuda = torch.cuda.is_available()

    if args.seed != 0:
        torch.manual_seed(args.seed)

    # Data
    print('==> Preparing data..')
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.testset=="CIFAR10":
        print('Chose CIFAR10 Dataset..')

        testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False, num_workers=8)
    elif args.testset=="CIFAR100":
        print('Chose CIFAR100 Dataset..')

        testset = datasets.CIFAR100(root='~/data', train=False, download=False,
                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False, num_workers=8)

    elif args.testset=="CIFAR10C":
        print('Chose CIFAR10 Corrupted Trainset..')
        corruptions = load_txt('./corruptions.txt')

        for i, cname in enumerate(corruptions):
            tmp_dataset = CIFAR10C(root=os.path.join(my_data_root, 'CIFAR-10-C'),name=cname,
                                        transform=transform_train)
            start= 20000
            stop = 30000
            indices = [i for i in range(start, stop)] # use sevirity 3
            sev3 = Subset(tmp_dataset, indices)
            split_lengths = [int(len(sev3)*0.833), int(len(sev3)*0.167)]
            _, sev3_testset = random_split(sev3, split_lengths)
            if i==0:
                testset_arr = [sev3_testset]
            else:
                testset_arr.append(sev3_testset)

        testset =  ConcatDataset(testset_arr)

        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False, num_workers=8)

    elif args.testset=="CIFAR100C":
        print('Chose CIFAR100 Corrupted Trainset..')
        corruptions = load_txt('./corruptions.txt')

        for i, cname in enumerate(corruptions):
            tmp_dataset = CIFAR100C(root=os.path.join(my_data_root, 'CIFAR-100-C'),name=cname,
                                        transform=transform_train)
            start= 20000
            stop = 30000
            indices = [i for i in range(start, stop)] # use sevirity 3
            sev3 = Subset(tmp_dataset, indices)
            split_lengths = [int(len(sev3)*0.833), int(len(sev3)*0.167)]
            _, sev3_testset = random_split(sev3, split_lengths)
            if i==0:
                testset_arr = [sev3_testset]
            else:
                testset_arr.append(sev3_testset)

        testset =  ConcatDataset(testset_arr)

        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False, num_workers=8)



    # Model
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.dir + '/'+ args.checkpoint)
    net = checkpoint['net']
    #best_acc = checkpoint['acc']
    #start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
     
    if not os.path.isdir('test_results'):
        os.mkdir('test_results')
    logname = ('test_results/log_' + net.__class__.__name__ + '_' + args.testset
            + '.csv')

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()


    def test():  
        #global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            with torch.no_grad():    
                outputs = net(inputs)
                loss = criterion(outputs, targets)

            # test_loss += loss.data[0]
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),
                        'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total,
                            correct, total))
        acc = 100.*correct/total

        return (test_loss/batch_idx, acc)


    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            #logwriter.writerow(['epoch', 'test loss', 'test acc', 'test err'])
            logwriter.writerow(['test loss', 'test acc', 'test err'])


    test_loss, test_acc = test()
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([test_loss,test_acc, (100-test_acc)])
    
if __name__ == '__main__':
    tic = time.perf_counter()
    main()
    toc = time.perf_counter()
    print(f"Run for: {(((toc - tic)/60)):0.4f} minutes.")