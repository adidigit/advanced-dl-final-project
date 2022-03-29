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
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--model', default="ResNet18", type=str,
                        help='model type (default: ResNet18)')
    # from paper: DenseNet190,RsNet18,
    # we want to check: ResNet101
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=200, type=int,
                        help='total epochs to run')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                        help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    #parser.add_argument('--dataset', default="CIFAR10C", type=str, help='data set')
    parser.add_argument('--dataset', default="CIFAR10", type=str, help='data set')
    parser.add_argument('--testset', default="None", type=str, help='test set')
    args = parser.parse_args()


    my_data_root = 'C:/Users/naama-alon/data'
    #my_data_root ='../data'

    use_cuda = torch.cuda.is_available()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    tot_val_time = 0

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

    if args.dataset=="CIFAR10":
        print('Chose CIFAR10 Dataset..')
        num_classes=10
        trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                                    transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=args.batch_size,
                                                shuffle=True, num_workers=8)
        if args.testset =="None":
            testset = datasets.CIFAR10(root='~/data', train=False, download=False,
                                    transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                    shuffle=False, num_workers=8)
    elif args.dataset=="CIFAR100":
        print('Chose CIFAR100 Dataset..')
        num_classes=100
        trainset = datasets.CIFAR100(root='~/data', train=True, download=True,
                                    transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=args.batch_size,
                                                shuffle=True, num_workers=8)
        if args.testset =="None":
            testset = datasets.CIFAR100(root='~/data', train=False, download=False,
                                    transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                    shuffle=False, num_workers=8)

    if args.testset=="CIFAR10C":
        print('Chose CIFAR10 Corrupted Testset..')
        num_classes=10
        corruptions = load_txt('./corruptions.txt')

        for i, cname in enumerate(corruptions):
            tmp_dataset = CIFAR10C(root=os.path.join(my_data_root, 'CIFAR-10-C'),name=cname,
                                        transform=transform_train)
            start= 20000
            stop = 30000
            indices = [i for i in range(start, stop)] # use sevirity 3
            sev3 = Subset(tmp_dataset, indices)
            split_lengths = [int(len(sev3)*0.833), int(len(sev3)*0.167)]
            #sev3_trainset, sev3_testset = random_split(sev3, split_lengths)
            _ , sev3_testset = random_split(sev3, split_lengths) 
            if i==0:
                #trainset_arr =  [sev3_trainset]
                testset_arr = [sev3_testset]
            else:
                #trainset_arr.append(sev3_trainset)
                testset_arr.append(sev3_testset)

        #trainset =  ConcatDataset(trainset_arr)
        testset =  ConcatDataset(testset_arr)

        #trainloader = torch.utils.data.DataLoader(trainset,
                                                #batch_size=args.batch_size,
                                                #shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False, num_workers=8)

    elif args.testset=="CIFAR100C":
        print('Chose CIFAR100 Corrupted Testset..')
        num_classes=100
        corruptions = load_txt('./corruptions.txt')

        for i, cname in enumerate(corruptions):
            tmp_dataset = CIFAR100C(root=os.path.join(my_data_root, 'CIFAR-100-C'),name=cname,
                                        transform=transform_train)
            start= 20000
            stop = 30000
            indices = [i for i in range(start, stop)] # use sevirity 3
            sev3 = Subset(tmp_dataset, indices)
            split_lengths = [int(len(sev3)*0.833), int(len(sev3)*0.167)]
            _ , sev3_testset = random_split(sev3, split_lengths) 
            if i==0:
                testset_arr = [sev3_testset]
            else:
                testset_arr.append(sev3_testset)

        testset =  ConcatDataset(testset_arr)

        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False, num_workers=8)



    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7' + args.name + '_'
                                + str(args.seed))
        net = checkpoint['net']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        rng_state = checkpoint['rng_state']
        torch.set_rng_state(rng_state)
    else:
        print('==> Building model..')
        net = models.__dict__[args.model](num_classes)
        print(f"Chose {args.model} Model..")
    
    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('results/log_' + net.__class__.__name__ + '_' + args.name + '_'
            + str(args.seed) + '.csv')

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                        weight_decay=args.decay)


    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                        args.alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                        targets_a, targets_b))
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # train_loss += loss.data[0]
            train_loss+=loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar(batch_idx, len(trainloader),
                        'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                            100.*correct/total, correct, total))
        return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


    def test(epoch, best_acc):
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
        if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
            checkpoint(acc, epoch)
        if acc > best_acc:
            best_acc = acc

        if acc > best_acc:
            checkpoint(acc, epoch)
            best_acc = acc
        if epoch == start_epoch + args.epoch - 1: #last epoch
            checkpoint(acc, epoch, last=True)

        return (test_loss/batch_idx, 100.*correct/total, best_acc)


    def checkpoint(acc, epoch, last=False):
        # Save checkpoint.
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if last:
            torch.save(state, './checkpoint/ckpt.t7_' + args.model + '_'
                + args.dataset  + '_' + args.testset + '_last')
        else:
            torch.save(state, './checkpoint/ckpt.t7_' + args.model + '_'
                    + args.dataset  + '_' + args.testset + '_best')


    def adjust_learning_rate(optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc', 'test err'])

    for epoch in range(start_epoch, args.epoch):
        train_loss, reg_loss, train_acc = train(epoch)
        tic = time.perf_counter()
        test_loss, test_acc, best_acc = test(epoch, best_acc)
        toc = time.perf_counter()
        tot_val_time += (toc - tic)
        adjust_learning_rate(optimizer, epoch)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc, (100-test_acc)])

    return tot_val_time
    

if __name__ == '__main__':
    tic = time.perf_counter()
    tot_val_time = main()
    toc = time.perf_counter()
    print(f"Validation run for: {(((tot_val_time)/60)):0.4f} minutes.")
    print(f"Run for: {(((toc - tic)/60)/60):0.4f} hours.")