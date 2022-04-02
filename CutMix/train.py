# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import resnet as RN
import pyramidnet as PYRM
import utils
import numpy as np

import warnings

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from torch.utils.data import random_split

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')

parser.add_argument('--net_type', default='resnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=18, type=int,
                    help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='cifar100', type=str,
                    help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
                    help='cutmix probability')
parser.add_argument('--my_test', default=0, type=float,
                    help='my test')


parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100

def load_data():
    if args.dataset.startswith('cifar'):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if args.dataset == 'cifar100':
            train_dataset = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR100('../data', train=False, transform=transform_test)
            numberofclass = 100
        elif args.dataset == 'cifar10':
            train_dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
            val_dataset = datasets.CIFAR10('../data', train=False, transform=transform_test)
            numberofclass = 10
        else:
            raise Exception('unknown dataset: {}'.format(args.dataset))

    elif args.dataset == 'imagenet':
        traindir = os.path.join('/home/data/ILSVRC/train')
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils.ColorJitter(brightness=0.4, contrast=0.4,
                                      saturation=0.4)
        lighting = utils.Lighting(alphastd=0.1,
                                  eigval=[0.2175, 0.0188, 0.0045],
                                  eigvec=[[-0.5675, 0.7192, 0.4009],
                                          [-0.5808, -0.0045, -0.8140],
                                          [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ]))

        numberofclass = 1000

    else:
        raise Exception('unknown dataset: {}'.format(args.dataset))


    return train_dataset, val_dataset , numberofclass

def train_cifar(config ,checkpoint_dir=None):
    train_subset, val_subset , numberofclass = load_data()

    if args.dataset == 'cifar100':
        train_loader = torch.utils.data.DataLoader(train_subset,
                                               batch_size=int(config["batch_size"]), shuffle=True, num_workers=args.workers,
                                               pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_subset,
                                                 batch_size=int(config["batch_size"]), shuffle=True, num_workers=args.workers,
                                                 pin_memory=True)

    if args.dataset == 'imagenet':
        valdir = os.path.join('/home/data/ILSVRC/val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=int(config["batch_size"]), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=int(config["batch_size"]), shuffle=False,
            num_workers=args.workers, pin_memory=True)

    print("=> creating model '{}'".format(args.net_type))
    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset, args.depth, numberofclass, args.bottleneck)  # for ResNet
    elif args.net_type == 'pyramidnet':
        model = PYRM.PyramidNet(args.dataset, args.depth, args.alpha, numberofclass,
                                args.bottleneck)
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()#.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader,model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        tune.report(loss=(val_loss), accuracy=err1)

    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)

def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    load_data()

    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
        "my_test": tune.choice([1,2,3])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    data_dir = os.path.abspath("./data")
    gpus_per_trial = 1
    num_samples = 10
    result = tune.run(
        partial(train_cifar),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))




def train(train_loader,model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            input = input.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            if(torch.cuda.is_available()):
                rand_index = torch.randperm(input.size()[0]).cuda()
            else:
                rand_index = torch.randperm(input.size()[0])
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

        if config["my_test"] == 1 and args.beta > 0:
            lam = np.random.beta(args.beta, args.beta)
            rand_col_row = np.random.choice(2,replace=False)  # choose cols or rows
            if (torch.cuda.is_available()):
                rand_index = torch.randperm(input.size()[0]).cuda()
            else:
                rand_index = torch.randperm(input.size()[0])
            target_a = target
            target_b = target[rand_index]

            if rand_col_row == 0:
                ratio = int(lam * input.shape[-2])
                rand_pixels = np.random.choice(list(range(input.shape[-2])), ratio, replace=False)
                input[:, :, rand_pixels, :] = (input[rand_index, :, :, :])[:, :, rand_pixels, :]
                lam = 1 - len(rand_pixels) * input.shape[-1] / (input.size()[-1] * input.size()[-2])
            else:
                ratio = int(lam * input.shape[-1])
                rand_pixels = np.random.choice(list(range(input.shape[-1])), ratio)
                input[:, :, :, rand_pixels] = (input[rand_index, :, :, :])[:, :, :, rand_pixels]
                lam = 1 - len(rand_pixels) * input.shape[-2] / (input.size()[-1] * input.size()[-2])

        if config["my_test"] == 2 and args.beta > 0:
            lam = np.random.beta(args.beta, args.beta)
            input_original_shape = input.shape
            input = input.view(input.shape[0], input.shape[1], input.shape[2] * input.shape[3])

            if (torch.cuda.is_available()):
                rand_index = torch.randperm(input.size()[0]).cuda()
            else:
                rand_index = torch.randperm(input.size()[0])
            target_a = target
            target_b = target[rand_index]

            ratio = int(lam * input.shape[2])
            rand_pixels = np.random.choice(list(range(input.shape[2])), input.shape[2], replace=False)
            input[:, :, rand_pixels] = (input[rand_index, :, :])[:, :, rand_pixels1]
            lam = 1 - len(rand_pixels) / (input.size()[2])
            input = input.view(input_original_shape)
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)


        if config["my_test"] == 3 and args.beta > 0:
            lam = np.random.beta(args.beta, args.beta)
            input_original_shape = input.shape
            input = input.view(input.shape[0], input.shape[1], input.shape[2] * input.shape[3])

            if (torch.cuda.is_available()):
                rand_index = torch.randperm(input.size()[0]).cuda()
            else:
                rand_index = torch.randperm(input.size()[0])
            target_a = target
            target_b = target[rand_index]

            ratio = int(lam * input.shape[2])
            rand_pixels = np.random.choice(list(range(input.shape[2])), input.shape[2], replace=False)
            rand_pixels1 = rand_pixels[:ratio]
            rand_pixels2 = rand_pixels[ratio:]

            input[:, :, rand_pixels1] = (input[rand_index, :, :])[:, :, rand_pixels1]
            input[:, :, rand_pixels2] = (input[:, :, :])[:, :, rand_pixels2]
            lam = 1 - len(rand_pixels2) / (input.size()[2])
            input = input.view(input_original_shape)
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)


        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if torch.cuda.is_available():
            target = target.cuda()
            input = input.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (args.expname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.expname) + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
