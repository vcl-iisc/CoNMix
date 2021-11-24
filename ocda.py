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

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import network
from mixup_utils import progress_bar
from helper.data_list import ImageList_idx, ImageList, ImageList_MixUp, ImageList_ocda
from torch.utils.data import DataLoader
import ml_collections
import wandb
import random 

def init_src_model_load(args):
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()
    elif args.net == 'vit':
        netF = network.ViT().cuda()
    elif args.net == 'deit_s':
        netF = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).cuda()
        netF.in_features = 1000

    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features,bottleneck_dim=256).cuda()
    netC = network.feat_classifier(type='wn', class_num=args.class_num, bottleneck_dim=256).cuda()

    if torch.cuda.device_count() >= 1:
        gpu_list = []
        for i in range(len(args.gpu_id.split(','))):
            gpu_list.append(i)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        netF = nn.DataParallel(netF, device_ids=gpu_list)
        netB = nn.DataParallel(netB, device_ids=gpu_list)
        netC = nn.DataParallel(netC, device_ids=gpu_list)
    return netF, netB, netC

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args):
    ## prepare data

    def image_train(resize_size=256, crop_size=224, alexnet=False):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        return transforms.Compose([
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            normalize
        ])

    dsets = {}
    dset_loaders = {}
    txt_tar = open(f'{args.txt_folder}/{args.dset}/{names[args.s]}.csv').readlines()

    print("Source Domain: ", names[args.s], "No. of Images: ", len(txt_tar))
    dsets['train'] = ImageList_ocda(txt_tar, transform=image_train(), target = args.t, train=True)
    print("Since ODCA only using: ", len(dsets['train']))
    dset_loaders['train'] = DataLoader(dsets['train'], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    
    if args.dset =='domain_net':
        txt_tar = []
        for i in range(6):
            if i == args.s:
                continue
            tmp = open(f'data/{args.dset}/{names[i].lower()}_test.txt').readlines()
            print("Domain:", names[i], "Number of Images:", len(tmp))

            txt_tar.extend(tmp)
        txt_test = txt_tar.copy()  
        print("Total Images:", len(txt_test)) 

        dsets["test"] = ImageList(txt_test, transform=image_train())
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.test_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    
    else:
        dsets['test'] = ImageList_ocda(txt_tar, transform=image_train(), target = args.t, train=False)
        print("target images: ", len(dsets['test']))
        dset_loaders['test'] = DataLoader(dsets['test'], batch_size=args.test_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    return dset_loaders,dsets

def separate_classwise_idx(args, dset, num_classes): #!@ args
    all_data_clswise = np.array(dset.imgs)
    numbers = np.array(list(map(int, all_data_clswise[:,2])))

    classwise_dset = {}
    classwise_loaders = {}

    for i in range(num_classes):
        idx_dict = np.argwhere(numbers==i).squeeze().tolist()
        classwise_dset[i] = torch.utils.data.Subset(dset, idx_dict)
        classwise_loaders[i] = DataLoader(classwise_dset[i], batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=True)        
    return classwise_loaders, classwise_dset

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    # if alpha > 0:
    #     lam = np.random.beta(alpha, alpha)
    # else:
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


def train(args, epoch,train_loader):
    print('\nEpoch: %d' % epoch)
    netF.train()
    netB.train()
    netC.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, pseudo_lbl, targets, domain) in enumerate(train_loader):
        if use_cuda:
            inputs, targets, pseudo_lbl, domain = inputs.cuda(), targets.cuda(),  pseudo_lbl.cuda(), domain.cuda()

        # optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min = 1e-6, last_epoch = epoch)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, pseudo_lbl, args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        outputs = netC(netB(netF(inputs)))
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += pseudo_lbl.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss': train_loss/(batch_idx+1), 'train_acc': 100.*correct/total})
    
        progress_bar(batch_idx, len(train_loader),
                    'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1),100.*correct/total, correct, total))
                # return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)
    # checkpoint(args, netF, netB, netC)
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch,testloader):
    global best_acc
    netF.eval()
    netB.eval()
    netC.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, pseudo_lbl, targets, domain) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(), 
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = netC(netB(netF(inputs)))
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader),'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        acc = 100.*correct/total
        # if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        #     checkpoint(args, netF, netB, netC)
        if acc > best_acc:
            best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)


def checkpoint(args, netF, netB, netC):
    # Save checkpoint.
    save_pth = os.path.join(args.save_weights, args.dset, names[args.s])
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    torch.save(netF.state_dict(), os.path.join(save_pth, "target_F.pt"))
    torch.save(netB.state_dict(), os.path.join(save_pth, "target_B.pt"))
    torch.save(netC.state_dict(), os.path.join(save_pth, "target_C.pt"))
    print('Model saved to',save_pth )
    

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--net', default="deit_s", type=str, help='model type (default: ResNet18)')
    parser.add_argument('--name', default='0', type=str, help='name of run')
    parser.add_argument('--suffix', default='ocda', type=str, help='wandb name of run')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--test_bs', default=256, type=int)
    parser.add_argument('--worker', type=int, default=8, help="number of workers")

    parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
    parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--alpha', default=1., type=float, help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--s', default=0, type=int)
    parser.add_argument('--t', default=1, type=int)
    parser.add_argument('--txt_folder', default='csv', type=str)
    parser.add_argument('--save_weights', default='OCDA_wts', type=str)
    parser.add_argument('--dset', type=str, default='office-home', choices=['visda-2017', 'office', 'office-home', 'office-caltech', 'pacs', 'domain_net'])

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    best_acc = 0  # best test accuracy
    start_epoch = 1  # start from epoch 0 or last checkpoint epoch

    if args.seed != 0:
        torch.manual_seed(args.seed)

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 7
    if args.dset == 'pacs':
        names = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.class_num = 7
    if args.dset =='domain_net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','sketch', 'real']
        args.class_num = 345

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

    if not os.path.isdir('results'):
        os.mkdir('results')


    print('==> Perparing Dataloaders and Building model..')
    all_loader,all_dset = data_load(args)   
    netF, netB, netC = init_src_model_load(args)

    if use_cuda:
        netF.cuda()
        netB.cuda()
        netC.cuda()
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]   

    optimizer = optim.SGD(param_group, lr=args.lr, momentum=0.9,
                        weight_decay=args.decay)


    logname = ('results/log' + '.csv')
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc'])

    mode = 'online' if args.wandb else 'disabled'
    wandb.init(project='OCDA', entity='vclab', name=f'{args.dset} - {names[args.s]} to {names[args.t]} {args.suffix}', mode=mode)#!@

    print(f'\nStarting training {names[args.s]} to others.')
    train_len = len(all_dset['train'])
    test_len = len(all_dset['test'])
    print(f'Training: {train_len} Images \t Testing: {test_len} Images')

    for epoch in range(start_epoch, args.epoch+1):

        train_loss, reg_loss, train_acc = train(args, epoch, all_loader['train'])
        checkpoint(args, netF, netB, netC)

        if epoch % args.interval == 0:
            print('\n Start Testing')
            test_loss, test_acc = test(epoch, all_loader['test'])
            wandb.log({ 'test_loss': test_loss,  'test_acc': test_acc})