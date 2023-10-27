"""
Membership Inference attack on AsymML.
The attack model can infer if a sample exists in the training dataset only via the label information

Ref:
    attack model: https://proceedings.mlr.press/v139/choquette-choo21a.html

Author:

Note:
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skimage import metrics
from itertools import cycle

sys.path.insert(0, './')
from attack.attack_model import membershipModel, aug_attack, train_aug_attack
from attack.attack_utils import split_data
from model import resnet
from data.dataset import *
from utils import resnet_conversion

parser = argparse.ArgumentParser()
parser.add_argument( '--dataset', default='cifar100', type=str, choices=['cifar10', 'imagenet'] )

# target model under attack
parser.add_argument( '--model', default='resnet18', type=str, help='target model' )
parser.add_argument( '--modeldir', default='./checkpoints/cifar100_resnet18/model.pt', type=str,
                     help='model path' )
parser.add_argument( '--with-residual', action='store_true',
                     help='if has residual as prior knowledge' )
parser.add_argument( '--epsilon', default=1.25, type=float, help='noise added to the residual' )
parser.add_argument( '--ndata', default=0.2, type=float, help='fraction of training data for the attacker' )

# train parameters
parser.add_argument( '--batch-size', default=100, type=int )
parser.add_argument( '--lr', default=0.01, type=float )
parser.add_argument( '--lr-decay', default=0.1, type=float, help='lr decay factor' )
parser.add_argument( '--decay-period', default=100, type=int, help='lr decay period' )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0001, type=float, help='weight decay' )
parser.add_argument( '--epochs', default=50, type=int )
parser.add_argument( '--workers', default=8, type=int, help='number of data loading workers' )
parser.add_argument( '--check-freq', default=20, type=int, help='checkpoint frequency' )
parser.add_argument( '--ckpdir', default='./checkpoints', type=str, help='checkpoint dir' )
parser.add_argument( '--log-dir', default='./log/attack/resnet18', type=str, help='train log dir' )

# DP noise parameters
parser.add_argument( '--nsr', default=0.1, type=float, help='noise to signal ratio' )
args = parser.parse_args()


def main():
    device = torch.device( "cuda:0" )
    # =========================================================================
    # membership attack model via labels only
    # model architecture is based on Label-only Membership attack:
    # (https://proceedings.mlr.press/v139/choquette-choo21a.html)

    """"""
    """create dataset"""
    if args.dataset == 'cifar100':
        trainset, testset = dataset_CIFAR100_train, dataset_CIFAR100_val
        tgt_trainset, tgt_testset, src_trainset, src_testset = split_data( trainset, testset, args.ndata )
        tgt_train_dl = torch.utils.data.DataLoader(
            tgt_trainset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        tgt_test_dl = torch.utils.data.DataLoader(
            tgt_testset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

        src_train_dl = torch.utils.data.DataLoader(
            src_trainset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )
        src_test_dl = torch.utils.data.DataLoader(
            src_testset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers
        )

    else:
        raise NotImplementedError

    """target model"""
    tgt_model = resnet.__dict__[ args.model ]( num_classes=100 )
    # args: model, dataset, bb point, svd, rank, dct, quant, noise, alpha
    tgt_model = resnet_conversion.resnet_asym(
        tgt_model, args.dataset, 1, True, 8, True, True, args.epsilon, 1
    )
    tgt_model.to( device )

    src_model = resnet.__dict__[ args.model ]( num_classes=100 )
    src_model.to( device )

    """define loss and optimization"""
    crit = torch.nn.CrossEntropyLoss()
    opt_tgt = torch.optim.SGD(
        tgt_model.parameters(),
        args.lr, momentum=args.momentum, weight_decay=args.wd,
    )
    opt_src = torch.optim.SGD(
        src_model.parameters(),
        args.lr, momentum=args.momentum, weight_decay=args.wd,
    )
    scheduler_tgt = torch.optim.lr_scheduler.CosineAnnealingLR( opt_tgt, args.epochs, eta_min=0.00001*args.lr )
    scheduler_src = torch.optim.lr_scheduler.CosineAnnealingLR( opt_src, args.epochs, eta_min=0.00001*args.lr )

    # =========================================================================
    """start training"""
    writer = SummaryWriter( args.log_dir )
    for epoch in range( args.epochs ):
        train(
            tgt_model, tgt_train_dl, tgt_test_dl,
            crit, opt_tgt,
            device, epoch, 'Target'
        )
        train(
            src_model, src_train_dl, src_test_dl,
            crit, opt_src,
            device, epoch, 'Source'
        )

        scheduler_tgt.step()
        scheduler_src.step()

    # =========================================================================
    # perform membership inference attack
    attack_test_set = aug_attack( tgt_model, tgt_train_dl, tgt_test_dl, args.batch_size, 'Target' )
    attack_train_set = aug_attack( src_model, src_train_dl, src_test_dl, args.batch_size, 'Source' )

    metrics = train_aug_attack( attack_train_set, attack_test_set )

    writer.close()


def train(
        model, trainloader, testloader,
        crit, opt,
        device, epoch, id ):
    """
    train wrapper for generator and discriminator
    :param model target model
    :param trainloader train data loader
    :param testloader public dataset loader
    :param crit loss function
    :param opt optimizer for discriminator
    :param device runtime device (GPUs)
    :param epoch current epoch
    :param id model identifier
    """
    avg_loss_train = AverageMeter()
    avg_acc_train = AverageMeter()

    # =========================================================================
    print( '\n\n' )
    for i, ( inputs, labels ) in enumerate( trainloader ):
        # train discriminator: netD
        # - first feed real data
        opt.zero_grad()
        inputs, labels = inputs.to( device ), labels.to( device )
        if id == 'Target':
            outputs, _, _, _, _ = model( inputs )
        else:
            outputs = model( inputs )
        loss = crit( outputs, labels )
        loss.backward()

        opt.step()
        prec1 = cal_acc( outputs, labels )[ 0 ]
        avg_acc_train.update( prec1.item(), inputs.size( 0 ) )
        avg_loss_train.update( loss.item(), inputs.size( 0 ) )

        if i % 100 == 0:
            print(
                '[Membership Attack {} Train] Epoch: {},{}/{}, \t Loss: {:.3f}, \t Acc: {:.3f}'.format(
                    id, epoch, i, len( trainloader ), avg_loss_train.avg, avg_acc_train.avg
                )
            )

    # =========================================================================
    # validate
    avg_loss_val, avg_acc_val = AverageMeter(), AverageMeter()

    model.eval()
    for i, (inputs, labels) in enumerate( testloader ):
        inputs, labels = inputs.to( device ), labels.to( device )

        with torch.no_grad():
            if id == 'Target':
                outputs, _, _, _, _ = model( inputs )
            else:
                outputs = model( inputs )
            loss = crit( outputs, labels )

        prec1 = cal_acc( outputs, labels )[ 0 ]
        avg_loss_val.update( loss.item(), inputs.size( 0 ) )
        avg_acc_val.update( prec1.item(), inputs.size( 0 ) )

    print(
        '\n[Membership Attack {} Val] Epoch: {},{}/{}, \t Loss: {:.3f}, \t Acc: {:.3f}'.format(
            id, epoch, i, len( testloader ), avg_loss_val.avg, avg_acc_val.avg
        )
    )


def cal_acc(output, target, topk=(1,)):
    """
    Calculate model accuracy
    :param output:
    :param target:
    :param topk:
    :return: topk accuracy
    """
    maxk = max(topk)
    batch_size = target.size( 0 )

    _, pred = output.topk( maxk, 1, True, True )
    pred = pred.t()
    correct = pred.eq( target.view( 1, -1 ).expand_as( pred ) )

    acc = []
    for k in topk:
        correct_k = correct[:k].view( -1 ).float().sum( 0 )
        acc.append( correct_k.mul_( 100.0 / batch_size ) )
    return acc


def cal_metrics( orig, recon ):
    """Calculate metrics such as PSNR and SSIM
    :param orig original data
    :param recon reconstructed data from the attack model
    """
    n_batch = orig.size( 0 )
    orig = orig.cpu().detach().numpy()
    orig = np.transpose( orig, ( 0,2,3,1 ) )
    recon = recon.cpu().detach().numpy()
    recon = np.transpose( recon, ( 0,2,3,1 ) )

    val_psnr, val_ssim = 0.0, 0.0
    for b in range( n_batch ):
        orig_i = orig[ b ]
        recon_i = recon[ b ]
        val_psnr += metrics.peak_signal_noise_ratio( orig_i, recon_i )
        val_ssim += metrics.structural_similarity( orig_i, recon_i, multichannel=True )

    return val_psnr/n_batch, val_ssim/n_batch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__( self ):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate( model, dl, crit ):
    # - model: model to be validated
    # - dl: val data loader
    # - crit: loss function
    # - epoch: current epoch
    avg_loss, avg_acc = AverageMeter(), AverageMeter()

    model.eval()
    for i, ( input, target ) in enumerate( dl ):
        input, target = input.cuda(), target.cuda()

        with torch.no_grad():
            output, output_res, x_backbone, x_svd, x_dct = model( input )
            loss = crit( output, target )

        prec1 = cal_acc( output, target )[ 0 ]
        avg_loss.update( loss.item(), input.size( 0 ) )
        avg_acc.update( prec1.item(), input.size( 0 ) )

    """
    print( 'Residual energy ratio: {:.3f}'.format( 1 - ratio_dct/len( dl ) ) )
    quit()
    """

    return avg_acc.avg, avg_loss.avg


if __name__ == '__main__':
    main()
