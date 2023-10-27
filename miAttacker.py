"""
Model Inversion (mi) attack on AsymML.
The attack model can access 1) well-trained model parameters; 2) data in GPU memory,
and try to reconstruct images that are similar to training dataset.

Ref:
    attack model: http://arxiv.org/abs/1911.07135

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
from attack.attack_model import miGenerator, miDiscriminator
from attack.attack_utils import init_netGD
from model import resnet
from data.dataset import *
from utils import resnet_conversion

parser = argparse.ArgumentParser()
parser.add_argument( '--dataset', default='cifar100', type=str, choices=['cifar10', 'imagenet'] )

# target model under attack
parser.add_argument( '--nz', default=512, type=int, help='size of latent vector' )
parser.add_argument( '--model', default='resnet18', type=str, help='target model' )
parser.add_argument( '--modeldir', default='./checkpoints/cifar100_resnet18/model.pt', type=str,
                     help='model path' )
parser.add_argument( '--with-residual', action='store_true',
                     help='if has residual as prior knowledge' )
parser.add_argument( '--noise', default=0.4, type=float,
                     help='noise added to the residual')

# train parameters
parser.add_argument( '--batch-size', default=64, type=int )
parser.add_argument( '--lr', default=0.01, type=float )
parser.add_argument( '--lr-decay', default=0.1, type=float, help='lr decay factor' )
parser.add_argument( '--decay-period', default=100, type=int, help='lr decay period' )
parser.add_argument( '--momentum', default=0.9, type=float )
parser.add_argument( '--wd', default=0.0005, type=float, help='weight decay' )
parser.add_argument( '--epochs', default=50, type=int )
parser.add_argument( '--workers', default=8, type=int, help='number of data loading workers' )
parser.add_argument( '--check-freq', default=20, type=int, help='checkpoint frequency' )
parser.add_argument( '--ckpdir', default='./checkpoints', type=str, help='checkpoint dir' )
parser.add_argument( '--log-dir', default='./log/attack/resnet18', type=str, help='train log dir' )

# DP noise parameters
parser.add_argument( '--nsr', default=0.1, type=float, help='noise to signal ratio' )
args = parser.parse_args()

# add mean and divide variance
invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(
            mean=[-0.485, -0.456, -0.406],
            std=[1., 1., 1.]
        ),
    ]
)


def main():
    device = torch.device( "cuda:0" )
    # =========================================================================
    # create attack model with 1) generator; 2) discriminator
    # model architecture is based on The Secret Revealer:
    # (http://arxiv.org/abs/1911.07135)

    """generator"""
    netG = miGenerator( 32, args.nz, 128 ).to( device )
    nn.DataParallel( netG )
    netG.apply( init_netGD )
    print('=========================Generator=========================')
    print( netG )

    """discriminator"""
    netD = miDiscriminator( 64 ).to( device )
    nn.DataParallel( netD )
    netD.apply( init_netGD )
    print( '=========================Discriminator=========================' )
    print( netD )

    """target model"""
    model = resnet.__dict__[ args.model ]( num_classes=100 )
    # args: model, dataset, bb point, svd, rank, dct, quant, noise, alpha
    model = resnet_conversion.resnet_asym(
        model, args.dataset, 1, True, 8, True, True, args.noise, 1
    )
    model.to( device )
    chkpoint = torch.load( args.modeldir )
    chkpoint_right_format = {}
    for key in chkpoint[ 'model_state_dict' ]:
        chkpoint_right_format[ key[ 7:: ] ] = \
            chkpoint[ 'model_state_dict' ][ key ]
    model.load_state_dict( chkpoint_right_format )
    model.eval()
    print('=========================Target Model=========================')
    print( model )

    """define loss and optimization"""
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam( netG.parameters(), lr=args.lr/10, betas=(0.5, 0.999) )
    optimizerD = torch.optim.Adam( netD.parameters(), lr=args.lr, betas=(0.5, 0.999) )
    # optimizerG = torch.optim.SGD( netG.parameters(), lr=args.lr/4,  momentum=0.9 )
    # optimizerD = torch.optim.SGD( netD.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001 )
    # lr_lambda = lambda epoch: args.lr_decay ** (epoch // args.decay_period)
    # schedulerG = torch.optim.lr_scheduler.LambdaLR( optimizerG, lr_lambda=lr_lambda)
    # schedulerD = torch.optim.lr_scheduler.LambdaLR( optimizerD, lr_lambda=lr_lambda )
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR( optimizerG, args.epochs, eta_min=0.001*args.lr )
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR( optimizerD, args.epochs, eta_min=0.001*args.lr )

    """create dataset"""
    if args.dataset == 'cifar100':
        # trainset = dataset_CIFAR100_train
        trainset = dataset_CIFAR100_sub1
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )
        valset = dataset_CIFAR100_val
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True
        )
        # pubset = dataset_Pub
        pubset = dataset_CIFAR100_sub2
        publoader = torch.utils.data.DataLoader(
            pubset,
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )
    else:
        raise NotImplementedError

    """Validate the target model's checkpoint"""
    print( 'Check the pretrain model ... ' )
    criterion_val = torch.nn.CrossEntropyLoss().cuda()
    pretrain_acc, pretrain_loss = validate( model, valloader, criterion_val )
    print( 'Pretrain accuracy: {:.3f}'.format( pretrain_acc ) )

    # =========================================================================
    """start training"""
    writer = SummaryWriter( args.log_dir )
    fix_noise = torch.randn( args.batch_size, args.nz, 1, 1, device=device )
    for epoch in range( args.epochs ):
        train(
            model, netD, netG, trainloader, publoader,
            criterion, optimizerD, optimizerG,
            device, epoch, writer
        )

        # schedulerG.step()
        # schedulerD.step()

        # - check generator's performance
        if epoch % 2 == 0:
            with torch.no_grad():
                for i, data in enumerate( publoader, 0 ):
                    images = data[ 0 ].to( device )

                    images_to_show = invTrans( images )

                    # generate the residuals
                    x_bb = model.backbone( images )
                    x_svd, x_svd_res = model.svd_layer( x_bb )
                    x_dct, x_dct_res = model.dct_layer( x_svd )
                    x_res = x_svd_res + x_dct_res
                    x_res_quant = model.quant_layer( x_res )
                    if not args.with_residual:
                        x_res_quant = torch.zeros_like( x_res_quant )

                    # feed to generative model
                    fake = netG( x_res_quant, fix_noise ).detach().cpu()
                    fake_to_show = invTrans( fake )
                    break

            writer.add_images( 'train-real', images_to_show, epoch )
            writer.add_images( 'train-recon', fake_to_show, epoch )

    # =========================================================================
    # perform model inversion ( with data in untrusted GPUs as prior information )
    n_tries = 100
    criterion = torch.nn.CrossEntropyLoss()
    gloss = AverageMeter()
    gacc = AverageMeter()
    g_psnr, g_ssim = AverageMeter(), AverageMeter()
    for i, data in enumerate( valloader, 0 ):
        images = data[ 0 ].to( device )
        labels = data[ 1 ].to( device )

        # generate the residuals
        x_bb = model.backbone(images)
        x_svd, x_svd_res = model.svd_layer(x_bb)
        x_dct, x_dct_res = model.dct_layer(x_svd)
        x_res = x_svd_res + x_dct_res
        x_res_quant = model.quant_layer(x_res)
        if not args.with_residual:
            x_res_quant = torch.zeros_like( x_res_quant )

        loss_min = float( 'inf' )
        acc_best = 0.0
        noise_j = torch.randn( args.batch_size, args.nz, 1, 1, requires_grad=True, device=device )
        optimizerIn = torch.optim.SGD( [ noise_j ], lr=args.lr*10, momentum=0.9 )
        for j in range( n_tries ):
            optimizerIn.zero_grad()

            fake = netG( x_res_quant, noise_j )
            output, _, _, _, _ = model( fake )
            loss = criterion( output, labels )

            # optimize noise_j
            loss.backward( retain_graph=True )
            optimizerIn.step()

            prec1 = cal_acc( output, labels )[ 0 ].item()
            if loss.item() < loss_min:
                loss_min = loss.item()
                acc_best = prec1
                images_recon = fake

        # - PSNR and SSIM
        # psnr_i, ssim_i = cal_metrics( images, images_recon )
        # g_psnr.update( psnr_i, args.batch_size )
        # g_ssim.update( ssim_i, args.batch_size )
        output, _, _, _, _ = model( images_recon )
        loss_final = criterion( output, labels )
        prec1_final = cal_acc( output, labels )[ 0 ].item()

        gloss.update( loss_final, args.batch_size )
        gacc.update( prec1_final, args.batch_size )
        img_to_show = invTrans( images )
        img_recon_to_show = invTrans( images_recon )
        writer.add_images( 'miAttack-real', F.interpolate( img_to_show, size=(64, 64) ), i )
        writer.add_images( 'miAttack-recon', F.interpolate( img_recon_to_show, size=(64, 64) ), i )
        writer.add_images( 'miAttack-residual', F.interpolate( x_res_quant[:,0:3,:,:], size=(64, 64) ), i )

        if i == 10:
            break
    print( 'generator loss: {:.4f}\t generator acc: {:.4f}'.format( gloss.avg, gacc.avg ) )
    print( 'generator PSNR: {:.4f}\t generator SSIM: {:.4f}'.format( g_psnr.avg, g_ssim.avg ) )

    writer.close()


def train(
        model, netD, netG,  trainloader, publoader,
        criterion, optimizerD, optimizerG,
        device, epoch, writer ):
    """
    train wrapper for generator and discriminator
    :param model target model
    :param netD discriminator model
    :param netG generator model
    :param trainloader train data loader
    :param publoader public dataset loader
    :param criterion loss function
    :param optimizerD optimizer for discriminator
    :param optimizerG optimizer for generator
    :param device runtime device (GPUs)
    :param epoch current epoch
    :param writer summary writer for tensorboard
    """
    avglossD = AverageMeter()
    avglossG = AverageMeter()

    # =========================================================================
    for i, (data1, data2) in enumerate( zip( publoader, trainloader ) ):
        # train discriminator: netD
        # - first feed real data
        netD.zero_grad()
        images_pub = data1[ 0 ].to( device )
        b = images_pub.size( 0 )
        labels = torch.full( (b,), 1.0, dtype=torch.float, device=device )
        output = netD( images_pub ).view( -1 )
        lossD_real = criterion( output, labels )
        lossD_real.backward()

        # - then feed fake data and prior information
        noise = torch.randn( b, args.nz, 1, 1, device=device )

        # generate the residuals
        images = data2[ 0 ].to( device )
        x_bb = model.backbone( images )
        x_svd, x_svd_res = model.svd_layer( x_bb )
        x_dct, x_dct_res = model.dct_layer( x_svd )
        x_res = x_svd_res + x_dct_res
        x_res_quant = model.quant_layer( x_res )
        if not args.with_residual:
            x_res_quant = torch.zeros_like( x_res_quant )

        images_fake = netG( x_res_quant.detach(), noise )
        labels.fill_( 0.0 )
        output = netD( images_fake.detach() ).view( -1 )
        lossD_fake = criterion( output, labels )
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optimizerD.step()
        avglossD.update( lossD.item(), b )

        # train generator: netG
        netG.zero_grad()
        labels.fill_( 1.0 )
        output = netD( images_fake ).view( -1 )
        lossG = criterion( output, labels )
        lossG.backward()

        optimizerG.step()
        avglossG.update( lossG.item(), b )

    # =========================================================================
    # print train progress
    print('Epoch: [{0}/{1}]\t'
          'Loss D {avglossD.avg:.4f}\t Loss G {avglossG.avg:.4f}'.format(
            epoch, args.epochs, avglossD=avglossD, avglossG=avglossG ) )
    writer.add_scalar( 'loss/netD', avglossD.avg, global_step=epoch*len(trainloader) )
    writer.add_scalar( 'loss/netG', avglossG.avg, global_step=epoch*len(trainloader) )


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
