# This is the main function of the project
import sys
import time
import torch
import logging

from torch.utils.tensorboard import SummaryWriter
from model import vgg, resnet, resnet_imagenet
from model.model_lowrank import resnet6, resnet8
from utils import vgg_conversion, resnet_conversion
from data.dataset import *
from args import get_args
from utils.meter import AverageMeter, cal_acc
from utils.regularization import add_l2_reg, add_orth_reg, add_convorth_reg
args = get_args().parse_args()

formatter = logging.Formatter( '%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S' )
root_logger = logging.getLogger()
logging.getLogger( "PIL" ).setLevel( 51 )
root_logger.setLevel( level=logging.DEBUG )
file_handler = logging.FileHandler( args.logdir+'/train.log' )
file_handler.setFormatter( formatter )
root_logger.addHandler( file_handler )
console_handler = logging.StreamHandler( sys.stdout )
console_handler.setFormatter( formatter )
root_logger.addHandler( console_handler )

logging.info( 30*'-' + 'Asymmetric Learning' + 30*'-' )
logging.info( 'Running hyper parameters...\n' )
for key in vars( args ):
    logging.info( '\t' + key + ': ' + str( getattr( args, key ) ) )
logging.info( 80 * '-' )

torch.set_printoptions( threshold=10_000 )


def main():

    # ----------------------------------- model/data/training settings ----------------------------------- #

    """create model
        - create the original model
        - add svd layer: obtain principal channels
        - add dct layer: obtain low-frequent components
    """
    model, trainset, valset = get_model( args.model, args.dataset, initmethod=args.init )
    if args.svd or args.dct:
        if 'vgg' in args.model:
            model = vgg_conversion.vgg_asym( model, 1, args.rank, initmethod=args.init )
        elif 'resnet' in args.model:
            model = resnet_conversion.resnet_asym( model, args.dataset, 1, args.svd, args.rank, args.dct, args.quant, args.noise )
    if args.cuda:
        model = torch.nn.DataParallel( model )
        model.cuda()

    # print( model ); quit()

    """Dataset"""
    train_dl, val_dl = create_dl( trainset, valset )

    crit = torch.nn.CrossEntropyLoss()
    if args.cuda:
        crit.cuda()

    """SGD optimizer"""
    opt = torch.optim.SGD(
        model.parameters(),
        args.lr, momentum=args.momentum, weight_decay=args.wd,
    )

    """ADAM optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.wd,
    )
    """

    # define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( opt, args.epochs, eta_min=0.00001 )
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR( opt, milestones=[50, 100, 150], gamma=0.1 )

    # ---------------------------------------------- train ---------------------------------------------- #

    best_acc, best_ep = 0.0, 0
    writer = SummaryWriter( log_dir=args.logdir )
    for ep in range( 0, args.epochs ):
        acc_train, loss_train, loss_reg = train_1_epoch( model, train_dl, crit, opt, ep )
        acc_val, loss_val, ratio_svd, ratio_dct = validate( model, val_dl,  crit, ep )
        lr_scheduler.step()

        if best_acc < acc_val:
            best_acc, best_ep = acc_val, ep

        # log progress
        add_log( writer, model, loss_train, acc_train, loss_val, acc_val, loss_reg, ratio_svd, ratio_dct, ep )

    logging.info( '[AsymML Test Stats] Best acc: {:.3f} at {} Epoch'.format( best_acc, best_ep ) )
    writer.close()


def get_model( name, dataset, initmethod='gaussian' ):
    model, trainset, valset = None, None, None
    if 'vgg' in name:
        if dataset == 'cifar10':
            model = vgg.__dict__[ name ]( len_feature=512, num_classes=10, initmethod=initmethod )
            trainset, valset = dataset_CIFAR10_train, dataset_CIFAR10_val
        elif dataset == 'cifar100':
            model = vgg.__dict__[ name ]( len_feature=512, num_classes=100, initmethod=initmethod )
            trainset, valset = dataset_CIFAR100_train, dataset_CIFAR100_val
        else:
            raise NotImplementedError
    if 'resnet' in name:
        if dataset == 'cifar10':
            model = resnet.__dict__[ name ]( num_classes=10 )
            # model = resnet8( num_classes=10 )
            trainset, valset = dataset_CIFAR10_train, dataset_CIFAR10_val
        elif dataset == 'cifar100':
            model = resnet.__dict__[ name ]( num_classes=100 )
            # model = resnet8( num_classes=100 )
            trainset, valset = dataset_CIFAR100_train, dataset_CIFAR100_val
        elif dataset == 'imagenet':
            model = resnet_imagenet.__dict__[ name ]( num_classes=1000 )
            trainset, valset = dataset_ImageNet_train, dataset_ImageNet_val
        else:
            raise NotImplementedError

    return model, trainset, valset


def create_dl( trainset, valset ):
    train_dl = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True
    )
    val_dl = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.workers, pin_memory=True
    )
    return train_dl, val_dl


def train_1_epoch( model, dl, crit, opt, epoch ):
    # - model: model to be trained
    # - dl: train data loader
    # - crit: loss function
    # - opt: optimizer
    # - epoch: current epoch

    time_1_batch, time_data = AverageMeter(), AverageMeter()
    avg_loss, avg_acc, avg_reg = AverageMeter(), AverageMeter(), AverageMeter()

    model.train()
    end = time.time()
    for i, ( input, target ) in enumerate( dl ):
        time_data.update( time.time() - end )

        # ---------------------------------------- fwd/bwd/opt:beg ---------------------------------------- #
        opt.zero_grad()
        if args.cuda:
            input, target = input.cuda(), target.cuda()

        if args.svd or args.dct:
            output, _, _, _ = model( input )
        else:
            output = model( input )
        loss_cls = crit( output, target )
        # add regularization (explicitly)
        loss_reg = 0.0
        if args.reg == 'orth':
            # print( "use orth reg" ); quit()
            loss_reg = add_orth_reg( model, args.regcoeff, args.svd )
        elif args.reg == 'conv':
            loss_reg = add_convorth_reg( model, args.regcoeff, args.svd )
        loss = loss_cls + loss_reg

        loss.backward()

        opt.step()
        # ---------------------------------------- fwd/bwd/opt:end ---------------------------------------- #

        """collect training stats"""
        prec1 = cal_acc( output, target )[ 0 ]
        avg_acc.update( prec1.item(), input.size( 0 ) )
        avg_loss.update( loss_cls.item(), input.size( 0 ) )
        if args.reg == 'orth' or args.reg == 'conv':
            avg_reg.update( loss_reg.item(), 1 )

        time_1_batch.update( time.time() - end )
        end = time.time()

        if i % 500 == 0:
            logging.info(
                '[AsymML Train Stats] Epoch: {},{}/{}, \t Acc: {:.3f}, \t Time: {:.3f}/{:.3f}'.format(
                    epoch, i, len( dl ), avg_acc.avg,  time_1_batch.avg, time_1_batch.avg * (i+1)
                )
            )

    return avg_acc.avg, avg_loss.avg, avg_reg.avg


def validate( model, dl, crit, epoch ):
    # - model: model to be validated
    # - dl: val data loader
    # - crit: loss function
    # - epoch: current epoch
    avg_loss, avg_acc = AverageMeter(), AverageMeter()
    ratio_svd, ratio_dct = 0.0, 0.0  # also profile the energy leakage ratio

    model.eval()
    for i, ( input, target ) in enumerate( dl ):
        if args.cuda:
            input, target = input.cuda(), target.cuda()

        with torch.no_grad():
            if args.svd or args.dct:
                output, x_backbone, x_svd, x_dct = model( input )
                ratio_svd += ( torch.norm(x_svd)**2 / torch.norm( x_backbone)**2 )
                ratio_dct += ( torch.norm(x_dct)**2 / torch.norm( x_svd)**2 )
            else:
                output = model( input )
            loss = crit( output, target )

        prec1 = cal_acc( output, target )[ 0 ]
        avg_loss.update( loss.item(), input.size( 0 ) )
        avg_acc.update( prec1.item(), input.size( 0 ) )

    return avg_acc.avg, avg_loss.avg, ratio_svd/len( dl ), ratio_dct/len( dl )


def add_log( writer, model, loss_train, acc_train, loss_val, acc_val, loss_reg, ratio_svd, ratio_dct, ep ):

    logging.info('[AsymML TRAIN] Epoch: [{0}/{1}]\t Loss: {2:.3f}, Accuracy: {3:.3f}'.format(
        ep, args.epochs, loss_train, acc_train ) )
    logging.info('[AsymML VAL  ] Epoch: [{0}/{1}]\t Loss: {2:.3f}, Accuracy: {3:.3f}'.format(
        ep, args.epochs, loss_val, acc_val ) )
    writer.add_scalar( 'loss/train', loss_train, ep )
    writer.add_scalar( 'loss/val', loss_val, ep )
    writer.add_scalar( 'loss/reg', loss_reg, ep )
    writer.add_scalar( 'acc/train', acc_train, ep )
    writer.add_scalar( 'acc/val', acc_val, ep )

    # profile energy leakage
    if args.svd or args.dct:
        writer.add_scalar( 'stats/svd_ratio', ratio_svd, ep )
        writer.add_scalar( 'stats/dct_ratio', ratio_dct, ep )

    # parameter norm
    pnorm = 0.0
    for m in model.parameters():
        if m.requires_grad:
            pnorm += torch.norm( m ) ** 2
    writer.add_scalar( 'stats/pnorm', torch.sqrt( pnorm ), ep )

    # dct energy ratio
    # writer.add_scalar( 'stats/dct_ratio', torch.sqrt( e_dct_ratio ), ep )


if __name__ == '__main__':
    main()
