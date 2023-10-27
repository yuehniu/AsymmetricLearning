#  register running arguments
import argparse


def get_args():
    parser = argparse.ArgumentParser( description='Asymmetric Learning running arguments' )
    parser.add_argument( '--cuda', action='store_true' )
    parser.add_argument( '--model', type=str, default='vgg11', choices=[ 'vgg11', 'vgg11_bn', 'resnet18', 'resnet34' ] )
    parser.add_argument( '--svd', action='store_true', help='add svd decomposition module'  )
    parser.add_argument( '--dct', action='store_true', help='add dct decomposition module' )
    parser.add_argument( '--quant', action='store_true', help='add quantization module' )
    parser.add_argument( '--epsilon', type=float, default=1, help='DP privacy budget' )
    parser.add_argument( '--dpdata', action='store_true', help='add noise directly to data' )
    parser.add_argument( '--p', type=float, default=0.001, help='sampling ratio')
    parser.add_argument( '--dataset', type=str, default='cifar10', choices=[ 'cifar10', 'cifar100', 'imagenet' ] )

    # train hparams
    parser.add_argument( '--lr',         type=float, default=0.1 )
    parser.add_argument( '--momentum',   type=float, default=0.9 )
    parser.add_argument( '--wd',         type=float, default=1e-4 )
    parser.add_argument( '--batch-size', type=int, default=64  )
    parser.add_argument( '--epochs',     type=int, default=150 )
    parser.add_argument( '--workers',    type=int, default=8 )
    parser.add_argument( '--init',       type=str, default='gaussian', choices=[ 'gaussian', 'orthogonal' ] )
    parser.add_argument( '--reg',        type=str, default='l2', choices=[ 'l2', 'orth', 'conv' ] )
    parser.add_argument( '--regcoeff',   type=float, default=0.0001 )
    parser.add_argument( '--seed',       type=int, default=1 )

    # asymmetric learning
    parser.add_argument( '--rank', type=int, default=4, help='rank at the splitting layer' )
    parser.add_argument( '--maxchannel', type=int, default=64, help='max number of output channels in low-rank part' )

    # print
    parser.add_argument( '--logdir', type=str, default='./log' )

    # save checkpoints
    parser.add_argument( '--save-dir', type=str, default='./checkpoints/' )

    return parser
