#  register running arguments
import argparse


def get_args():
    parser = argparse.ArgumentParser( description='Asymmetric Learning running arguments' )
    parser.add_argument( '--cuda', action='store_true' )
    parser.add_argument( '--model', type=str, default='vgg11', choices=[ 'vgg11', 'vgg11_bn', 'resnet18', 'resnet6', 'resnet8' ] )
    parser.add_argument( '--svd', action='store_true', help='add svd decomposition module'  )
    parser.add_argument( '--dct', action='store_true', help='add dct decomposition module' )
    parser.add_argument( '--quant', action='store_true', help='add quantization module' )
    parser.add_argument( '--noise', type=float, default=1, help='noise level added when apply quantization' )
    parser.add_argument( '--dataset', type=str, default='cifar10', choices=[ 'cifar10', 'cifar100', 'imagenet' ] )

    # train hparams
    parser.add_argument( '--lr',         type=float, default=0.1 )
    parser.add_argument( '--momentum',   type=float, default=0.9 )
    parser.add_argument( '--wd',         type=float, default=1e-4 )
    parser.add_argument( '--batch-size', type=int, default=128  )
    parser.add_argument( '--epochs',     type=int, default=150 )
    parser.add_argument( '--workers',    type=int, default=8 )
    parser.add_argument( '--init',       type=str, default='gaussian', choices=[ 'gaussian', 'orthogonal' ] )
    parser.add_argument( '--reg',        type=str, default='l2', choices=[ 'l2', 'orth', 'conv' ] )
    parser.add_argument( '--regcoeff',   type=float, default=0.0001 )

    # asymmetric learning
    parser.add_argument( '--rank', type=int, default=4, help='rank at the splitting layer' )
    parser.add_argument( '--maxchannel', type=int, default=64, help='max number of output channels in low-rank part' )

    # print
    parser.add_argument( '--logdir', type=str, default='./log' )

    return parser
