import torch
import torch.nn as nn
import numpy as np

def init_netGD( m ):
    """
    custom weight init function for generator
    """
    classname = m.__class__.__name__
    if classname.find( 'Conv' ) != -1:
        nn.init.normal_( m.weight.data, 0.0, 0.02 )
    elif classname.find( 'BatchNorm' ) != -1:
        nn.init.normal_( m.weight.data, 1.0, 0.02 )
        nn.init.constant_( m.bias.data, 0 )


def split_data( trainset, testset, ndata ):
    ndata_train, ndata_test = int( ndata * len( trainset ) ), int( ndata * len( testset ) )
    src_train_indices = list( range( 0, ndata_train//2 ) )
    tgt_train_indices = list( range( ndata_train, len( trainset ) ) )  # len( trainset )
    src_train_set = torch.utils.data.Subset( trainset, src_train_indices )
    tgt_train_set = torch.utils.data.Subset( trainset, tgt_train_indices )

    src_test_indices = list( range( 0, ndata_test//2 ) )
    tgt_test_indices = list( range( ndata_test, len( testset ) ) )
    src_test_set = torch.utils.data.Subset( testset, src_test_indices )
    tgt_test_set = torch.utils.data.Subset( testset, tgt_test_indices )

    return tgt_train_set, tgt_test_set, src_train_set, src_test_set


def create_rotates( r ):
    rotates = np.linspace( -r, r, ( r * 2 + 1 ) )

    return rotates


def create_translates( d ):

    def all_shifts( mshift ):
        if mshift == 0:
            return [(0, 0, 0, 0)]
        all_pairs = []
        start = (0, mshift, 0, 0)
        end = (0, mshift, 0, 0)
        vdir = -1
        hdir = -1
        first_time = True
        while (start[1] != end[1] or start[2] != end[2]) or first_time:
            all_pairs.append(start)
            start = (0, start[1] + vdir, start[2] + hdir, 0)
            if abs(start[1]) == mshift:
                vdir *= -1
            if abs(start[2]) == mshift:
                hdir *= -1
            first_time = False
        all_pairs = [(0, 0, 0, 0)] + all_pairs  # add no shift
        return all_pairs

    translates = all_shifts( d )
    return translates


import scipy.ndimage.interpolation as interpolation
def apply_augment( dl, aug, type ):
    for i, (inputs, labels) in enumerate( dl ):
        if type == 'd':
            ds = ( interpolation.shift( inputs, aug, mode='nearest' ), labels )
        else:
            ds = ( interpolation.rotate( inputs, aug, ( 1, 2 ), reshape=False ), labels )