#  a customized nn layer that exploit spatial correlations using DCT transform
import math

import numpy as np
import torch
import torch.nn as nn

"""generate DCT transform matrix"""
in_sz, s_in_sz = 32, 16    # input / lowrank input feature size to DCT layer 56/28, 32/16
M, m = 16, 8    # DCT / lowrank DCT transform block size 14/7, 16/8
T, t = torch.zeros( M, M ).cuda(), torch.zeros( m, m ).cuda()
for i in range( M ):
    for j in range( M ):
        if i == 0:
            T[ i, j ] = 1 / math.sqrt( M )
        else:
            T[ i, j ] = math.sqrt( 2 / M ) * math.cos( math.pi * (2 * j + 1) * i / ( 2 * M ) )

for i in range( m ):
    for j in range( m ):
        if i == 0:
            t[ i, j ] = 1 / math.sqrt( m )
        else:
            t[ i, j ] = math.sqrt( 2 / m ) * math.cos( math.pi * (2 * j + 1) * i / ( 2 * m ) )


# define an autograd function to perform forward and backward of DCT transform
class asym_dct( torch.autograd.Function ):

    @staticmethod
    def forward( ctx, input ):
        # - ctx:
        # - input: lowrank input
        # - input2: svd residuals

        """apply block-wise DCT transform"""
        b, c, _, _ = input.shape
        input_mean = torch.mean( input, dim=( 2, 3 ), keepdim=True )
        input_zero_mean = input
        x_lowfreq = torch.zeros( b, c, s_in_sz, s_in_sz, device='cuda', dtype=input.dtype )
        X_highfreq = torch.zeros( b, c, in_sz, in_sz, device='cuda', dtype=input.dtype )
        for i in range( 0, in_sz // M ):
            for j in range( 0, in_sz // M ):
                X_blk = input_zero_mean[ :, :, i*M:(i+1)*M, j*M:(j+1)*M ]
                X_blk_dct = T.cuda() @ X_blk @ T.cuda().t()

                """compute low-frequency part (1st way)"""
                x_blk_dct = X_blk_dct[ :, :, 0:m, 0:m ]
                x_blk_idct = t.cuda().t() @ x_blk_dct @ t.cuda()
                x_lowfreq[ :, :, i*m:(i+1)*m, j*m:(j+1)*m ].copy_( x_blk_idct )

                """compute low-frequency part (2nd way)
                x_blk_dct = torch.zeros_like( X_blk_dct )
                x_blk_dct[ :, :, 0:m, 0:m].copy_( X_blk_dct[:, :, 0:m, 0:m] )
                x_blk_idct = T.t() @ x_blk_dct @ T
                x_lowfreq[ :, :, i * M:(i + 1) * M, j * M:(j + 1) * M ].copy_( x_blk_idct )
                """

                """compute high-frequency part"""
                X_blk_dct_highfreq = X_blk_dct.clone()
                X_blk_dct_highfreq[ :, :, 0:m, 0:m ] = 0.0
                X_blk_idct_highfreq = T.cuda().t() @ X_blk_dct_highfreq @ T.cuda()
                X_highfreq[ :, :, i*M:(i+1)*M, j*M:(j+1)*M ].copy_( X_blk_idct_highfreq )

        # print('dct: ', torch.norm( X_highfreq ) ** 2 / torch.norm( input ) ** 2); quit()

        return x_lowfreq, X_highfreq

    @staticmethod
    def backward( ctx, grad_out, grad_out2 ):
        # - ctx:
        # - grad_out: gradient from top-left corner
        # - grad_out2: gradient from the residual

        b, c, _, _ = grad_out.shape

        # compute gradient on inputs
        grad_in = torch.zeros( b, c, in_sz, in_sz, device='cuda', dtype=grad_out.dtype )
        for i in range( 0, s_in_sz // m ):  # need to change to 'm' if using the first way
            for j in range( 0, s_in_sz // m ):
                """low-frequency backward (1st way)"""
                g_blk = grad_out[ :, :, i*m:(i+1)*m, j*m:(j+1)*m ]
                g_blk_1 = t.cuda() @ g_blk @ t.t().cuda()

                G_blk_1 = torch.zeros( b, c, M, M, device='cuda', dtype=grad_out.dtype )
                G_blk_1[ :, :, 0:m, 0:m ].copy_( g_blk_1 )

                """low-frequency backward (2nd way)
                g_blk = grad_out[ :, :, i*M:(i+1)*M, j*M:(j+1)*M ]
                G_blk_1 = T @ g_blk @ T.t()
                G_blk_1[ :, :, m:M, : ].zero_()
                G_blk_1[ :, :, :, m:M ].zero_()
                """

                G_blk_2 = T.cuda().t() @ G_blk_1 @ T.cuda()
                grad_in[ :, :, i*M:(i+1)*M, j*M:(j+1)*M ].copy_( G_blk_2 )

        # if mean is subtract before DCT
        # grad_in -= torch.mean( grad_in, dim=( 2, 3 ), keepdim=True )

        return grad_in + grad_out2


#  define an asymmetric decomposition layer
class AsymDCT( nn.Module ):

    def __init__( self ):
        super( AsymDCT, self ).__init__()
        self.name = 'AsymDCT'

    def forward( self, input ):
        # X_dct = self.T @ input @ self.T.t()
        # return X_dct[ :, :, 0:self.M_keep, 0:self.M_keep ]
        return asym_dct.apply( input )
