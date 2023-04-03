"""orthogonalization layer
A custom nn layer that orthogonalize input features,
output orthogonal features

"""
import torch
import torch.nn as nn


def gs_projection( u, v ):
    uv, uu = torch.mul( u, v ), torch.mul( u, u )
    return uv.sum( dim=(1, 2),  keepdim=True ) / uu.sum( dim=(1, 2),  keepdim=True ) * u


class GSOrth( nn.Module ):
    def __init__( self, r, dim ):
        """
        r: number of orthogonal channels to calculate
        dim: height and width of input feature
        """
        super( GSOrth, self ).__init__()
        self.r = r
        self.h, self.w = dim[ 0 ], dim[ 1 ]

        """initialize the basis"""
        self.basis = torch.zeros( self.r, self.h, self.w ).cuda()
        self.vec = self.basis.view( self.r, -1 )
        for i in range( self.r ):
            stride = self.r
            self.vec[ i, i::stride ] = 1.0
            vec_norm = torch.norm( self.vec[ i ] )
            self.vec[ i ].div_( vec_norm )

    def forward( self, input ):
        b, c, h, w = input.shape

        """project feature into the pre-define basis"""
        coeff_0 = torch.mul( input, self.basis[ 0 ] ).sum( dim=(1, 2, 3), keepdim=True )
        input_orth = coeff_0 * self.basis[ 0 ]
        for i in range( 1, self.r ):
            coeff_i = torch.mul( input, self.basis[ i ] ).sum( dim=(1, 2, 3), keepdim=True )
            input_orth_i = coeff_i * self.basis[ i ]
            input_orth = torch.concat( (input_orth, input_orth_i), 1 )

        return input_orth
