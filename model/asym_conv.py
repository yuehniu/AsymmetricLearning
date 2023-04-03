#  In this file, we implement the customized conv op that
#  - decompose intermediate actions in an asymmetric way;
#  - read low-rank part as input and apply convolution op on it
import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_lowrank( nn.Module ):
    """convolution with low-rank input"""

    def __init__( self, n_kernels, n_channels, sz_kernel, stride, padding ):
        super( conv_lowrank ).__init__()

        self.n_kernels, self.n_channels, self.sz_kernel = n_kernels, n_channels, sz_kernel
        self.stride, self.padding = stride, padding

        self.w = nn.Parameter( torch.empty( n_kernels, n_channels, sz_kernel, sz_kernel ) )
        self.bias = nn.Parameter( torch.empty( n_kernels ) )

        nn.init.kaiming_normal_( self.w, mode='fan_out', nonlinearity='relu' )
        nn.init.constant( self.bias, 0 )

    def forward( self, v, u ):
        """forward
        :param v, principal channels
        :param u, coefficients of the principal channels
        """

        # transform self.w given u
        b, p, c = u.shape
        m, n, k, _ = self.w.shape
        u_kernel = u.view( b*p, c, 1, 1 )
        w_u = F.conv2d( self.w, u_kernel ).view( b*m, n, k, k )

        # apply convolution
        _, _, h, w = v.shape
        x = v.view( 1, b*c, h, w )
        y = F.conv2d( x, w_u, groups=b, stride=self.stride, padding=self.padding )
        y = y.view( b, m, y.size( 2 ), y.size( 3 ) )

        y += self.bias
