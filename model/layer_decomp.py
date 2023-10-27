#  a customized nn layer that decomposes intermediate representations (IR) into
#  - low-rank part and
#  - residuals
import torch
import torch.nn as nn
from utils.decomp import svd_approx, svd_exact, svd_lowrank


#  define an autograd function to decompose IR in an asymmetric manner
class asym_svd( torch.autograd.Function ):

    @staticmethod
    def forward( ctx, input, r, freeze ):
        # - ctx:
        # - input: input IR
        # - r: rank

        # u, x_lowrank, x_residual = svd_exact( input, r )
        u, x_lowrank, x_residual = svd_approx( input, r )
        # u, x_lowrank, x_residual = svd_lowrank( input, r )

        ctx.constant = freeze
        ctx.save_for_backward( u )  # coefficient for orthogonal channels

        return x_lowrank, x_residual

    @staticmethod
    def backward( ctx, grad_out1, grad_out2 ):
        # - ctx:
        # - grad_out1: gradient from low-rank part
        # - grad_out2: gradient from residual part

        """transform gradient on low-rank part
        b, r, h, w = grad_out1.shape
        m_grad1 = torch.reshape( grad_out1, ( b, r, -1 ) )
        u, = ctx.saved_tensors  # b x c x r
        _, c, _ = u.shape
        m_grad1_x = torch.matmul( u, m_grad1 ).view( b, c, h, w )
        """
        freeze = ctx.constant

        #  combine gradients
        if freeze:
            grad_in = torch.zeros_like( grad_out1 )
        else:
            grad_in = grad_out1 + grad_out2

        return grad_in, None, None


#  define an asymmetric decomposition layer
class AsymSVD( nn.Module ):

    def __init__( self, r, freeze=False ):
        #  - r: rank
        #  - freeze: freeze the gradient to backbone

        super( AsymSVD, self ).__init__()
        self.r = r
        self.freeze = freeze

    def forward( self, input ):
        return asym_svd.apply( input, self.r, self.freeze )