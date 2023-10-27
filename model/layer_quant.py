"""A customized (randomized) quantization layer that
    - apply probabilitistic quantization on residuals after the backbone

"""
import math
import torch
import torch.nn as nn


def quant_op( input, epsilon=1, p=0.0001 ):
    b, c = input.size( 0 ), input.size( 1 )
    input_reduced = torch.mean( input, dim=1, keepdim=True )
    input_norm = torch.norm( input_reduced, p=1 ) / b
    noise_std = p * input_norm / epsilon

    # noise_std = min( 0.8, noise_std )
    # noise = torch.randn_like( input ) * noise_std
    Lap = torch.distributions.laplace.Laplace( 0, noise_std )
    noise = Lap.sample( input_reduced.shape ).cuda()

    input_noisy = input_reduced + noise
    input_quant = torch.sign( input_noisy )

    return input_quant


class rand_quant( torch.autograd.Function ):

    @staticmethod
    def forward( ctx, input, sigma, p ):
        ctx.save_for_backward( input )

        return quant_op( input, sigma, p )

    @staticmethod
    def backward( ctx, grad_out ):
        input, = ctx.saved_tensors

        grad_in = torch.zeros_like( input )
        return grad_in, None, None


"""define a quantization layer"""
class RandQuant( nn.Module ):

    def __init__( self, sigma=1, p=0.001 ):
        super( RandQuant, self ).__init__()
        self.sigma=sigma
        self.p=p

    def forward( self, input ):
        return rand_quant.apply( input, self.sigma, self.p )