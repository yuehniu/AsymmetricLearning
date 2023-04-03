"""A customized (randomized) quantization layer that
    - apply probabilitistic quantization on residuals after the backbone

"""
import torch
import torch.nn as nn


def quant_op( input, sigma=1 ):
    input_mean = torch.mean( torch.abs( input ) )
    noise_std = sigma * torch.sqrt( input_mean )
    noise = torch.randn_like( input ) * noise_std
    input_noisy = input + noise
    input_quant = torch.sign( input_noisy )

    return input_quant


"""define a autograd function for quantization"""
class rand_quant( torch.autograd.Function ):

    @staticmethod
    def forward( ctc, input, sigma ):
        return quant_op( input, sigma )

    @staticmethod
    def backward( ctx, grad_out ):
        grad_in = torch.zeros_like( grad_out )
        return grad_out, None


"""define a quantization layer"""
class RandQuant( nn.Module ):

    def __init__( self, sigma=1 ):
        super( RandQuant, self ).__init__()
        self.sigma=sigma

    def forward( self, input ):
        return rand_quant.apply( input, self.sigma )