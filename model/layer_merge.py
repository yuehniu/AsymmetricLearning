"""A customized (randomized) quantization layer that
    - apply probabilitistic quantization on residuals after the backbone

"""
import torch
import torch.nn as nn


class merge_logits( torch.autograd.Function ):
    @staticmethod
    def forward( ctx, input1, input2, sigma ):
        prob_m2 = nn.functional.softmax( input2, dim=1 )

        ctx.constant = sigma
        ctx.save_for_backward( prob_m2 )
        return input1 + input2

    @staticmethod
    def backward( ctx, grad_out ):
        b = grad_out.size( 0 )
        grad_in1 = grad_out.clone()

        """1st way: separate update"""
        y = grad_out <= 0
        label = y.to( grad_out.dtype )
        prob_m2, = ctx.saved_tensors
        grad_in2 = ( prob_m2 - label ) / b

        """2nd way: directly add noise
        sigma = ctx.constant
        g_norm = torch.norm( grad_out, dim=1, keepdim=True )
        noise = sigma * g_norm * torch.randn_like( grad_out )
        grad_in2 = grad_out.clone() + noise
        """

        return grad_in1, grad_in2, None


"""define a quantization layer"""
class MergeLogits( nn.Module ):

    def __init__( self ):
        super( MergeLogits, self ).__init__()
        self.sigma = 0.0  # noise added to gradient  CIFAR-10: 0.0002

    def forward( self, input1, input2 ):
        return merge_logits.apply( input1, input2, self.sigma )