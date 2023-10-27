# several regularization implementations, including
# - orthogonality regularization

import torch
import torch.nn as nn
import numpy as np
from model.resnet import BasicBlock
from model.model_lowrank import Bottleneck, ResNetLowRank


def norm_diff( w ):
    # compute fro norm of diff to identity matrix
    #
    # - w: weight in a layer  -->  torch Tensor

    co, ci = w.shape[ 0 ], w.shape[ 1 ]
    w2d = w.view( co, -1 )
    co, ci = w2d.shape
    if co > ci: w2d = w2d.t()  # convert to fat matrix
    wwT = w2d @ w2d.t()
    diff = wwT -  torch.eye( w2d.shape[ 0 ] ).cuda()

    return torch.norm( diff ) ** 2


def norm_conv_diff( w, stride, padding ):
    # compute fro norm of convolution diff
    #
    # - w: conv weight in a layer

    co, ci, k = w.shape[ 0 ], w.shape[ 1 ], w.shape[ 2 ]
    if co > ci*k*k: w = torch.permute( w, ( 1, 0, 2, 3 ) )
    conv_k = nn.functional.conv2d( w, w, stride=stride, padding=padding )
    target = torch.zeros_like( conv_k )
    ct = int( np.floor( conv_k.shape[ -1 ] / 2 ) )
    target[ :, :, ct, ct ] = torch.eye( conv_k.shape[ 0 ] ).cuda()
    diff = conv_k - target

    return torch.norm( diff ) ** 2


def add_orth_reg( model, alpha, svd=False ):
    # orthogonality regularization
    #
    # - model: model to be regularized
    # - alpha: regularization coefficient  -->  float
    # - svd: asymmetric learning with two branches (M1, M2)

    loss_orth = 0.0
    for name, w in model.named_parameters():
        if len( w.shape ) == 1:
            continue

        elif len( w.shape ) == 2:  # fc layer
            continue

        elif len( w.shape ) == 4:  # conv layer
            co, ci, k, _ = w.shape
            if svd and co == ci * 2 and co <= 256 and k > 1 and 'm_lowrank' in name:
                loss_orth += ( alpha * norm_diff( w ) )

    return loss_orth


def add_l2_reg( model, alpha ):
    # L2 regularization
    #
    # - model: model to be regularized
    # - alpha: regularization coefficient  -->  float

    loss_l2 = 0.0
    for name, param in model.named_parameters():
        loss_l2 += ( 0.5 * alpha * torch.norm( param ) ** 2 )

    return loss_l2


def add_convorth_reg( model, alpha, svd=False ):
    # add conv orthogonality regularization
    #
    # - model: model to be regularized
    # - alpha: regularization coefficient  -> float
    # - svd: asymmetric learning with two branches (M1, M2)

    loss_orth = 0.0

    def __orth_reg( m ):
        nonlocal loss_orth
        for mm in m.children():
            if isinstance( mm, nn.Sequential ) or isinstance( mm, ResNetLowRank ) or \
                    isinstance( mm, BasicBlock ) or isinstance( mm, Bottleneck ):
                __orth_reg( mm )

            elif isinstance( mm, nn.Linear ):
                """
                w = mm.weight
                loss_orth += ( alpha * norm_diff( w ) )
                """
                continue

            elif isinstance( mm, nn.Conv2d ):
                stride, padding, w = mm.stride, mm.padding, mm.weight
                co, ci, k, _ = w.shape

                if svd and co == ci * 2 and co <= 256 and k > 1:
                    loss_orth += ( alpha * norm_conv_diff( w, stride, padding ) )

                    print( w.shape )

    __orth_reg( model )

    quit()

    return loss_orth
