#  convert a resnet-like model to its low-rank counterpart
import copy
import math
import torch
import torch.nn as nn
from model.layer_decomp import AsymSVD
from model.layer_dct import AsymDCT
from model.layer_quant import RandQuant
from model.resnet import BasicBlock
from model.model_lowrank import resnet6, resnet8


class ResNet_Asym( nn.Module ):
    # create a ResNet model compatible with asymmetric learning

    def __init__( self, backbone, m_lowrank, m_residual, svd_layer, dct_layer, quant_layer ):
        super( ResNet_Asym, self ).__init__()
        self.backbone   = backbone
        self.svd_layer, self.dct_layer, self.quant_layer = svd_layer, dct_layer, quant_layer
        self.m_lowrank  = m_lowrank
        self.m_residual = m_residual

        # self._initialize_weights( initmethod )

    def forward( self, x ):
        x_backbone = self.backbone( x )
        x_svd, x_svd_res = self.svd_layer( x_backbone )
        if self.dct_layer:
            x_dct, x_dct_res = self.dct_layer( x_svd )
        else:
            x_dct, x_dct_res = x_svd, 0
        x_residual = x_svd_res + x_dct_res
        if self.quant_layer:
            x_residual_quant = self.quant_layer( x_residual )
        else:
            x_residual_quant = x_residual

        out_lowrank = self.m_lowrank( x_dct )

        if self.m_residual:
            out_residual = self.m_residual( x_residual_quant )
        else:
            out_residual = 0.0

        return out_lowrank+out_residual, x_backbone, x_svd, x_dct

    def _initialize_weights( self, initmethod ):
        for m in self.modules():
            if isinstance( m, nn.Conv2d ):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if initmethod == 'orthogonal':
                    nn.init.orthogonal( m.weight )
                else:
                    m.weight.data.normal_( 0, math.sqrt( 2. / n ) )
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance( m, nn.BatchNorm2d ):
                m.weight.data.fill_( 1 )
                m.bias.data.zero_()
            elif isinstance( m, nn.Linear ):
                if initmethod == 'orthogonal':
                    nn.init.orthogonal( m.weight )
                else:
                    n = m.weight.size( 1 )
                    m.weight.data.normal_( 0, math.sqrt( 2. / n ) )
                m.bias.data.zero_()


def resnet_asym( model, dataset, split_layer, svd, r, dct, quant, sigma ):
    # convert an original model into three sub-models:
    #    - backbone model
    #    - m_lowrank: read low-rank data as inputs
    #    - m_residual: read residuals as inputs
    # backbone & m_lowrank are secured in TEEs, while m_residual is offloaded to the server
    #
    # - model: original model definition
    # - dataset: which dataset is used, split point varies with datasets
    # - split_layer: at which layer to split the model --> int
    # - svd: if add svd layer
    # - r: rank at the split layer --> int
    # - dct: if add dct layer
    # - quant: if add quantization layer
    # - sigma: noise level

    lyr_indx = 0  # convolution layer index
    is_backbone, scaling = True, 1
    backbone, m_residual = [], []
    out_channels, num_classes = 0, 0

    def __convert( m ):
        nonlocal lyr_indx, is_backbone, scaling
        nonlocal backbone, m_residual, num_classes, out_channels
        for mm in m.children():
            if isinstance( mm, nn.Sequential ):
                __convert( mm )
            else:
                if isinstance( mm, BasicBlock ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )
                    lyr_indx += 1

                elif isinstance( mm, nn.Conv2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                        out_channels = mm.out_channels
                    else:
                        m_residual.append( copy.deepcopy( mm ) )

                    lyr_indx += 1

                elif isinstance( mm, nn.ReLU ) or isinstance( mm, nn.Sigmoid ) or \
                        isinstance( mm, nn.PReLU ) or isinstance( mm, nn.SELU ):
                    if is_backbone:
                        backbone.append( nn.PReLU( num_parameters=out_channels ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )

                    if lyr_indx == split_layer and 'cifar' in dataset:
                        is_backbone = False

                elif isinstance( mm, nn.AdaptiveAvgPool2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )

                elif isinstance( mm, nn.MaxPool2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        backbone.append( copy.deepcopy( mm ) )

                    if lyr_indx == split_layer and 'imagenet' in dataset:
                        is_backbone = False

                elif isinstance( mm, nn.BatchNorm2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )

                elif isinstance( mm, nn.Linear ):
                    m_residual.append( nn.Flatten() )
                    m_residual.append( copy.deepcopy( mm ) )
                    num_classes = mm.out_features

    __convert( model )

    backbone = nn.Sequential( *backbone )
    m_lowrank, m_residual = resnet8( inplanes=out_channels, rank=r, num_classes=num_classes, has_dct=dct ), None
    svd_layer, dct_layer, quant_layer = None, None, None
    if svd: svd_layer = AsymSVD( r )
    if dct: dct_layer = AsymDCT()
    if quant: quant_layer = RandQuant( sigma=sigma )
    model_new = ResNet_Asym( backbone, m_lowrank, m_residual, svd_layer, dct_layer, quant_layer )

    return model_new
