#  convert a resnet-like model to its low-rank counterpart
import copy
import math
import torch
import torch.nn as nn
from model.layer_decomp import AsymSVD
from model.layer_dct import AsymDCT
from model.layer_quant import RandQuant
from model.layer_merge import MergeLogits
from model.resnet import BasicBlock
from model.model_lowrank import resnet6, resnet8, resnet12


class ResNet_Asym( nn.Module ):
    # create a ResNet model compatible with asymmetric learning

    def __init__( self, backbone, m_lowrank, m_residual, svd_layer, dct_layer, quant_layer ):
        super( ResNet_Asym, self ).__init__()
        self.backbone    = backbone
        self.svd_layer   = svd_layer
        self.dct_layer   = dct_layer
        self.quant_layer = quant_layer
        self.m_lowrank   = m_lowrank
        self.m_residual  = m_residual
        self.merge_layer = MergeLogits()

        self.lr_only = False  # if train with low-rank model only

        # self.scale = nn.Parameter( torch.FloatTensor( [ 1.0 ] ) )

        # self._initialize_weights( initmethod )

    def forward( self, x ):
        x_bb = self.backbone( x )

        x_svd, x_svd_res = self.svd_layer( x_bb )
        if self.dct_layer:
            x_dct, x_dct_res = self.dct_layer( x_svd )
        else:
            x_dct, x_dct_res = x_svd, 0
        x_res = x_svd_res + x_dct_res

        # print( 'l1 norm, x_bb: ', torch.norm( x_bb[ 0 ], 1 ) )
        # print( 'l1 norm, x_res: ', torch.norm( x_res[ 0 ], 1 ) ); quit()
        # print( 'res energy ratio: ', torch.norm( x_res[ 0 ] ) ** 2 / torch.norm( x_bb[ 0 ] ) ** 2 ); quit()

        if self.quant_layer:
            x_res_quant = self.quant_layer( x_res )
        else:
            x_res_quant = x_res

        out_lwr = self.m_lowrank( x_dct )

        if self.m_residual and not self.lr_only:
            out_res = self.m_residual( x_res_quant )
            out = self.merge_layer( out_lwr, out_res )
        else:
            out_res = torch.zeros_like( out_lwr )
            out = out_lwr

        return out, out_res, x_bb, x_svd, x_dct

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


def resnet_asym( model, dataset, split_layer, svd, r, dct, quant, sigma, p ):
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
    # - p: sampling ratio

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

    backbone   = nn.Sequential( *backbone )
    if len( m_residual ) == 11:
        M1  = resnet8( inplanes=out_channels, rank=r, num_classes=num_classes, has_dct=dct )
    else:
        M1 = resnet12( inplanes=out_channels, rank=r, num_classes=num_classes, has_dct=dct )
    M2 = None

    svd_layer, dct_layer, quant_layer, fc_layer = None, None, None, None
    if svd: svd_layer = AsymSVD( r )
    if dct: dct_layer = AsymDCT()
    if quant:
        quant_layer = RandQuant( sigma=sigma, p=p )

        conv_map = nn.Conv2d( 1, out_channels, kernel_size=3, stride=1, padding=1, bias=False )
        nn.init.kaiming_normal_( conv_map.weight, mode='fan_out', nonlinearity='relu' )

        bn_map, relu_map = nn.BatchNorm2d( out_channels ), nn.ReLU( inplace=True )
        nn.init.constant_( bn_map.weight, 1 )
        nn.init.constant_( bn_map.bias, 0 )

        M2 = nn.Sequential( *[ conv_map, bn_map, relu_map, *m_residual]  )

    model_new = ResNet_Asym( backbone, M1, M2, svd_layer, dct_layer, quant_layer )

    return model_new
