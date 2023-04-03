#  convert a vgg-like model to its low-rank counterpart
import copy
import math
import torch
import torch.nn as nn
from model.layer_decomp import AsymSVD


class VGG_Asym( nn.Module ):
    # create a VGG model compatible with asymmetric learning

    def __init__( self, backbone, m_lowrank, m_residual, initmethod='gaussian' ):
        super( VGG_Asym, self ).__init__()
        self.backbone   = backbone
        self.m_lowrank  = m_lowrank
        self.m_residual = m_residual

        self._initialize_weights( initmethod )

    def forward( self, x ):
        x_lowrank, x_residual = self.backbone( x )
        out_lowrank  = self.m_lowrank( x_lowrank )
        out_residual = self.m_residual( x_residual )

        return out_lowrank + out_residual, x_lowrank, x_residual

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


def vgg_asym( model, split_layer, r, initmethod='gaussian' ):
    # convert an orginal model into three sub-models:
    #    - backbone model
    #    - m_lowrank: read low-rank data as inputs
    #    - m_residual: read residuals as inputs
    # backbone & m_lowrank are secured in TEEs, while m_residual is offloaded to the server
    #
    # - model: original model definition
    # - split_layer: at which layer to split the model --> int
    # - r: rank at the split layer --> int
    # - initmethod: weight initialization method --> string

    lyr_indx = 0  # convolution layer index
    is_backbone, scaling = True, 1
    backbone, m_lowrank, m_residual = [], [], []

    def __convert( m ):
        nonlocal lyr_indx, is_backbone, scaling
        nonlocal backbone, m_lowrank, m_residual
        for mm in m.children():
            if isinstance( mm, nn.Sequential ):
                __convert( mm )
            else:
                if isinstance( mm, nn.Conv2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:  # create two branches: lowrank and residual
                        m_residual.append( copy.deepcopy( mm ) )
                        kern_sz, stride, padding = mm.kernel_size[0], mm.stride[0], mm.padding
                        in_channels, out_channels = min( 32, r * scaling ), min( 32, 2 * r * scaling )
                        m_lowrank.append( nn.Conv2d( in_channels, out_channels, kern_sz, stride=stride, padding=padding ) )
                        scaling *= 2

                    lyr_indx += 1
                elif isinstance( mm, nn.LeakyReLU ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )
                        m_lowrank.append( nn.LeakyReLU() )

                    if lyr_indx == split_layer:
                        backbone.append( AsymDecomp( r ) )
                        is_backbone = False
                elif isinstance( mm, nn.MaxPool2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )
                        kern_sz = mm.kernel_size
                        m_lowrank.append( nn.MaxPool2d( kernel_size=kern_sz ) )
                elif isinstance( mm, nn.BatchNorm2d ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )
                        channels = min( 32, r * scaling )
                        m_lowrank.append( nn.BatchNorm2d( channels ) )
                elif isinstance( mm, nn.Flatten ):
                    if is_backbone:
                        backbone.append( copy.deepcopy( mm ) )
                    else:
                        m_residual.append( copy.deepcopy( mm ) )
                        m_lowrank.append( nn.Flatten() )
                elif isinstance( mm, nn.Linear ):
                    m_residual.append( copy.deepcopy( mm ) )
                    out_features = mm.out_features
                    in_channels = min( 32, r * scaling )
                    m_lowrank.append( nn.Linear( in_channels, out_features ) )

    __convert( model )

    backbone = nn.Sequential( *backbone )
    m_lowrank, m_residual = nn.Sequential( *m_lowrank ), nn.Sequential( *m_residual )
    model_new = VGG_Asym( backbone, m_lowrank, m_residual, initmethod )

    return model_new
