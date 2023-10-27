import torch
import torch.nn as nn
from model.resnet import BasicBlock
from model.layer_dct import AsymDCT


class Bottleneck( nn.Module ):
    expansion = 4

    def __init__( self, inplanes, planes, stride=1, downsample=None ):
        super( Bottleneck, self ).__init__()
        self.conv1 = nn.Conv2d( inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False )
        self.bn1   = nn.BatchNorm2d( planes )
        # self.relu1 = nn.PReLU( num_parameters=planes )
        self.conv2 = nn.Conv2d( planes, 2 * planes, kernel_size=3, stride=stride, padding=1, bias=False )
        self.bn2   = nn.BatchNorm2d( 2 * planes )
        # self.relu2 = nn.PReLU( num_parameters=2*planes )
        self.conv3 = nn.Conv2d( 2 * planes, planes * self.expansion, kernel_size=1, bias=False )
        self.bn3   = nn.BatchNorm2d( planes * self.expansion )
        self.relu  = nn.ReLU( inplace=True )
        # self.relu3 = nn.PReLU( num_parameters=planes * self.expansion )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetLowRank( nn.Module ):

    def __init__(
            self, block, layers,
            inplanes=64, rank=4, num_classes=10, feature_size=32, mask_size=32, has_dct=False
    ):
        super( ResNetLowRank, self ).__init__()
        self.inplanes, self.in_feature_size, self.in_mask_size = inplanes, feature_size, mask_size
        self.has_dct = has_dct

        self.layer1 = self._make_layer( block, 2 * rank, layers[0], stride=1 )
        if not has_dct:
            self.layer2 = self._make_layer( block, 4 * rank, layers[1], stride=2 )
        self.layer3 = self._make_layer( block, 8 * rank, layers[2], stride=2 )
        self.layer4 = self._make_layer( block, 16 * rank, layers[3], stride=2 )
        self.avgpool = nn.AdaptiveAvgPool2d( 1 )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear( 16 * rank * block.expansion, num_classes )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer( self, block, planes, blocks, stride=1 ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d( self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False ),
                nn.BatchNorm2d( planes * block.expansion ),
            )

        layers = []
        layers.append( block( self.inplanes, planes, stride, downsample ) )
        self.inplanes = planes * block.expansion
        for i in range( 1, blocks ):
            layers.append( block( self.inplanes, planes ) )

        return nn.Sequential( *layers )

    def forward( self, x ):
        x_lyr = self.layer1( x )
        if not self.has_dct:
            x_lyr = self.layer2( x_lyr )
        x_lyr = self.layer3( x_lyr )
        x_lyr = self.layer4( x_lyr )

        x_pool = self.avgpool( x_lyr )
        x_pool = self.flatten( x_pool )

        x_fc = self.fc( x_pool )

        return x_fc


class MlpBlock( nn.Module ):

    def __init__( self, in_features, out_features ):
        super( MlpBlock, self ).__init__()
        self.fc1  = nn.Linear( in_features, out_features )
        self.fc2  = nn.Linear( out_features, out_features )
        self.norm = nn.LayerNorm( in_features )
        self.relu = nn.ReLU()

    def forward( self, x ):
        residual = x

        x_fc1 = self.relu( self.fc1( x ) )
        x_fc2 = self.fc2( x_fc1 )

        out = self.norm( x_fc2 + residual )

        return out


class MlpLowRank( nn.Module ):

    def __init__( self, inplanes=8, num_classes=10, feature_size=32, mask_size=32 ):
        # - inplanes: number of channels in inputs
        # - num_classes: number of target classes
        # - feature_size: size of input feature
        # - mask_size: size of region to be masked

        super( MlpLowRank, self ).__init__()
        self.inplanes, self.in_feature_size, self.in_mask_size = inplanes, feature_size, mask_size

        self.layer_dct = AsymDCT( self.in_feature_size, self.in_mask_size )
        self.flatten   = nn.Flatten()
        self.fc1       = nn.Linear( inplanes * mask_size * mask_size, 512 )
        self.relu      = nn.ReLU()
        self.mlp_blk1  = MlpBlock( 512, 512 )
        self.mlp_blk2  = MlpBlock( 512, 512 )
        self.mlp_blk3  = MlpBlock( 512, 512 )
        self.fc2       = nn.Linear( 512, num_classes )

    def forward( self, x ):
        x_dct  = self.layer_dct( x )

        x_1d   = self.flatten( x_dct )

        x_fc1  = self.relu( self.fc1( x_1d ) )

        x_blk1 = self.mlp_blk1( x_fc1 )
        x_blk2 = self.mlp_blk2( x_blk1 )
        x_blk3 = self.mlp_blk3( x_blk2 )

        out    = self.fc2( x_blk3 )

        return out


def resnet6( pretrained=False, **kwargs ):
    """Constructs a ResNet-6 model."""

    model = ResNetLowRank( BasicBlock, [2, 2, 2], **kwargs )
    return model


def resnet8( pretrained=False, **kwargs ):
    """Constructs a ResNet-6 model."""

    model = ResNetLowRank( Bottleneck, [ 2, 2, 2, 2 ], **kwargs )
    return model


def resnet12( pretrained=False, **kwargs ):
    """Constructs a ResNet-6 model."""

    model = ResNetLowRank( Bottleneck, [ 3, 4, 6, 3 ], **kwargs )
    return model


def mlp3( **kwargs ):
    """Constructs a MLP model"""
    model = MlpLowRank( **kwargs )
    return model