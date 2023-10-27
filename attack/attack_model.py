"""
Attack model in model inversion (mi) attacks.
It is essentially a GAN model with 1) generator; and 2) discriminator.

Ref:
    attack model: http://arxiv.org/abs/1911.07135

Author:

Note:
"""
import numpy as np
import torch
import torch.nn as nn
import scipy.ndimage.interpolation as interpolation
from utils.meter import cal_acc_f1
from attack.attack_utils import create_rotates, create_translates, apply_augment


class miGenerator( nn.Module ):
    def __init__( self, nc_base, nc_latent, nc_dec ):
        """
        :param nc_base number of kernels in first conv layer
        :param nc_latent number of channel in latent vectors
        :param nc_dec number of channel to decoder
        """
        super( miGenerator, self ).__init__()
        self.prior = nn.Sequential(
            nn.Conv2d( 64, nc_base, 3, 1, padding=1, bias=False ),
            nn.BatchNorm2d( nc_base ),
            nn.ReLU(),
            nn.Conv2d( nc_base, 2*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 2*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 2*nc_base, 4*nc_base, 3, 1, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 4*nc_base, 4*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, padding=1 ),
            # nn.BatchNorm2d( 4*nc_base ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=2, padding=1  ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=4, padding=1 ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=8, padding=1 ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=16, padding=1 ),
        )
        self.enc = nn.Sequential(
            nn.ConvTranspose2d( nc_latent, 2*nc_dec, 4, 1, 0, bias=False  ),  # output: 4x4
            nn.BatchNorm2d( 2*nc_dec ),
            nn.ReLU(),
            nn.ConvTranspose2d( 2*nc_dec, nc_dec, 4, 2, 1, bias=False ),  # output: 8x8
            nn.BatchNorm2d( nc_dec ),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d( nc_dec, nc_dec, 4, 2, 1, bias=False ),  # output: 16x16
            # nn.ConvTranspose2d( 2*nc_dec, nc_dec, 4, 2, 1 ),  # output: 16x16
            nn.BatchNorm2d( nc_dec ),
            nn.ReLU(),
            nn.ConvTranspose2d( nc_dec, nc_dec//2, 4, 2, 1, bias=False ),  # output: 32x32
            nn.BatchNorm2d( nc_dec//2 ),
            nn.ReLU(),
            nn.Conv2d( nc_dec//2, nc_dec//4, 3, padding=1, bias=False ),
            nn.BatchNorm2d( nc_dec//4 ),
            nn.ReLU(),
            nn.Conv2d( nc_dec//4, 3, 3, padding=1, bias=False ),
            nn.Tanh(),
        )

    def forward( self, xu, z ):
        # x = torch.cat( ( self.prior( xu ), self.enc( z ) ), 1 )
        x = self.enc( z ) + self.prior( xu )
        return self.dec( x )


"""Conv-based Discriminator"""
class miDiscriminator( nn.Module ):
    def __init__( self, nc_base ):
        super( miDiscriminator, self ).__init__()

        self.model = nn.Sequential(
            nn.Conv2d( 3, nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( nc_base ),
            nn.ReLU(),
            nn.Conv2d( nc_base, 2*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 2*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 2*nc_base, 4*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 4*nc_base, 8*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 8*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 8*nc_base, 1, 2, bias=False ),
            nn.Sigmoid(),
        )

    def forward( self, x ):
        return self.model( x )


"""Linear layer discriminator
class miDiscriminator( nn.Module ):
    def __init__( self, nc_base ):
        super( miDiscriminator, self ).__init__()

        self.model = nn.Sequential(
            nn.Linear( 3 * 32 * 32, 512 ),
            nn.LeakyReLU( negative_slope=0.2, inplace=True ),

            nn.Linear( 512, 256 ),
            nn.LeakyReLU( negative_slope=0.2, inplace=True ),

            nn.Linear( 256, 1 ),
            nn.Sigmoid()
        )

    def forward( self, x ):
        x = torch.flatten( x, 1 )
        return self.model( x )
"""


"""Membership Attack Model"""
class membershipModel( nn.Module ):
    def __init__( self ):
        super( membershipModel, self ).__init__()

        self.model = nn.Sequential(
            nn.LazyLinear( 10 ),
            nn.LeakyReLU( negative_slope=1e-2 ),
            nn.LazyLinear( 10 ),
            nn.LeakyReLU( negative_slope=1e-2 ),
            nn.LazyLinear( 2 )
        )

    def forward( self, x ):
        return self.model( x )


def aug_attack( model, train_dl, test_dl,  bs, id ):
    aug_rotates = create_rotates( r=9 )
    aug_translates = create_translates( d=1 )
    print( len( aug_rotates ) )
    print( len( aug_translates ) ); quit()

    m = np.concatenate( [ np.ones( len( train_dl ) * bs ), np.zeros( len( test_dl ) * bs ) ], axis=0 )

    attack_in = np.zeros( ( len( train_dl ) * bs, len( aug_rotates ) + len( aug_translates ) ) )
    labels_in = np.zeros( len( train_dl ) * bs )
    attack_out = np.zeros( ( len( test_dl ) * bs, len( aug_rotates ) + len( aug_translates ) ) )
    labels_out = np.zeros( len( test_dl ) * bs )

    """Prepare training and test examples for the attack model"""
    for i, aug in enumerate( aug_rotates ):
        for j, ( inputs, labels ) in enumerate( train_dl ):
            inputs = inputs.numpy()
            inputs = interpolation.rotate( inputs, aug, ( 1, 2 ), reshape=False )
            inputs = torch.from_numpy( inputs ).cuda()

            if id == 'Target':
                outputs, _, _, _, _ = model( inputs )
            else:
                outputs = model( inputs )

            outputs, labels = outputs.detach().cpu().numpy(), labels.numpy()
            pred_correct = np.equal( labels, np.argmax( outputs, axis=1 ) ).squeeze()

            offset = j * bs
            attack_in[ offset:offset+bs, i ] = pred_correct
            labels_in[ offset:offset+bs ] = labels

            # break

        for j, ( inputs, labels ) in enumerate( test_dl ):
            inputs = inputs.numpy()
            inputs = interpolation.rotate( inputs, aug, ( 1, 2 ), reshape=False )
            inputs = torch.from_numpy( inputs ).cuda()

            if id == 'Target':
                outputs, _, _, _, _ = model(inputs)
            else:
                outputs = model(inputs)

            outputs, labels = outputs.detach().cpu().numpy(), labels.numpy()
            pred_correct = np.equal( labels, np.argmax( outputs, axis=1 ) ).squeeze()

            offset = j * bs
            attack_out[ offset:offset + bs, i ] = pred_correct
            labels_out[ offset:offset + bs ] = labels

            # break

        print( '{}, {}-th aug'.format( id, i ) )

    attack_set = (
        np.concatenate( [ attack_in, attack_out ], 0 ),
        np.concatenate( [ labels_in, labels_out ], 0 ),
        m
    )

    return attack_set


def train_aug_attack( train_set, test_set, n_classes=100 ):
    models = []
    for i in range( n_classes ):
        model = membershipModel()
        models.append( model )

    acc_labels_train, f1_labels_train = [], []
    acc_labels_test, f1_labels_test = [], []
    for i in range( len( models ) ):
        m = models[ i ]
        crit = torch.nn.CrossEntropyLoss()
        opt = torch.optim.Adam(
            m.parameters(),
            0.001,
        )

        """Create a sub dataset for each label"""
        train_set_sel = train_set[ 1 ].flatten() == i
        inputs_sub = torch.from_numpy( train_set[ 0 ][ train_set_sel ].astype( np.float32 ) )
        labels_sub = torch.from_numpy( train_set[ 2 ][ train_set_sel ].astype( np.int ) )
        if len( labels_sub ) < 50: continue

        train_set_sub = torch.utils.data.TensorDataset( inputs_sub, labels_sub )
        train_dl_sub = torch.utils.data.DataLoader(
            train_set_sub,
            batch_size=16, shuffle=True
        )

        """Train an attacker model for each label"""
        tot_tp, tot_fp, tot_tn, tot_fn = None, None, None, None
        for j, (inputs, labels) in enumerate( train_dl_sub ):
            outputs = m( inputs )
            loss = crit( outputs, labels )
            loss.backward()
            opt.step()

            labels = labels.numpy().astype( np.bool ).squeeze()
            preds = np.argmax( outputs.detach().numpy(), axis=1 ).astype( np.bool )
            tot_tp, tot_fp, tot_tn, tot_fn = cal_acc_f1( labels, preds, tot_tp, tot_fp, tot_tn, tot_fn )

        acc = ( np.sum( tot_tp ) + np.sum( tot_tn ) ) / \
              ( np.sum( tot_tp ) + np.sum( tot_tn ) + np.sum( tot_fp ) + np.sum( tot_fn ) )

        r = np.sum( tot_tp ) / ( np.sum( tot_tp ) + np.sum( tot_fn ) )
        p = np.sum( tot_tp ) / ( np.sum( tot_tp ) + np.sum( tot_fp ) )
        f1 = ( 2 * ( r * p ) ) / ( r + p )
        if np.isnan( f1 ): continue

        acc_labels_train.append( acc )
        f1_labels_train.append( f1 )

        print( '[Train] {}-th label: Acc: {:.4f}, \t F1: {:.4f}'.format( i, acc, f1 ) )

        """---- Train Ends ----"""

        """Test the attacker model for each label"""
        test_set_sel = test_set[ 1 ].flatten() == i
        inputs_sub = torch.from_numpy( test_set[ 0 ][ test_set_sel ].astype( np.float32 ) )
        labels_sub = torch.from_numpy( test_set[ 2 ][ test_set_sel ].astype( np.int ) )
        test_set_sub = torch.utils.data.TensorDataset( inputs_sub, labels_sub )
        test_dl_sub = torch.utils.data.DataLoader(
            test_set_sub,
            batch_size=100, shuffle=False
        )

        tot_tp, tot_fp, tot_tn, tot_fn = None, None, None, None
        for j, ( inputs, labels ) in enumerate( test_dl_sub ):
            outputs = m( inputs )

            labels = labels.numpy().astype( np.bool ).squeeze()
            preds  = np.argmax( outputs.detach().numpy(), axis=1 ).astype( np.bool )
            tot_tp, tot_fp, tot_tn, tot_fn = cal_acc_f1( labels, preds, tot_tp, tot_fp, tot_tn, tot_fn )

        acc = ( np.sum( tot_tp ) + np.sum( tot_tn ) ) / \
              ( np.sum( tot_tp ) + np.sum( tot_tn ) + np.sum( tot_fp ) + np.sum( tot_fn ) )

        r = np.sum( tot_tp ) / ( np.sum( tot_tp ) + np.sum( tot_fn ) )
        p = np.sum( tot_tp ) / (np.sum( tot_tp ) + np.sum( tot_fp ) )
        f1 = ( 2 * ( r * p ) ) / ( r + p )
        if np.isnan( f1 ): continue

        print( '[Test] {}-th label: Acc: {:.4f}, \t F1: {:.4f}'.format( i, acc, f1 ) )

        acc_labels_test.append( acc )
        f1_labels_test.append( f1 )

    acc_labels_avg, f1_labels_avg = np.average( acc_labels_train ), np.average( f1_labels_train )
    print( 'Global Train Acc: {:.4f}, \t Global F1: {:.4f}'.format( acc_labels_avg, f1_labels_avg ) )

    acc_labels_avg, f1_labels_avg = np.average( acc_labels_test ), np.average( f1_labels_test )
    print( 'Global Test Acc: {:.4f}, \t Global F1: {:.4f}'.format( acc_labels_avg, f1_labels_avg ) )
