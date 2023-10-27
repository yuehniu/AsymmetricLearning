import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__( self ):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_acc( output, target, topk=(1,) ):
    """
    Calculate model accuracy
    :param output:
    :param target:
    :param topk:
    :return: topk accuracy
    """
    maxk = max( topk )
    batch_size = target.size( 0 )

    _, pred = output.topk( maxk, 1, True, True )
    pred = pred.t()
    correct = pred.eq( target.view( 1, -1 ).expand_as( pred ) )

    acc = []
    for k in topk:
        correct_k = correct[ :k ].view( -1 ).float().sum( 0 )
        acc.append( correct_k.mul_( 100.0 / batch_size ) )
    return acc


def cal_acc_f1( preds, labels, tot_tp, tot_fp, tot_tn, tot_fn ):
    tp = np.logical_and(
        np.equal(labels, True), np.equal( preds, True)
    ).astype(np.int)
    fp = np.logical_and(
        np.equal(labels, False), np.equal( preds, True)
    ).astype(np.int)
    tn = np.logical_and(
        np.equal(labels, False), np.equal( preds, False)
    ).astype(np.int)
    fn = np.logical_and(
        np.equal(labels, True), np.equal( preds, False)
    ).astype(np.int)

    if tot_tp is None:
        tot_tp, tot_fp, tot_tn, tot_fn = tp, fp, tn, fn
    else:
        tot_tp = np.append( tot_tp, tp )
        tot_fp = np.append( tot_fp, fp )
        tot_tn = np.append( tot_tn, tn )
        tot_fn = np.append( tot_fn, fn )

    return tot_tp, tot_fp, tot_tn, tot_fn
