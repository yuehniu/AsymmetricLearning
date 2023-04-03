import torch


def svd_approx( x, r, iters=2, eps=1e-6 ):
    """
    Approximated SVD
    :param x, data to be decomposed --> [batch, channel, height, weight ]
    :param r, rank --> int
    :param iters, number of iteration needed --> int
    :param eps --> float
    :return low-rank output and residual
    """
    b, c, h, w = x.shape
    x_copy = x.clone()
    x_copy = x_copy.view( b, c, h*w )
    x_u, x_v = [], []
    i = 0
    while i < r:
        u_i = torch.randn( b, c, 1 ).cuda()
        u_i = torch.nn.functional.normalize( u_i, dim=1 )
        v_i = torch.mean( x_copy, dim=1, keepdim=True ).transpose( 1, 2 )

        # Alternate optimization
        for j in range( iters ):
            u_i = torch.div(
                torch.matmul( x_copy, v_i ),
                torch.matmul( v_i.transpose( 1, 2 ), v_i ) + eps
            )

            v_i = torch.div(
                torch.matmul( x_copy.transpose( 1, 2 ), u_i ),
                torch.matmul( u_i.transpose( 1, 2 ), u_i ) + eps
            )
        x_rank_i = torch.matmul( u_i, v_i.transpose( 1, 2 ) )
        x_copy -= x_rank_i
        x_u.append( u_i.view( b, c, 1 ) ), x_v.append( v_i.view( b, h, w ) )
        i += 1

    x_u, x_v = torch.cat( x_u, dim=2 ), torch.stack( x_v )
    x_v = torch.transpose( x_v, 0, 1 )  # --> [ b, r, h, w ]
    x_residual = x_copy.view( b, c, h, w )

    # print( 'approx svd: ', torch.norm( x_residual ) ** 2 / torch.norm( x ) ** 2 )

    return x_u, x-x_residual, x_residual


def svd_exact( x, r ):
    """
    Exact SVD
    :param x, data to be decomposed --> [ batch, channel, h, w ]
    :param r, rank --> int
    :return low-rank output and residual
    """
    b, c, h, w = x.shape
    x = x.view( b, c, h*w )
    U, s, V = torch.linalg.svd( x, full_matrices=False )

    x_u = U[ :, :, 0:r ] @ torch.diag_embed( s[ :, 0:r ] )
    x_v = V[ :, 0:r, : ]

    x_residual = U[ :, :, r:: ] @ torch.diag_embed( s[ :, r:: ] ) @ V[ :, r::, : ]

    # reshape
    x_v = x_v.view( b, r, h, w )
    x_residual = x_residual.view( b, c, h, w )

    # print( 'exact svd: ', torch.norm( x_residual ) ** 2 / torch.norm( x ) ** 2 )

    return x_u, x_v, x_residual


def svd_lowrank( x, r ):
    """
    Low-rank SVD (another svd approximation method)
    :param x, data to be decomposed --> [ batch, channel, h, w ]
    :param r, rank --> int
    :return low-rank output and residual
    """
    b, c, h, w = x.shape
    x = x.view( b, c, h*w )
    U, s, V = torch.svd_lowrank( x, q=r )
    V = torch.transpose( V, 1, 2 )

    x_u = U
    x_v = V

    x_lowrank  = x_u @ torch.diag_embed( s ) @ x_v
    x_residual = x - x_lowrank

    # reshape
    x_v = x_v.view( b, r, h, w )
    x_lowrank = x_lowrank.view( b, c, h, w )
    x_residual = x_residual.view( b, c, h, w )

    # print( 'lowrank svd: ', torch.norm( x_residual ) ** 2 / torch.norm( x ) ** 2 )

    return x_u, x_lowrank, x_residual
