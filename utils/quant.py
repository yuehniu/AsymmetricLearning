import random
from scipy.stats import norm

sigma = 0.02


def prob_quant( x ):
    x_vec = x.reshape( -1 )
    l = x.size
    for i in range( l ):
        x_i = x_vec[ i ]
        p = norm.cdf( x_i/sigma )
        r = random.uniform( 0, 1 )
        if p < r:
            x_vec[ i ] = 1.0
        else:
            x_vec[ i ] = 0.0

    return x
