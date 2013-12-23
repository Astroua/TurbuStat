import numpy as np

def kl_divergence(P, Q):
    '''
    Kullback Leidler Divergence

    INPUTS
    ------

    P,Q - array
          Two Discrete Probability distributions
    '''

    return np.sum(np.where(P!=0, P*np.log(P / Q), 0))
