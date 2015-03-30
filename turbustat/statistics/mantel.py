
import numpy as np
import numpy.random as ra
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform


def mantel_test(dist1, dist2, corr_func='pearson', nperm=1e3,
                seed=2904100, pval_type='greater'):
    '''
    Perform the Mantel test to compare 2 distance matrices.
    '''

    if corr_func is 'pearson':
        corr_func = pearsonr
    elif corr_func is 'spearman':
        corr_func = spearmanr
    else:
        raise TypeError('corr_func must be: pearson or spearman.')

    # Convert distance matrixes to condensed 1D form.

    dist1_flat = squareform(dist1)
    dist2_flat = squareform(dist2)

    orig_cor = corr_func(dist1_flat, dist2_flat)[0]

    if nperm == 0:
        pval = np.NaN
        return orig_cor, pval

    # Set seed
    ra.seed(seed)

    perm_cors = np.empty((nperm, ))

    for p in range(nperm):
        dist1_perm = ra.permutation(dist1_flat)

        perm_cors[p] = corr_func(dist1_perm, dist2_flat)[0]

    if pval_type == 'two-tail':
        n_higher = (np.abs(perm_cors) >= np.abs(orig_cor)).sum()
    elif pval_type == 'greater':
        n_higher = (perm_cors >= orig_cor).sum()
    elif pval_type == 'less':
        n_higher = (perm_cors <= orig_cor).sum()
    else:
        raise TypeError('pval_type must be: two-tail, greater, or less.')

    pval = (n_higher + 1) / float(nperm + 1)

    return orig_cor, pval
