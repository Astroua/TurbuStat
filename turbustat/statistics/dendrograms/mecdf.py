# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division


'''

Implementation of a Multivariable ECDF

Limited to step function for now.

'''

import numpy as np


def mecdf(arr):
    '''

    Parameters
    ----------

    arr : np.ndarray
          Array containing where each row is a PDF.

    Returns
    -------

    ecdf - np.ndarray
           Array containing the ECDF of each row.

    '''
    assert isinstance(arr, np.ndarray)

    nrows = arr.shape[0]
    ecdf = np.empty(arr.shape)

    for n in range(nrows):
        ecdf[n, :] = np.cumsum(arr[n, :]/np.sum(arr[n, :].astype(float)))

    return ecdf
