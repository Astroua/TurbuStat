
import numpy as np


def hellinger(data1, data2):
    '''
    Calculate the Hellinger Distance between two datasets.

    Parameters
    ----------
    data1 : numpy.ndarray
        1D array.
    data2 : numpy.ndarray
        1D array.

    Returns
    -------
    distance : float
        Distance value.
    '''
    distance = (1 / np.sqrt(2)) * \
        np.sqrt(np.nansum((np.sqrt(data1) - np.sqrt(data2)) ** 2.))
    return distance


def standardize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)
