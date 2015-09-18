
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


def kl_divergence(P, Q):
    '''
    Kullback Leidler Divergence

    Parameters
    ----------

    P,Q : numpy.ndarray
        Two Discrete Probability distributions

    Returns
    -------

    kl_divergence : float
    '''
    P = P[~np.isnan(P)]
    Q = Q[~np.isnan(Q)]
    P = P[np.isfinite(P)]
    Q = Q[np.isfinite(Q)]
    return np.nansum(np.where(Q != 0, P * np.log(P / Q), 0))


def common_histogram_bins(dataset1, dataset2, nbins=None, logscale=False):
    '''
    Returns bins appropriate for both given datasets. If nbins is not
    specified, the number is set by the square root of the average
    number of elements in the datasets. This assumes that there are many
    (~>100) elements in each dataset.
    '''

    if dataset1.ndim > 1 or dataset2.ndim > 1:
        raise ValueError("dataset1 and dataset2 should be 1D arrays.")

    global_min = min(np.nanmin(dataset1), np.nanmin(dataset2))
    global_max = max(np.nanmax(dataset1), np.nanmax(dataset2))

    if nbins is None:
        nbins = int(np.sqrt((dataset1.size + dataset2.size)/2.))

    if logscale:
        return np.logspace(np.log10(global_min),
                           np.log10(global_max), num=nbins)

    return np.linspace(global_min, global_max, num=nbins)
