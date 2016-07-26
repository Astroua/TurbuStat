
import numpy as np
import astropy.wcs as wcs


def hellinger(data1, data2, bin_width=1.0):
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
    distance = (bin_width / np.sqrt(2)) * \
        np.sqrt(np.nansum((np.sqrt(data1) -
                           np.sqrt(data2)) ** 2.))
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


def common_histogram_bins(dataset1, dataset2, nbins=None, logscale=False,
                          return_centered=False):
    '''
    Returns bins appropriate for both given datasets. If nbins is not
    specified, the number is set by the square root of the average
    number of elements in the datasets. This assumes that there are many
    (~>100) elements in each dataset.

    Parameters
    ----------
    dataset1 : 1D numpy.ndarray
        Dataset to use in finding matching set of bins.
    dataset2 : 1D numpy.ndarray
        Same as above.
    nbins : int, optional
        Specify the number of bins to use.
    logscale : bool, optional
        Return logarithmically spaced bins.
    return_centered : bool, optional
        Return the centers of the bins along the the usual edge output.
    '''

    if dataset1.ndim > 1 or dataset2.ndim > 1:
        raise ValueError("dataset1 and dataset2 should be 1D arrays.")

    global_min = min(np.nanmin(dataset1), np.nanmin(dataset2))
    global_max = max(np.nanmax(dataset1), np.nanmax(dataset2))

    if nbins is None:
        avg_num = np.sqrt((dataset1.size + dataset2.size)/2.)
        nbins = np.floor(avg_num).astype(int)

    if logscale:
        bins = np.logspace(np.log10(global_min),
                           np.log10(global_max), num=nbins)
    else:
        bins = np.linspace(global_min, global_max, num=nbins)

    if return_centered:
        center_bins = (bins[:-1] + bins[1:]) / 2
        return bins, center_bins

    return bins


def common_scale(wcs1, wcs2, tol=1e-5):
    '''
    Return the factor to make the pixel scales in the WCS objects the same.

    Assumes pixels a near to being square and distortions between the grids
    are minimal. If they are distorted an error is raised.

    For laziness, the celestial scales should be the same (so pixels are
    squares). Otherwise this approach of finding a common scale will not work
    and a reprojection would be a better approach before running any
    comparisons.

    Parameters
    ----------
    wcs1 : astropy.wcs.WCS
        WCS Object to match to.
    wcs2 : astropy.wcs.WCS
        WCS Object.

    Returns
    -------
    scale : float
        Factor between the pixel scales.
    '''

    if wcs.utils.is_proj_plane_distorted(wcs1):
        raise wcs.WcsError("First WCS object is distorted.")

    if wcs.utils.is_proj_plane_distorted(wcs2):
        raise wcs.WcsError("Second WCS object is distorted.")

    scales1 = np.abs(wcs.utils.proj_plane_pixel_scales(wcs1.celestial))
    scales2 = np.abs(wcs.utils.proj_plane_pixel_scales(wcs2.celestial))

    # Forcing near square pixels
    if scales1[0] - scales1[1] > tol:
        raise ValueError("Pixels in first WCS are not square. Recommend "
                         "reprojecting to the same grid.")

    if scales2[0] - scales2[1] > tol:
        raise ValueError("Pixels in second WCS are not square. Recommend "
                         "reprojecting to the same grid.")

    scale = scales2[0] / scales1[0]

    return scale
