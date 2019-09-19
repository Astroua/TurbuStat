# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from scipy.stats import binned_statistic
import astropy.units as u
from astropy.coordinates import Angle
from scipy.stats import t as t_dist


def pspec(psd2, nbins=None, return_stddev=False, binsize=1.0,
          logspacing=True, max_bin=None, min_bin=None, return_freqs=True,
          theta_0=None, delta_theta=None, boot_iter=None,
          mean_func=np.nanmean):
    '''
    Calculate the radial profile using scipy.stats.binned_statistic.

    Parameters
    ----------
    psd2 : np.ndarray
        2D Spectral power density.
    nbins : int, optional
        Number of bins to use. If None, it is calculated based on the size
        of the given arrays.
    return_stddev : bool, optional
        Return the standard deviations in each bin.
    binsize : float, optional
        Size of bins to be used. If logspacing is enabled, this will increase
        the number of bins used by the inverse of the given binsize.
    logspacing : bool, optional
        Use logarithmically spaces bins.
    max_bin : float, optional
        Give the maximum value to bin to.
    min_bin : float, optional
        Give the minimum value to bin to.
    return_freqs : bool, optional
        Return spatial frequencies.
    theta_0 : `~astropy.units.Quantity`, optional
        The center angle of the azimuthal mask. Must have angular units.
    delta_theta : `~astropy.units.Quantity`, optional
        The width of the azimuthal mask. This must be given when
        a `theta_0` is given. Must have angular units.
    boot_iter : int, optional
        Number of bootstrap iterations for estimating the standard deviation
        in each bin. Require `return_stddev=True`.
    mean_func : function, optional
        Define the function used to create the 1D power spectrum. The default
        is `np.nanmean`.

    Returns
    -------
    bins_cents : np.ndarray
        Centre of the bins.
    ps1D : np.ndarray
        1D binned power spectrum.
    ps1D_stddev : np.ndarray
        Returned when return_stddev is enabled. Standard deviations
        within each of the bins.
    '''

    yy, xx = make_radial_arrays(psd2.shape)

    dists = np.sqrt(yy**2 + xx**2)
    if theta_0 is not None:

        if delta_theta is None:
            raise ValueError("Must give delta_theta.")

        theta_0 = theta_0.to(u.rad)
        delta_theta = delta_theta.to(u.rad)

        theta_limits = Angle([theta_0 - 0.5 * delta_theta,
                              theta_0 + 0.5 * delta_theta])

        # Define theta array
        thetas = Angle(np.arctan2(yy, xx) * u.rad)

        # Wrap around pi
        theta_limits = theta_limits.wrap_at(np.pi * u.rad)

    if nbins is None:
        nbins = int(np.round(dists.max() / binsize) + 1)

    if return_freqs:
        yy_freq, xx_freq = make_radial_freq_arrays(psd2.shape)

        freqs_dist = np.sqrt(yy_freq**2 + xx_freq**2)

        zero_freq_val = freqs_dist[np.nonzero(freqs_dist)].min() / 2.
        freqs_dist[freqs_dist == 0] = zero_freq_val

    if max_bin is None:
        if return_freqs:
            max_bin = 0.5
        else:
            max_bin = dists.max()

    if min_bin is None:
        if return_freqs:
            min_bin = 1.0 / min(psd2.shape)
        else:
            min_bin = 0.5

    if logspacing:
        bins = np.logspace(np.log10(min_bin), np.log10(max_bin), nbins + 1)
    else:
        bins = np.linspace(min_bin, max_bin, nbins + 1)

    if return_freqs:
        dist_arr = freqs_dist
    else:
        dist_arr = dists

    if theta_0 is not None:
        if theta_limits[0] < theta_limits[1]:
            azim_mask = np.logical_and(thetas >= theta_limits[0],
                                       thetas <= theta_limits[1])
        else:
            azim_mask = np.logical_or(thetas >= theta_limits[0],
                                      thetas <= theta_limits[1])

        azim_mask = np.logical_or(azim_mask, azim_mask[::-1, ::-1])

        # Fill in the middle angles
        ny = np.floor(psd2.shape[0] / 2.).astype(int)
        nx = np.floor(psd2.shape[1] / 2.).astype(int)

        azim_mask[ny - 1:ny + 1, nx - 1:nx + 1] = True
    else:
        azim_mask = None

    ps1D, bin_edge, cts = binned_statistic(dist_arr[azim_mask].ravel(),
                                           psd2[azim_mask].ravel(),
                                           bins=bins,
                                           statistic=mean_func)

    bin_cents = (bin_edge[1:] + bin_edge[:-1]) / 2.

    if not return_stddev:
        if theta_0 is not None:
            return bin_cents, ps1D, azim_mask
        else:
            return bin_cents, ps1D
    else:

        if boot_iter is None:

            stat_func = lambda x: np.nanstd(x, ddof=1)

        else:
            from astropy.stats import bootstrap

            stat_func = lambda data: np.mean(bootstrap(data, boot_iter,
                                                       bootfunc=np.std))

        ps1D_stddev = binned_statistic(dist_arr[azim_mask].ravel(),
                                       psd2[azim_mask].ravel(),
                                       bins=bins,
                                       statistic=stat_func)[0]

        # We're dealing with variations in the number of samples for each bin.
        # Add a correction based on the t distribution
        bin_cts = binned_statistic(dist_arr[azim_mask].ravel(),
                                   psd2[azim_mask].ravel(),
                                   bins=bins,
                                   statistic='count')[0]

        # Two-tail CI for 85% (~1 sigma)
        alpha = 1 - (0.15 / 2.)

        # Correction factor to convert to the standard error
        A = t_dist.ppf(alpha, bin_cts - 1) / np.sqrt(bin_cts)

        # If the standard error is larger than the standard deviation,
        # use it instead
        ps1D_stddev[A > 1] *= A[A > 1]

        # Mask out bins that have 1 or fewer points
        mask = bin_cts <= 1

        ps1D_stddev[mask] = np.NaN
        ps1D[mask] = np.NaN

        # ps1D_stddev[ps1D_stddev == 0.] = np.NaN

        if theta_0 is not None:
            return bin_cents, ps1D, ps1D_stddev, azim_mask
        else:
            return bin_cents, ps1D, ps1D_stddev


def make_radial_arrays(shape, y_center=None, x_center=None):

    if y_center is None:
        y_center = np.floor(shape[0] / 2.).astype(int)
    else:
        y_center = int(y_center)

    if x_center is None:
        x_center = np.floor(shape[1] / 2.).astype(int)
    else:
        x_center = int(x_center)

    y = np.arange(-y_center, shape[0] - y_center)
    x = np.arange(-x_center, shape[1] - x_center)

    yy, xx = np.meshgrid(y, x, indexing='ij')

    return yy, xx


def make_radial_freq_arrays(shape):

    yfreqs = np.fft.fftshift(np.fft.fftfreq(shape[0]))
    xfreqs = np.fft.fftshift(np.fft.fftfreq(shape[1]))

    yy_freq, xx_freq = np.meshgrid(yfreqs, xfreqs, indexing='ij')

    return yy_freq[::-1], xx_freq[::-1]
