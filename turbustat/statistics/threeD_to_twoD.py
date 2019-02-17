# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

'''
Routines for transforming data cubes to 2D representations
'''

import numpy as np
from astropy.utils.console import ProgressBar


def intensity_data(cube, p=0.2, noise_lim=-np.inf, norm=True):
    '''
    Clips off channels below the given noise limit and keep the
    upper percentile specified.

    Parameters
    ----------
    cube : numpy.ndarray
        Data cube.
    p : float, optional
        Sets the fraction of data to keep in each channel.
    noise_lim : float, optional
        The noise limit used to reject channels in the cube.

    Returns
    -------

    intensity_vecs : numpy.ndarray
        2D dataset of size (# channels, p * cube.shape[1] * cube.shape[2]).
    '''
    vec_length = int(round(p * cube.shape[1] * cube.shape[2]))
    intensity_vecs = np.empty((cube.shape[0], vec_length))

    delete_channels = []

    if norm:
        maxval = np.nanmax(cube)
    else:
        maxval = 1.0

    for dv in range(cube.shape[0]):
        vec_vec = cube[dv, :, :]
        # Remove nans from the slice
        vel_vec = vec_vec[np.isfinite(vec_vec)]
        # Apply noise limit
        vel_vec = vel_vec[vel_vec > noise_lim]
        vel_vec = np.sort(vel_vec)[::-1]
        if len(vel_vec) < vec_length:
            diff = vec_length - len(vel_vec)
            vel_vec = np.append(vel_vec, [0.0] * diff)
        else:
            vel_vec = vel_vec[:vec_length]

        # Return the normalized, shortened vector
        if maxval != 0.0:
            intensity_vecs[dv, :] = vel_vec / maxval
        else:
            delete_channels.append(dv)
    # Remove channels
    intensity_vecs = np.delete(intensity_vecs, delete_channels, axis=0)

    return intensity_vecs


def _format_data(cube, data_format='intensity', num_spec=1000,
                 noise_lim=-np.inf, p=0.2, normalize=True):
    '''
    Rearrange data into a 2D object using the given format.
    '''

    if data_format is "spectra":
        if num_spec is None:
            raise ValueError('Must specify num_spec for data format',
                             'spectra.')

        # Find the brightest spectra in the cube
        mom0 = np.nansum(cube, axis=0)

        bright_spectra = \
            np.argpartition(mom0.ravel(), -num_spec)[-num_spec:]

        x = np.empty((num_spec,), dtype=int)
        y = np.empty((num_spec,), dtype=int)

        for i in range(num_spec):
            x[i] = int(bright_spectra[i] / cube.shape[1])
            y[i] = int(bright_spectra[i] % cube.shape[2])

        data_matrix = cube[:, x, y]

    elif data_format is "intensity":
        data_matrix = intensity_data(cube, noise_lim=noise_lim,
                                     p=p)

    else:
        raise NameError(
            "data_format must be either 'spectra' or 'intensity'.")

    # Normalize by rescaling the data to an interval between 0 and 1
    # Ignore all values of 0, since they're just filled in.
    if normalize:
        data_matrix /= np.linalg.norm(data_matrix, ord=2)

    return data_matrix


def var_cov_cube(cube, mean_sub=False, progress_bar=True):
    '''
    Compute the variance-covariance matrix of a data cube, with proper
    handling of NaNs.

    Parameters
    ----------
    cube : numpy.ndarray
        PPV cube. Spectral dimension assumed to be 0th axis.
    mean_sub : bool, optional
        Subtract column means.
    progress_bar : bool, optional
        Show a progress bar, since this operation could be slow for large
        cubes.

    Returns
    -------
    cov_matrix : numpy.ndarray
        Computed covariance matrix.
    '''

    n_velchan = cube.shape[0]

    cov_matrix = np.zeros((n_velchan, n_velchan))

    if progress_bar:
        bar = ProgressBar(n_velchan)

    for i, chan in enumerate(_iter_2D(cube)):
        # Set the nans to tiny values
        chan[np.isnan(chan)] = np.finfo(chan.dtype).eps

        norm_chan = chan
        if mean_sub:
            norm_chan -= np.nanmean(chan)
        for j, chan2 in enumerate(_iter_2D(cube[:i + 1, :, :])):
            norm_chan2 = chan2
            if mean_sub:
                norm_chan2 -= np.nanmean(chan2)

            divisor = np.sum(np.isfinite(norm_chan * norm_chan2))

            # Apply Bessel's correction when mean subtracting
            if mean_sub:
                divisor -= 1.0

            cov_matrix[i, j] = \
                np.nansum(norm_chan * norm_chan2) / divisor

        # Variances
        # Divided in half to account for doubling in line below
        var_divis = np.sum(np.isfinite(norm_chan))
        if mean_sub:
            var_divis -= 1.0

        cov_matrix[i, i] = 0.5 * \
            np.nansum(norm_chan * norm_chan) / var_divis

        if progress_bar:
            bar.update(i + 1)

    cov_matrix = cov_matrix + cov_matrix.T

    return np.nan_to_num(cov_matrix)


def _iter_2D(arr):
    '''
    Flatten a 3D cube into 2D by its channels.
    '''

    for chan in arr.reshape((arr.shape[0], -1)):
        yield chan.copy()
