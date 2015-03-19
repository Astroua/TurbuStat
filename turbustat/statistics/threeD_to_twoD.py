
'''
Routines for transforming data cubes to 2D representations
'''

import numpy as np


def intensity_data(cube, p=0.1, noise_lim=0.1):
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

    for dv in range(cube.shape[0]):
        vec_vec = cube[dv, :, :]
        # Remove nans from the slice
        vel_vec = vec_vec[np.isfinite(vec_vec)]
        # Apply noise limit
        vel_vec = vel_vec[vel_vec > noise_lim]
        vel_vec.sort()
        if len(vel_vec) < vec_length:
            diff = vec_length - len(vel_vec)
            vel_vec = np.append(vel_vec, [0.0] * diff)
        else:
            vel_vec = vel_vec[:vec_length]

        # Return the normalized, shortened vector
        maxval = np.max(vel_vec)
        if maxval != 0.0:
            intensity_vecs[dv, :] = vel_vec / maxval
        else:
            delete_channels.append(dv)
    # Remove channels
    intensity_vecs = np.delete(intensity_vecs, delete_channels, axis=0)

    return intensity_vecs