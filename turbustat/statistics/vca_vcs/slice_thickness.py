# Licensed under an MIT open source license - see LICENSE


import numpy as np


def change_slice_thickness(cube, slice_thickness=1.0):
    '''

    Degrades the velocity resolution of a data cube. This is to avoid
    shot noise by removing velocity fluctuations at small thicknesses.

    Parameters
    ----------
    cube : numpy.ndarray
           3D data cube to degrade
    slice_thickness : float, optional
        Thicknesses of the new slices. Minimum is 1.0
        Thickness must be integer multiple of the original cube size

    Returns
    -------
    degraded_cube : numpy.ndarray
        Data cube degraded to new slice thickness
    '''

    assert isinstance(slice_thickness, float)
    if slice_thickness < 1:
        slice_thickness == 1
        print "Slice Thickness must be at least 1.0. Returning original cube."

    if slice_thickness == 1:
        return cube

    if cube.shape[0] % slice_thickness != 0:
        raise TypeError("Slice thickness must be integer multiple of dimension"
                        " size % s" % (cube.shape[0]))

    slice_thickness = int(slice_thickness)

    # Want to average over velocity channels
    new_channel_indices = np.arange(0, cube.shape[0] / slice_thickness)
    degraded_cube = np.ones(
        (cube.shape[0] / slice_thickness, cube.shape[1], cube.shape[2]))

    for channel in new_channel_indices:
        old_index = int(channel * slice_thickness)
        channel = int(channel)
        degraded_cube[channel, :, :] = \
            np.nanmean(cube[old_index:old_index + slice_thickness], axis=0)

    return degraded_cube
