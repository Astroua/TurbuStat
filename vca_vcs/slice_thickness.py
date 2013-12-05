

import numpy as np

def change_slice_thickness(cube, slice_thickness=1.0):
    '''

    Degrades the velocity resolution of a data cube. This is to avoid
    shot noise by removing velocity fluctuations at small thicknesses.

    INPUTS
    ------

    cube - array
           3D data cube to degrade

    slice_thickness - float
                      Thicknesses of the new slices. Minimum is 1.0
                      Thickness must be integer multiple of the original cube size

    OUTPUT
    ------

    degraded_cube - array
                    Data cube degraded to new slice thickness

    '''

    from scipy.stats import nanmean

    assert isinstance(slice_thickness, float)
    if slice_thickness < 1.0:
        slice_thickness == 1.0
        print "Slice Thickness must be at least 1.0. Returning original cube."

    if slice_thickness==1.0:
        return cube

    if cube.shape[0]%slice_thickness != 0:
        raise TypeError("Slice thickness must be integer multiple of dimension size % s" % (cube.shape[0]))

    ## Want to average over velocity channels
    new_channel_indices = np.arange(0, cube.shape[0]/slice_thickness)
    degraded_cube = np.ones((cube.shape[0]/slice_thickness, cube.shape[1], cube.shape[2]))

    for channel in new_channel_indices:
        old_index = channel*slice_thickness
        degraded_cube[channel,:,:] = nanmean(cube[old_index:old_index+slice_thickness], axis=0)

    return degraded_cube