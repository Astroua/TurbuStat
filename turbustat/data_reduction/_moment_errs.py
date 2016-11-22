
import numpy as np
import astropy.units as u

from spectral_cube._moments import _moment_shp
from spectral_cube import SpectralCube
from spectral_cube.cube_utils import iterator_strategy

# from spectral_cube.np_compat import allbadtonan

'''
Functions for making moment error maps.

Borrows heavily from the functionality in _moments.py from spectral-cube.

Functions require, at minimum, a SpectralCube object and a scale value that
characterizes the noise.
'''


def _slice0(cube, axis, scale):
    """
    0th moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float

    Returns
    -------
    moment0_error : array
    """
    if isinstance(scale, SpectralCube):
        # scale should then have the same shape as the cube.
        if cube.shape != scale.shape:
            raise IndexError("When scale is a SpectralCube, it must have the"
                             " same shape as the cube.")
        _scale_cube = True
    else:
        _scale_cube = False

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp) * cube.unit ** 2

    view = [slice(None)] * 3
    valid = np.zeros(shp, dtype=np.bool)

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = cube._mask.include(data=cube._data, wcs=cube._wcs, view=view)
        valid |= plane

        if _scale_cube:
            noise_plane = np.nan_to_num(scale.filled_data[view])
        else:
            noise_plane = scale

        result += plane * np.power(noise_plane, 2)

    out_result = np.sqrt(result) * cube._pix_size_slice(axis)
    out_result[~valid] = np.nan

    return out_result


def _slice1(cube, axis, scale, moment0=None, moment1=None):
    """
    1st moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float
    moment0 : 0th moment
    moment1 : 1st moment

    Returns
    -------
    moment1_error : array
    """

    if isinstance(scale, SpectralCube):
        # scale should then have the same shape as the cube.
        if cube.shape != scale.shape:
            raise IndexError("When scale is a SpectralCube, it must have the"
                             " same shape as the cube.")
        _scale_cube = True
    else:
        _scale_cube = False

    # I don't think there is a way to do this with one pass.
    # The first 2 moments always have to be pre-computed.

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how='slice', axis=axis)
    if moment1 is None:
        moment1 = cube.moment1(how='slice', axis=axis)

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    spec_unit = cube.spectral_axis.unit
    axis_sum = u.Quantity(moment0.copy() /
                          (cube._pix_size_slice(axis) * spec_unit))

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp) * spec_unit ** 2

    view = [slice(None)] * 3
    pix_cen = u.Quantity(cube._pix_cen()[axis] * spec_unit)
    pix_size = cube._pix_size_slice(axis) * spec_unit

    for i in range(cube.shape[axis]):
        view[axis] = i

        term1 = (pix_cen[view] + pix_size) / axis_sum
        term2 = moment1 / axis_sum

        if _scale_cube:
            noise_plane = \
                np.nan_to_num(scale.filled_data[view])
        else:
            noise_plane = scale

        result += np.power((term1 - term2) * noise_plane, 2)

    return np.sqrt(result)


def _slice2(cube, axis, scale, moment0=None, moment1=None, moment2=None,
            moment1_err=None):
    """
    2nd moment error along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float
    moment0 : 0th moment
    moment1 : 1st moment
    moment2 : 2nd moment
    moment1_err : 1st moment error

    Returns
    -------
    moment1_error : array
    """

    if isinstance(scale, SpectralCube):
        # scale should then have the same shape as the cube.
        if cube.shape != scale.shape:
            raise IndexError("When scale is a SpectralCube, it must have the"
                             " same shape as the cube.")
        _scale_cube = True
    else:
        _scale_cube = False

    # I don't think there is a way to do this with one pass.
    # The first 2 moments always have to be pre-computed.

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how='slice', axis=axis)
    if moment1 is None:
        moment1 = cube.moment1(how='slice', axis=axis)
    if moment2 is None:
        moment2 = cube.moment2(how='slice', axis=axis)
    if moment1_err is None:
        moment1_err = _slice1(cube, axis, scale, moment0=moment0,
                              moment1=moment1)

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    spec_unit = cube.spectral_axis.unit
    axis_sum = u.Quantity(moment0.copy() /
                          (cube._pix_size_slice(axis) * spec_unit))

    shp = _moment_shp(cube, axis)
    term1 = np.zeros(shp) * spec_unit ** 4
    term2 = np.zeros(shp) * spec_unit * cube.unit

    view = [slice(None)] * 3
    pix_cen = cube._pix_cen()[axis] * spec_unit
    pix_size = cube._pix_size_slice(axis) * spec_unit

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = np.nan_to_num(cube.filled_data[view])

        term11 = np.power((pix_cen[view] + pix_size - moment1), 2) / axis_sum
        term12 = moment2 / axis_sum

        if _scale_cube:
            noise_plane = \
                np.nan_to_num(scale.filled_data[view])
        else:
            noise_plane = scale

        term1 += np.power((term11 - term12) * noise_plane, 2)

        term2 += np.nan_to_num(plane) * (pix_cen[view] + pix_size - moment1)

    term2 = 4 * np.power((moment1_err * term2) / (axis_sum), 2)

    return np.sqrt(term1 + term2)


def _cube0(cube, axis, scale):
    '''
    Moment 0 error computed cube-wise.
    '''

    if isinstance(scale, SpectralCube):
        # scale should then have the same shape as the cube.
        if cube.shape != scale.shape:
            raise IndexError("When scale is a SpectralCube, it must have the"
                             " same shape as the cube.")
        noise_plane = np.nan_to_num(scale.filled_data[:])
    else:
        noise_plane = scale

    error_arr = \
        np.sqrt(
            np.sum(cube._mask.include(data=cube._data, wcs=cube._wcs) *
                   noise_plane**2, axis=axis)) * cube._pix_size_slice(axis)

    return error_arr


def _cube1(cube, axis, scale, moment0=None, moment1=None):
    '''
    Moment 1 error computed cube-wise.
    '''

    if isinstance(scale, SpectralCube):
        # scale should then have the same shape as the cube.
        if cube.shape != scale.shape:
            raise IndexError("When scale is a SpectralCube, it must have the"
                             " same shape as the cube.")
        noise_plane = scale.filled_data[:]
    else:
        noise_plane = scale

    # I don't think there is a way to do this with one pass.
    # The first 2 moments always have to be pre-computed.

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how='cube', axis=axis)
    if moment1 is None:
        moment1 = cube.moment1(how='cube', axis=axis)

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    spec_unit = cube.spectral_axis.unit
    axis_sum = u.Quantity(moment0.copy() /
                          (cube._pix_size_slice(axis) * spec_unit))

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp) * spec_unit

    view = [slice(None)] * 3
    pix_cen = cube._pix_cen()[axis] * spec_unit
    pix_size = cube._pix_size_slice(axis) * spec_unit

    term1 = (pix_cen[view] + pix_size) / axis_sum
    term2 = moment1 / axis_sum

    result = np.sqrt(np.sum(np.power((term1 - term2) *
                                     np.nan_to_num(noise_plane), 2),
                            axis=axis))

    good_pix = np.isfinite(moment0) + np.isfinite(moment1)

    result[~good_pix] = np.NaN

    return result


def _cube2(cube, axis, scale, moment0=None, moment1=None, moment2=None,
           moment1_err=None):
    '''
    '''

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how='cube', axis=axis)
    if moment1 is None:
        moment1 = cube.moment1(how='cube', axis=axis)
    if moment2 is None:
        moment2 = cube.moment2(how='cube', axis=axis)
    if moment1_err is None:
        moment1_err = _cube1(cube, axis, scale, moment0=moment0,
                             moment1=moment1)

    spec_unit = cube.spectral_axis.unit
    pix_cen = cube._pix_cen()[axis] * spec_unit
    pix_size = cube._pix_size_slice(axis) * spec_unit

    if isinstance(scale, SpectralCube):
        # scale should then have the same shape as the cube.
        if cube.shape != scale.shape:
            raise IndexError("When scale is a SpectralCube, it must have the"
                             " same shape as the cube.")
        noise_plane = np.nan_to_num(scale.filled_data[:])
    else:
        noise_plane = scale

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    axis_sum = u.Quantity(moment0.copy() /
                          (cube._pix_size_slice(axis) * spec_unit))

    plane = np.nan_to_num(cube.filled_data[:])

    term11 = np.power((pix_cen + pix_size - moment1), 2) / axis_sum
    term12 = moment2 / axis_sum

    term1 = np.sum(np.power((term11 - term12) * noise_plane, 2), axis=axis)

    term21 = u.Quantity(np.nan_to_num(plane) * (pix_cen + pix_size - moment1))

    term2 = 4 * np.power((moment1_err * np.sum(term21, axis=axis)) / axis_sum, 2)

    result = np.sqrt(term1 + term2)

    good_pix = np.isfinite(moment0) + np.isfinite(moment1) + \
        np.isfinite(moment2)

    result[~good_pix] = np.NaN

    # result[result == 0] = np.NaN

    return result


def linewidth_sigma_err(cube, scale, how='auto', moment0=None, moment1=None,
                        moment2=None, moment1_err=None):
    '''
    Error on the line width.
    '''

    if how == "auto":
        how = iterator_strategy(cube, 0)

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how=how, axis=0)
    if moment1 is None:
        moment1 = cube.moment1(how=how, axis=0)
    if moment2 is None:
        moment2 = cube.moment2(how=how, axis=0)
    if moment1_err is None:
        moment1_err = _cube1(cube, 0, scale, moment0=moment0,
                             moment1=moment1)

    if how == "cube":
        mom2_err = _cube2(cube, 0, scale, moment0, moment1, moment2,
                          moment1_err)
    elif how == "slice":
        mom2_err = _slice2(cube, 0, scale, moment0, moment1, moment2,
                           moment1_err)
    elif how == "ray":
        raise NotImplementedError("")
    else:
        raise ValueError("how must be cube, slice, or ray.")

    return mom2_err / (2 * np.sqrt(moment2))


def linewidth_fwhm_err(cube, scale, how='auto', moment0=None, moment1=None,
                       moment2=None, moment1_err=None):
    '''
    Error on the FWHM line width.
    '''

    SIGMA2FWHM = 2. * np.sqrt(2. * np.log(2.))

    del_lwidth_sig = linewidth_sigma_err(cube, scale, how, moment0, moment1,
                                         moment2, moment1_err)

    return SIGMA2FWHM * del_lwidth_sig


def moment_raywise(cube, order, axis):
    """
    Compute moments by accumulating the answer one ray at a time
    """
    shp = _moment_shp(cube, axis)
    out = np.zeros(shp) * np.nan

    pix_cen = cube._pix_cen()[axis]
    pix_size = cube._pix_size()[axis]

    for x, y, slc in cube._iter_rays(axis):
        # the intensity, i.e. the weights
        include = cube._mask.include(data=cube._data, wcs=cube._wcs,
                                     view=slc)
        if not include.any():
            continue

        data = cube.flattened(slc).value * pix_size[slc][include]

        if order == 0:
            out[x, y] = data.sum()
            continue

        order1 = (data * pix_cen[slc][include]).sum() / data.sum()
        if order == 1:
            out[x, y] = order1
            continue

        ordern = (data * (pix_cen[slc][include] - order1) ** order).sum()
        ordern /= data.sum()

        out[x, y] = ordern
    return out