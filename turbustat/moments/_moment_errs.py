# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import astropy.units as u
from warnings import warn

from spectral_cube._moments import _moment_shp
from spectral_cube import SpectralCube
from spectral_cube.lower_dimensional_structures import Projection
from spectral_cube.cube_utils import iterator_strategy
from spectral_cube.wcs_utils import drop_axis

# from spectral_cube.np_compat import allbadtonan

'''
Functions for making moment error maps.

Borrows heavily from the functionality in _moments.py from spectral-cube.

Functions require, at minimum, a SpectralCube object and a scale value that
characterizes the noise.
'''

# convenience structures to keep track of the reversed index
# conventions between WCS and numpy
np2wcs = {2: 0, 1: 1, 0: 2}


def moment0_error(cube, scale, axis=0, how='auto'):
    '''
    Compute the zeroth moment error.

    Parameters
    ----------
    cube : SpectralCube
        Data cube.
    scale : SpectralCube or `~astropy.units.Quantity`
        The noise level in the data, either as a single value (with the same
        units as the cube) or a SpectralCube of noise values.
    axis : int
        Axis to compute moment over.
    how : {'auto', 'cube', 'slice'}, optional
        The computational method to use.

    Returns
    -------
    moment0 :  `~spectral_cube.lower_dimensional_structures.Projection`

    '''

    if how == "auto":
        how = iterator_strategy(cube, 0)

        if how == "ray":
            warn("Automatically selected 'ray' which isn't implemented. Using"
                 " 'slice' instead.")
            how = 'slice'

    if how == "cube":
        moment0_err = _cube0(cube, axis, scale)
    elif how == "slice":
        moment0_err = _slice0(cube, axis, scale)
    else:
        raise ValueError("how must be 'cube' or 'slice'.")

    # Multiply by spectral unit
    moment0_err *= cube.spectral_axis.unit

    meta = {'moment_order': 0,
            'moment_axis': axis,
            'moment_method': how}

    cube_meta = cube.meta.copy()
    meta.update(cube_meta)

    new_wcs = drop_axis(cube._wcs, np2wcs[axis])

    return Projection(moment0_err, copy=False, wcs=new_wcs, meta=meta,
                      header=cube._nowcs_header)


def moment1_error(cube, scale, axis=0, how='auto', moment0=None, moment1=None):
    '''
    Compute the first moment error.

    Parameters
    ----------
    cube : SpectralCube
        Data cube.
    scale : SpectralCube or `~astropy.units.Quantity`
        The noise level in the data, either as a single value (with the same
        units as the cube) or a SpectralCube of noise values.
    axis : int
        Axis to compute moment over.
    how : {'auto', 'cube', 'slice'}, optional
        The computational method to use.

    Returns
    -------
    moment1_err :  `~spectral_cube.lower_dimensional_structures.Projection`

    '''

    if how == "auto":
        how = iterator_strategy(cube, 0)

        if how == "ray":
            warn("Automatically selected 'ray' which isn't implemented. Using"
                 " 'slice' instead.")
            how = 'slice'

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how=how, axis=axis)
    if moment1 is None:
        moment1 = cube.moment1(how=how, axis=axis)

    # Remove velocity offset from centroid to match cube._pix_cen
    # Requires converting to a Quantity
    moment1 = u.Quantity(moment1)
    moment1 -= cube.spectral_axis[0]

    if how == "cube":
        moment1_err = _cube1(cube, axis, scale, moment0, moment1)
    elif how == "slice":
        moment1_err = _slice1(cube, axis, scale, moment0, moment1)
    else:
        raise ValueError("how must be 'cube' or 'slice'.")

    meta = {'moment_order': 1,
            'moment_axis': axis,
            'moment_method': how}

    cube_meta = cube.meta.copy()
    meta.update(cube_meta)

    new_wcs = drop_axis(cube._wcs, np2wcs[axis])

    return Projection(moment1_err, copy=False, wcs=new_wcs, meta=meta,
                      header=cube._nowcs_header)


def moment2_error(cube, scale, axis=0, how='auto', moment0=None, moment1=None,
                  moment2=None, moment1_err=None):
    '''
    Compute the second moment error.

    Parameters
    ----------
    cube : SpectralCube
        Data cube.
    scale : SpectralCube or `~astropy.units.Quantity`
        The noise level in the data, either as a single value (with the same
        units as the cube) or a SpectralCube of noise values.
    axis : int
        Axis to compute moment over.
    how : {'auto', 'cube', 'slice'}, optional
        The computational method to use.

    Returns
    -------
    moment2_err :  `~spectral_cube.lower_dimensional_structures.Projection`

    '''

    if how == "auto":
        how = iterator_strategy(cube, 0)

        if how == "ray":
            warn("Automatically selected 'ray' which isn't implemented. Using"
                 " 'slice' instead.")
            how = 'slice'

    # Compute moments if they aren't given.
    if moment0 is None:
        moment0 = cube.moment0(how='cube', axis=axis)
    if moment1 is None:
        moment1 = cube.moment1(how='cube', axis=axis)

    # Remove velocity offset to match cube._pix_cen
    # Requires converting to a Quantity
    moment1 = u.Quantity(moment1)
    moment1 -= cube.spectral_axis[0]

    if moment2 is None:
        moment2 = cube.moment2(how='cube', axis=axis)
    if moment1_err is None:
        moment1_err = _cube1(cube, axis, scale, moment0=moment0,
                             moment1=moment1)

    if how == "cube":
        moment2_err = _cube2(cube, axis, scale, moment0, moment1, moment2,
                             moment1_err)
    elif how == "slice":
        moment2_err = _slice2(cube, axis, scale, moment0, moment1, moment2,
                              moment1_err)
    else:
        raise ValueError("how must be 'cube' or 'slice'.")

    meta = {'moment_order': 2,
            'moment_axis': axis,
            'moment_method': how}

    cube_meta = cube.meta.copy()
    meta.update(cube_meta)

    new_wcs = drop_axis(cube._wcs, np2wcs[axis])

    return Projection(moment2_err, copy=False, wcs=new_wcs, meta=meta,
                      header=cube._nowcs_header)


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


def _slice1(cube, axis, scale, moment0, moment1):
    """
    1st moment along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float or SpectralCube
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

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    spec_unit = cube.spectral_axis.unit
    axis_sum = u.Quantity(moment0 /
                          (cube._pix_size_slice(axis) * spec_unit))

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp) * spec_unit ** 2

    view = [slice(None)] * 3
    pix_cen = u.Quantity(cube._pix_cen()[axis] * spec_unit)

    # term2 does not depend on the plane.
    term2 = moment1 / axis_sum

    for i in range(cube.shape[axis]):
        view[axis] = i

        term1 = pix_cen[view] / axis_sum

        if _scale_cube:
            noise_plane = \
                np.nan_to_num(scale.filled_data[view])
        else:
            noise_plane = scale

        result += np.power((term1 - term2) * noise_plane, 2)

    return np.sqrt(result)


def _slice2(cube, axis, scale, moment0, moment1, moment2,
            moment1_err):
    """
    2nd moment error along an axis, calculated slicewise

    Parameters
    ----------
    cube : SpectralCube
    axis : int
    scale : float or SpectralCube
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

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    spec_unit = cube.spectral_axis.unit
    axis_sum = u.Quantity(moment0 /
                          (cube._pix_size_slice(axis) * spec_unit))

    shp = _moment_shp(cube, axis)
    term1 = np.zeros(shp) * spec_unit ** 4
    term2 = np.zeros(shp) * spec_unit * cube.unit

    view = [slice(None)] * 3
    pix_cen = cube._pix_cen()[axis] * spec_unit

    # term12 does not depend on plane.
    term12 = moment2 / axis_sum

    for i in range(cube.shape[axis]):
        view[axis] = i
        plane = np.nan_to_num(cube.filled_data[view])

        term11 = np.power((pix_cen[view] - moment1), 2) / axis_sum

        if _scale_cube:
            noise_plane = \
                np.nan_to_num(scale.filled_data[view])
        else:
            noise_plane = scale

        term1 += np.power((term11 - term12) * noise_plane, 2)

        term2 += np.nan_to_num(plane) * (pix_cen[view] - moment1)

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


def _cube1(cube, axis, scale, moment0, moment1):
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

    # Divide moment0 by the pixel size in the given axis so it represents the
    # sum.
    spec_unit = cube.spectral_axis.unit
    axis_sum = u.Quantity(moment0 /
                          (cube._pix_size_slice(axis) * spec_unit))

    shp = _moment_shp(cube, axis)
    result = np.zeros(shp) * spec_unit

    pix_cen = cube._pix_cen()[axis] * spec_unit

    term1 = pix_cen / axis_sum
    term2 = moment1 / axis_sum

    result = np.sqrt(np.sum(np.power((term1 - term2) *
                                     np.nan_to_num(noise_plane), 2),
                            axis=axis))

    good_pix = np.isfinite(moment0) + np.isfinite(moment1)

    result[~good_pix] = np.NaN

    return result


def _cube2(cube, axis, scale, moment0, moment1, moment2,
           moment1_err):
    '''
    '''

    spec_unit = cube.spectral_axis.unit
    pix_cen = cube._pix_cen()[axis] * spec_unit

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
    axis_sum = u.Quantity(moment0 /
                          (cube._pix_size_slice(axis) * spec_unit))

    plane = np.nan_to_num(cube.filled_data[:])

    term11 = np.power((pix_cen - moment1), 2) / axis_sum
    term12 = moment2 / axis_sum

    term1 = np.sum(np.power((term11 - term12) * noise_plane, 2), axis=axis)

    term21 = u.Quantity(np.nan_to_num(plane) * (pix_cen - moment1))

    term2 = \
        4 * np.power((moment1_err * np.sum(term21, axis=axis)) / axis_sum, 2)

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

    if moment2 is None:
        moment2 = cube.moment2(how=how, axis=0)

    mom2_err = moment2_error(cube, scale, axis=0, how=how,
                             moment0=moment0,
                             moment1=moment1,
                             moment2=moment2,
                             moment1_err=moment1_err)

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
