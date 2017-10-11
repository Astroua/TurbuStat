# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u


from ..io.sim_tools import create_cube_header, create_image_header, create_fits_hdu


def test_create_cube_header():

    pixel_scale = 0.001 * u.deg
    spec_pixel_scale = 1000. * u.m / u.s
    beamfwhm = 0.003 * u.deg
    imshape = (2, 3, 4)
    restfreq = 1.4 * u.GHz
    bunit = u.K

    hdr = create_cube_header(pixel_scale, spec_pixel_scale, beamfwhm, imshape,
                             restfreq, bunit)

    assert hdr['CDELT1'] == -pixel_scale.value
    assert hdr['CDELT2'] == pixel_scale.value
    assert hdr['CDELT3'] == spec_pixel_scale.value

    assert hdr['CUNIT1'] == 'deg'
    assert hdr['CUNIT2'] == 'deg'
    assert u.Unit(hdr['CUNIT3']).is_equivalent(spec_pixel_scale.unit)

    assert hdr["RESTFRQ"] == restfreq.to(u.Hz).value


def test_create_image_header():

    pixel_scale = 0.001 * u.deg
    beamfwhm = 0.003 * u.deg
    imshape = (3, 4)
    restfreq = 1.4 * u.GHz
    bunit = u.K

    hdr = create_image_header(pixel_scale, beamfwhm, imshape,
                              restfreq, bunit)

    assert hdr['CDELT1'] == -pixel_scale.value
    assert hdr['CDELT2'] == pixel_scale.value

    assert hdr['CUNIT1'] == 'deg'
    assert hdr['CUNIT2'] == 'deg'

    assert hdr["RESTFRQ"] == restfreq.to(u.Hz).value


def test_create_cube_hdu():

    cube = np.zeros((2, 3, 4))

    pixel_scale = 0.001 * u.deg
    spec_pixel_scale = 1000. * u.m / u.s
    beamfwhm = 0.003 * u.deg
    imshape = (2, 3, 4)
    restfreq = 1.4 * u.GHz
    bunit = u.K

    hdu = create_fits_hdu(cube, pixel_scale, spec_pixel_scale, beamfwhm, imshape, restfreq, bunit)

    assert hdu.header['NAXIS'] == 3
    assert hdu.header['NAXIS1'] == 4
    assert hdu.header['NAXIS2'] == 3
    assert hdu.header['NAXIS3'] == 2


def test_create_image_hdu():

    img = np.zeros((3, 4))

    pixel_scale = 0.001 * u.deg
    beamfwhm = 0.003 * u.deg
    imshape = (3, 4)
    restfreq = 1.4 * u.GHz
    bunit = u.K

    hdu = create_fits_hdu(img, pixel_scale, beamfwhm, imshape, restfreq, bunit)


    assert hdu.header['NAXIS'] == 2
    assert hdu.header['NAXIS1'] == 4
    assert hdu.header['NAXIS2'] == 3