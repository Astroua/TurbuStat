# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest
import astropy.units as u
import numpy.testing as npt
import numpy as np

from turbustat.statistics.base_statistic import BaseStatisticMixIn

from ._testing_data import header
from .helpers import assert_allclose


@pytest.mark.xfail(raises=AttributeError)
def test_no_header():

    obj = BaseStatisticMixIn()

    obj.header


def test_header():

    obj = BaseStatisticMixIn()

    obj.header = header

    obj.header


def test_angles():

    obj = BaseStatisticMixIn()

    obj.header = header

    obj._ang_size

    obj._angular_equiv


def test_distance():

    obj = BaseStatisticMixIn()

    obj.header = header
    obj.distance = 1 * u.cm

    obj.distance
    obj._physical_equiv
    obj._physical_size


@pytest.mark.xfail(raises=AttributeError)
def test_distance_no_header():

    obj = BaseStatisticMixIn()

    obj.distance = 1 * u.cm

    obj.distance


@pytest.mark.xfail(raises=AttributeError)
def test_no_distance():

    obj = BaseStatisticMixIn()

    obj.distance


@pytest.mark.xfail(raises=u.UnitConversionError)
def test_bad_distance_unit():

    obj = BaseStatisticMixIn()

    obj.header = header
    obj.distance = 1 * u.K


def test_spectral():
    obj = BaseStatisticMixIn()

    obj.header = header

    obj._spectral_size
    obj._spectral_equiv


@pytest.mark.xfail(raises=ValueError)
def test_no_spectral():
    obj = BaseStatisticMixIn()

    from astropy.wcs import WCS

    celestial_header = WCS(header).celestial.to_header()

    obj.header = celestial_header

    obj._spectral_size


def test_pixel_to_spatial_conversions():

    obj = BaseStatisticMixIn()

    obj.header = header
    obj.distance = 1 * u.pc

    assert_allclose(obj._to_angular(1 * u.pix, u.deg),
                    header["CDELT2"] * u.deg)

    assert_allclose(obj._to_physical(1 * u.pix, u.pc),
                    (header["CDELT2"] * np.pi / 180.) * u.pc)

    assert_allclose(obj._spatial_unit_conversion(1 * u.pix, u.deg),
                    header["CDELT2"] * u.deg)

    assert_allclose(obj._spatial_unit_conversion(1 * u.pix, u.pc),
                    (header["CDELT2"] * np.pi / 180.) * u.pc)

    assert_allclose(obj._to_pixel(header["CDELT2"] * u.deg),
                    1.0 * u.pix)

    assert_allclose(obj._to_pixel((header["CDELT2"] * np.pi / 180.) * u.pc),
                    1.0 * u.pix)

    # Now area conversions
    assert_allclose(obj._to_pixel_area((header["CDELT2"] * u.deg)**2),
                    1.0 * u.pix**2)

    assert_allclose(obj._to_pixel_area(((header["CDELT2"] * np.pi / 180.) * u.pc)**2),
                    1.0 * u.pix**2)


def test_pixel_to_spatial_freq_conversions():

    obj = BaseStatisticMixIn()

    obj.header = header
    obj.distance = 1 * u.pc

    assert_allclose(obj._spatial_freq_unit_conversion(1 / u.pix, 1 / u.deg),
                    1 / (header["CDELT2"] * u.deg))

    assert_allclose(obj._spatial_freq_unit_conversion(1 / u.pix, 1 / u.pc),
                    1 / ((header["CDELT2"] * np.pi / 180.) * u.pc))

    assert_allclose(obj._to_pixel_freq(1 / (header["CDELT2"] * u.deg)),
                    1 / u.pix)

    assert_allclose(obj._to_pixel_freq(1 / ((header["CDELT2"] * np.pi / 180.) * u.pc)),
                    1 / u.pix)


def test_pixel_to_spectral_conversions():

    obj = BaseStatisticMixIn()

    obj.header = header

    assert_allclose(obj._to_spectral(1 * u.pix, u.m / u.s),
                    np.abs(header["CDELT3"]) * u.m / u.s)

    assert_allclose(obj._to_spectral(np.abs(header["CDELT3"]) * u.m / u.s, u.pix),
                    1 * u.pix)


def test_pixel_to_spectral_freq_conversions():

    obj = BaseStatisticMixIn()

    obj.header = header

    assert_allclose(1 / obj._spectral_freq_unit_conversion(1 / u.pix, u.s / u.m),
                    np.abs(header["CDELT3"]) * u.m / u.s)

    assert_allclose(obj._spectral_freq_unit_conversion( 1/ (np.abs(header["CDELT3"]) * u.m / u.s), 1 / u.pix),
                    1 / u.pix)
