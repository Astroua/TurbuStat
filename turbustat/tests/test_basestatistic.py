
import pytest
import astropy.units as u

from turbustat.statistics.base_statistic import BaseStatisticMixIn

from ._testing_data import header


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

    obj.ang_size

    obj.angular_equiv


def test_distance():

    obj = BaseStatisticMixIn()

    obj.header = header
    obj.distance = 1 * u.cm

    obj.distance
    obj.distance_equiv
    obj.distance_size


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

    obj.spectral_size
    obj.spectral_equiv


@pytest.mark.xfail(raises=ValueError)
def test_no_spectral():
    obj = BaseStatisticMixIn()

    from astropy.wcs import WCS

    celestial_header = WCS(header).celestial.to_header()

    obj.header = celestial_header

    obj.spectral_size
