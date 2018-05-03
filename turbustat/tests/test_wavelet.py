# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
import os

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False

from ..statistics import Wavelet, Wavelet_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_Wavelet_method():
    tester = Wavelet(dataset1["moment0"])
    tester.run()
    npt.assert_almost_equal(tester.values, computed_data['wavelet_val'])

    npt.assert_almost_equal(tester.slope, computed_data['wavelet_slope'])

    # Test loading and saving
    tester.save_results("wave_output.pkl", keep_data=False)

    saved_tester = Wavelet.load_results("wave_output.pkl")

    # Remove the file
    os.remove("wave_output.pkl")

    npt.assert_almost_equal(saved_tester.values, computed_data['wavelet_val'])

    npt.assert_almost_equal(saved_tester.slope, computed_data['wavelet_slope'])


def test_Wavelet_method_withbreak():
    tester = Wavelet(dataset1["moment0"])
    tester.run(xhigh=7 * u.pix, brk=5.5 * u.pix)

    npt.assert_almost_equal(tester.slope, computed_data['wavelet_slope_wbrk'])
    npt.assert_almost_equal(tester.brk.value,
                            computed_data['wavelet_brk_wbrk'])


def test_Wavelet_method_failbreak():
    '''
    Fit a segmented linear model where there isn't a break. Should revert to
    fit without a break.
    '''

    tester = Wavelet(dataset1["moment0"])
    tester.run(xlow=2.7 * u.pix, xhigh=4 * u.pix, brk=3. * u.pix)

    # No break and only 1 slope
    assert tester.brk is None
    assert isinstance(tester.slope, np.float)


def test_Wavelet_method_fitlimits():

    distance = 250 * u.pc

    xlow = 2 * u.pix
    xhigh = 10 * u.pix

    tester = Wavelet(dataset1["moment0"])
    tester.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.value * dataset1['moment0'][1]['CDELT2'] * u.deg
    xhigh = xhigh.value * dataset1['moment0'][1]['CDELT2'] * u.deg

    tester2 = Wavelet(dataset1["moment0"])
    tester2.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.to(u.rad).value * distance
    xhigh = xhigh.to(u.rad).value * distance

    tester3 = Wavelet(dataset1["moment0"], distance=distance)
    tester3.run(xlow=xlow, xhigh=xhigh)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_Wavelet_method_customscales():

    distance = 250 * u.pc

    scales = np.arange(1, 11, 2) * u.pix

    tester = Wavelet(dataset1["moment0"], scales=scales)
    tester.run()

    scales = scales.value * dataset1['moment0'][1]['CDELT2'] * u.deg

    tester2 = Wavelet(dataset1["moment0"], scales=scales)
    tester2.run()

    scales = scales.to(u.rad).value * distance

    tester3 = Wavelet(dataset1["moment0"], scales=scales, distance=distance)
    tester3.run()

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_Wavelet_distance():
    tester_dist = \
        Wavelet_Distance(dataset1["moment0"],
                         dataset2["moment0"]).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['wavelet_distance'])


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_Wavelet_method_fftw():
    tester = Wavelet(dataset1["moment0"])
    tester.run(use_pyfftw=True, threads=1)
    npt.assert_almost_equal(tester.values, computed_data['wavelet_val'])

    npt.assert_almost_equal(tester.slope, computed_data['wavelet_slope'])
