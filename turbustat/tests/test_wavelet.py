# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
import os
from astropy.io import fits

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False

from ..statistics import Wavelet, Wavelet_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances
from ..simulator import make_extended


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


@pytest.mark.parametrize(('plaw', 'ellip'),
                         [(plaw, ellip) for plaw in [2, 3, 4]
                          for ellip in [0.2, 0.75, 1.0]])
def test_wavelet_plaw_img(plaw, ellip):
    '''
    The slopes with azimuthal constraints should be the same. When elliptical,
    the power will be different along the different directions, but the slope
    should remain the same.
    '''

    imsize = 256
    theta = 0

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=False)

    test = Wavelet(fits.PrimaryHDU(img))
    # The turn-over occurs near ~1/16 of the axis size.
    test.run(xhigh=imsize / 16. * u.pix)

    # Ensure slopes are consistent to within 2%
    # Relation to the power law slope is (plaw - 2) / 2.

    # Use an abs difference for cases where plaw - 2. = 0.
    # Wavelets gives some scatter on scales smaller than 16 pix. Allow a
    # reasonable range for the slope.
    npt.assert_allclose(0.5 * (plaw - 2.), test.slope, atol=0.1)
