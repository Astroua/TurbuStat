# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy.testing as npt
import astropy.units as u
import os
from astropy.io import fits

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False

from ..statistics import DeltaVariance, DeltaVariance_Distance
from ..simulator import make_extended
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_DelVar_method():
    # There is a difference in the convolution of astropy 1.x and 2.x on the
    # large-scales. Restrict the fitting region to where the convolution
    # agrees.
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1["moment0_error"][0])
    tester.run(xhigh=11. * u.pix, nan_treatment='interpolate',
               verbose=True, save_name='test.png')
    os.system("rm test.png")

    # Run the fit again with bootstrap resampling
    tester.fit_plaw(bootstrap=True, xhigh=11. * u.pix)

    # The slice is again to restrict where the convolution functions both give
    # the same value
    npt.assert_allclose(tester.delta_var[:-7],
                        computed_data['delvar_val'][:-7])
    npt.assert_almost_equal(tester.slope, computed_data['delvar_slope'])

    # Test the save and load
    tester.save_results("delvar_output.pkl", keep_data=False)
    saved_tester = DeltaVariance.load_results("delvar_output.pkl")

    # Remove the file
    os.remove("delvar_output.pkl")

    npt.assert_allclose(saved_tester.delta_var[:-7],
                        computed_data['delvar_val'][:-7])
    npt.assert_almost_equal(saved_tester.slope, computed_data['delvar_slope'])


def test_DelVar_method_fill():
    # There is a difference in the convolution of astropy 1.x and 2.x on the
    # large-scales. Restrict the fitting region to where the convolution
    # agrees.
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1["moment0_error"][0])
    tester.run(boundary='fill', xhigh=11. * u.pix, nan_treatment='interpolate')
    # The slice is again to restrict where the convolution functions both give
    # the same value
    npt.assert_allclose(tester.delta_var[:-7],
                        computed_data['delvar_fill_val'][:-7])
    npt.assert_almost_equal(tester.slope, computed_data['delvar_fill_slope'])


def test_DelVar_method_fitlimits():

    distance = 250 * u.pc

    xlow = 4 * u.pix
    xhigh = 30 * u.pix

    tester = DeltaVariance(dataset1["moment0"])
    tester.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.value * (dataset1['moment0'][1]['CDELT2'] * u.deg)
    xhigh = xhigh.value * (dataset1['moment0'][1]['CDELT2'] * u.deg)

    tester2 = DeltaVariance(dataset1["moment0"])
    tester2.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.to(u.rad).value * distance
    xhigh = xhigh.to(u.rad).value * distance

    tester3 = DeltaVariance(dataset1["moment0"], distance=distance)
    tester3.run(xlow=xlow, xhigh=xhigh)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_DelVar_method_wbrk():

    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1['moment0_error'][0])
    tester.run(xhigh=11 * u.pix, brk=6 * u.pix)

    npt.assert_almost_equal(tester.slope, computed_data['delvar_slope_wbrk'])
    npt.assert_almost_equal(tester.brk.value, computed_data['delvar_brk'])

    # Make sure the break fits have correct units
    assert tester.brk.unit == u.pix
    assert tester.brk_err.unit == u.pix


def test_DelVar_distance():
    tester_dist = \
        DeltaVariance_Distance(dataset1["moment0"],
                               dataset2["moment0"],
                               weights1=dataset1["moment0_error"][0],
                               weights2=dataset2["moment0_error"][0],
                               delvar_kwargs=dict(xhigh=11 * u.pix))
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.curve_distance,
                            computed_distances['delvar_curve_distance'],
                            decimal=3)
    npt.assert_almost_equal(tester_dist.slope_distance,
                            computed_distances['delvar_slope_distance'],
                            decimal=3)

    # Try distance metric when giving DeltaVariance classes as inputs
    tester_dist2 = \
        DeltaVariance_Distance(tester_dist.delvar1,
                               tester_dist.delvar2)
    tester_dist2.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist2.curve_distance,
                            computed_distances['delvar_curve_distance'],
                            decimal=3)
    npt.assert_almost_equal(tester_dist2.slope_distance,
                            computed_distances['delvar_slope_distance'],
                            decimal=3)

    # And test when given DeltaVariance classes that haven't been run yet.
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1['moment0_error'][0])
    tester2 = \
        DeltaVariance(dataset2["moment0"],
                      weights=dataset2['moment0_error'][0])

    tester_dist3 = \
        DeltaVariance_Distance(tester, tester2,
                               delvar_kwargs=dict(xhigh=11 * u.pix))
    tester_dist3.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist3.curve_distance,
                            computed_distances['delvar_curve_distance'],
                            decimal=3)
    npt.assert_almost_equal(tester_dist3.slope_distance,
                            computed_distances['delvar_slope_distance'],
                            decimal=3)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_DelVar_method_fftw():
    # There is a difference in the convolution of astropy 1.x and 2.x on the
    # large-scales. Restrict the fitting region to where the convolution
    # agrees.
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1["moment0_error"][0])
    tester.run(xhigh=11. * u.pix, use_pyfftw=True, threads=1)
    # The slice is again to restrict where the convolution functions both give
    # the same value
    npt.assert_allclose(tester.delta_var[:-7],
                        computed_data['delvar_val'][:-7])
    npt.assert_almost_equal(tester.slope, computed_data['delvar_slope'])


@pytest.mark.parametrize(('plaw', 'ellip'),
                         [(plaw, ellip) for plaw in [2, 3, 4]
                          for ellip in [0.2, 0.75, 1.0]])
def test_delvar_plaw_img(plaw, ellip):
    '''
    The slopes with azimuthal constraints should be the same. When elliptical,
    the power will be different along the different directions, but the slope
    should remain the same.
    '''

    imsize = 128
    theta = 0

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=False)

    test = DeltaVariance(fits.PrimaryHDU(img))
    test.run(xhigh=imsize / 4. * u.pix)

    # Ensure slopes are consistent to within 2%
    # Relation to the power law slope is plaw - 2
    # Highly elliptical structure (0.2) leads to ~3% deviations

    npt.assert_allclose(plaw, test.slope + 2., rtol=0.04)
