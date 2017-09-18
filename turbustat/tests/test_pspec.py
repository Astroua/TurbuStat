# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest
import numpy.testing as npt
import astropy.units as u
from astropy.io import fits

from ..statistics import PowerSpectrum, PSpec_Distance
from ._testing_data import (dataset1, dataset2, computed_data,
                            computed_distances, make_extended, assert_between)


def test_PSpec_method():
    tester = \
        PowerSpectrum(dataset1["moment0"])
    tester.run()
    npt.assert_allclose(tester.ps1D, computed_data['pspec_val'])
    npt.assert_allclose(tester.slope, computed_data['pspec_slope'])
    npt.assert_allclose(tester.slope2D, computed_data['pspec_slope2D'])


def test_Pspec_method_fitlimits():

    distance = 250 * u.pc

    low_cut = 0.02 / u.pix
    high_cut = 0.1 / u.pix

    tester = PowerSpectrum(dataset1["moment0"])
    tester.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.value / (dataset1['moment0'][1]['CDELT2'] * u.deg)
    high_cut = high_cut.value / (dataset1['moment0'][1]['CDELT2'] * u.deg)

    tester2 = PowerSpectrum(dataset1["moment0"])
    tester2.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.to(u.rad**-1).value / distance
    high_cut = high_cut.to(u.rad**-1).value / distance

    tester3 = PowerSpectrum(dataset1["moment0"], distance=distance)
    tester3.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_PSpec_distance():
    tester_dist = \
        PSpec_Distance(dataset1["moment0"],
                       dataset2["moment0"])
    tester_dist.distance_metric()

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['pspec_distance'])


def test_pspec_nonequal_shape():

    mom0_sliced = dataset1["moment0"][0][:16, :]
    mom0_hdr = dataset1["moment0"][1]

    test = PowerSpectrum((mom0_sliced, mom0_hdr)).run()
    test_T = PowerSpectrum((mom0_sliced.T, mom0_hdr)).run()

    npt.assert_almost_equal(test.slope, test_T.slope, decimal=7)
    npt.assert_almost_equal(test.slope2D, test_T.slope2D, decimal=3)


@pytest.mark.parametrize(('plaw', 'ellip'),
                         [(plaw, ellip) for plaw in [3, 4]
                          for ellip in [0.2, 0.5, 0.75, 0.9, 1.0]])
def test_pspec_azimlimits(plaw, ellip):
    '''
    The slopes with azimuthal constraints should be the same. When elliptical,
    the power will be different along the different directions, but the slope
    should remain the same.
    '''

    imsize = 256
    theta = 0

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_psd=False)

    test = PowerSpectrum(fits.PrimaryHDU(img))
    test.run(radial_pspec_kwargs={"theta_0": 0 * u.deg,
                                  "delta_theta": 30 * u.deg},
             fit_2D=False, weighted_fit=True)

    test2 = PowerSpectrum(fits.PrimaryHDU(img))
    test2.run(radial_pspec_kwargs={"theta_0": 90 * u.deg,
                                   "delta_theta": 30 * u.deg},
              fit_2D=False, weighted_fit=True)

    test3 = PowerSpectrum(fits.PrimaryHDU(img))
    test3.run(radial_pspec_kwargs={},
              fit_2D=False, weighted_fit=True)

    # Ensure slopes are consistent to within 5%
    assert_between(test3.slope, - 1.05 * plaw, - 0.95 * plaw)
    assert_between(test2.slope, - 1.05 * plaw, - 0.95 * plaw)
    assert_between(test.slope, - 1.05 * plaw, - 0.95 * plaw)
