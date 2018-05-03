# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy.testing as npt
import astropy.units as u
from astropy.io import fits
import pytest
import numpy as np
import os

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False

from ..statistics import MVC, MVC_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances, make_extended, assert_between


def test_MVC_method():
    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run()

    # Tiny discrepancy introduced when using rfft_to_fft instead of np.fft.fft2
    npt.assert_allclose(tester.ps1D, computed_data['mvc_val'], rtol=1e-3)
    npt.assert_allclose(tester.slope, computed_data['mvc_slope'], rtol=1e-4)
    npt.assert_allclose(tester.slope2D, computed_data['mvc_slope2D'],
                        rtol=1e-4)

    # Test loading and saving
    tester.save_results("mvc_output.pkl", keep_data=False)

    saved_tester = MVC.load_results("mvc_output.pkl")

    # Remove the file
    os.remove("mvc_output.pkl")

    npt.assert_allclose(saved_tester.ps1D, computed_data['mvc_val'], rtol=1e-3)
    npt.assert_allclose(saved_tester.slope, computed_data['mvc_slope'],
                        rtol=1e-4)
    npt.assert_allclose(saved_tester.slope2D, computed_data['mvc_slope2D'],
                        rtol=1e-4)

def test_MVC_method_fitlimits():

    distance = 250 * u.pc

    low_cut = 0.02 / u.pix
    high_cut = 0.1 / u.pix

    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.value / (dataset1['cube'][1]['CDELT2'] * u.deg)
    high_cut = high_cut.value / (dataset1['cube'][1]['CDELT2'] * u.deg)

    tester2 = MVC(dataset1["centroid"],
                  dataset1["moment0"],
                  dataset1["linewidth"],
                  dataset1["centroid"][1])
    tester2.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.to(u.rad**-1).value / distance
    high_cut = high_cut.to(u.rad**-1).value / distance

    tester3 = MVC(dataset1["centroid"],
                  dataset1["moment0"],
                  dataset1["linewidth"],
                  dataset1["centroid"][1], distance=distance)
    tester3.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_MVC_distance():
    tester_dist = \
        MVC_Distance(dataset1, dataset2).distance_metric()
    # Tiny discrepancy introduced when using rfft_to_fft instead of np.fft.fft2
    npt.assert_allclose(tester_dist.distance,
                        computed_distances['mvc_distance'],
                        rtol=2e-3)


@pytest.mark.parametrize(('plaw', 'ellip'),
                         [(plaw, ellip) for plaw in [3, 4]
                          for ellip in [0.2, 0.5, 0.75, 0.9, 1.0]])
def test_mvc_azimlimits(plaw, ellip):
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

    ones = np.ones_like(img)

    test = MVC(fits.PrimaryHDU(img), fits.PrimaryHDU(ones),
               fits.PrimaryHDU(ones))
    test.run(radial_pspec_kwargs={"theta_0": 0 * u.deg,
                                  "delta_theta": 40 * u.deg},
             fit_2D=False,
             fit_kwargs={'weighted_fit': True})

    test2 = MVC(fits.PrimaryHDU(img), fits.PrimaryHDU(ones),
                fits.PrimaryHDU(ones))
    test2.run(radial_pspec_kwargs={"theta_0": 90 * u.deg,
                                   "delta_theta": 40 * u.deg},
              fit_2D=False,
              fit_kwargs={'weighted_fit': True})

    test3 = MVC(fits.PrimaryHDU(img), fits.PrimaryHDU(ones),
                fits.PrimaryHDU(ones))
    test3.run(radial_pspec_kwargs={},
              fit_2D=False,
              fit_kwargs={'weighted_fit': True})

    # Ensure slopes are consistent to within 7%
    assert_between(test3.slope, - 1.07 * plaw, - 0.93 * plaw)
    assert_between(test2.slope, - 1.07 * plaw, - 0.93 * plaw)
    assert_between(test.slope, - 1.07 * plaw, - 0.93 * plaw)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_MVC_method_fftw():
    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run(use_pyfftw=True, threads=1)

    # Tiny discrepancy introduced when using rfft_to_fft instead of np.fft.fft2
    npt.assert_allclose(tester.ps1D, computed_data['mvc_val'], rtol=1e-3)
    npt.assert_allclose(tester.slope, computed_data['mvc_slope'], rtol=1e-4)
    npt.assert_allclose(tester.slope2D, computed_data['mvc_slope2D'],
                        rtol=1e-4)
