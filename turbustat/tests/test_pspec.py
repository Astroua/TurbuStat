# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import numpy.testing as npt
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve_fft
import os

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False

try:
    from radio_beam.beam import Beam
    RADIO_BEAM_INSTALLED = True
except ImportError:
    RADIO_BEAM_INSTALLED = False

from ..statistics import PowerSpectrum, PSpec_Distance
from ._testing_data import (dataset1, dataset2, computed_data,
                            computed_distances)
from ..simulator import make_extended


def test_PSpec_method():
    tester = \
        PowerSpectrum(dataset1["moment0"])
    tester.run(verbose=True, save_name='test.png')
    os.system("rm test.png")

    # Test fitting with bootstrapping
    tester.fit_pspec(bootstrap=True)

    npt.assert_allclose(tester.ps1D, computed_data['pspec_val'])
    npt.assert_allclose(tester.slope, computed_data['pspec_slope'])
    npt.assert_allclose(tester.slope2D, computed_data['pspec_slope2D'])

    # Test loading and saving
    tester.save_results("pspec_output.pkl", keep_data=False)

    saved_tester = PowerSpectrum.load_results("pspec_output.pkl")

    # Remove the file
    os.remove("pspec_output.pkl")

    npt.assert_allclose(saved_tester.ps1D, computed_data['pspec_val'])
    npt.assert_allclose(saved_tester.slope, computed_data['pspec_slope'])
    npt.assert_allclose(saved_tester.slope2D, computed_data['pspec_slope2D'])


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
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['pspec_distance'])

    # With pre-computed pspec

    tester_dist2 = \
        PSpec_Distance(tester_dist.pspec1,
                       tester_dist.pspec2)
    tester_dist2.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist2.distance,
                            computed_distances['pspec_distance'])

    # With fresh pspec instances
    tester = PowerSpectrum(dataset1["moment0"])
    tester2 = PowerSpectrum(dataset2["moment0"])

    tester_dist3 = \
        PSpec_Distance(tester,
                       tester2)
    tester_dist3.distance_metric(verbose=False)

    npt.assert_almost_equal(tester_dist3.distance,
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

    imsize = 128
    theta = 0

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=False)

    test = PowerSpectrum(fits.PrimaryHDU(img))
    test.run(radial_pspec_kwargs={"theta_0": 0 * u.deg,
                                  "delta_theta": 40 * u.deg},
             fit_kwargs={'weighted_fit': False},
             fit_2D=False)

    test2 = PowerSpectrum(fits.PrimaryHDU(img))
    test2.run(radial_pspec_kwargs={"theta_0": 90 * u.deg,
                                   "delta_theta": 40 * u.deg},
              fit_kwargs={'weighted_fit': False},
              fit_2D=False)

    test3 = PowerSpectrum(fits.PrimaryHDU(img))
    test3.run(radial_pspec_kwargs={},
              fit_kwargs={'weighted_fit': False},
              fit_2D=False)

    # Ensure slopes are consistent to within 2%
    npt.assert_allclose(-plaw, test3.slope, rtol=0.02)
    npt.assert_allclose(-plaw, test2.slope, rtol=0.02)
    npt.assert_allclose(-plaw, test.slope, rtol=0.02)


@pytest.mark.parametrize('plaw', [2, 3, 4])
def test_pspec_weightfit(plaw):
    '''
    The slopes with azimuthal constraints should be the same. When elliptical,
    the power will be different along the different directions, but the slope
    should remain the same.
    '''

    imsize = 64
    theta = 0

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=1., theta=theta,
                        return_fft=False)

    test = PowerSpectrum(fits.PrimaryHDU(img))
    test.run(fit_kwargs={'weighted_fit': True},
             fit_2D=False)

    # Ensure slopes are consistent to within 2%
    npt.assert_allclose(-plaw, test.slope, rtol=0.02)


@pytest.mark.parametrize('theta',
                         [0., np.pi / 4., np.pi / 2., 7 * np.pi / 8.])
def test_pspec_fit2D(theta):
    '''
    Since test_elliplaw tests everything, only check for consistent theta
    here.
    '''

    imsize = 64
    ellip = 0.5
    plaw = 4.

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=False)

    test = PowerSpectrum(fits.PrimaryHDU(img))
    test.run(fit_2D=True)

    try:
        npt.assert_allclose(theta, test.theta2D,
                            atol=0.08)
    except AssertionError:
        npt.assert_allclose(theta, test.theta2D - np.pi,
                            atol=0.08)

    npt.assert_allclose(-plaw, test.slope2D, rtol=0.02)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_PSpec_method_fftw():
    tester = \
        PowerSpectrum(dataset1["moment0"])
    tester.run(use_pyfftw=True, threads=1)
    npt.assert_allclose(tester.ps1D, computed_data['pspec_val'])
    npt.assert_allclose(tester.slope, computed_data['pspec_slope'])
    npt.assert_allclose(tester.slope2D, computed_data['pspec_slope2D'])


@pytest.mark.skipif("not RADIO_BEAM_INSTALLED")
def test_PSpec_beamcorrect():

    imsize = 128
    theta = 0
    plaw = 3.0
    ellip = 1.0

    beam = Beam(30 * u.arcsec)

    plane = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                          theta=theta,
                          return_fft=False)
    plane = convolve_fft(plane, beam.as_kernel(10 * u.arcsec),
                         boundary='wrap')

    # Generate a header
    hdu = fits.PrimaryHDU(plane)

    hdu.header['CDELT1'] = (10 * u.arcsec).to(u.deg).value
    hdu.header['CDELT2'] = - (10 * u.arcsec).to(u.deg).value
    hdu.header['BMAJ'] = beam.major.to(u.deg).value
    hdu.header['BMIN'] = beam.major.to(u.deg).value
    hdu.header['BPA'] = 0.0
    hdu.header['CRPIX1'] = imsize / 2.,
    hdu.header['CRPIX2'] = imsize / 2.,
    hdu.header['CRVAL1'] = 0.0,
    hdu.header['CRVAL2'] = 0.0,
    hdu.header['CTYPE1'] = 'GLON-CAR',
    hdu.header['CTYPE2'] = 'GLAT-CAR',
    hdu.header['CUNIT1'] = 'deg',
    hdu.header['CUNIT2'] = 'deg',

    hdu.header.update(beam.to_header_keywords())

    test = PowerSpectrum(hdu)
    test.run(beam_correct=True,
             low_cut=10**-1.5 / u.pix,
             high_cut=1 / (6 * u.pix),
             fit_2D=False)

    npt.assert_allclose(-plaw, test.slope, rtol=0.02)


@pytest.mark.parametrize(('apod_type'),
                         ['splitcosinebell', 'hanning', 'tukey',
                          'cosinebell'])
def test_PSpec_apod_kernel(apod_type):

    imsize = 256
    theta = 0
    plaw = 3.0
    ellip = 1.0

    plane = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                          theta=theta,
                          return_fft=False)

    # Generate a header
    hdu = fits.PrimaryHDU(plane)

    test = PowerSpectrum(hdu)

    # Effects large scales
    low_cut = 10**-1.8 / u.pix

    test.run(apodize_kernel=apod_type, alpha=0.3, beta=0.8, fit_2D=False,
             low_cut=low_cut, verbose=False)

    npt.assert_allclose(-plaw, test.slope, rtol=0.02)
