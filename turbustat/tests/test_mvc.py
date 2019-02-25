# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy.testing as npt
import astropy.units as u
from astropy.io import fits
from astropy.convolution import convolve_fft
import pytest
import numpy as np
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

from ..statistics import MVC, MVC_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances
from ..simulator import make_extended
from .testing_utilities import assert_between


def test_MVC_method():
    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester.run(verbose=True, save_name='test.png')
    os.system("rm test.png")

    # Test fit with bootstrap resampling
    tester.fit_pspec(bootstrap=True)

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
        MVC_Distance(dataset1, dataset2)
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    # Tiny discrepancy introduced when using rfft_to_fft instead of np.fft.fft2
    npt.assert_allclose(tester_dist.distance,
                        computed_distances['mvc_distance'],
                        rtol=2e-3)

    # Test with MVC classes as inputs
    tester_dist2 = \
        MVC_Distance(tester_dist.mvc1, tester_dist.mvc2)
    tester_dist2.distance_metric(verbose=False)

    # Tiny discrepancy introduced when using rfft_to_fft instead of np.fft.fft2
    npt.assert_allclose(tester_dist2.distance,
                        computed_distances['mvc_distance'],
                        rtol=2e-3)

    # Now from fresh MVC classes that need to be computed
    tester = MVC(dataset1["centroid"],
                 dataset1["moment0"],
                 dataset1["linewidth"],
                 dataset1["centroid"][1])
    tester2 = MVC(dataset2["centroid"],
                  dataset2["moment0"],
                  dataset2["linewidth"],
                  dataset2["centroid"][1])
    tester_dist3 = \
        MVC_Distance(tester, tester2)
    tester_dist3.distance_metric(verbose=False)

    # Tiny discrepancy introduced when using rfft_to_fft instead of np.fft.fft2
    npt.assert_allclose(tester_dist3.distance,
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

    imsize = 128
    theta = 0

    # Generate a red noise model
    img = make_extended(imsize, powerlaw=plaw, ellip=ellip, theta=theta,
                        return_fft=False)

    ones = np.ones_like(img)

    test = MVC(fits.PrimaryHDU(img), fits.PrimaryHDU(ones),
               fits.PrimaryHDU(ones))
    test.run(radial_pspec_kwargs={"theta_0": 0 * u.deg,
                                  "delta_theta": 40 * u.deg},
             fit_2D=False,
             fit_kwargs={'weighted_fit': False})

    test2 = MVC(fits.PrimaryHDU(img), fits.PrimaryHDU(ones),
                fits.PrimaryHDU(ones))
    test2.run(radial_pspec_kwargs={"theta_0": 90 * u.deg,
                                   "delta_theta": 40 * u.deg},
              fit_2D=False,
              fit_kwargs={'weighted_fit': False})

    test3 = MVC(fits.PrimaryHDU(img), fits.PrimaryHDU(ones),
                fits.PrimaryHDU(ones))
    test3.run(radial_pspec_kwargs={},
              fit_2D=False,
              fit_kwargs={'weighted_fit': False})

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


@pytest.mark.skipif("not RADIO_BEAM_INSTALLED")
def test_MVC_beamcorrect():

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

    ones = np.ones_like(plane)

    test = MVC(hdu, fits.PrimaryHDU(ones),
               fits.PrimaryHDU(ones))

    test.run(beam_correct=True,
             low_cut=10**-1.5 / u.pix,
             high_cut=1 / (6 * u.pix),
             fit_2D=False)

    npt.assert_allclose(-plaw, test.slope, rtol=0.02)


@pytest.mark.parametrize(('apod_type'),
                         ['splitcosinebell', 'hanning', 'tukey',
                          'cosinebell'])
def test_MVC_apod_kernel(apod_type):

    imsize = 256
    theta = 0
    plaw = 3.0
    ellip = 1.0

    plane = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                          theta=theta,
                          return_fft=False)

    # Generate a header
    hdu = fits.PrimaryHDU(plane)

    ones = np.ones_like(plane)

    test = MVC(hdu, fits.PrimaryHDU(ones),
               fits.PrimaryHDU(ones))

    # Avoid shot noise scatter at large scales
    if apod_type == 'cosinebell':
        low_cut = 10**-0.8 / u.pix
    else:
        low_cut = 10**-1.4 / u.pix

    test.run(apodize_kernel=apod_type, alpha=0.3, beta=0.8, fit_2D=False,
             low_cut=low_cut)
    npt.assert_allclose(-plaw, test.slope, rtol=0.02)
