# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy.testing as npt
import numpy as np
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


from ..statistics import VCA, VCA_Distance
from ..statistics.vca_vcs.slice_thickness import spectral_regrid_cube
from ..io.input_base import to_spectral_cube
from ._testing_data import (dataset1, dataset2, computed_data,
                            computed_distances)
from ..simulator import make_extended
from .testing_utilities import assert_between


def test_VCA_method():
    tester = VCA(dataset1["cube"])
    tester.run(verbose=True, save_name='test.png')
    os.system("rm test.png")

    # Test fitting with bootstrapping
    tester.fit_pspec(bootstrap=True)

    npt.assert_allclose(tester.ps1D, computed_data['vca_val'])
    npt.assert_almost_equal(tester.slope, computed_data['vca_slope'],
                            decimal=3)
    npt.assert_almost_equal(tester.slope2D, computed_data['vca_slope2D'],
                            decimal=3)

    # Test loading and saving
    tester.save_results("vca_output.pkl", keep_data=False)

    saved_tester = VCA.load_results("vca_output.pkl")

    # Remove the file
    os.remove("vca_output.pkl")

    npt.assert_allclose(saved_tester.ps1D, computed_data['vca_val'])
    npt.assert_almost_equal(saved_tester.slope, computed_data['vca_slope'],
                            decimal=3)
    npt.assert_almost_equal(saved_tester.slope2D, computed_data['vca_slope2D'],
                            decimal=3)


def test_VCA_method_change_chanwidth():

    orig_width = np.abs(dataset1['cube'][1]["CDELT3"]) * u.m / u.s

    tester = VCA(dataset1["cube"], channel_width=2 * orig_width)

    # Should have 250 channels now
    assert tester.data.shape[0] == 250

    tester.run()


def test_VCA_method_fitlimits():

    distance = 250 * u.pc

    low_cut = 0.02 / u.pix
    high_cut = 0.1 / u.pix

    tester = VCA(dataset1["cube"])
    tester.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.value / (dataset1['cube'][1]['CDELT2'] * u.deg)
    high_cut = high_cut.value / (dataset1['cube'][1]['CDELT2'] * u.deg)

    tester2 = VCA(dataset1["cube"])
    tester2.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    low_cut = low_cut.to(u.rad**-1).value / distance
    high_cut = high_cut.to(u.rad**-1).value / distance

    tester3 = VCA(dataset1["cube"], distance=distance)
    tester3.run(low_cut=low_cut, high_cut=high_cut, fit_2D=False)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_VCA_distance():
    tester_dist = \
        VCA_Distance(dataset1["cube"],
                     dataset2["cube"])
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['vca_distance'])

    # With pre-computed VCA classes as inputs
    tester_dist2 = \
        VCA_Distance(tester_dist.vca1,
                     tester_dist.vca2)
    tester_dist2.distance_metric()

    npt.assert_almost_equal(tester_dist2.distance,
                            computed_distances['vca_distance'])

    # With fresh VCA classes as inputs
    tester = VCA(dataset1["cube"])
    tester2 = VCA(dataset2["cube"])

    tester_dist3 = \
        VCA_Distance(tester,
                     tester2)
    tester_dist3.distance_metric()

    npt.assert_almost_equal(tester_dist3.distance,
                            computed_distances['vca_distance'])


@pytest.mark.parametrize(("regrid_type", "channel_width"),
                         [['downsample', 2 * u.pix], ['downsample', 2],
                          ['downsample', 80.1 * u.m / u.s],
                          ['regrid', 2 * u.pix],
                          ['regrid', 80.1 * u.m / u.s]])
def test_spectral_regrid(regrid_type, channel_width):

    # All of the different choice should regrid the cube to channels of
    # 2 pixels.

    sc_cube = to_spectral_cube(*dataset1['cube'])

    # Regrid to different factors of the channel width
    sc_regrid = spectral_regrid_cube(sc_cube, channel_width,
                                     method=regrid_type)

    assert sc_regrid.shape[0] == sc_cube.shape[0] / 2
    assert sc_regrid.shape[1] == sc_cube.shape[1]
    assert sc_regrid.shape[2] == sc_cube.shape[2]

    npt.assert_allclose(sc_regrid.sum() / float(sc_regrid.size),
                        sc_cube.sum() / float(sc_cube.size),
                        atol=1e-3)

    # Check if flux is conserved. The interpolation built into spectral-cube
    # (right now) will NOT. Avoid testing this until the improved
    # interpolation is added.
    # npt.assert_allclose(sc_pix_regrid.sum() * channel_width.value,
    #                     sc_cube.sum())

    npt.assert_allclose(sc_regrid.header['CDELT3'],
                        sc_cube.header["CDELT3"] * 2,
                        atol=0.2)


@pytest.mark.parametrize(('plaw', 'ellip'),
                         [(plaw, ellip) for plaw in [3, 4]
                          for ellip in [0.2, 0.5, 0.75, 0.9, 1.0]])
def test_vca_azimlimits(plaw, ellip):
    '''
    The slopes with azimuthal constraints should be the same. When elliptical,
    the power will be different along the different directions, but the slope
    should remain the same.
    '''

    imsize = 128
    theta = 0

    nchans = 10
    # Generate a red noise model cube
    cube = np.empty((nchans, imsize, imsize))
    for i in range(nchans):
        cube[i] = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                                theta=theta,
                                return_fft=False)

    # Use large bins to minimize shot noise since the number of samples is
    # limited
    # Also cut-off the largest scale which seems to get skewed up in the
    # power-law image.
    test = VCA(fits.PrimaryHDU(cube))
    test.run(radial_pspec_kwargs={'binsize': 8.,
                                  "theta_0": 0 * u.deg,
                                  "delta_theta": 40 * u.deg},
             fit_2D=False,
             fit_kwargs={'weighted_fit': False},
             low_cut=(4. / imsize) / u.pix)

    test2 = VCA(fits.PrimaryHDU(cube))
    test2.run(radial_pspec_kwargs={'binsize': 8.,
                                   "theta_0": 90 * u.deg,
                                   "delta_theta": 40 * u.deg},
              fit_2D=False,
              fit_kwargs={'weighted_fit': False},
              low_cut=(4. / imsize) / u.pix)

    test3 = VCA(fits.PrimaryHDU(cube))
    test3.run(radial_pspec_kwargs={'binsize': 8.},
              fit_2D=False,
              fit_kwargs={'weighted_fit': False},
              low_cut=(4. / imsize) / u.pix)

    # Ensure slopes are consistent to within 0.1. Shot noise with the
    # limited number of points requires checking within a range.
    npt.assert_allclose(-plaw, test3.slope, rtol=0.02)
    npt.assert_allclose(-plaw, test2.slope, rtol=0.02)
    npt.assert_allclose(-plaw, test.slope, rtol=0.02)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_VCA_method_fftw():
    tester = VCA(dataset1["cube"])
    tester.run(use_pyfftw=True, threads=1)
    npt.assert_allclose(tester.ps1D, computed_data['vca_val'])
    npt.assert_almost_equal(tester.slope, computed_data['vca_slope'],
                            decimal=3)
    npt.assert_almost_equal(tester.slope2D, computed_data['vca_slope2D'],
                            decimal=3)


@pytest.mark.skipif("not RADIO_BEAM_INSTALLED")
def test_VCA_beamcorrect():

    imsize = 128
    theta = 0
    plaw = 3.0
    ellip = 1.0

    beam = Beam(30 * u.arcsec)

    nchans = 10
    # Generate a red noise model cube
    cube = np.empty((nchans, imsize, imsize))
    for i in range(nchans):
        plane = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                              theta=theta,
                              return_fft=False)
        cube[i] = convolve_fft(plane, beam.as_kernel(10 * u.arcsec),
                               boundary='wrap')

    # Generate a header
    hdu = fits.PrimaryHDU(cube)

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

    test = VCA(hdu)
    test.run(beam_correct=True, high_cut=1 / (6 * u.pix),
             fit_2D=False)

    npt.assert_allclose(-plaw, test.slope, rtol=0.02)


@pytest.mark.parametrize(('apod_type'),
                         ['splitcosinebell', 'hanning', 'tukey',
                          'cosinebell'])
def test_VCA_apod_kernel(apod_type):

    imsize = 256
    theta = 0
    plaw = 3.0
    ellip = 1.0

    nchans = 10
    # Generate a red noise model cube
    cube = np.empty((nchans, imsize, imsize))
    for i in range(nchans):
        cube[i] = make_extended(imsize, powerlaw=plaw, ellip=ellip,
                                theta=theta,
                                return_fft=False)

    # Generate a header
    hdu = fits.PrimaryHDU(cube)

    test = VCA(hdu)

    # Effects large scales
    if apod_type == 'cosinebell':
        low_cut = 10**-1.8 / u.pix
    else:
        low_cut = None

    test.run(apodize_kernel=apod_type, alpha=0.3, beta=0.8, fit_2D=False,
             low_cut=low_cut)
    npt.assert_allclose(-plaw, test.slope, rtol=0.02)
