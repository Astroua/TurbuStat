# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest
import warnings

import numpy as np
import numpy.testing as npt
import astropy.units as u
from astropy.io import fits
from scipy.stats import linregress
import os

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False

from ..statistics import (Bispectrum, Bispectrum_Distance,
                          BiSpectrum, BiSpectrum_Distance)
from ._testing_data import dataset1,\
    dataset2, computed_data, computed_distances
from ..simulator import make_extended


def test_Bispec_method():
    tester = Bispectrum(dataset1["moment0"])
    tester.run(verbose=True, save_name='test.png')
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val'])
    os.system("rm test.png")

    # Test the save and load
    tester.save_results("bispec_output.pkl", keep_data=False)
    saved_tester = Bispectrum.load_results("bispec_output.pkl")

    # Remove the file
    os.remove("bispec_output.pkl")

    assert np.allclose(saved_tester.bicoherence,
                       computed_data['bispec_val'])


def test_Bispec_method_meansub():
    tester = Bispectrum(dataset1["moment0"])
    tester.run(mean_subtract=True)
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val_meansub'])


def test_Bispec_distance():
    tester_dist = \
        Bispectrum_Distance(dataset1["moment0"],
                            dataset2["moment0"])
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    npt.assert_almost_equal(tester_dist.surface_distance,
                            computed_distances['bispec_surface_distance'])

    npt.assert_almost_equal(tester_dist.mean_distance,
                            computed_distances['bispec_mean_distance'])

    # With pre-computed Bispectrum inputs

    tester_dist2 = \
        Bispectrum_Distance(tester_dist.bispec1,
                            tester_dist.bispec2)
    tester_dist2.distance_metric()

    npt.assert_almost_equal(tester_dist2.surface_distance,
                            computed_distances['bispec_surface_distance'])

    npt.assert_almost_equal(tester_dist2.mean_distance,
                            computed_distances['bispec_mean_distance'])

    # With fresh Bispectrum instances
    tester = Bispectrum(dataset1["moment0"])
    tester2 = Bispectrum(dataset2["moment0"])

    tester_dist3 = \
        Bispectrum_Distance(tester,
                            tester2)
    tester_dist3.distance_metric()

    npt.assert_almost_equal(tester_dist3.surface_distance,
                            computed_distances['bispec_surface_distance'])

    npt.assert_almost_equal(tester_dist3.mean_distance,
                            computed_distances['bispec_mean_distance'])


def test_bispec_azimuthal_slicing():

    tester = Bispectrum(dataset1["moment0"])
    tester.run()

    azimuthal_slice = tester.azimuthal_slice(16, 10,
                                             value='bispectrum_logamp',
                                             bin_width=5 * u.deg)

    npt.assert_allclose(azimuthal_slice[16][0],
                        computed_data['bispec_azim_bins'])
    npt.assert_allclose(azimuthal_slice[16][1],
                        computed_data['bispec_azim_vals'])
    npt.assert_allclose(azimuthal_slice[16][2],
                        computed_data['bispec_azim_stds'])


@pytest.mark.parametrize('plaw',
                         [plaw for plaw in [2, 3, 4]])
def test_bispec_radial_slicing(plaw):

    img = make_extended(128, powerlaw=plaw)

    bispec = Bispectrum(fits.PrimaryHDU(img))
    bispec.run(nsamples=100)

    # Extract a radial profile
    rad_prof = bispec.radial_slice(45 * u.deg, 20 * u.deg,
                                   value='bispectrum_logamp',
                                   bin_width=5)

    rad_bins = rad_prof[45][0]
    rad_vals = rad_prof[45][1]

    # Remove empty bins and avoid the increased value at largest wavenumbers.
    mask = np.isfinite(rad_vals) & (rad_bins < 140.)
    # Lack of power at small wavenumber?
    mask = mask & (rad_bins > 3.)

    # Do a quick fit to get the slope and test against the expected value
    out = linregress(np.log10(rad_bins[mask]),
                     rad_vals[mask])

    # Bispectrum is the FFT^3. Since the powerlaw above corresponds to the
    # power-spectrum slopes, we expect the bispectrum slope to be:
    # (powerlaw / 2.) * 3
    # Because of the phase information causing distortions, we're going to be
    # liberal with the allowed range.
    npt.assert_allclose(out.slope, -plaw * 1.5, atol=0.3)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_Bispec_method_fftw():
    tester = Bispectrum(dataset1["moment0"])
    tester.run(use_pyfftw=True, threads=1)
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val'])


def test_BiSpec_deprec():
    '''
    Check for deprecation warnings on old-named classes.
    '''

    with warnings.catch_warnings(record=True) as w:
        bispec = BiSpectrum(dataset1['cube'])

    assert len(w) >= 1

    good_catch = False
    for wn in w:
        if "Use the new 'Bispectrum'" in str(wn.message):
            assert wn.category == Warning
            good_catch = True
            break

    if not good_catch:
        raise ValueError("Did not receive deprecation warning.")

    with warnings.catch_warnings(record=True) as w:
        bispec = BiSpectrum_Distance(dataset1['moment0'],
                                     dataset2['moment0'])

    assert len(w) >= 1

    good_catch = False
    for wn in w:
        if "Use the new 'Bispectrum_Distance'" in str(wn.message):
            assert wn.category == Warning
            good_catch = True
            break

    if not good_catch:
        raise ValueError("Did not receive deprecation warning.")
