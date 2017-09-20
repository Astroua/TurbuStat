# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
from astropy.io import fits
from scipy.stats import linregress

from ..statistics import BiSpectrum, BiSpectrum_Distance
from ._testing_data import dataset1,\
    dataset2, computed_data, computed_distances, make_extended


def test_Bispec_method():
    tester = BiSpectrum(dataset1["moment0"])
    tester.run()
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val'])


def test_Bispec_method_meansub():
    tester = BiSpectrum(dataset1["moment0"])
    tester.run(mean_subtract=True)
    assert np.allclose(tester.bicoherence,
                       computed_data['bispec_val_meansub'])


def test_Bispec_distance():
    tester_dist = \
        BiSpectrum_Distance(dataset1["moment0"],
                            dataset2["moment0"])
    tester_dist.distance_metric()

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['bispec_distance'])


def test_bispec_azimuthal_slicing():

    tester = BiSpectrum(dataset1["moment0"])
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

    img = make_extended(256, powerlaw=plaw)

    bispec = BiSpectrum(fits.PrimaryHDU(img))
    bispec.run(nsamples=100)

    # Extract a radial profile
    rad_prof = bispec.radial_slice(45 * u.deg, 20 * u.deg,
                                   value='bispectrum_logamp',
                                   bin_width=5)

    rad_bins = rad_prof[45][0]
    rad_vals = rad_prof[45][1]

    # Remove empty bins and avoid the increased value at largest wavenumbers.
    mask = np.isfinite(rad_vals) & (np.log10(rad_bins) < 2.)

    # Do a quick fit to get the slope and test against the expected value
    out = linregress(np.log10(rad_bins[mask]),
                     rad_vals[mask])

    # Bispectrum is the FFT^3. Since the powerlaw above corresponds to the
    # power-spectrum slopes, we expect the bispectrum slope to be:
    # (powerlaw / 2.) * 3
    # Because of the phase information causing distortions, we're going to be
    # liberal with the allowed range.
    npt.assert_allclose(out.slope, -plaw * 1.5, atol=0.3)
