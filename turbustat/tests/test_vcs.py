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

from ..statistics import VCS, VCS_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_VCS_method():
    tester = VCS(dataset1["cube"]).run(high_cut=0.3 / u.pix,
                                       low_cut=3e-2 / u.pix)

    # Test fitting with bootstrapping
    tester.fit_pspec(bootstrap=True,
                     high_cut=0.3 / u.pix,
                     low_cut=3e-2 / u.pix)

    npt.assert_allclose(tester.ps1D, computed_data['vcs_val'])

    npt.assert_allclose(tester.slope, computed_data['vcs_slopes'])

    # Test loading and saving
    tester.save_results("vcs_output.pkl", keep_data=False)

    saved_tester = VCS.load_results("vcs_output.pkl")

    # Remove the file
    os.remove("vcs_output.pkl")

    npt.assert_allclose(saved_tester.ps1D, computed_data['vcs_val'])

    npt.assert_allclose(saved_tester.slope, computed_data['vcs_slopes'])


def test_VCS_distance():
    tester_dist = \
        VCS_Distance(dataset1["cube"], dataset2["cube"],
                     fit_kwargs=dict(high_cut=0.3 / u.pix,
                                     low_cut=3e-2 / u.pix))
    tester_dist = tester_dist.distance_metric()

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['vcs_distance'])


def test_VCS_method_fitlimits():

    high_cut = 0.17 / u.pix
    low_cut = 0.02 / u.pix

    tester = VCS(dataset1["cube"])
    tester.run(high_cut=high_cut, low_cut=low_cut)

    # Convert to spectral units
    high_cut = \
        high_cut.value / np.abs(dataset1['cube'][1]['CDELT3'] * u.m / u.s)
    low_cut = \
        low_cut.value / np.abs(dataset1['cube'][1]['CDELT3'] * u.m / u.s)
    tester2 = VCS(dataset1["cube"])
    tester2.run(high_cut=high_cut, low_cut=low_cut)

    npt.assert_allclose(tester.slope, tester2.slope, atol=0.02)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_VCS_method_fftw():
    tester = VCS(dataset1["cube"]).run(high_cut=0.3 / u.pix,
                                       low_cut=3e-2 / u.pix,
                                       use_pyfftw=True,
                                       threads=1)

    npt.assert_allclose(tester.ps1D, computed_data['vcs_val'])

    npt.assert_allclose(tester.slope, computed_data['vcs_slopes'])
