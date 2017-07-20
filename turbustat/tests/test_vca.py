# Licensed under an MIT open source license - see LICENSE

'''
Test functions for VCA
'''

import pytest

import numpy.testing as npt
import numpy as np
import astropy.units as u

from ..statistics import VCA, VCA_Distance
from ..statistics.vca_vcs.slice_thickness import spectral_regrid_cube
from ..io.input_base import to_spectral_cube
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_VCA_method():
    tester = VCA(dataset1["cube"])
    tester.run()
    npt.assert_allclose(tester.ps1D, computed_data['vca_val'])
    npt.assert_almost_equal(tester.slope, computed_data['vca_slope'],
                            decimal=3)


def test_VCA_distance():
    tester_dist = \
        VCA_Distance(dataset1["cube"],
                     dataset2["cube"]).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['vca_distance'])


@pytest.mark.parametrize("channel_width",
                         [2, 5, 10, 25] * u.pix)
def test_spectral_regrid(channel_width):

    sc_cube = to_spectral_cube(*dataset1['cube'])

    # Convert those pixel widths into spectral units
    spec_width = np.abs(sc_cube.header["CDELT3"]) * \
        channel_width.value * u.m / u.s

    # Regrid to different factors of the channel width
    sc_pix_regrid = spectral_regrid_cube(sc_cube, channel_width)
    sc_spec_regrid = spectral_regrid_cube(sc_cube, spec_width)

    assert sc_pix_regrid.shape == sc_spec_regrid.shape

    npt.assert_allclose(sc_pix_regrid.sum(), sc_spec_regrid.sum())

    # Check if flux is conserved. The interpolation built into spectral-cube
    # (right now) will NOT. Avoid testing this until the improved
    # interpolation is added.
    # npt.assert_allclose(sc_pix_regrid.sum() * channel_width.value,
    #                     sc_cube.sum())

    npt.assert_allclose(sc_pix_regrid.header['CDELT3'],
                        sc_spec_regrid.header["CDELT3"])
