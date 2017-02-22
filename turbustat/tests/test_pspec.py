# Licensed under an MIT open source license - see LICENSE


'''
Test functions for PSpec
'''

import numpy.testing as npt

from ..statistics import PowerSpectrum, PSpec_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_PSpec_method():
    tester = \
        PowerSpectrum(dataset1["moment0"])
    tester.run()
    npt.assert_allclose(tester.ps1D, computed_data['pspec_val'])


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
