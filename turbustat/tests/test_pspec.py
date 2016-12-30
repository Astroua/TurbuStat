# Licensed under an MIT open source license - see LICENSE


'''
Test functions for PSpec
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import PowerSpectrum, PSpec_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testPSpec(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_PSpec_method(self):
        self.tester = \
            PowerSpectrum(dataset1["moment0"],
                          weights=dataset1["moment0_error"][0] ** 2.)
        self.tester.run()
        npt.assert_allclose(self.tester.ps1D, computed_data['pspec_val'])

    def test_PSpec_distance(self):
        self.tester_dist = \
            PSpec_Distance(dataset1["moment0"],
                           dataset2["moment0"],
                           weights1=dataset1["moment0_error"][0] ** 2.,
                           weights2=dataset2["moment0_error"][0] ** 2.)
        self.tester_dist.distance_metric()

        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['pspec_distance'])


def test_pspec_nonequal_shape():

    mom0_sliced = dataset1["moment0"][0][:16, :]
    mom0_hdr = dataset1["moment0"][1]

    test = PowerSpectrum((mom0_sliced, mom0_hdr)).run()
    test_T = PowerSpectrum((mom0_sliced.T, mom0_hdr)).run()

    npt.assert_almost_equal(test.slope, test_T.slope, decimal=7)
