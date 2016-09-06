# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Tsallis
'''

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from ..statistics import Tsallis, Tsallis_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


class testTsallis(TestCase):

    def setUp(self):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def test_Tsallis_method(self):
        self.tester = Tsallis(dataset1["moment0"],
                              lags=[1, 2, 4, 8, 16], num_bins=100)
        self.tester.run()
        npt.assert_allclose(self.tester.tsallis_fits,
                            computed_data['tsallis_val'], atol=0.01)

    def test_Tsallis_distance(self):
        self.tester_dist = \
            Tsallis_Distance(dataset1["moment0"],
                             dataset2["moment0"],
                             lags=[1, 2, 4, 8, 16],
                             num_bins=100).distance_metric()
        npt.assert_almost_equal(self.tester_dist.distance,
                                computed_distances['tsallis_distance'],
                                decimal=4)
